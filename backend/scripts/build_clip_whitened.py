"""
Build the PCA-whitening transform for CLIP image embeddings and populate the
gold_ornaments_clip_w collection that live image search actually queries.

Why this exists: raw CLIP embeddings live in a narrow cone (a well-known
property of contrastively-trained vision-language models), so cosine
similarity between two UNRELATED jewelry photos is already high (~0.6-0.8+)
and similarity between near-duplicates is barely higher - scores are
"compressed" near 1.0 regardless of actual visual similarity, and rankings
become unreliable. PCA-whitening centers the embedding distribution and
rescales each principal axis to unit variance, spreading the corpus back
out so cosine similarity actually reflects visual similarity again.

Critical detail - dimensionality MUST be truncated, not just regularized:
measured on this corpus (13868 x 512), the eigenvalue spectrum drops from
~0.119 (component 0) to ~0.00000 (component 511) - the top 128 components
already explain 96.8% of the variance, and the trailing ~380 components are
almost pure noise. A first attempt at whitening all 512 dims with only a
small epsilon floor let 1/sqrt(eigval) blow those near-zero-variance noise
directions up to dominate the L2-normalized output - the query image no
longer even matched itself. The fix (used here) is the standard PCA-
whitening-for-retrieval approach (Jegou & Chum 2012): keep only the top K
principal components and discard the noisy tail entirely, rather than
trying to rescale it.

This script was referenced throughout services/similarity_search.py
(_get_whitening's docstring, the WARNING when the .npz is missing, and
find_similar_clip's docstring) but never existed - so every image search
had been silently running on raw, uncalibrated CLIP the whole time.

Steps:
  1. Pull every vector + payload currently in gold_ornaments_clip.
  2. Fit truncated PCA whitening: mean + a (D x K) projection so that
     y = (x - mean) @ projection has K decorrelated, unit-variance
     components (K=128 by default - see WHITEN_DIM below).
  3. Save (mean, projection) to models/clip_whitening.npz - this is what
     _get_whitening() loads at query time so a live search query is
     whitened with the exact same transform the corpus was built with.
  4. Apply the transform to every corpus vector, L2-normalize, and upsert
     into gold_ornaments_clip_w (recreated at the new K-dim size, WITH the
     category/features payload indexes - without these, any category-
     filtered search 400s and find_similar_clip() silently falls back to
     the ConvNeXt collection instead of using the whitened results).
  5. Run a sanity check: every image should match itself (~1.0), and a
     random-pair sample should show a much wider spread than raw CLIP's
     compressed 0.6-0.9 band. Aborts before touching Qdrant if this fails.

Usage: python scripts/build_clip_whitened.py [K]   (K defaults to 128)
"""
import os
import sys
import time

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from services.similarity_search import (
    _get_qdrant_client,
    COLLECTION_NAME_CLIP,
    COLLECTION_NAME_CLIP_W,
    WHITEN_PATH,
)

SCROLL_BATCH = 500
UPSERT_BATCH = 200
WHITEN_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def fetch_all_points(client, collection):
    ids, vectors, payloads = [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        for p in points:
            ids.append(p.id)
            vectors.append(p.vector)
            payloads.append(p.payload)
        print(f"  fetched {len(ids)}...")
        if offset is None:
            break
    return ids, np.asarray(vectors, dtype=np.float32), payloads


def fit_whitening(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order, cov is symmetric PSD
    # Keep the top-k (largest eigenvalues = most real signal), drop the noisy tail.
    eigvals = eigvals[::-1][:k]
    eigvecs = eigvecs[:, ::-1][:, :k]
    eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)

    projection = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))  # (D, k)
    explained = eigvals.sum()
    return mean.astype(np.float32), projection.astype(np.float32), explained


def main():
    client = _get_qdrant_client()

    print(f"Fetching all points from '{COLLECTION_NAME_CLIP}'...")
    ids, vectors, payloads = fetch_all_points(client, COLLECTION_NAME_CLIP)
    total = len(ids)
    print(f"Fetched {total} vectors, dim={vectors.shape[1]}.")
    if total < 50:
        print("Too few vectors to fit a meaningful whitening transform. Aborting.")
        return

    print(f"Fitting PCA whitening transform (top {WHITEN_DIM} components)...")
    mean, projection, explained = fit_whitening(vectors, WHITEN_DIM)

    print("Applying transform to all vectors...")
    whitened = (vectors - mean) @ projection
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    whitened = whitened / np.clip(norms, 1e-12, None)

    # Sanity check BEFORE touching Qdrant. Each vector must match itself
    # (~1.0) and a random sample must show real spread, not near-zero noise.
    self_sims = np.sum(whitened * whitened, axis=1)  # should all be ~1.0 (post-normalize)
    print(f"Self-similarity (should be ~1.0): min={self_sims.min():.4f}, mean={self_sims.mean():.4f}")

    sample_idx = np.random.RandomState(0).choice(total, size=min(500, total), replace=False)
    sample = whitened[sample_idx]
    sims = sample @ sample.T
    off_diag = sims[~np.eye(len(sample), dtype=bool)]
    print(f"Random-pair spread on {len(sample)}-item sample: "
          f"mean={off_diag.mean():.3f}, p50={np.percentile(off_diag,50):.3f}, "
          f"p90={np.percentile(off_diag,90):.3f}, p99={np.percentile(off_diag,99):.3f}")

    # Re-embed the very first corpus image as a "fresh query" through the
    # same transform and confirm it finds itself as the top match - this is
    # exactly the failure mode the first (untruncated) attempt hit.
    test_vec = whitened[0]
    sims_to_test = whitened @ test_vec
    best_idx = np.argmax(sims_to_test)
    if best_idx != 0 or sims_to_test[0] < 0.99:
        print(f"ABORT: sanity check failed - vector 0 does not match itself "
              f"(best_idx={best_idx}, self_score={sims_to_test[0]:.4f}). "
              f"Not writing the .npz or touching Qdrant.")
        return
    print("Self-match sanity check passed.")

    os.makedirs(os.path.dirname(WHITEN_PATH), exist_ok=True)
    np.savez(WHITEN_PATH, mean=mean, projection=projection)
    print(f"Saved whitening transform to {WHITEN_PATH} (dim={WHITEN_DIM})")

    print(f"Recreating collection '{COLLECTION_NAME_CLIP_W}' at dim={WHITEN_DIM}...")
    if client.collection_exists(COLLECTION_NAME_CLIP_W):
        client.delete_collection(COLLECTION_NAME_CLIP_W)
    client.create_collection(
        collection_name=COLLECTION_NAME_CLIP_W,
        vectors_config=VectorParams(size=WHITEN_DIM, distance=Distance.COSINE),
    )
    # Without these, any category-filtered search 400s (Qdrant requires an
    # index to filter on a payload field) and find_similar_clip() silently
    # falls back to the ConvNeXt collection instead of using this one.
    client.create_payload_index(
        collection_name=COLLECTION_NAME_CLIP_W, field_name="category", field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME_CLIP_W, field_name="features", field_schema=PayloadSchemaType.KEYWORD
    )

    print("Upserting whitened vectors...")
    t0 = time.time()
    for i in range(0, total, UPSERT_BATCH):
        batch_ids = ids[i:i + UPSERT_BATCH]
        batch_vecs = whitened[i:i + UPSERT_BATCH]
        batch_payloads = payloads[i:i + UPSERT_BATCH]
        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in zip(batch_ids, batch_vecs, batch_payloads)
        ]
        client.upsert(collection_name=COLLECTION_NAME_CLIP_W, points=points)
        print(f"  upserted {min(i + UPSERT_BATCH, total)}/{total} ({time.time()-t0:.0f}s)")

    print(f"\nDone. {COLLECTION_NAME_CLIP_W} now has {total} whitened vectors (dim={WHITEN_DIM}).")
    print("Live image search will automatically start using it (via _get_whitening()).")


if __name__ == "__main__":
    main()
