"""
Measure how well the (now-fixed) whitened-CLIP embedding space separates
all 13 categories: for every point in gold_ornaments_clip_w, find its
nearest OTHER point (excluding itself) and check whether it shares the
same category. This tests the actual search embedding space directly,
complementing the earlier ConvNeXt-classifier confusion matrix.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from collections import Counter
from services.similarity_search import _get_qdrant_client, COLLECTION_NAME_CLIP_W, CLASS_NAMES

client = _get_qdrant_client()

print("Fetching all vectors + categories from gold_ornaments_clip_w...")
vectors, categories = [], []
offset = None
while True:
    points, offset = client.scroll(
        collection_name=COLLECTION_NAME_CLIP_W, limit=500, offset=offset,
        with_payload=['category'], with_vectors=True,
    )
    if not points:
        break
    for p in points:
        vectors.append(p.vector)
        categories.append(p.payload.get('category', 'Unknown'))
    if offset is None:
        break

X = np.asarray(vectors, dtype=np.float32)
cats = np.asarray(categories)
N = X.shape[0]
print(f"Loaded {N} vectors, dim={X.shape[1]}.")

matrix = {c: Counter() for c in CLASS_NAMES}
per_class_total = Counter()

BATCH = 1000
correct = 0
for i in range(0, N, BATCH):
    batch = X[i:i+BATCH]
    sims = batch @ X.T  # (batch, N)
    for row_offset in range(batch.shape[0]):
        idx = i + row_offset
        sims[row_offset, idx] = -1.0  # exclude self
        nn_idx = np.argmax(sims[row_offset])
        true_c = cats[idx]
        nn_c = cats[nn_idx]
        if true_c in matrix:
            matrix[true_c][nn_c] += 1
            per_class_total[true_c] += 1
            if nn_c == true_c:
                correct += 1
    print(f"  processed {min(i+BATCH, N)}/{N}...")

print(f"\nOverall nearest-neighbor same-category rate: {correct}/{N} = {100*correct/N:.1f}%\n")

print("=" * 100)
print("WHITENED-CLIP NEAREST-NEIGHBOR CONFUSION MATRIX (rows=true category, cols=NN's category)")
print("=" * 100)
row_label = "TRUE / NN"
header = f"{row_label:<22}" + "".join(f"{c[:6]:>8}" for c in CLASS_NAMES)
print(header)
print("-" * len(header))
for true_c in CLASS_NAMES:
    total = per_class_total.get(true_c, 0)
    if total == 0:
        print(f"{true_c:<22}  (no samples)")
        continue
    row = matrix[true_c]
    cells = "".join(f"{row.get(pred_c,0):>8}" if row.get(pred_c,0) else f"{'.':>8}" for pred_c in CLASS_NAMES)
    acc = 100.0 * row.get(true_c, 0) / total
    print(f"{true_c:<22}{cells}   ({acc:.0f}% same-category NN, n={total})")

print("\n" + "=" * 100)
print("TOP CROSS-CATEGORY CONFUSIONS (true -> nearest neighbor's category)")
print("=" * 100)
confusions = []
for true_c in CLASS_NAMES:
    total = per_class_total.get(true_c, 0)
    if not total:
        continue
    for pred_c, n in matrix[true_c].items():
        if pred_c != true_c and n > 0:
            confusions.append((n, true_c, pred_c, 100.0*n/total))
confusions.sort(reverse=True)
for n, true_c, pred_c, pct in confusions[:20]:
    print(f"  {true_c:<22} -> nearest neighbor in {pred_c:<22}  {n:>4} times  ({pct:.1f}%)")
