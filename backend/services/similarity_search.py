import os
import threading
import time
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Paths
# Use absolute path for Qdrant DB and models to ensure it's found regardless of where app.py is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
QDRANT_DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "gold_ornaments"
COLLECTION_NAME_CLIP = "gold_ornaments_clip"
# Whitened-CLIP collection: the index image search actually runs on.
# See backend/scripts/build_clip_whitened.py for why whitening is required.
COLLECTION_NAME_CLIP_W = "gold_ornaments_clip_w"
WHITEN_PATH = os.path.join(BASE_DIR, "models", "clip_whitening.npz")
NUM_CLASSES = 13

load_dotenv(os.path.join(BASE_DIR, '.env'))

# ---------------------------------------------------------------------------
# Score calibration
#
# Raw cosine is not a "match percentage". Measured on the whitened-CLIP index,
# comparing items inside a single category:
#     unrelated items          p50 ~ 0.06, p90 ~ 0.25, p99 ~ 0.46
#     same product, re-shot    ~ 0.76 - 0.85
#     byte-identical image     1.00
# EXACT_COS sits just below the same-product range so a re-shot or lightly
# edited photo of a catalogue item still lands in "exact"; SIMILAR_COS sits
# between the unrelated p90 and p99 so only the genuinely close designs are
# promoted to "similar".
#
# EXACT_BAND_PERCENT/SIMILAR_BAND_PERCENT decide which UI section a result
# lands in (Similar Ornaments vs Other Models). These were originally 99/85 -
# but the percent formula below actually maps SIMILAR_COS itself to 80%, not
# 85% (an internal inconsistency in the original calibration), and live
# testing across Necklaces/Ear Rings/Rings/Bangles showed genuine (non-
# duplicate) visual matches for a real uploaded photo topping out around
# 48-77% in THIS corpus's whitened embedding space - so every search was
# landing 100% of its results in "other" and showing nothing under Similar
# Ornaments. Lowered to match what real searches actually produce; this only
# changes which bucket a score falls into, not the score itself.
# ---------------------------------------------------------------------------
EXACT_COS = float(os.getenv("EXACT_COS", "0.75"))
SIMILAR_COS = float(os.getenv("SIMILAR_COS", "0.35"))
EXACT_BAND_PERCENT = float(os.getenv("EXACT_BAND_PERCENT", "99.0"))
SIMILAR_BAND_PERCENT = float(os.getenv("SIMILAR_BAND_PERCENT", "95.0"))


# (Old calibrate_score logic removed per user request to show raw similarities directly)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing ML Service on device:", device)

from services.blob_storage import strip_dataset_prefix

BASE_DIR_FWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/') + '/'
_KNOWN_PREFIXES = ["D:/Gold searching/", "E:/Projects/GitHub/GSS/", BASE_DIR_FWD]

def _strip_image_url_prefix(url):
    """Strip known absolute prefixes from image URLs so they work with Flask endpoint.
    Dataset images come out as <Category>/<filename>, matching the blob container."""
    # Some older ingested payloads carry a stray leading/trailing quote
    # character (a quoting bug in a past ingestion run) - strip it so it
    # doesn't end up baked into the served path / blob key.
    url_normalized = url.strip().strip("'\"").replace('\\', '/')
    for prefix in _KNOWN_PREFIXES:
        if url_normalized.startswith(prefix):
            url_normalized = url_normalized[len(prefix):]
            break
    return strip_dataset_prefix(url_normalized)

# Torch defaults to one intra-op thread per core. With several requests in
# flight that oversubscribes the CPU and makes every inference slower, so cap it.
_TORCH_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))
if _TORCH_THREADS <= 0:
    _TORCH_THREADS = max(1, min(4, os.cpu_count() or 2))
torch.set_num_threads(_TORCH_THREADS)
print("Torch intra-op threads:", _TORCH_THREADS)

import torchvision.transforms.functional as F_vision


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        # Pad left, top, right, bottom
        padding = (
            int((max_wh - w) / 2),
            int((max_wh - h) / 2),
            max_wh - w - int((max_wh - w) / 2),
            max_wh - h - int((max_wh - h) / 2)
        )
        return F_vision.pad(image, padding, fill=255, padding_mode='constant')


transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    "Bangles",
    "Bracelets",
    "Chains",
    "Ear Cuff Chains",
    "Ear Rings",
    "Hair Braid Ornaments",
    "Maang Tikka",
    "Necklaces",
    "Nose Pin",
    "Rings",
    "Straps(Pattilu)",
    "Toe Rings",
    "Waist Chain"
]
CONFIDENCE_THRESHOLD = 0.70

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Lazy model loading
#
# The three models total well over a gigabyte and, on a cold container, timm and
# sentence-transformers download their weights over the network. Building them
# at import time meant the process could not answer a single request for
# minutes: the platform health probe failed, the container was recycled, and the
# site served 503 until an attempt happened to be fast enough. Loading is
# deferred behind a lock now, and warmup() starts it in the background once the
# server is already listening, so /health answers immediately.
# ---------------------------------------------------------------------------

_model_lock = threading.RLock()
_qdrant_lock = threading.Lock()
_MODEL_STATE = {"imagenet": "pending", "classifier": "pending", "clip": "pending"}


class FeatureExtractorWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        # Depending on whether the model is torchvision or timm, extract features
        if getattr(self.model, 'is_torchvision', False):
            # Torchvision ConvNeXt feature extraction without final linear layer
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = self.model.classifier[0](x) # LayerNorm2d
            x = self.model.classifier[1](x) # Flatten
            return x
        else:
            # Timm ConvNeXt feature extraction without final linear layer
            # forward_head with pre_logits=True gets the features before the final fc
            return self.model.forward_head(self.model.forward_features(x), pre_logits=True)

    def eval(self):
        pass

    def to(self, device):
        pass

def _ensure_model_imagenet():
    """ConvNeXt feature extractor used for image-to-image search.
    Now shares memory with the classification model to prevent OOM."""
    global model_imagenet
    if "model_imagenet" in globals():
        return model_imagenet
    with _model_lock:
        if "model_imagenet" not in globals():
            print("Wrapping ConvNeXt model for feature extraction...")
            _MODEL_STATE["imagenet"] = "loading"
            try:
                base_model = _ensure_class_model()
                model_imagenet = FeatureExtractorWrapper(base_model)
                _MODEL_STATE["imagenet"] = "ready"
            except Exception as e:
                _MODEL_STATE["imagenet"] = "error: {}".format(e)
                raise
    return model_imagenet


def _ensure_class_model():
    """Fine-tuned ConvNeXt used to predict the ornament category."""
    global class_model
    if "class_model" in globals():
        return class_model
    with _model_lock:
        if "class_model" not in globals():
            print("Loading ConvNeXt classification model into memory...")
            _MODEL_STATE["classifier"] = "loading"
            try:
                import torchvision
                import timm
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                is_torchvision = any(k.startswith('features.') for k in state_dict.keys())
                if is_torchvision:
                    m = torchvision.models.convnext_base(num_classes=NUM_CLASSES)
                    m.load_state_dict(state_dict)
                else:
                    m = timm.create_model("convnext_base", pretrained=False, num_classes=NUM_CLASSES)
                    m.load_state_dict(state_dict)
                m.to(device)
                m.eval()
                m.is_torchvision = is_torchvision
                class_model = m
                _MODEL_STATE["classifier"] = "ready"
            except Exception as e:
                _MODEL_STATE["classifier"] = "error: {}".format(e)
                raise
    return class_model


def _ensure_clip_model():
    """CLIP model backing text and voice search."""
    global clip_model
    if "clip_model" in globals():
        return clip_model
    with _model_lock:
        if "clip_model" not in globals():
            print("Loading CLIP model for text/voice search...")
            _MODEL_STATE["clip"] = "loading"
            try:
                clip_model = SentenceTransformer('clip-ViT-B-32', device=str(device))
                _MODEL_STATE["clip"] = "ready"
            except Exception as e:
                _MODEL_STATE["clip"] = "error: {}".format(e)
                raise
    return clip_model


_whiten_state = {"loaded": False, "mean": None, "projection": None}


def _get_whitening():
    """Load the PCA-whitening transform fitted by
    backend/scripts/build_clip_whitened.py. Returns (mean, projection), or
    (None, None) if the artifact is missing - in which case image search falls
    back to the raw CLIP collection instead of failing."""
    if not _whiten_state["loaded"]:
        with _model_lock:
            if not _whiten_state["loaded"]:
                if os.path.exists(WHITEN_PATH):
                    data = np.load(WHITEN_PATH)
                    _whiten_state["mean"] = data["mean"].astype(np.float32)
                    _whiten_state["projection"] = data["projection"].astype(np.float32)
                    print("Loaded CLIP whitening transform:", _whiten_state["projection"].shape)
                else:
                    print("WARNING: {} not found. Image search will use raw CLIP "
                          "(scores will be compressed). Run "
                          "backend/scripts/build_clip_whitened.py.".format(WHITEN_PATH))
                _whiten_state["loaded"] = True
    return _whiten_state["mean"], _whiten_state["projection"]


def _get_qdrant_client():
    """Qdrant Cloud client, built on first use so a slow or unreachable Qdrant
    cannot stall process startup."""
    global qdrant_client
    if "qdrant_client" in globals():
        return qdrant_client
    with _qdrant_lock:
        if "qdrant_client" not in globals():
            print("Connecting to Qdrant Cloud...")
            try:
                # check_compatibility issues a blocking version request on init.
                qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=QDRANT_TIMEOUT,
                    check_compatibility=False,
                )
            except TypeError:
                # Older qdrant-client without check_compatibility.
                qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=QDRANT_TIMEOUT,
                )
    return qdrant_client


# PEP 562 hook: keeps `from services.similarity_search import clip_model` (used
# by routes.py and llm_helper.py) working, loading on first access rather than
# at import time.
_LAZY_ATTRS = {
    "model_imagenet": _ensure_model_imagenet,
    "class_model": _ensure_class_model,
    "clip_model": _ensure_clip_model,
    "qdrant_client": _get_qdrant_client,
}


def __getattr__(name):
    loader = _LAZY_ATTRS.get(name)
    if loader is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    return loader()


def model_status():
    """Per-model load state, surfaced by the /ready endpoint."""
    return dict(_MODEL_STATE)


def models_ready():
    return all(state == "ready" for state in _MODEL_STATE.values())


def warmup(background=True):
    """Load every model. Called right after the web server starts so the first
    real search does not pay the load cost."""
    def _run():
        started = time.time()
        try:
            _get_qdrant_client()
        except Exception as e:
            print("[warmup] Qdrant connection failed:", e)
        for label, loader in (
            ("ImageNet ConvNeXt", _ensure_model_imagenet),
            ("ConvNeXt classifier", _ensure_class_model),
            ("CLIP", _ensure_clip_model),
        ):
            try:
                loader()
            except Exception as e:
                print("[warmup] Failed to load {}: {}".format(label, e))
        print("[warmup] Finished in {:.1f}s. State: {}".format(time.time() - started, _MODEL_STATE))

    if not background:
        _run()
        return None
    thread = threading.Thread(target=_run, name="model-warmup", daemon=True)
    thread.start()
    return thread


from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct


def _embed_image_clip(image_path):
    """CLIP image embedding for the query, whitened with the same transform used
    to build the index so query and corpus live in the same space."""
    clip = _ensure_clip_model()
    image = Image.open(image_path).convert("RGB")
    vec = np.asarray(clip.encode(image), dtype=np.float32)

    mean, projection = _get_whitening()
    if mean is None or projection is None:
        # No whitening artifact: search the raw CLIP collection instead.
        norm = np.linalg.norm(vec)
        return (vec / max(norm, 1e-12)).tolist(), False

    y = (vec - mean) @ projection
    norm = np.linalg.norm(y)
    return (y / max(norm, 1e-12)).tolist(), True


def find_similar_clip(image_path, top_k=1000, offset=0, category=None):
    """Image-to-image search on whitened CLIP embeddings.

    This replaces the old ConvNeXt-feature search. The ConvNeXt features came
    from a 13-way category classifier, so within a category they were almost
    constant (median cosine 0.991 between two unrelated earrings) and every
    result rendered as "100% match". Whitened CLIP separates the bands - see
    backend/scripts/build_clip_whitened.py for the measurements.

    Returns (results, db_time_ms) where each result carries the raw cosine plus
    the calibrated `match_percent` and `band` the UI groups on.
    """
    query_vector, whitened = _embed_image_clip(image_path)
    collection = COLLECTION_NAME_CLIP_W if whitened else COLLECTION_NAME_CLIP

    query_filter = None
    if category:
        query_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    try:
        t0 = time.time()
        all_hits = _get_qdrant_client().search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            offset=offset,
            with_payload=["image_url", "category", "features"],
        )
        db_time_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        print("Qdrant CLIP search error on {!r}: {}".format(collection, e))
        print("Falling back to ConvNeXt ImageNet search on 'gold_ornaments'...")
        return find_similar(image_path, top_k=top_k, offset=offset, category=category)

    if not all_hits:
        print(f"No hits in '{collection}', falling back to ConvNeXt ImageNet search on 'gold_ornaments'...")
        return find_similar(image_path, top_k=top_k, offset=offset, category=category)

    # Strict category search: once a category filter is applied, every
    # result must come from that category, even if the top match is weak.
    # (Previously this fell back to an unfiltered search when the filtered
    # top hit was weak, to route around classifier misclassification - that
    # traded category purity for recall. Category purity is now the
    # explicit requirement, so no unfiltered fallback happens here; a weak
    # filtered match still only returns same-category results.)

    results = []
    for rank, hit in enumerate(all_hits, 1):
        score_val = float(hit.score)
        percent = round(score_val * 100.0, 2)
        if percent >= EXACT_BAND_PERCENT:
            band = "exact"
        elif percent >= SIMILAR_BAND_PERCENT:
            band = "similar"
        else:
            band = "other"

        results.append({
            "rank": rank + offset,
            "similarity": score_val,
            "match_percent": percent,
            "band": band,
            "image_url": _strip_image_url_prefix(hit.payload.get("image_url", "")),
            "category": hit.payload.get("category", ""),
            "features": hit.payload.get("features", []),
        })

    return results, db_time_ms


def find_similar(image_path, top_k=1000, offset=0, model_type="imagenet", category=None):
    """
    Given an image path, extracts features and queries Qdrant DB for cosine similarity.
    Uses paginated search calls (max 100 per call) to support large result sets.
    """
    model = _ensure_model_imagenet()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(image)
        feature = feature.view(feature.size(0), -1)
        feature = F.normalize(feature, p=2, dim=1)

    query_vector = feature.squeeze().cpu().numpy().tolist()

    query_filter = None
    if category:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        )

    try:
        t0 = time.time()

        all_hits = _get_qdrant_client().search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            offset=offset,
            # Only the fields the response actually uses; pulling the whole
            # payload for up to 1000 hits is wasted bandwidth on every search.
            with_payload=["image_url", "category", "features"],
        )

        db_time_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        print(f"Qdrant search error (Has the migration script been run?): {e}")
        return [], 0

    results = []
    for rank, hit in enumerate(all_hits, 1):
        score_val = float(hit.score)
        percent = round(score_val * 100.0, 2)
        if percent >= EXACT_BAND_PERCENT:
            band = "exact"
        elif percent >= SIMILAR_BAND_PERCENT:
            band = "similar"
        else:
            band = "other"
            
        results.append({
            "rank": rank + offset,
            "similarity": score_val,
            "match_percent": percent,
            "band": band,
            "image_url": _strip_image_url_prefix(hit.payload.get("image_url", "")),
            "category": hit.payload.get("category", ""),
            "features": hit.payload.get("features", [])
        })

    return results, db_time_ms


def find_similar_by_text(query_text, top_k=1000, offset=0, category=None):
    """
    Given a text query, extracts features using CLIP and queries Qdrant DB for cosine similarity.
    Uses paginated search calls (max 100 per call) to support large result sets.
    """
    model = _ensure_clip_model()

    with torch.no_grad():
        embedding = model.encode(query_text).tolist()

    query_filter = None
    if category and category.lower() != 'all':
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        )

    try:
        t0 = time.time()

        all_hits = _get_qdrant_client().search(
            collection_name=COLLECTION_NAME_CLIP,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k,
            offset=offset,
            with_payload=["image_url", "category", "features"],
        )

        db_time_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        print(f"Qdrant text search error on '{COLLECTION_NAME_CLIP}': {e}")
        scroll_filter = None
        if category and category.lower() != 'all':
            scroll_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )
        print(f"Falling back to scroll search (category={category}) in '{COLLECTION_NAME}'...")
        try:
            hits, _ = _get_qdrant_client().scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=top_k,
                offset=offset,
                with_payload=["image_url", "category", "features"],
            )
            results = []
            for rank, hit in enumerate(hits, 1):
                score_val = getattr(hit, 'score', 0.85)
                percent = round(score_val * 100.0, 2)
                if percent >= EXACT_BAND_PERCENT:
                    band = "exact"
                elif percent >= SIMILAR_BAND_PERCENT:
                    band = "similar"
                else:
                    band = "other"

                results.append({
                    "rank": rank + offset,
                    "similarity": float(score_val),
                    "match_percent": percent,
                    "band": band,
                    "image_url": _strip_image_url_prefix(hit.payload.get("image_url", "")),
                    "category": hit.payload.get("category", ""),
                    "features": hit.payload.get("features", [])
                })
            return results, 0
        except Exception as scroll_err:
            print(f"Fallback scroll error: {scroll_err}")
            return [], 0

    if not all_hits:
        print(f"No hits in '{COLLECTION_NAME_CLIP}', falling back to '{COLLECTION_NAME}' (category={category})...")
        scroll_filter = None
        if category and category.lower() != 'all':
            scroll_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )
        try:
            all_hits, _ = _get_qdrant_client().scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=top_k,
                offset=offset,
                with_payload=["image_url", "category", "features"],
            )
        except Exception as scroll_err:
            print(f"Fallback scroll error: {scroll_err}")

    results = []
    for rank, hit in enumerate(all_hits, 1):
        score_val = getattr(hit, 'score', 0.85)
        percent = round(score_val * 100.0, 2)
        if percent >= 99.0:
            band = "exact"
        elif percent >= 85.0:
            band = "similar"
        else:
            band = "other"

        results.append({
            "rank": rank + offset,
            "similarity": float(score_val),
            "match_percent": percent,
            "band": band,
            "image_url": _strip_image_url_prefix(hit.payload.get("image_url", "")),
            "category": hit.payload.get("category", ""),
            "features": hit.payload.get("features", [])
        })

    return results, db_time_ms


def classify_image(image_path):
    model = _ensure_class_model()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probability = F.softmax(output, dim=1)
    confidence, prediction = torch.max(probability, dim=1)
    return confidence.item(), CLASS_NAMES[prediction.item()]


def add_inventory(image_paths):
    successful_uploads = 0
    points_to_upsert = []
    points_to_upsert_clip = []
    points_to_upsert_clip_w = []

    from services.jewelry_validator import validate_jewelry_image

    feature_model = _ensure_model_imagenet()
    clip = _ensure_clip_model()
    client = _get_qdrant_client()

    for image_path in image_paths:
        try:
            # 0. CLIP Zero-Shot Validation
            is_jewelry, j_score, nj_score = validate_jewelry_image(image_path, clip, device)
            print(f"CLIP Validation [Inventory] - Image: {os.path.basename(image_path)} | Jewelry Score: {j_score:.4f} | Non-Jewelry Score: {nj_score:.4f} | Accepted: {is_jewelry}")
            if not is_jewelry:
                print(f"Skipping {os.path.basename(image_path)}: Not a jewelry image.")
                continue

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = feature_model(image_tensor)
                feature = feature.view(feature.size(0), -1)
                feature = F.normalize(feature, p=2, dim=1)

            vector = feature.squeeze().cpu().numpy().tolist()
            clip_vec = np.asarray(clip.encode(image), dtype=np.float32)

            # Must be a real absolute path, not whatever string the caller
            # passed in - a relative path like "../Jewellery_Data/Bangles/x.png"
            # (e.g. from a script invoked with a relative directory argument)
            # would get stored verbatim as image_url, and neither
            # _strip_image_url_prefix() nor serve_image()'s prefix matching
            # recognizes a leading "../", so every such image 404s in the UI.
            abs_path = os.path.abspath(image_path).replace('\\', '/')
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, abs_path))

            conf, predicted_cat = classify_image(abs_path)
            payload = {"image_url": abs_path, "category": predicted_cat}

            # Best-effort copy to Blob Storage under the same relative key
            # serve_image() derives from image_url, so it can fall back to
            # Blob when the local file is ever missing. Never blocks or
            # fails ingestion - see blob_storage.py for the no-op-if-
            # unconfigured behavior.
            from services import blob_storage
            blob_storage.upload_file(abs_path)

            # Duplicate check must run on CLIP vectors, not the ConvNeXt
            # classifier feature. Those features cluster so tightly within a
            # category (median cosine ~0.991 between two UNRELATED items -
            # see find_similar_clip's docstring) that almost any new image in
            # an existing category was scoring above 0.99999 against some
            # unrelated already-indexed item and getting wrongly skipped
            # entirely, before it ever reached the CLIP/whitened-CLIP upsert
            # below. CLIP similarity is well separated (unrelated ~0.06-0.46,
            # same product re-shot ~0.76-0.85, byte-identical ~1.00) so 0.999
            # only matches genuine re-uploads of the same image.
            try:
                search_result = client.search(
                    collection_name=COLLECTION_NAME_CLIP,
                    query_vector=clip_vec.tolist(),
                    limit=1,
                    with_payload=False,
                )
                if search_result and search_result[0].score > 0.999:
                    print(f"Exact duplicate detected! Skipping {abs_path}")
                    continue
            except Exception as e:
                print(f"Similarity search error: {e}")
                pass

            # 1. Add to ConvNeXt Collection
            points_to_upsert.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

            # 2. Add to CLIP Collection
            points_to_upsert_clip.append(
                PointStruct(id=point_id, vector=clip_vec.tolist(), payload=payload)
            )

            # 3. Add to the whitened-CLIP collection that image search queries.
            # Without this, newly added inventory would be invisible to search.
            w_mean, w_proj = _get_whitening()
            if w_mean is not None and w_proj is not None:
                wv = (clip_vec - w_mean) @ w_proj
                wv = wv / max(float(np.linalg.norm(wv)), 1e-12)
                points_to_upsert_clip_w.append(
                    PointStruct(id=point_id, vector=wv.tolist(), payload=payload)
                )

            successful_uploads += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if points_to_upsert:
        batch_size = 50
        for j in range(0, len(points_to_upsert), batch_size):
            batch = points_to_upsert[j:j+batch_size]
            batch_clip = points_to_upsert_clip[j:j+batch_size]
            batch_clip_w = points_to_upsert_clip_w[j:j+batch_size]

            try:
                # Upsert ConvNeXt
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
            except Exception as e:
                print(f"Error upserting ConvNeXt batch to Qdrant: {e}")

            try:
                # Upsert CLIP
                client.upsert(
                    collection_name=COLLECTION_NAME_CLIP,
                    points=batch_clip
                )
            except Exception as e:
                print(f"Error upserting CLIP batch to Qdrant: {e}")

            try:
                # Upsert whitened CLIP (the collection image search reads)
                if batch_clip_w:
                    if not client.collection_exists(COLLECTION_NAME_CLIP_W):
                        # Graceful handling if it doesn't exist during live upload
                        from qdrant_client.http.models import Distance, VectorParams
                        client.create_collection(
                            collection_name=COLLECTION_NAME_CLIP_W,
                            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                        )
                    client.upsert(
                        collection_name=COLLECTION_NAME_CLIP_W,
                        points=batch_clip_w
                    )
            except Exception as e:
                print(f"Error upserting Whitened CLIP batch to Qdrant: {e}")

    return successful_uploads
