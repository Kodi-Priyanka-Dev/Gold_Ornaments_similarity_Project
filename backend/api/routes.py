import os
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import mimetypes
from flask import Blueprint, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from services.similarity_search import find_similar, find_similar_clip, find_similar_by_text, classify_image, CONFIDENCE_THRESHOLD, CLASS_NAMES
from services import blob_storage

api_blueprint = Blueprint('api', __name__)

# Dataset images never change once written, so let the browser cache them.
# Without this every result grid re-downloaded each thumbnail through Python.
IMAGE_CACHE_SECONDS = int(os.getenv('IMAGE_CACHE_SECONDS', str(7 * 24 * 3600)))

# Upper bound on how many hits a single search may pull back. The frontend asks
# for 1000; anything larger is a client mistake and just burns Qdrant bandwidth.
MAX_SEARCH_LIMIT = int(os.getenv('MAX_SEARCH_LIMIT', '1000'))

# Translation is a blocking call to an external service. It used to run with no
# timeout at all, so one slow response tied up a request thread indefinitely.
TRANSLATE_TIMEOUT_SECONDS = float(os.getenv('TRANSLATE_TIMEOUT_SECONDS', '6'))
_translate_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='translate')


def _clamp_int(value, default, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def translate_to_english(query):
    """Translate to English, falling back to the original text if the service is
    slow or unavailable. Never blocks the request thread for more than
    TRANSLATE_TIMEOUT_SECONDS."""
    def _translate():
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='auto', target='en').translate(query)

    try:
        translated = _translate_pool.submit(_translate).result(timeout=TRANSLATE_TIMEOUT_SECONDS)
        return translated or query
    except FutureTimeoutError:
        print(f"Translation timed out after {TRANSLATE_TIMEOUT_SECONDS}s; using original query.")
        return query
    except Exception as e:
        print(f"Translation Error: {e}")
        return query

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load image metadata dynamically to capture background updates
METADATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'metadata.json')
_last_metadata_mtime = 0
IMAGE_METADATA = {}
IMAGE_METADATA_BY_FILENAME = {}
IMAGE_METADATA_BY_RELPATH = {}  # keyed by path with known prefixes stripped

# Known prefixes that Qdrant or the search code may strip from image_url
BASE_DIR_FWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/') + '/'
_KNOWN_PREFIXES = ["D:/Gold searching/", "E:/Projects/GitHub/GSS/", BASE_DIR_FWD]

def _strip_known_prefixes(path):
    """Strip known absolute prefixes to get a relative path for matching."""
    p = path.replace('\\', '/')
    for prefix in _KNOWN_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return blob_storage.strip_dataset_prefix(p)

_metadata_lock = threading.Lock()
_inventory_lock = threading.Lock()

def get_current_metadata():
    global _last_metadata_mtime, IMAGE_METADATA, IMAGE_METADATA_BY_FILENAME, IMAGE_METADATA_BY_RELPATH
    if not os.path.exists(METADATA_FILE):
        return {}, {}, {}

    current_mtime = os.path.getmtime(METADATA_FILE)
    if current_mtime <= _last_metadata_mtime:
        return IMAGE_METADATA, IMAGE_METADATA_BY_FILENAME, IMAGE_METADATA_BY_RELPATH

    # Without this lock every concurrent request re-parsed the same ~2 MB of
    # JSON at once after a background inventory update touched the file.
    with _metadata_lock:
        if current_mtime <= _last_metadata_mtime:
            return IMAGE_METADATA, IMAGE_METADATA_BY_FILENAME, IMAGE_METADATA_BY_RELPATH
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                IMAGE_METADATA = json.load(f)

            IMAGE_METADATA_BY_FILENAME = {}
            IMAGE_METADATA_BY_RELPATH = {}
            for full_path, meta in IMAGE_METADATA.items():
                norm = full_path.replace('\\', '/')
                basename = os.path.basename(norm)
                stem = os.path.splitext(basename)[0]
                IMAGE_METADATA_BY_FILENAME[basename] = meta
                IMAGE_METADATA_BY_FILENAME[stem] = meta
                # Also index by stripped relative path
                rel = _strip_known_prefixes(norm)
                IMAGE_METADATA_BY_RELPATH[rel] = meta

            _last_metadata_mtime = current_mtime
            print(f"Reloaded metadata for {len(IMAGE_METADATA)} images ({len(IMAGE_METADATA_BY_RELPATH)} relative paths).")
        except Exception as e:
            print(f"Error reloading metadata: {e}")

    return IMAGE_METADATA, IMAGE_METADATA_BY_FILENAME, IMAGE_METADATA_BY_RELPATH

def _build_default_features(category, image_url, meta):
    feats = []
    if meta:
        for key in ['product_type', 'metal_color', 'gold_purity', 'stone_type', 'shape', 'type_of_wear']:
            val = meta.get(key)
            if val and str(val).strip() and str(val).strip() not in feats:
                feats.append(str(val).strip())
    
    cat = category or 'Gold Ornament'
    if cat not in feats:
        feats.insert(0, cat)
    if not any('gold' in f.lower() for f in feats):
        feats.append('22K Gold')
    if not any(w in ' '.join(feats).lower() for w in ['traditional', 'bridal', 'modern', 'handcrafted']):
        feats.append('Traditional Design')
    if not any(w in ' '.join(feats).lower() for w in ['fine', 'premium', 'ornament', 'craftsmanship']):
        feats.append('Fine Jewelry')
    return feats[:5]

def _log_response_summary(endpoint, results, extra=None, top_n=5):
    """Print a compact summary of exactly what is being sent back to the
    frontend for this request: result count, response time, and the top few
    matches with their category/score, so search behaviour is visible in the
    terminal without opening dev tools."""
    print(f"\n===== [{endpoint}] Response to frontend =====")
    if extra:
        for key, val in extra.items():
            print(f"  {key}: {val}")
    print(f"  results: {len(results)}")
    for r in results[:top_n]:
        pct = r.get('match_percent', round(r.get('similarity', 0) * 100, 1))
        band = r.get('band', '-')
        cat = r.get('meta_category') or r.get('category') or '-'
        url = os.path.basename(r.get('image_url', '') or '')
        print(f"    #{r.get('rank', '?')}  {pct}% [{band}]  category={cat}  file={url}")
    if len(results) > top_n:
        print(f"    ... and {len(results) - top_n} more")
    print("=" * (len(endpoint) + 32))


def enrich_results_with_metadata(results, force_category=None):
    """Attach category and features from Qdrant payloads or fallback to metadata.json or auto-generated tags."""
    meta_dict, meta_by_filename, meta_by_relpath = get_current_metadata()
    for result in results:
        image_url = result.get('image_url', '').replace('\\', '/')
        meta = meta_dict.get(image_url)
        if not meta:
            rel = _strip_known_prefixes(image_url)
            meta = meta_by_relpath.get(rel)
        if not meta:
            meta = meta_by_filename.get(os.path.basename(image_url))
        if not meta:
            stem = os.path.splitext(os.path.basename(image_url))[0]
            meta = meta_by_filename.get(stem, {})

        # Each result's own category (from its Qdrant payload) is ground truth
        # for that specific image - it must win over force_category (the
        # query image's own classified class). force_category exists only as
        # a last-resort fallback for a result that has no category of its
        # own. Previously this was `force_category or result.get('category')`,
        # which let the query's predicted class silently overwrite every
        # result's true category (e.g. a Rings/Waist Chain result showing a
        # "Necklaces" badge just because that's what the uploaded photo was
        # classified as).
        category = result.get('category') or force_category or meta.get('category') or ''
        
        # Check existing features
        features = result.get('features') or meta.get('features') or []
        if not features:
            features = _build_default_features(category, image_url, meta)

        result['meta_category'] = category
        result['meta_features'] = features
        # Attach the full raw metadata dict so the compare endpoint can
        # pass it directly into extract_comprehensive_features without
        # re-scanning the 56 MB metadata.json a second time.
        if meta:
            result['raw_metadata'] = meta
    return results


@api_blueprint.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
        
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        filename = name[:100] + ext
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        import time
        start_time = time.time()

        import services.similarity_search as ss
        clip_model = ss._ensure_clip_model()
        device = ss.device
        from services.jewelry_validator import validate_jewelry_image

        # CLIP validation and ConvNeXt classification run on different models
        # against the same uploaded image. Running them concurrently with PyTorch
        # can cause OpenMP threading deadlocks on CPU, so they must run sequentially.
        is_jewelry, j_score, nj_score = validate_jewelry_image(filepath, clip_model, device)
        confidence, predicted_category = classify_image(filepath)

        print(f"CLIP Validation - Image: {filename} | Jewelry Score: {j_score:.4f} | Non-Jewelry Score: {nj_score:.4f} | Accepted: {is_jewelry}")

        if not is_jewelry:
            return jsonify({"error": "Please upload a jewelry image such as an earring, necklace, bangle, ring, chain, or bracelet."}), 400

        # 1. Classification Check (ConvNeXt)
        # Strict category search: the top-1 predicted category is always used
        # to filter results, even below CONFIDENCE_THRESHOLD - every result
        # must belong to one single category, never an unfiltered mix.
        # `confidence` is still returned to the frontend for transparency.

        # 2. Run the actual ML model similarity search with strict category filtering!
        offset = _clamp_int(request.form.get('offset', 0), 0, 0, 100000)
        limit = _clamp_int(request.form.get('limit', 20), 20, 1, MAX_SEARCH_LIMIT)

        # Use the ConvNeXt 'find_similar' implementation to match the terminal behavior 
        # and provide the raw cosine similarity score directly.
        results, db_time_ms = find_similar(filepath, top_k=limit, offset=offset, category=predicted_category)
        enrich_results_with_metadata(results, force_category=predicted_category)

        # The true "exact match" is the image the user just uploaded, always
        # 100% - not a database entry. Any DB hit that lands in the "exact"
        # band (>=99% match, i.e. a near-duplicate photo) IS that same
        # image already present in the catalog, so it is dropped here rather
        # than shown a second time inside Similar Ornaments. What remains is
        # split into the two groups the UI renders as separate sections, in
        # a fixed order: Similar Ornaments (band=="similar") always above
        # Other Models (band=="other").
        exact_match_result = next((r for r in results if r.get('band') == 'exact'), None)
        has_exact_match = exact_match_result is not None
        similar_results = [r for r in results if r.get('band') == 'similar']
        other_model_results = [r for r in results if r.get('band') == 'other']

        # Fallback logic: If no 95%+ results, display 91-94% as Similarity Results
        if not similar_results:
            fallback_similar = []
            remaining_other = []
            for r in other_model_results:
                percent = r.get('match_percent', round(r.get('similarity', 0) * 100, 1))
                if percent >= 91.0:
                    fallback_similar.append(r)
                else:
                    remaining_other.append(r)
            
            if fallback_similar:
                similar_results = fallback_similar
                other_model_results = remaining_other

        _log_response_summary("IMAGE SEARCH", similar_results + other_model_results, extra={
            "uploaded_file": filename,
            "predicted_category": predicted_category,
            "classifier_confidence": round(confidence, 4),
            "duplicates_of_upload_excluded": len(results) - len(similar_results) - len(other_model_results),
            "search_time_ms": db_time_ms,
        })

        return jsonify({
            "message": "Search completed successfully",
            "predicted_category": predicted_category,
            "similar_results": similar_results,
            "other_model_results": other_model_results,
            "has_exact_match": has_exact_match,
            "exact_match_meta_category": exact_match_result.get('meta_category') if exact_match_result else None,
            "exact_match_meta_features": exact_match_result.get('meta_features') if exact_match_result else None,
            "search_time_ms": db_time_ms,
            # Bare filename of the saved upload (backend/uploads/<filename>) -
            # renderExactMatchCard() turns this into a server-resolvable URL
            # so Compare can ask the backend to open the file. The Exact
            # Match card itself still displays it via a local object URL for
            # speed; this is only needed for the Compare round trip.
            "uploaded_file": filename
        })
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": str(e)}), 500

def infer_category_from_text(text):
    text_lower = text.lower()
    if 'ear cuff' in text_lower or 'earcuff' in text_lower: return 'Ear Cuff Chains'
    if 'hair braid' in text_lower or 'hairbraid' in text_lower or 'savaram' in text_lower: return 'Hair Braid Ornaments'
    if 'maang' in text_lower or 'tikka' in text_lower: return 'Maang Tikka'
    if 'nose' in text_lower: return 'Nose Pin'
    if 'toe' in text_lower: return 'Toe Rings'
    if 'waist' in text_lower or 'hip chain' in text_lower or 'vaddanam' in text_lower: return 'Waist Chain'
    if 'pattilu' in text_lower or 'strap' in text_lower or 'anklet' in text_lower: return 'Straps(Pattilu)'
    if 'bangle' in text_lower: return 'Bangles'
    if 'bracelet' in text_lower: return 'Bracelets'
    if 'necklace' in text_lower or 'choker' in text_lower: return 'Necklaces'
    if 'chain' in text_lower or 'bead' in text_lower or 'beed' in text_lower or 'mangalsutra' in text_lower: return 'Chains'
    if 'earring' in text_lower or 'ear ring' in text_lower or 'ear' in text_lower or 'jhumka' in text_lower: return 'Ear Rings'
    if 'ring' in text_lower: return 'Rings'
    return None


# Single source of truth for resolving ANY raw category string (exact
# canonical name in any casing, or a free-text synonym) to the precise
# value stored in Qdrant's `category` payload field. Qdrant's category
# filter is an exact string match, so an unnormalized value (e.g. lowercase
# "necklaces", or "Earrings" instead of "Ear Rings") would silently match
# zero points instead of erroring - this is what the frontend's explicit
# `category` field (as opposed to a free-text query) is normalized through.
_CLASS_NAMES_LOWER = {c.lower(): c for c in CLASS_NAMES}


def normalize_category(raw):
    if not raw:
        return None
    raw = raw.strip()
    exact = _CLASS_NAMES_LOWER.get(raw.lower())
    if exact:
        return exact
    return infer_category_from_text(raw)

@api_blueprint.route('/search_text', methods=['POST'])
def search_text():
    data = request.get_json(silent=True) or request.form
    query = data.get('query')
    category = normalize_category(data.get('category', None))

    if query and not category:
        category = infer_category_from_text(query)

    offset = _clamp_int(data.get('offset', 0), 0, 0, 100000)
    limit = _clamp_int(data.get('limit', 20), 20, 1, MAX_SEARCH_LIMIT)

    if not query:
        return jsonify({"error": "No text query provided"}), 400

    try:
        import time
        start_time = time.time()
        
        # Multi-Language Translation (Google Translate)
        translated_query = translate_to_english(query)
        print(f"[Text Search] Translated: '{translated_query}'")

        # LLM Understanding (ChatGPT-like intent extraction)
        from services.llm_helper import extract_jewelry_keywords
        final_search_query = extract_jewelry_keywords(translated_query)
        
        import services.similarity_search as ss
        clip_model = ss._ensure_clip_model()
        device = ss.device
        from services.jewelry_validator import validate_jewelry_text

        is_jewelry, j_score, nj_score = validate_jewelry_text(final_search_query, clip_model, device)
        print(f"CLIP Text Validation - Query: '{final_search_query}' | Jewelry: {j_score:.4f} | Non-Jewelry: {nj_score:.4f} | Accepted: {is_jewelry}")

        if final_search_query.strip() == "NOT_JEWELRY" or not is_jewelry:
            return jsonify({"error": "Enter the Prompt Related to Jewellery"}), 400

        results, db_time_ms = find_similar_by_text(final_search_query, top_k=limit, offset=offset, category=category)
        enrich_results_with_metadata(results, force_category=category)

        # Re-rank based on stone/color keywords
        query_lower = final_search_query.lower()
        boost_keywords = []
        if any(w in query_lower for w in ['red', 'ruby']): boost_keywords.append('ruby')
        if any(w in query_lower for w in ['green', 'emerald']): boost_keywords.append('emerald')
        if any(w in query_lower for w in ['diamond', 'cz', 'white stone', 'white']): boost_keywords.extend(['diamond', 'cz'])
        if 'pearl' in query_lower: boost_keywords.append('pearl')
        if any(w in query_lower for w in ['plain', 'no stone', 'gold only']): boost_keywords.append('no stone')
        
        if boost_keywords:
            def get_boost(r):
                feats_lower = [str(f).lower() for f in r.get('meta_features', [])]
                path_lower = r.get('image_url', '').lower()
                for kw in boost_keywords:
                    if any(kw in f for f in feats_lower) or kw in path_lower:
                        return 1
                return 0
            
            results.sort(key=lambda r: (get_boost(r), r.get('similarity', 0)), reverse=True)
            for idx, r in enumerate(results, 1):
                r['rank'] = idx + offset

        _log_response_summary("TEXT SEARCH", results, extra={
            "query": query,
            "translated_query": translated_query,
            "category_filter": category,
            "search_time_ms": db_time_ms,
        })

        return jsonify({
            "message": "Text search completed successfully",
            "query": query,
            "translated_query": translated_query,
            "llm_query": final_search_query,
            "category_filter": category,
            "results": results,
            "search_time_ms": db_time_ms
        })
    except Exception as e:
        print(f"Error during text search: {e}")
        return jsonify({"error": str(e)}), 500

def _send_cached_image(path):
    """Serve a dataset image with long-lived cache headers and conditional
    requests, so repeat views become 304s instead of full re-downloads."""
    response = send_file(path, conditional=True, max_age=IMAGE_CACHE_SECONDS)
    response.headers['Cache-Control'] = f'public, max-age={IMAGE_CACHE_SECONDS}, immutable'
    return response


def _send_blob_image(blob_key):
    """Fall back to Blob Storage for an image not found on local disk. Returns
    None (caller falls through to 404) if Blob Storage isn't configured or
    doesn't have this key either."""
    data = blob_storage.download_bytes(blob_key)
    if data is None:
        return None
    content_type = mimetypes.guess_type(blob_key)[0] or 'application/octet-stream'
    response = Response(data, mimetype=content_type)
    response.headers['Cache-Control'] = f'public, max-age={IMAGE_CACHE_SECONDS}, immutable'
    return response


@api_blueprint.route('/images/<path:filepath>')
def serve_image(filepath):
    """
    Serves images from the local dataset securely.
    Supports absolute paths, relative paths, URL-encoded spaces, and Jewellery_Data.
    """
    try:
        import urllib.parse
        decoded = urllib.parse.unquote(filepath)
        
        # Fix leading slash on Windows drive letters, e.g. "/E:/Projects/..." -> "E:/Projects/..."
        if decoded.startswith('/') and len(decoded) > 2 and decoded[2] == ':':
            decoded = decoded[1:]
            
        normalized = decoded.replace('\\', '/')
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # 1. Direct check for full path
        if os.path.exists(decoded):
            return _send_cached_image(decoded)
            
        # 2. Check normalised path
        if os.path.exists(normalized):
            return _send_cached_image(normalized)

        # 3. Local Jewellery_Data/ copy, for both <Category>/<filename> and
        # legacy Jewellery_Data/<Category>/<filename> URLs
        dataset_rel = blob_storage.strip_dataset_prefix(normalized).lstrip('/')
        mapped = os.path.join(base_dir, 'Jewellery_Data', dataset_rel.replace('/', os.sep))
        if os.path.exists(mapped):
            return _send_cached_image(mapped)

        # 4. Handle new_inventory / dataset relative mapping
        if 'new_inventory/' in normalized.lower():
            rel_part = normalized[normalized.lower().find('new_inventory/'):]
            mapped = os.path.join(base_dir, 'backend', rel_part.replace('/', os.sep))
            if os.path.exists(mapped):
                return _send_cached_image(mapped)

        # 4b. Handle uploads/ (the search-query image the user just submitted,
        # saved by /api/search to backend/uploads/ - needed so Compare can
        # reference the uploaded photo by a real, server-resolvable URL
        # instead of a browser-only blob: URL the backend can never open).
        if 'uploads/' in normalized.lower():
            rel_part = normalized[normalized.lower().find('uploads/'):]
            mapped = os.path.join(base_dir, 'backend', rel_part.replace('/', os.sep))
            if os.path.exists(mapped):
                return _send_cached_image(mapped)

        # 5. Check relative to base_dir
        rel_mapped = os.path.join(base_dir, normalized.lstrip('/'))
        if os.path.exists(rel_mapped):
            return _send_cached_image(rel_mapped)

        # 6. Fallback search by filename inside Jewellery_Data
        target_name = os.path.basename(normalized)
        jewel_dir = os.path.join(base_dir, 'Jewellery_Data')
        if os.path.exists(jewel_dir):
            for root, _, files in os.walk(jewel_dir):
                if target_name in files:
                    return _send_cached_image(os.path.join(root, target_name))

        # 7. Not on local disk at all - try Blob Storage under the same
        # relative key (see blob_storage.to_blob_key). No-op if unconfigured.
        blob_response = _send_blob_image(blob_storage.to_blob_key(normalized))
        if blob_response is not None:
            return blob_response

        print(f"Warning: Image not found on disk for '{filepath}' (decoded: '{decoded}')")
        return "Image not found", 404
    except Exception as e:
        print(f"Error serving image {filepath}: {e}")
        return "Image not found", 404

@api_blueprint.route('/upload_inventory', methods=['POST'])
def upload_inventory():
    if 'images' not in request.files:
        return jsonify({"error": "No images part in the request"}), 400
        
    files = request.files.getlist('images')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No images selected"}), 400
        
    inventory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'new_inventory'))
    if not os.path.exists(inventory_dir):
        os.makedirs(inventory_dir)
        
    saved_count = 0
    file_paths = []
    for file in files:
        if file.filename:
            # We want to preserve folder structure if possible, but werkzeug's secure_filename drops paths.
            # To handle folders safely but preserve paths, we can use the original filename and secure parts.
            # For simplicity, we just save flat if secure_filename is used, or preserve dirs.
            # Let's save flat for now to ensure security.
            filename = secure_filename(os.path.basename(file.filename))
            filepath = os.path.join(inventory_dir, filename)
            file.save(filepath)
            saved_count += 1
            file_paths.append(filepath)
            
    if saved_count == 0:
        return jsonify({"error": "Failed to save any images"}), 500
        
    try:
        # Define the background task
        def process_in_background(paths):
            with _inventory_lock:
                try:
                    from services.similarity_search import add_inventory
                    from services.llm_helper import generate_image_metadata
                    
                    print(f"[Background Task] Starting processing for {len(paths)} images in chunks...")
                    
                    metadata_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'metadata.json'))
                    import json as _json
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as _f:
                            all_metadata = _json.load(_f)
                    else:
                        all_metadata = {}

                    from api.routes import IMAGE_METADATA_BY_FILENAME
                    
                    # Process in small batches so UI gets live updates
                    chunk_size = 50
                    for i in range(0, len(paths), chunk_size):
                        chunk_paths = paths[i:i+chunk_size]
                        
                        print(f"[Background Task] Processing chunk {i//chunk_size + 1}...")
                        
                        # 1. Add chunk to Qdrant
                        add_inventory(chunk_paths)
                        
                        # 2. Generate Metadata for chunk
                        for fpath in chunk_paths:
                            norm_path = fpath.replace('\\', '/')
                            if norm_path not in all_metadata:
                                meta = generate_image_metadata(fpath)
                                if meta:
                                    all_metadata[norm_path] = meta
                                    IMAGE_METADATA_BY_FILENAME[os.path.basename(fpath)] = meta
                                    print(f"[Background] Generated metadata for {os.path.basename(fpath)}")
                                    
                        # Save metadata after each chunk. Write to a temp file and
                        # rename: a plain open('w') left the ~2 MB file truncated
                        # for a moment, and any search served during that window
                        # read invalid JSON and lost all result metadata.
                        try:
                            tmp_file = metadata_file + '.tmp'
                            with open(tmp_file, 'w', encoding='utf-8') as _f:
                                _json.dump(all_metadata, _f, indent=2, ensure_ascii=False)
                            os.replace(tmp_file, metadata_file)
                        except Exception as meta_err:
                            print(f"[Background Task] Warning: Could not save metadata.json: {meta_err}")

                    print(f"[Background Task] Finished processing all {len(paths)} images!")
                except Exception as e:
                    print(f"[Background Task] Error processing inventory: {e}")

        # Start the background thread
        import threading
        thread = threading.Thread(target=process_in_background, args=(file_paths,))
        thread.daemon = True
        thread.start()

        return jsonify({
            "message": f"Successfully uploaded {saved_count} images! They are now being processed in the background (which may take a few minutes).",
            "logs": ""
        })
    except Exception as e:
        print(f"Error starting background processing: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/voice_search', methods=['POST'])
def voice_search():
    """
    Accepts an audio file, transcribes it using faster-whisper, translates it to English,
    and performs a similarity search on Qdrant using the translated text.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio selected"}), 400

    limit = int(request.form.get('limit', 100))
    offset = int(request.form.get('offset', 0))
    
    # language hint can be 'te-IN', we just take the first two letters 'te'
    voice_lang = request.form.get('voice_lang', 'en')
    language_hint = voice_lang.split('-')[0] if '-' in voice_lang else voice_lang
    
    # Save audio to a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    try:
        import time
        start_time = time.time()
        
        # 1. Transcribe audio
        from services.voice_service import transcribe_audio
        query = transcribe_audio(temp_audio_path, language_hint=language_hint)
        print(f"[Voice Search] Transcribed Text: '{query}'")
        
        if not query:
            return jsonify({"error": "Could not recognize any speech."}), 400
            
        # 2a. Multi-Language Translation (Google Translate)
        translated_query = translate_to_english(query)
        print(f"[Voice Search] Translated: '{translated_query}'")

        # 2b. LLM Understanding (ChatGPT-like intent extraction)
        from services.llm_helper import extract_jewelry_keywords
        final_search_query = extract_jewelry_keywords(translated_query)
        
        import services.similarity_search as ss
        clip_model = ss._ensure_clip_model()
        device = ss.device
        from services.jewelry_validator import validate_jewelry_text
        
        is_jewelry, j_score, nj_score = validate_jewelry_text(final_search_query, clip_model, device)
        print(f"CLIP Text Validation - Query: '{final_search_query}' | Jewelry: {j_score:.4f} | Non-Jewelry: {nj_score:.4f} | Accepted: {is_jewelry}")
        
        if final_search_query.strip() == "NOT_JEWELRY" or not is_jewelry:
            return jsonify({"error": "Enter the Prompt Related to Jewellery"}), 400

        # 3. Perform Similarity Search
        # Strict category search: infer the jewellery category from the
        # transcribed/translated query the same way text search does, so
        # voice search results never mix categories either. Inferred from
        # translated_query (not the LLM-extracted keyword phrase below,
        # which can drop the literal category word entirely).
        category = infer_category_from_text(translated_query)
        results, db_time_ms = find_similar_by_text(final_search_query, top_k=limit, offset=offset, category=category)
        enrich_results_with_metadata(results, force_category=category)

        _log_response_summary("VOICE SEARCH", results, extra={
            "transcribed": query,
            "translated_query": translated_query,
            "llm_query": final_search_query,
            "category_filter": category,
            "search_time_ms": db_time_ms,
        })

        return jsonify({
            "message": "Voice search completed successfully",
            "query": query,
            "translated_query": translated_query,
            "llm_query": final_search_query,
            "category_filter": category,
            "results": results,
            "search_time_ms": db_time_ms
        })
    except Exception as e:
        print(f"Error during voice search: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@api_blueprint.route('/compare', methods=['POST'])
def compare():
    data = request.get_json(silent=True) or request.form
    img1_path = data.get('img1')
    img2_path = data.get('img2')
    
    if not img1_path or not img2_path:
        return jsonify({"error": "Both img1 and img2 are required"}), 400

    temp_files = []
    try:
        from services.llm_helper import compare_jewelry_images
        
        # Paths sent by frontend might be relative URLs like /api/images/... or absolute http://...
        if 'api/images/' in img1_path:
            img1_path = img1_path.split('api/images/')[-1]
        if 'api/images/' in img2_path:
            img2_path = img2_path.split('api/images/')[-1]
            
        import urllib.parse
        img1_path = urllib.parse.unquote(img1_path)
        img2_path = urllib.parse.unquote(img2_path)
        
        # Resolve to real local paths (same logic as serve_image)
        def resolve_path(p):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            p_norm = p.replace('\\', '/')
            if os.path.exists(p): return p
            if os.path.exists(p_norm): return p_norm
            
            dataset_rel = blob_storage.strip_dataset_prefix(p_norm).lstrip('/')
            mapped = os.path.join(base_dir, 'Jewellery_Data', dataset_rel.replace('/', os.sep))
            if os.path.exists(mapped): return mapped
                
            if 'dataset/' in p_norm.lower():
                rel_part = p_norm[p_norm.lower().find('dataset/'):]
                mapped = os.path.join(base_dir, 'backend', rel_part.replace('/', os.sep))
                if os.path.exists(mapped): return mapped
                
            if 'new_inventory/' in p_norm.lower():
                rel_part = p_norm[p_norm.lower().find('new_inventory/'):]
                mapped = os.path.join(base_dir, 'backend', rel_part.replace('/', os.sep))
                if os.path.exists(mapped): return mapped

            if 'uploads/' in p_norm.lower():
                rel_part = p_norm[p_norm.lower().find('uploads/'):]
                mapped = os.path.join(base_dir, 'backend', rel_part.replace('/', os.sep))
                if os.path.exists(mapped): return mapped

            rel_mapped = os.path.join(base_dir, p_norm.lstrip('/'))
            if os.path.exists(rel_mapped): return rel_mapped

            # Not on local disk - materialize from Blob Storage into a temp
            # file so downstream code (which needs a real path) still works.
            # No-op / returns p unchanged if Blob Storage isn't configured.
            blob_key = blob_storage.to_blob_key(p_norm)
            data = blob_storage.download_bytes(blob_key)
            if data is not None:
                import tempfile
                suffix = os.path.splitext(blob_key)[1] or '.jpg'
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(data)
                tmp.close()
                temp_files.append(tmp.name)
                return tmp.name

            return p
            
        real_img1 = resolve_path(img1_path)
        real_img2 = resolve_path(img2_path)
        
        # Pre-fetch metadata for both images using the already-cached in-memory
        # metadata so compare_jewelry_images doesn't have to re-read the 56 MB
        # metadata.json on every compare request.
        meta_dict, meta_by_filename, meta_by_relpath = get_current_metadata()
        
        def fetch_meta_for(img_real_path, img_url_path):
            """Try every lookup strategy to find the metadata record. The URL
            path is tried too: a Blob Storage image resolves to a temp file
            whose name has nothing to do with its catalog entry."""
            for candidate in (img_real_path, img_url_path):
                norm = candidate.replace('\\', '/')
                # 1. Full absolute path
                m = meta_dict.get(norm)
                if m: return m
                # 2. Relative path (strips known prefixes)
                rel = _strip_known_prefixes(norm)
                m = meta_by_relpath.get(rel)
                if m: return m
                # 3. Basename
                m = meta_by_filename.get(os.path.basename(norm))
                if m: return m
                # 4. Stem (no extension)
                stem = os.path.splitext(os.path.basename(norm))[0]
                m = meta_by_filename.get(stem)
                if m: return m
            return {}
        
        pre_meta1 = fetch_meta_for(real_img1, img1_path) or {}
        pre_meta2 = fetch_meta_for(real_img2, img2_path) or {}
        
        result = compare_jewelry_images(real_img1, real_img2,
                                        pre_meta1=pre_meta1, pre_meta2=pre_meta2)
        if "error" in result:
             return jsonify(result), 500
             
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /compare: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
