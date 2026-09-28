import os
import json
from PIL import Image as PILImage

# Load API key from .env if not in environment
API_KEY = os.environ.get("GROQ_API_KEY", os.environ.get("GEMINI_API_KEY", os.environ.get("OpenAI_API_KEY", "")))
if not API_KEY:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(BASE_DIR, '.env'))
    API_KEY = os.environ.get("GROQ_API_KEY", os.environ.get("GEMINI_API_KEY", os.environ.get("OpenAI_API_KEY", "")))

if API_KEY:
    print("Groq API key found. Using Groq for LLM and Vision tasks.")
else:
    print("Warning: No API_KEY found. Vision metadata and LLM features will be disabled.")

# ===== PERFORMANCE OPTIMIZATION: CLIP Embedding Cache =====
# Cache for CLIP text embeddings to avoid recomputing same descriptions
_CLIP_TEXT_EMBEDDING_CACHE = {}

def get_cached_clip_embeddings(clip_model, texts):
    """
    Returns CLIP embeddings for text queries, using cache to avoid recomputation.
    texts: list of text strings
    Returns: list of embeddings (same order as texts)
    """
    uncached = [t for t in texts if t not in _CLIP_TEXT_EMBEDDING_CACHE]
    
    if uncached:
        new_embeddings = clip_model.encode(uncached, convert_to_tensor=True)
        for text, embedding in zip(uncached, new_embeddings):
            _CLIP_TEXT_EMBEDDING_CACHE[text] = embedding
    
    return [_CLIP_TEXT_EMBEDDING_CACHE[t] for t in texts]


def extract_jewelry_keywords(user_query: str) -> str:
    """
    Acts like ChatGPT to understand the user's intent and extract visual keywords
    for our CLIP similarity search.
    """
    prompt = f"""
    You are an expert jewelry assistant. A user is searching for jewelry in a database.
    Your job is to understand their intent and output ONLY a comma-separated list of visual keywords 
    that describe the jewelry they are looking for. Do not include introductory text.

    IMPORTANT: If the user's query is completely unrelated to jewelry or gold ornaments (for example, asking for "dog images", "cars", "food", etc.), you MUST output EXACTLY the word "NOT_JEWELRY" and nothing else.
    
    User Query: "{user_query}"
    
    Example 1:
    User: "I need something heavy to wear for a south indian wedding"
    Output: heavy traditional design, temple jewelry, yellow gold, intricate patterns
    
    Example 2: 
    User: "mujhe simple daily wear chain chahiye"
    Output: simple chain, minimalist modern design, lightweight gold

    Example 3:
    User: "show me some dog images"
    Output: NOT_JEWELRY
    
    Now, process the following query and output only the visual keywords (or NOT_JEWELRY if unrelated):
    User: "{user_query}"
    Output:
    """

    if API_KEY:
        try:
            import requests
            import time
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": [{"role": "user", "content": prompt}]
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload,
                    timeout=15
                )
                if response.status_code == 429 and attempt < max_retries - 1:
                    # Handle rate limiting
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    print(f"Groq text rate limit (429). Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                break
                
            keywords = response.json()['choices'][0]['message']['content'].strip()
            print(f"[LLM Insight] Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            print(f"LLM Error (Groq): {e}")

    # Fallback: return query as-is with jewelry appended
    print("Warning: Groq API unavailable. Falling back to exact query.")
    return user_query + " jewelry"


def generate_image_metadata(image_path):
    """
    Generates metadata using the Gemini API or OpenRouter API.
    """
    try:
        import os
        import json
        import base64
        import requests
        from PIL import Image as PILImage
        
        if not API_KEY:
            print("API Key not initialized. Cannot generate metadata.")
            return {
                "category": "Unknown",
                "product_type": "Unknown",
                "metal_color": "Unknown",
                "gold_purity": "Unknown",
                "shape": "Unknown",
                "type_of_wear": "Unknown",
                "stone_type": "Unknown",
                "stone_cut": "Unknown",
                "gross_weight_g": 0,
                "net_weight_g": 0,
                "stone_weight_g": 0,
                "estimated_cost_inr": 0,
                "gold_rate_inr_g": 0,
                "gst_amount_inr": 0,
                "total_price_incl_gst_inr": 0,
                "gst_rate": "Unknown",
                "size": "Unknown",
                "confidence_score": 0,
                "description": "Metadata generation disabled (No API Key).",
                "info": "Metadata failed",
                "features": ["metadata_failed"],
                "api_error": "No API Key"
            }

        prompt = """
        Analyze this jewelry image and extract detailed structured metadata.
        Output ONLY valid JSON matching this exact schema:
        {
            "category": "String (e.g., Bangles, Bracelets, Chains, Ear Cuff Chains, Ear Rings, Hair Braid Ornaments, Maang Tikka, Necklaces, Nose Pin, Rings, Straps(Pattilu), Toe Rings, Waist Chain)",
            "product_type": "String (e.g., Yellow Gold Bangle, Diamond Necklace)",
            "metal_color": "String (e.g., Yellow Gold, White Gold, Rose Gold, Platinum) or 'Unknown'",
            "gold_purity": "String (e.g., 18K, 22K, 24K) or 'Unknown'",
            "shape": "String (e.g., Round, Oval, Teardrop, Square, Floral) or 'Unknown'",
            "type_of_wear": "String (e.g., Traditional, Modern, Bridal, Daily Wear) or 'Unknown'",
            "stone_type": "String (e.g., Diamond, Ruby, Emerald, Pearl, CZ, No Stone) or 'Unknown'",
            "stone_cut": "String (e.g., Round, Princess, Pear) or 'Unknown'",
            "gross_weight_g": "Number (estimate weight in grams based on visual appearance) or 0",
            "net_weight_g": "Number (estimate net weight in grams) or 0",
            "stone_weight_g": "Number (estimate stone weight in grams) or 0",
            "estimated_cost_inr": "Number (estimate price in INR based on weight and standard gold rate) or 0",
            "gold_rate_inr_g": "Number (e.g., 7000 or current rate) or 0",
            "gst_amount_inr": "Number (estimate 3% of cost) or 0",
            "total_price_incl_gst_inr": "Number (estimate total cost including GST) or 0",
            "gst_rate": "String (e.g., '3%')",
            "size": "String (e.g., 'Medium', 'Large', '2.4', etc.) or 'Unknown'",
            "confidence_score": "Number (0.8 to 1.0 confidence of this visual analysis)",
            "description": "String (Full descriptive sentence of the item)",
            "info": "String (Up to 4 words summarizing the image)",
            "features": ["list", "of", "visual", "keywords", "e.g.", "Ruby", "Round", "Bridal"]
        }
        
        Important Guidelines:
        1. Always provide an estimated value for weights and prices based on the visual bulk and type of the jewelry. Do not use null for these numerical fields so they can be compared properly.
        2. Ensure the "features" array includes key visually identifying tags (e.g. stones, styles) as these are used for text-search keyword boosting.
        3. ONLY return the JSON object, with no markdown formatting or extra text.
        """
        
        text = ""
        
        # Use Groq Vision via requests
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        import time
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload,
                timeout=(10, float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))),
            )
            if response.status_code == 429 and attempt < max_retries - 1:
                # Handle rate limiting
                retry_after = float(response.headers.get("Retry-After", 3 ** attempt))
                print(f"Groq vision rate limit (429). Retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            break
            
        text = response.json()['choices'][0]['message']['content'].strip()
        
        if text.startswith('```json'):
            text = text[7:-3].strip()
        elif text.startswith('```'):
            text = text[3:-3].strip()
            
        result = json.loads(text)
        print(f"[Vision Metadata] {os.path.basename(image_path)} -> {result}")
        return result
    except Exception as e:
        print(f"[Vision Metadata] Error for {image_path}: {e}")
        return {
            "category": "Unknown",
            "product_type": "Unknown",
            "metal_color": "Unknown",
            "gold_purity": "Unknown",
            "shape": "Unknown",
            "type_of_wear": "Unknown",
            "stone_type": "Unknown",
            "stone_cut": "Unknown",
            "gross_weight_g": 0,
            "net_weight_g": 0,
            "stone_weight_g": 0,
            "estimated_cost_inr": 0,
            "gold_rate_inr_g": 0,
            "gst_amount_inr": 0,
            "total_price_incl_gst_inr": 0,
            "gst_rate": "Unknown",
            "size": "Unknown",
            "confidence_score": 0,
            "description": "Metadata generation failed due to API error.",
            "info": "Metadata failed",
            "features": ["metadata_failed"],
            "api_error": str(e)
        }

def detect_jewelry_attributes(image_path, image_features=None):
    """
    Detects jewelry attributes like stone type and shape using CLIP and visual analysis.
    Optimized for fast inference with caching and vectorized operations.
    image_features: optional pre-computed CLIP image embedding (shape [1, dim]) to
    avoid re-opening/re-encoding the image when the caller already has it.
    """
    try:
        from services.similarity_search import clip_model, device
        from PIL import Image
        import numpy as np
        import torch

        # Stone type detection
        stone_queries = {
            "ruby": ["ruby stone", "red gemstone", "red precious stone"],
            "emerald": ["emerald stone", "green gemstone", "green precious stone"],
            "sapphire": ["sapphire stone", "blue gemstone", "blue precious stone"],
            "diamond": ["diamond stone", "white gemstone", "brilliant stone"],
            "pearl": ["pearl", "pearlescent", "lustrous pearl"],
            "no stones": ["plain gold", "no gemstones", "minimalist design"],
        }
        
        # Shape detection - aligned with catalog features
        shape_queries = {
            "round": ["round shape", "circular", "spherical", "round design"],
            "oval": ["oval shape", "elliptical", "elongated round"],
            "square": ["square shape", "boxy design"],
            "rectangular": ["rectangular shape", "long rectangular"],
            "teardrop": ["teardrop shape", "pear shaped", "drop shaped", "drop"],
            "heart shape": ["heart shape", "love shape"],
            "floral pattern": ["flower pattern", "floral design", "botanical pattern"],
            "geometric pattern": ["geometric pattern", "angular design", "geometric shapes"]
        }
        
        # Encode image with CLIP (single pass) unless the caller already has it
        if image_features is None:
            img = Image.open(image_path).convert("RGB")
            image_features = clip_model.encode([img], convert_to_tensor=True)

        detected_attrs = {"stones": [], "shapes": []}
        
        # Vectorized stone detection using cache
        all_stone_descs = []
        stone_type_map = {}
        for stone_type, descriptions in stone_queries.items():
            for desc in descriptions:
                all_stone_descs.append(desc)
                stone_type_map[desc] = stone_type
        
        # Get all embeddings at once (cached)
        text_features = get_cached_clip_embeddings(clip_model, all_stone_descs)
        
        # Compute similarities in batch
        if isinstance(text_features, list):
            text_features = torch.stack(text_features)
        similarities = image_features[0] @ text_features.T
        
        # Aggregate scores by stone type
        stone_scores = {}
        for i, desc in enumerate(all_stone_descs):
            stone_type = stone_type_map[desc]
            if stone_type not in stone_scores:
                stone_scores[stone_type] = []
            stone_scores[stone_type].append(float(similarities[i]))
        
        for stone_type, scores in stone_scores.items():
            avg_score = np.mean(scores)
            if avg_score > 0.35:
                detected_attrs["stones"].append((stone_type, avg_score))
        
        # Vectorized shape detection using cache
        all_shape_descs = []
        shape_type_map = {}
        for shape_type, descriptions in shape_queries.items():
            for desc in descriptions:
                all_shape_descs.append(desc)
                shape_type_map[desc] = shape_type
        
        text_features = get_cached_clip_embeddings(clip_model, all_shape_descs)
        
        if isinstance(text_features, list):
            text_features = torch.stack(text_features)
        similarities = image_features[0] @ text_features.T
        
        # Aggregate scores by shape type
        shape_scores = {}
        for i, desc in enumerate(all_shape_descs):
            shape_type = shape_type_map[desc]
            if shape_type not in shape_scores:
                shape_scores[shape_type] = []
            shape_scores[shape_type].append(float(similarities[i]))
        
        for shape_type, scores in shape_scores.items():
            avg_score = np.mean(scores)
            if avg_score > 0.32:
                detected_attrs["shapes"].append((shape_type, avg_score))
        
        # Sort by score and get top attributes
        detected_attrs["stones"] = sorted(detected_attrs["stones"], key=lambda x: x[1], reverse=True)[:2]
        detected_attrs["shapes"] = sorted(detected_attrs["shapes"], key=lambda x: x[1], reverse=True)[:2]
        
        return detected_attrs
    except Exception as e:
        print(f"Error in attribute detection: {e}")
        import traceback
        traceback.print_exc()
        return {"stones": [], "shapes": []}


def _build_default_metadata_features(category, meta):
    """Generate a reasonable set of catalog tags when metadata.json has no
    (or an empty) "features" list for an image - the same fallback the
    search results list already applies via routes._build_default_features,
    so the Compare page never shows a bare "Not cataloged" for an image that
    simply lacks hand-labeled features."""
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


def extract_comprehensive_features(image_path, pre_loaded_meta=None, category=None, image_features=None):
    """
    Extracts comprehensive features for gold jewelry using:
    1. Metadata features (pre-cataloged)
    2. CLIP semantic features (visual properties)
    3. ConvNeXt visual characteristics (texture, weight appearance, craftsmanship)

    Returns a structured feature dictionary with scoring.
    pre_loaded_meta: optional dict already fetched from metadata.json (avoids re-reading)
    image_features: optional pre-computed CLIP image embedding (shape [1, dim]) to
    avoid re-opening/re-encoding the image when the caller already has it.
    """
    try:
        from services.similarity_search import clip_model, device
        from PIL import Image
        import numpy as np
        import os
        import json
        
        # Load metadata
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        METADATA_FILE = os.path.join(BASE_DIR, 'metadata.json')
        
        features_dict = {
            "metadata_features": [],
            "metadata_price": None,
            "stone_features": [],
            "design_features": [],
            "color_features": [],
            "craftsmanship_features": [],
            "style_features": [],
            "weight_features": []
        }
        
        # 1. Use pre-loaded metadata if provided, otherwise scan metadata.json
        if pre_loaded_meta:
            meta = pre_loaded_meta
            if "features" in meta:
                features_dict["metadata_features"] = meta["features"]
            if "total_price_incl_gst_inr" in meta:
                features_dict["metadata_price"] = meta["total_price_incl_gst_inr"]
            elif "estimated_cost_inr" in meta:
                features_dict["metadata_price"] = meta["estimated_cost_inr"]
            features_dict["raw_metadata"] = meta
        elif os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Extract image info
            abs_path = os.path.abspath(image_path).replace('\\', '/')
            img_basename = os.path.basename(image_path)
            path_parts = abs_path.split('/')
            
            # Find the category (directory name) - usually the part after 'dataset/'
            img_category = None
            for i, part in enumerate(path_parts):
                if part.lower() == 'dataset' and i + 1 < len(path_parts):
                    img_category = path_parts[i + 1].lower()
                    break
            
            # Try to find matching entry in metadata
            found = False
            for full_path, meta in metadata.items():
                full_path_lower = full_path.lower()
                meta_basename = os.path.basename(full_path_lower)
                
                # Strategy 1: Direct filename match with category verification
                if meta_basename == img_basename.lower():
                    # Verify category matches if we extracted it
                    is_match = False
                    if img_category:
                        meta_parts = full_path_lower.split('/')
                        if len(meta_parts) > 1:
                            meta_category = meta_parts[-2]  # directory before filename
                            if meta_category.lower() == img_category:
                                is_match = True
                    else:
                        # No category info - just match basename
                        is_match = True
                        
                    if is_match:
                        if "features" in meta:
                            features_dict["metadata_features"] = meta["features"]
                        if "total_price_incl_gst_inr" in meta:
                            features_dict["metadata_price"] = meta["total_price_incl_gst_inr"]
                        elif "estimated_cost_inr" in meta:
                            features_dict["metadata_price"] = meta["estimated_cost_inr"]
                        features_dict["raw_metadata"] = meta
                        found = True
                        break

        # No hand-labeled "features" in metadata.json for this image (or no
        # catalog entry found at all) - generate the same default tags the
        # search results list shows instead of leaving it blank.
        if not features_dict["metadata_features"]:
            features_dict["metadata_features"] = _build_default_metadata_features(
                category, features_dict.get("raw_metadata")
            )

        # 2. Load image for CLIP analysis unless the caller already has the embedding
        if image_features is None:
            img = Image.open(image_path).convert("RGB")
            image_features = clip_model.encode([img], convert_to_tensor=True)

        # 3. Define DISTINCTIVE feature queries for better discrimination
        stone_queries = {
            "ruby stones": ["beautiful red ruby gemstones", "deep crimson ruby precious stones", "bright red rubies"],
            "emerald stones": ["green emerald precious stones", "rich green emerald gems", "deep green emeralds"],
            "sapphire stones": ["blue sapphire gemstones", "deep blue sapphire stones", "sapphire precious gems"],
            "diamond stones": ["sparkling clear diamonds", "brilliant diamond gemstones", "shiny diamonds"],
            "pearl": ["lustrous white pearl", "smooth pearl gemstones", "pearlescent white gems"],
            "semi-precious stones": ["semi-precious mixed stones", "colored semi-precious gems", "affordable stones"],
            "unset gold": ["plain smooth gold surface", "gold without any stones", "blank unmarked gold"]
        }
        
        design_queries = {
            "heavy traditional": ["heavy ornate intricate traditional Indian design", "temple style ornamental jewelry", "elaborate traditional work"],
            "minimalist modern": ["minimalist simple modern contemporary design", "sleek clean lines", "simple modern jewelry"],
            "floral pattern": ["flower petal botanical designs", "floral motif patterns", "rose and flower designs"],
            "geometric": ["geometric angular mathematical shapes", "angular sharp patterns", "geometric symmetric design"],
            "filigree": ["delicate fine filigree wirework", "intricate filigree detail", "ornate filigree patterns"],
            "layered": ["multiple layered tiers and levels", "stacked layered design", "multi-tier construction"],
            "carved": ["hand carved sculptural relief", "carved relief ornamental details", "carved patterns"]
        }
        
        color_queries = {
            "yellow gold": ["warm bright yellow gold", "pure yellow gold color", "traditional yellow gold"],
            "white gold": ["pale white silver-toned metal", "whitened precious metal", "white color alloy"],
            "rose gold": ["warm copper-pink rose gold", "rose pink toned gold", "copper rose color"],
            "polished": ["smooth shiny mirror polished surface", "high gloss bright finish", "reflective polished"],
            "matte": ["dull matte non-reflective surface", "brushed satin finish", "flat matte appearance"]
        }
        
        craftsmanship_queries = {
            "high precision": ["perfect flawless precision execution", "excellent high quality craftsmanship", "perfectly done work"],
            "handcrafted": ["hand-made artisan handcrafted", "manual artisanal workmanship", "hand crafted by artisan"],
            "intricate details": ["very detailed intricate fine work", "delicate detailed ornaments", "highly detailed elements"],
            "weight/substantial": ["heavy dense solid construction", "substantial sturdy weight", "heavy looking piece"],
            "lightweight": ["light delicate thin construction", "lightweight airy design", "delicate thin piece"]
        }
        
        style_queries = {
            "traditional Indian": ["traditional South Indian temple style", "ethnic Indian cultural design", "traditional Indian jewelry"],
            "contemporary": ["contemporary modern trendy design", "up-to-date current style", "modern contemporary"],
            "vintage/antique": ["vintage antique retro historical style", "antique looking classic", "vintage period design"],
            "bohemian": ["bohemian free-spirited artistic style", "boho eclectic design", "bohemian creative"],
            "bridal/formal": ["bridal wedding formal ceremonial", "formal wedding jewelry", "bridal elaborate"],
            "casual/daily": ["casual everyday simple lightweight", "daily simple wear", "casual simple design"]
        }
        
        weight_queries = {
            "heavy": ["visually heavy dense appearance", "heavy looking substantial piece", "dense heavy construction"],
            "medium weight": ["medium balanced standard appearance", "moderate weight", "balanced construction"],
            "lightweight": ["light delicate thin appearance", "lightweight thin design", "delicate light piece"]
        }
        
        # 4. Smart scoring with normalization for accurate discrimination
        def score_category_discriminative(queries_dict, category_name):
            results = []
            feature_scores = {}
            
            # First pass: score all features
            for feature_name, descriptions in queries_dict.items():
                text_features = get_cached_clip_embeddings(clip_model, descriptions)
                if isinstance(text_features, list):
                    import torch
                    text_features = torch.stack(text_features)
                
                # Compute similarity for each description
                similarities = image_features[0] @ text_features.T
                scores = [float(s) for s in similarities]
                
                # Use mean of scores
                avg_score = float(np.mean(scores))
                feature_scores[feature_name] = avg_score
            
            # Normalize scores within this category to spread them out
            if feature_scores:
                scores_list = list(feature_scores.values())
                min_score = min(scores_list)
                max_score = max(scores_list)
                score_range = max_score - min_score
                
                # Apply discriminative transformation
                for feature_name, raw_score in feature_scores.items():
                    # Normalize to 0-1 range within category
                    if score_range > 0.01:  # Avoid division by very small numbers
                        normalized = (raw_score - min_score) / score_range
                    else:
                        # All scores similar - spread them slightly
                        normalized = 0.5
                    
                    # Apply selective filtering - only keep meaningful differences
                    # Features in top 50% of category scores
                    if normalized > 0.25:
                        # Map to realistic confidence range (30-95%)
                        final_score = 0.30 + (normalized * 0.65)
                        final_score = min(final_score, 0.95)  # Cap at 95% to avoid 100%
                        results.append((feature_name, round(final_score, 3)))
            
            # Sort by score descending and limit to top 3 per category
            return sorted(results, key=lambda x: x[1], reverse=True)[:3]
        
        # 5. Extract all feature categories with improved discrimination
        features_dict["stone_features"] = score_category_discriminative(stone_queries, "stones")
        features_dict["design_features"] = score_category_discriminative(design_queries, "design")
        features_dict["color_features"] = score_category_discriminative(color_queries, "color")
        features_dict["craftsmanship_features"] = score_category_discriminative(craftsmanship_queries, "craftsmanship")
        features_dict["style_features"] = score_category_discriminative(style_queries, "style")
        features_dict["weight_features"] = score_category_discriminative(weight_queries, "weight")
        
        return features_dict
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return {
            "metadata_features": [],
            "stone_features": [],
            "design_features": [],
            "color_features": [],
            "craftsmanship_features": [],
            "style_features": [],
            "weight_features": []
        }


def compare_jewelry_images(image_path_1, image_path_2, pre_meta1=None, pre_meta2=None):
    """
    Enhanced comparison of two jewelry images using local ML models.
    Now includes comprehensive feature analysis from metadata, ConvNeXt, and CLIP.
    pre_meta1 / pre_meta2: metadata dicts pre-loaded by the routes layer (avoids
    re-reading the 56 MB metadata.json inside this function).
    """
    try:
        from services.similarity_search import transform, model_imagenet, device, clip_model, CLASS_NAMES, _ensure_class_model
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import os

        # Ensure paths exist
        if not os.path.exists(image_path_1) or not os.path.exists(image_path_2):
             return {"error": "One or both images could not be found."}

        # 2. Load and transform images once, reused for classification, CLIP
        # encoding and similarity scoring below (each image was previously
        # opened/encoded up to 4 separate times across these steps).
        img1_pil = Image.open(image_path_1).convert("RGB")
        img2_pil = Image.open(image_path_2).convert("RGB")

        img1_tensor = transform(img1_pil).unsqueeze(0).to(device)
        img2_tensor = transform(img2_pil).unsqueeze(0).to(device)

        # 1. Classify both images in a single batched forward pass
        class_model = _ensure_class_model()
        with torch.no_grad():
            class_batch = torch.cat([img1_tensor, img2_tensor], dim=0)
            class_output = class_model(class_batch)
            class_probs = F.softmax(class_output, dim=1)
        class_conf, class_pred = torch.max(class_probs, dim=1)
        conf1, cat1 = class_conf[0].item(), CLASS_NAMES[class_pred[0].item()]
        conf2, cat2 = class_conf[1].item(), CLASS_NAMES[class_pred[1].item()]

        # Batch-encode both images with CLIP once, shared by attribute
        # detection and comprehensive feature extraction below instead of
        # each re-opening the file and re-running CLIP independently.
        clip_features = clip_model.encode([img1_pil, img2_pil], convert_to_tensor=True)
        img1_clip_features = clip_features[0:1]
        img2_clip_features = clip_features[1:2]

        # 1b. Use pre-loaded metadata when provided by the routes layer.
        # Fall back to scanning metadata.json only if nothing was pre-loaded.
        import json
        meta1 = pre_meta1 if pre_meta1 else {}
        meta2 = pre_meta2 if pre_meta2 else {}

        if not meta1 or not meta2:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            METADATA_FILE = os.path.join(BASE_DIR, 'metadata.json')
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                def find_meta(img_path):
                    basename = os.path.basename(img_path).lower()
                    for k, v in metadata.items():
                        if os.path.basename(k).lower() == basename:
                            return v
                    return {}

                if not meta1:
                    meta1 = find_meta(image_path_1)
                if not meta2:
                    meta2 = find_meta(image_path_2)

        # Fallback: generate metadata via LLM vision for images not in catalog
        # (e.g. user-uploaded images that were never indexed). Both are
        # independent network calls to Groq (up to ~60s each with retries),
        # so run them concurrently instead of back-to-back to avoid doubling
        # the worst-case wait.
        if not meta1 or not meta2:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(generate_image_metadata, image_path_1) if not meta1 else None
                future2 = executor.submit(generate_image_metadata, image_path_2) if not meta2 else None

                if future1 is not None:
                    print(f"[Compare] meta1 not in catalog, generating via LLM for {os.path.basename(image_path_1)}")
                    generated = future1.result()
                    # A failed vision call returns a placeholder error dict (features:
                    # ["metadata_failed"]) - treat that the same as no metadata at all
                    # so extract_comprehensive_features falls back to auto-generated
                    # tags instead of showing "metadata_failed" as a real feature.
                    # However, we STILL want the other placeholder keys (like estimated_cost_inr=0)
                    # to be present so the UI doesn't show an empty column.
                    if generated:
                        meta1 = generated
                if future2 is not None:
                    print(f"[Compare] meta2 not in catalog, generating via LLM for {os.path.basename(image_path_2)}")
                    generated = future2.result()
                    if generated:
                        meta2 = generated

        if meta1 and meta1.get("category"):
            cat1 = meta1.get("category")
        if meta2 and meta2.get("category"):
            cat2 = meta2.get("category")

        # 3. Extract ConvNeXt features in batch (images/tensors already loaded above)
        with torch.no_grad():
            # Process both images together for better GPU utilization
            batch = torch.cat([img1_tensor, img2_tensor], dim=0)
            features = model_imagenet(batch)
            features = features.view(features.size(0), -1)
            features = F.normalize(features, p=2, dim=1)

            f1 = features[0:1]
            f2 = features[1:2]

        # 4. Compute similarity
        sim = F.cosine_similarity(f1, f2).item()

        # 5. Detect attributes (stone type and shape), reusing the batched CLIP embeddings
        attrs1 = detect_jewelry_attributes(image_path_1, image_features=img1_clip_features)
        attrs2 = detect_jewelry_attributes(image_path_2, image_features=img2_clip_features)

        # 6. Extract comprehensive features, passing pre-loaded metadata and the
        # already-computed CLIP embedding to avoid re-reading metadata.json and
        # re-encoding the image.
        comp_features1 = extract_comprehensive_features(image_path_1, pre_loaded_meta=meta1 or None, category=cat1, image_features=img1_clip_features)
        comp_features2 = extract_comprehensive_features(image_path_2, pre_loaded_meta=meta2 or None, category=cat2, image_features=img2_clip_features)
        
        # Prioritize shape from catalog metadata if available
        catalog_shapes = ["round", "oval", "square", "rectangular", "teardrop", "heart shape", "drop", "floral pattern", "floral", "geometric pattern", "geometric", "layered"]
        meta_shape1 = next((f.lower() for f in comp_features1.get("metadata_features", []) if f.lower() in catalog_shapes), None)
        meta_shape2 = next((f.lower() for f in comp_features2.get("metadata_features", []) if f.lower() in catalog_shapes), None)
        
        if meta_shape1 and meta_shape1 not in [s[0] for s in attrs1["shapes"]]:
            attrs1["shapes"].insert(0, (meta_shape1, 1.0))
        elif meta_shape1:
            attrs1["shapes"].sort(key=lambda x: (x[0]==meta_shape1, x[1]), reverse=True)
            
        if meta_shape2 and meta_shape2 not in [s[0] for s in attrs2["shapes"]]:
            attrs2["shapes"].insert(0, (meta_shape2, 1.0))
        elif meta_shape2:
            attrs2["shapes"].sort(key=lambda x: (x[0]==meta_shape2, x[1]), reverse=True)

        # Extract top stone and shape
        stone1 = attrs1["stones"][0][0] if attrs1["stones"] else "Unknown"
        stone2 = attrs2["stones"][0][0] if attrs2["stones"] else "Unknown"
        shape1 = attrs1["shapes"][0][0] if attrs1["shapes"] else "Unknown"
        shape2 = attrs2["shapes"][0][0] if attrs2["shapes"] else "Unknown"
        
        # 7. Build dynamic text based on comparisons
        diffs = []
        sims = []
        
        if cat1 == cat2:
            sims.append(f"Both items belong to the '{cat1}' category.")
        else:
            diffs.append(f"Image 1 is categorized as '{cat1}', while Image 2 is '{cat2}'.")
        
        # Stone comparison
        if stone1 == stone2:
            sims.append(f"Both feature {stone1} stones.")
        else:
            diffs.append(f"Image 1 has {stone1}, while Image 2 has {stone2}.")
        
        # Shape comparison
        if shape1 == shape2:
            sims.append(f"Both have a {shape1} shape/design.")
        else:
            diffs.append(f"Image 1 features a {shape1} shape, Image 2 has {shape2}.")
        
        # Overall design similarity
        if sim > 0.95:
            sims.append("The designs are almost mathematically identical.")
        elif sim > 0.85:
            sims.append("The designs share very strong visual patterns and structure.")
            if shape1 != shape2:
                diffs.append("Despite similar aesthetics, they have different shapes.")
        elif sim > 0.70:
            sims.append("They share some overarching style elements.")
            diffs.append("They have noticeable differences in their core design.")
        else:
            diffs.append("These items look completely different in structure and style.")

        return {
            "image1_info": f"Analysis -> Detected Category: {cat1}",
            "image2_info": f"Analysis -> Detected Category: {cat2}",
            "summary": "Comparison completed successfully.",
            "differences": diffs,
            "similarities": sims,
            "match_score": round(sim * 100),
            "attributes": {
                "image1": {
                    "category": cat1,
                    "stone_type": stone1,
                    "shape": shape1,
                    "all_stones": [s[0] for s in attrs1["stones"]],
                    "all_shapes": [sh[0] for sh in attrs1["shapes"]]
                },
                "image2": {
                    "category": cat2,
                    "stone_type": stone2,
                    "shape": shape2,
                    "all_stones": [s[0] for s in attrs2["stones"]],
                    "all_shapes": [sh[0] for sh in attrs2["shapes"]]
                }
            },
            "comprehensive_features": {
                "image1": comp_features1,
                "image2": comp_features2
            }
        }
    except Exception as e:
        print(f"Error during local comparison: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

