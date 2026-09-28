import os
import json
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
NEW_INVENTORY_DIR = os.path.join(BASE_DIR, "backend", "new_inventory")
OUTPUT_FILE = os.path.join(BASE_DIR, "metadata.json")

print("Initializing ML Service for Metadata Extraction...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP model
print("Loading CLIP model...")
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)

# 13 Classes Feature Vocabularies
CATEGORIES = [
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

FEATURES = [
    "round", "teardrop", "square", "floral pattern", "geometric design",
    "layered", "choker", "stud", "dangle", "chain", "ruby stones",
    "emerald stones", "diamond accents", "heavy traditional design",
    "minimalist modern design", "heart shape", "kundan work",
    "pearl embellishment", "antique gold finish", "temple jewelry"
]

# Pre-encode text features for speed
print("Pre-encoding text features...")
with torch.no_grad():
    cat_embeddings = clip_model.encode([f"a photo of a gold {c.lower()}" for c in CATEGORIES], convert_to_tensor=True)
    feat_embeddings = clip_model.encode([f"jewelry with {f}" for f in FEATURES], convert_to_tensor=True)

def get_image_paths(directories):
    paths = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    paths.append(os.path.join(root, file))
    return paths

def infer_category_from_path(filepath):
    # Match path components after 'dataset' or 'new_inventory'
    lower_path = filepath.lower().replace('\\', '/')
    if "ear cuff" in lower_path or "earcuff" in lower_path: return "Ear Cuff Chains"
    if "hair braid" in lower_path or "hairbraid" in lower_path or "savaram" in lower_path: return "Hair Braid Ornaments"
    if "maang" in lower_path or "tikka" in lower_path: return "Maang Tikka"
    if "nose" in lower_path: return "Nose Pin"
    if "toe" in lower_path: return "Toe Rings"
    if "waist" in lower_path or "vaddanam" in lower_path: return "Waist Chain"
    if "pattilu" in lower_path or "strap" in lower_path: return "Straps(Pattilu)"
    if "bangle" in lower_path: return "Bangles"
    if "bracelet" in lower_path: return "Bracelets"
    if "necklace" in lower_path: return "Necklaces"
    if "chain" in lower_path: return "Chains"
    if "ear ring" in lower_path or "earring" in lower_path or "/ear" in lower_path: return "Ear Rings"
    if "/ring" in lower_path or "ring" in lower_path: return "Rings"
    return None

def process_images(image_paths, max_images=None):
    if max_images:
        image_paths = image_paths[:max_images]
        
    metadata = {}
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        if i % 10 == 0:
            print(f"Processing {i}/{total}...")
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            with torch.no_grad():
                img_emb = clip_model.encode(image, convert_to_tensor=True)
                
                # 1. Determine Category (Prefer folder path, fallback to CLIP)
                category = infer_category_from_path(img_path)
                if not category:
                    cat_scores = torch.nn.functional.cosine_similarity(img_emb, cat_embeddings)
                    best_cat_idx = cat_scores.argmax().item()
                    category = CATEGORIES[best_cat_idx]
                
                # 2. Extract Features
                feat_scores = torch.nn.functional.cosine_similarity(img_emb, feat_embeddings)
                
                # Get top 3 features
                top_scores, top_indices = torch.topk(feat_scores, k=3)
                
                image_features = []
                for score, idx in zip(top_scores, top_indices):
                    # Only include feature if score is reasonably high (e.g., > 0.22 for CLIP)
                    if score.item() > 0.22:
                        image_features.append(FEATURES[idx.item()])
            
            # Use absolute path as key for portability
            abs_path = img_path.replace('\\', '/')
            metadata[abs_path] = {
                "category": category,
                "features": image_features
            }
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    return metadata

if __name__ == "__main__":
    print(f"Scanning for images in:\n - {DATASET_DIR}\n - {NEW_INVENTORY_DIR}")
    all_images = get_image_paths([DATASET_DIR, NEW_INVENTORY_DIR])
    print(f"Found {len(all_images)} total images.")
    
    # Process all images
    results = process_images(all_images)
    
    print(f"Saving metadata to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done! Metadata extraction is complete.")
