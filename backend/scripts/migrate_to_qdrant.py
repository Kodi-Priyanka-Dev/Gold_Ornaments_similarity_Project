import os
import sys
import pickle
import json
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# Adjust paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")

IMAGENET_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "imagenet_image_features.npy")
IMAGENET_PATHS_PATH = os.path.join(EMBEDDINGS_DIR, "imagenet_image_paths.pkl")
QDRANT_DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

COLLECTION_NAME = "gold_ornaments"
VECTOR_SIZE = 1024

def extract_category(path):
    path_lower = path.lower().replace('\\', '/')
    if 'ear cuff' in path_lower or 'earcuff' in path_lower: return 'Ear Cuff Chains'
    if 'hair braid' in path_lower or 'hairbraid' in path_lower or 'savaram' in path_lower: return 'Hair Braid Ornaments'
    if 'maang' in path_lower or 'tikka' in path_lower: return 'Maang Tikka'
    if 'nose' in path_lower: return 'Nose Pin'
    if 'toe' in path_lower: return 'Toe Rings'
    if 'waist' in path_lower or 'vaddanam' in path_lower: return 'Waist Chain'
    if 'pattilu' in path_lower or 'strap' in path_lower: return 'Straps(Pattilu)'
    if 'bangle' in path_lower: return 'Bangles'
    if 'bracelet' in path_lower: return 'Bracelets'
    if 'necklace' in path_lower: return 'Necklaces'
    if 'chain' in path_lower: return 'Chains'
    if 'earring' in path_lower or 'ear ring' in path_lower or '/ear' in path_lower: return 'Ear Rings'
    if '/ring' in path_lower or 'ring' in path_lower: return 'Rings'
    return 'Unknown'

def migrate():
    print(f"Loading vectors from: {IMAGENET_EMBEDDINGS_PATH}")
    if not os.path.exists(IMAGENET_EMBEDDINGS_PATH):
        print("Error: Embeddings not found. Please ensure feature extraction has been run.")
        sys.exit(1)
        
    features = np.load(IMAGENET_EMBEDDINGS_PATH)
    
    print(f"Loading paths from: {IMAGENET_PATHS_PATH}")
    with open(IMAGENET_PATHS_PATH, "rb") as f:
        paths = pickle.load(f)
        
    total_images = len(paths)
        
    print(f"Loaded {total_images} records.")
    
    metadata_dict = {}
    if os.path.exists(METADATA_PATH):
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
    else:
        print(f"Warning: {METADATA_PATH} not found. Fallback to basic paths.")
    
    from dotenv import load_dotenv
    QDRANT_URL = os.getenv("QDRANT_URL", "https://50179a5e-43ed-4eca-99bf-61004fc188e3.sa-east-1-0.aws.cloud.qdrant.io")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NDQyZTlkODEtYzY5Mi00MmRmLTkwZjItOWRlZWE3OTg4OTRmIn0.V4TEcRNZx_o8hY3JCUBqblYGz5nicPja-jxjDyO4hbY")
    print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
    
    # Increased timeout to 180 seconds to allow for larger batch processing
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=180.0)
    
    # Check if collection exists
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
        
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    
    print("Creating payload indexes for 'category' and 'features'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="features",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    print("Uploading points to Qdrant in batches...")
    
    # Set to a safe 250 batch size. Larger batches take longer on Qdrant's end.
    batch_size = 250 
    for i in range(0, total_images, batch_size):
        end_idx = min(i + batch_size, total_images)
        
        points = []
        for j in range(i, end_idx):
            point_id = j + 1
            # Normalizing the path separators for consistency
            abs_path = paths[j].replace('\\', '/')
            
            file_meta = metadata_dict.get(abs_path, {})
            cat = file_meta.get("category", extract_category(abs_path))
            feat_list = file_meta.get("features", [])
            
            payload = {
                "image_url": abs_path, 
                "category": cat,
                "features": feat_list
            }
            
            # The features array is likely already normalized by torch.F.normalize
            vector = features[j].tolist()
            
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
            
        retries = 3
        for attempt in range(retries):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Upsert warning: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Upsert failed after {retries} attempts: {e}")
                    
        print(f"  Uploaded batch {i} to {end_idx}...")
        
    print(f"Migration complete! Uploaded {total_images} records into Qdrant.")
    
    print("\n---------------------------------------------------------")
    print("Core dataset migration complete! Now restoring new inventory items...")
    add_script = os.path.join(os.path.dirname(__file__), "add_new_inventory.py")
    if os.path.exists(add_script):
        import subprocess
        subprocess.run([sys.executable, add_script])
    else:
        print("add_new_inventory.py not found.")

if __name__ == "__main__":
    migrate()
