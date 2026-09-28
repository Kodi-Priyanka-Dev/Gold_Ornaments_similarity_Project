import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLLECTION_NAME = "gold_ornaments"

# ==========================================================
# QDRANT CONNECTION
# ==========================================================
load_dotenv(os.path.join(BASE_DIR, '.env'))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    total_count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Total vectors in database: {total_count}")
except Exception as e:
    print(f"Error accessing collection '{COLLECTION_NAME}': {e}")
    sys.exit(1)

print("\nFetching all vectors...")
points = []
offset = None
while True:
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )
    points.extend(result[0])
    offset = result[1]
    if offset is None:
        break

print(f"Successfully fetched {len(points)} vectors.")

print("\nPerforming similarity search for all vectors to find duplicates (score > 0.999)...")
duplicates_found = 0
processed_pairs = set()

for point in tqdm(points):
    # Perform similarity search using the vector of this point
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=point.vector,
        limit=5  # Top 5 should be enough to find immediate duplicates
    )
    
    current_url = point.payload.get("image_url")
    
    for match in search_result:
        match_url = match.payload.get("image_url")
        
        # Skip self-matches
        if match.id == point.id or current_url == match_url:
            continue
            
        # Only care about very high similarities
        if match.score > 0.999:
            # Create a unique key for the pair to avoid printing A->B and B->A
            pair_key = tuple(sorted([current_url, match_url]))
            if pair_key not in processed_pairs:
                processed_pairs.add(pair_key)
                duplicates_found += 1
                print(f"\n[DUPLICATE] Score: {match.score:.4f}")
                print(f"  -> {current_url}")
                print(f"  -> {match_url}")

print(f"\nSearch complete! Found {duplicates_found} unique duplicate pairs across all {total_count} items in the database.")
