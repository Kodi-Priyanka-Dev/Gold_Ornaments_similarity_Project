"""
Upload the existing local image dataset to Azure Blob Storage so it has a
durable copy off local disk, without touching Qdrant or metadata.json at
all: the blob key for each file is derived from the same relative path
serve_image()/add_inventory() already use, so nothing needs re-indexing.

This is a pure copy - it does not read/modify Qdrant, does not re-embed
anything, and does not delete the local files (serve_image() keeps
preferring local disk; Blob Storage is only the fallback / durable copy).

Usage (from anywhere - it fixes up sys.path):
    python scripts/backfill_blob_storage.py                  # Jewellery_Data + new_inventory
    python scripts/backfill_blob_storage.py <dir> [<dir>...]  # specific folders
"""
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    except Exception:
        pass

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from services import blob_storage

DEFAULT_DIRS = [
    os.path.join(PROJECT_ROOT, "Jewellery_Data"),
    os.path.join(BACKEND_DIR, "new_inventory"),
]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(directory):
    if not os.path.isdir(directory):
        print(f"  (skipping - not a directory: {directory})")
        return []
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                found.append(os.path.join(root, f))
    return found


def main():
    if not blob_storage.is_enabled():
        print("Blob Storage is not configured (no AZURE_STORAGE_ACCOUNT_URL or "
              "AZURE_STORAGE_CONNECTION_STRING in .env). Nothing to do.")
        return

    dirs = sys.argv[1:] or DEFAULT_DIRS

    images = []
    for d in dirs:
        found = collect_images(d)
        print(f"{d}: {len(found)} images")
        images.extend(found)

    total = len(images)
    if total == 0:
        print("No images found. Nothing to do.")
        return

    print(f"\nUploading {total} images to container "
          f"'{blob_storage._CONTAINER_NAME}'...\n")

    started = time.time()
    uploaded = 0
    failed = 0
    for i, path in enumerate(images, 1):
        if blob_storage.upload_file(path):
            uploaded += 1
        else:
            failed += 1
        if i % 100 == 0 or i == total:
            elapsed = time.time() - started
            print(f"[{i}/{total}]  uploaded={uploaded} failed={failed}  ({elapsed:.1f}s)")

    print(f"\nDone in {time.time() - started:.1f}s. "
          f"Uploaded {uploaded}/{total} ({failed} failed).")


if __name__ == "__main__":
    main()
