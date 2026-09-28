"""
Backfill images into all three Qdrant search collections
(gold_ornaments, gold_ornaments_clip, gold_ornaments_clip_w) using the
exact same add_inventory() pipeline the live web app uses.

Why this exists:
1. backend/new_inventory/ - uploads normally go through /api/upload_inventory,
   which processes images in a background thread inside the Flask process.
   Every dev-server restart killed that thread mid-batch, leaving most of
   these images un-embedded.
2. Jewellery_Data/Necklaces/ - this folder was fully indexed into
   gold_ornaments (990/993) but has ZERO points in gold_ornaments_clip -
   the collection live image search actually queries - because whatever
   one-time migration built that CLIP collection ran before this folder
   was included. Live image search can never find these necklaces until
   they are embedded here too.

Both were previously blocked by a separate bug in add_inventory()'s
duplicate check (now fixed) that compared ConvNeXt classifier features
with too tight a threshold, wrongly flagging almost every new image as a
duplicate of some unrelated already-indexed item. Safe to re-run: the
duplicate check (now CLIP-based) skips images that are already properly
indexed.

Usage (from anywhere - it fixes up sys.path):
    python scripts/reprocess_new_inventory.py                 # new_inventory + Necklaces
    python scripts/reprocess_new_inventory.py <dir> [<dir>...] # specific folders
"""
import os
import sys
import time

# Filenames scraped from real product listings routinely contain emoji
# (e.g. a literal U+1F496 in a filename crashed a print() mid-run here,
# and - worse - the `except Exception: print(...)` handler around it ALSO
# crashed trying to print the same filename in its error message, which
# is an uncaught fatal crash that kills the whole multi-hour job instead
# of just skipping one file). errors='replace' makes every print() safe
# against this on Windows' cp1252 console.
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

from services.similarity_search import add_inventory

DEFAULT_DIRS = [
    os.path.join(BACKEND_DIR, "new_inventory"),
    os.path.join(PROJECT_ROOT, "Jewellery_Data", "Necklaces"),
]
CHUNK_SIZE = 50
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(directory):
    if not os.path.isdir(directory):
        print(f"  (skipping - not a directory: {directory})")
        return []
    # Recursive: some categories (Maang Tikka, Waist Chain, ...) store images
    # in nested sub-folders (e.g. "Maang Tikka/ad_stone_tikka/*.jpg"), not
    # flat in the category directory itself - os.listdir() alone silently
    # found 0 images for those.
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                found.append(os.path.join(root, f))
    return found


def main():
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

    print(f"\nProcessing {total} images in chunks of {CHUNK_SIZE}...")
    print("(Already-indexed images are detected via CLIP similarity and skipped.)\n")

    started = time.time()
    total_added = 0
    for i in range(0, total, CHUNK_SIZE):
        chunk = images[i:i + CHUNK_SIZE]
        t0 = time.time()
        added = add_inventory(chunk)
        total_added += added
        elapsed = time.time() - t0
        done = min(i + CHUNK_SIZE, total)
        print(f"[{done}/{total}]  +{added} new this chunk  "
              f"({elapsed:.1f}s)  total added so far: {total_added}")

    print(f"\nDone in {time.time() - started:.1f}s. "
          f"Added {total_added} new embeddings out of {total} images scanned.")


if __name__ == "__main__":
    main()
