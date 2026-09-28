# ============================================================
# JEWELLERY IMAGE CLASSIFICATION - ConvNeXt @ 320
# LOCAL TRAINING SCRIPT (VS Code / Antigravity / any terminal)
# ============================================================
# This is the Colab notebook cell rewritten as a normal, runnable
# Python program. What changed and why:
#
#   1. No google.colab import at module level. Drive mounting is
#      attempted only if the Colab package is actually importable.
#   2. Every path is derived from the repo location, or from a CLI
#      flag / env var. No hard-coded /content or /content/drive.
#   3. `cp -r` replaced with shutil.copytree, so the optional local
#      cache works on Windows as well as Linux/macOS.
#   4. The whole pipeline lives under `if __name__ == "__main__"`.
#      On Windows and macOS, DataLoader workers are spawned by
#      re-importing this file; without that guard, num_workers > 0
#      re-runs the entire script inside every worker.
#   5. CLI flags for everything that used to be a constant to edit,
#      so the same file serves a real run and a 60-second smoke test.
#   6. CPU / CUDA / MPS are all detected, AMP is enabled only where
#      it actually exists (CUDA), and channels_last is only applied
#      on CUDA where it helps.
#   7. All text files are written as UTF-8 explicitly - the Windows
#      default (cp1252) mangles non-ASCII characters in reports.
#   8. The product catalog is READ from the spreadsheets in <repo>/metadata
#      (--metadata-dir), not generated and not fetched. No network call,
#      no API key. Rows are joined onto the images on disk by relative
#      path; anything the catalog does not cover is reported, and only
#      then filled in from a fixed offline rate table.
#
# Metadata columns removed by request:
#   Metal Type, Product URL, Stone Clarity
#
# ------------------------------------------------------------
# QUICK START (from the repo root)
# ------------------------------------------------------------
#   pip install -r requirements.txt
#
#   # smoke test: proves the pipeline end-to-end in a few minutes
#   python main.py/convnext_train.py --model convnext_tiny --image-size 224 \
#       --epochs 1 --limit-per-class 8 --batch-size 4 --workers 0
#
#   # real run (needs a CUDA GPU to be practical)
#   python main.py/convnext_train.py
#
#   python main.py/convnext_train.py --help    # every knob
# ============================================================

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # never try to open a GUI window
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Scraped catalogues almost always contain a few half-downloaded JPEGs.
# Without this, one truncated file kills an epoch mid-way.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 1. STATIC CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

GST_RATE = 0.03                 # standard GST on gold jewellery in India

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Directory names that are NEVER a class. Without this guard, a stray
# nested "train" folder is silently picked up by ImageFolder as a class.
NON_CLASS_DIR_NAMES = {
    "train", "training", "val", "valid", "validation",
    "test", "testing", "tmp", "temp", "results", "output", "outputs",
    ".ipynb_checkpoints", "__macosx",
}

SCRATCH_DIR_NAMES = ("results", "output", "outputs")

# Where the dataset is looked for when --data-root is not given.
DATA_ROOT_CANDIDATES = [
    os.environ.get("JEWELLERY_DATA_ROOT"),
    os.path.join(PROJECT_ROOT, "dataset"),
    os.path.join(PROJECT_ROOT, "Jewellery_Data"),
    os.path.join(PROJECT_ROOT, "Jewellery_Data_Cleaned"),
    os.path.join(PROJECT_ROOT, "Jewellery_Dataset"),
    "/content/drive/MyDrive/Jewellary_Dataset",   # still works if run in Colab
]


# ============================================================
# 2. CLI
# ============================================================

def build_arg_parser():

    parser = argparse.ArgumentParser(
        description="Train a ConvNeXt jewellery classifier locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--data-root", default=None,
        help="Dataset root holding train/ (+ optional val/), or one folder per "
             "class. Default: first existing of $JEWELLERY_DATA_ROOT, "
             "<repo>/dataset, <repo>/Jewellery_Data, <repo>/Jewellery_Dataset.",
    )
    paths.add_argument(
        "--results-root", default=None,
        help="Where checkpoints, logs, plots and reports go. "
             "Default: <repo>/output/<model>_<image-size>.",
    )
    paths.add_argument(
        "--copy-to-local", default=None, metavar="DIR",
        help="Copy the dataset into DIR before training and read from there. "
             "Only worth it when the dataset sits on a slow drive "
             "(Google Drive, SMB share). Off by default - local disk is fast.",
    )
    paths.add_argument(
        "--force-recopy", action="store_true",
        help="Discard an existing --copy-to-local cache and copy again.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model", default="convnext_base",
        help="Any timm model name, e.g. convnext_tiny, convnext_small.",
    )
    model_group.add_argument("--image-size", type=int, default=320)
    model_group.add_argument(
        "--no-pretrained", action="store_true",
        help="Random init instead of downloading ImageNet weights (offline runs).",
    )

    optim_group = parser.add_argument_group("optimisation")
    optim_group.add_argument("--epochs", type=int, default=25)
    optim_group.add_argument(
        "--batch-size", type=int, default=0,
        help="Physical batch size fed to the device. 0 = pick from GPU memory.",
    )
    optim_group.add_argument(
        "--effective-batch-size", type=int, default=16,
        help="Batch size the optimizer sees. Anything above --batch-size is "
             "made up with gradient accumulation.",
    )
    optim_group.add_argument("--lr", type=float, default=1e-4)
    optim_group.add_argument("--weight-decay", type=float, default=0.05)
    optim_group.add_argument("--label-smoothing", type=float, default=0.1)
    optim_group.add_argument("--warmup-epochs", type=int, default=1)
    optim_group.add_argument(
        "--val-split", type=float, default=0.20,
        help="Only used when there is no val/ folder.",
    )
    optim_group.add_argument("--seed", type=int, default=42)
    optim_group.add_argument(
        "--no-class-weights", action="store_true",
        help="Disable inverse-frequency class weighting.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto",
                         choices=["auto", "cuda", "cpu", "mps"])
    runtime.add_argument(
        "--workers", type=int, default=None,
        help="DataLoader workers. Default: min(4, cpu_count). "
             "Use 0 if you hit worker crashes.",
    )
    runtime.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed precision (it is CUDA-only anyway).",
    )
    runtime.add_argument(
        "--resume", action="store_true",
        help="Continue from checkpoint.pth instead of starting fresh.",
    )
    runtime.add_argument(
        "--limit-per-class", type=int, default=0,
        help="Keep at most N images per class. For smoke tests.",
    )

    catalog = parser.add_argument_group("catalog metadata")
    catalog.add_argument(
        "--no-metadata", action="store_true",
        help="Skip the product catalog step entirely.",
    )
    catalog.add_argument(
        "--metadata-dir", default=None,
        help="Folder holding the provided catalog spreadsheet "
             "(jewellery_metadata.csv / .xlsx, plus the optional "
             "class_index_mapping sheet). "
             f"Default: {os.path.join('<repo>', 'metadata')}.",
    )
    catalog.add_argument(
        "--strict-metadata", action="store_true",
        help="Abort if any training/validation image has no row in the "
             "provided catalog, instead of filling those rows in offline.",
    )

    return parser


# ============================================================
# 3. SEED / DEVICE
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # benchmark=True is a real speed win, and bit-exact
    # reproducibility is not needed here
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(requested):
    """auto -> cuda, else Apple MPS, else cpu."""

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda was requested but torch.cuda.is_available() is "
                f"False.\nInstalled torch: {torch.__version__} "
                f"(CUDA build: {torch.version.cuda})\n"
                "A CPU-only wheel is installed. Reinstall a CUDA build, e.g.:\n"
                "  pip install --index-url https://download.pytorch.org/whl/cu124 "
                "torch torchvision"
            )
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def auto_batch_size(gpu_gb):
    """Conservative, safe batch sizes for ConvNeXt-Base at 320."""
    if gpu_gb >= 38:
        return 32           # A100 40/80GB
    if gpu_gb >= 22:
        return 16           # L4 / A10 / 3090
    if gpu_gb >= 14:
        return 8            # T4 / V100 16GB
    if gpu_gb > 0:
        return 4
    return 4                # CPU / MPS


# ============================================================
# 4. DATASET LAYOUT HELPERS
# ============================================================

def resolve_data_root(explicit):
    """Return the dataset root, or exit with something actionable."""

    if explicit:
        root = os.path.abspath(explicit)
        if not os.path.isdir(root):
            raise SystemExit(
                f"--data-root does not exist:\n  {root}\n"
                "Check the spelling and that any network drive is mounted."
            )
        return root

    for candidate in DATA_ROOT_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)

    tried = "\n".join(f"  {c}" for c in DATA_ROOT_CANDIDATES if c)
    raise SystemExit(
        "No dataset folder found. Tried:\n" + tried +
        "\n\nPass one explicitly:\n"
        "  python main.py/convnext_train.py --data-root D:/path/to/dataset\n"
        "or set the JEWELLERY_DATA_ROOT environment variable."
    )


def resolve_split_dirs(root):
    """
    Supports either:
        root/train/<class>/*.jpg   (+ optional root/val, root/valid, root/test)
    or:
        root/<class>/*.jpg         (single folder, split at runtime)
    """
    train_candidates = ["train", "Train", "training", "Training"]
    val_candidates = ["val", "valid", "validation", "Val", "Valid", "Validation",
                      "test", "Test"]

    train_dir = None
    for name in train_candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            train_dir = path
            break

    val_dir = None
    if train_dir is not None:
        for name in val_candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                val_dir = path
                break
    else:
        # flat layout: root itself holds the class folders
        train_dir = root

    return train_dir, val_dir


def count_images_recursive(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        # never count the script's own output folder
        dirs[:] = [d for d in dirs if d.lower() not in SCRATCH_DIR_NAMES]
        total += sum(1 for f in files if f.lower().endswith(IMAGE_EXTENSIONS))
    return total


def per_class_counts(split_dir):
    """
    {class_name: image_count} for every class folder in a split dir.

    A total-image count is not enough to verify a copy: a network drive can
    return an incomplete directory listing, so the source itself gets
    undercounted, the copy copies that much, and a total-vs-total check
    happily agrees with its own bad number. Comparing per class catches it.
    """
    counts = {}
    if not os.path.isdir(split_dir):
        return counts
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if entry.name.lower() in NON_CLASS_DIR_NAMES or entry.name.startswith("."):
            continue
        counts[entry.name] = count_images_recursive(entry.path)
    return counts


def validate_split_dir(split_dir, split_name, local_cache_dir=None):
    """
    Refuse to proceed when a class folder exists but holds no images.

    Without this, ImageFolder fails several frames deep with
    "Found no valid file for the classes ..." which says nothing about
    whether the problem is the source or the local cache.
    """
    counts = per_class_counts(split_dir)

    if not counts:
        raise SystemExit(
            f"No class folders found in {split_name} dir:\n  {split_dir}\n"
            "Expected one subfolder per class."
        )

    empty = [name for name, n in counts.items() if n == 0]

    if empty:
        lines = [
            f"{len(empty)} of {len(counts)} class folders in {split_name}/ contain "
            "no images.\n",
            f"  dir: {split_dir}\n",
            "Per-class counts:",
        ]
        for name, n in counts.items():
            mark = "   <-- EMPTY" if n == 0 else ""
            lines.append(f"  {name:<28} {n:>6}{mark}")
        if local_cache_dir:
            lines += [
                "",
                "If the images ARE present at the source, the cache is the problem:",
                f"  delete {local_cache_dir}",
                "  ...then re-run (or pass --force-recopy), or drop --copy-to-local",
                "  entirely to read straight from the source.",
            ]
        lines += [
            "",
            "If they are missing at the source too, the train/val/test split did",
            "not finish - it created the folders but never moved the files.",
        ]
        raise SystemExit("\n".join(lines))

    print(f"  {split_name}: {len(counts)} classes, {sum(counts.values())} images")
    return counts


def local_copy(src, dst_root, dst_name, force_recopy=False):
    """
    Copy src -> dst_root/dst_name with per-class verification.

    A completion sentinel is written only after the copy succeeds. Without
    it, an interrupted copy leaves a non-empty directory that looks 'done'
    and gets reused forever - which silently trains on partial data.
    """
    if src is None:
        return None

    dst = os.path.join(dst_root, dst_name)
    sentinel = os.path.join(dst, ".copy_complete")

    source_counts = per_class_counts(src)
    source_total = sum(source_counts.values())

    empty_at_source = [c for c, n in source_counts.items() if n == 0]
    if empty_at_source:
        raise RuntimeError(
            f"{len(empty_at_source)} class folder(s) are empty at the SOURCE "
            f"({src}):\n  " + ", ".join(empty_at_source) +
            "\nNothing to copy for those. Fix the dataset before training."
        )

    def mismatches(other_counts):
        """Classes whose local count does not match the source."""
        bad = []
        for class_name, n in source_counts.items():
            if other_counts.get(class_name, 0) != n:
                bad.append((class_name, n, other_counts.get(class_name, 0)))
        return bad

    if os.path.exists(sentinel) and not force_recopy:
        local_counts = per_class_counts(dst)
        bad = mismatches(local_counts)
        if not bad:
            print(
                f"  {dst_name}: complete copy present "
                f"({sum(local_counts.values())} images, "
                f"{len(local_counts)} classes) - reusing."
            )
            return dst
        print(f"  {dst_name}: cache is incomplete - recopying. Mismatched classes:")
        for class_name, want, got in bad[:8]:
            print(f"      {class_name:<28} source {want:>6}  local {got:>6}")
        if len(bad) > 8:
            print(f"      ... and {len(bad) - 8} more")
    elif os.path.isdir(dst):
        reason = "--force-recopy set" if force_recopy else "no completion sentinel"
        print(f"  {dst_name}: {reason} - discarding partial copy and recopying.")

    # Remove any partial/stale tree so the copy cannot merge into it.
    # This only ever deletes the cache under --copy-to-local, never the source.
    if os.path.isdir(dst):
        shutil.rmtree(dst)

    os.makedirs(dst, exist_ok=True)

    print(f"  {dst_name}: copying {source_total} images "
          f"across {len(source_counts)} classes...")

    # shutil, not `cp -r`: this has to work on Windows too.
    shutil.copytree(src, dst, dirs_exist_ok=True)

    # Verify PER CLASS, not just the total.
    local_counts = per_class_counts(dst)
    bad = mismatches(local_counts)
    if bad:
        detail = "\n".join(
            f"  {class_name:<28} source {want:>6}  local {got:>6}"
            for class_name, want, got in bad
        )
        raise RuntimeError(
            f"copy verification failed for {dst_name} - "
            f"{len(bad)} class(es) did not copy fully:\n{detail}"
        )

    with open(sentinel, "w", encoding="utf-8") as f:
        json.dump(source_counts, f, indent=2)

    print(
        f"  {dst_name}: verified {sum(local_counts.values())} images "
        f"across {len(local_counts)} classes."
    )
    return dst


# ============================================================
# 5. DATASETS
# ============================================================

class CleanImageFolder(datasets.ImageFolder):
    """
    ImageFolder that refuses to treat split/scratch directories as classes.

    Plain ImageFolder makes a class out of every subdirectory it finds, so a
    nested "train/" folder becomes an extra class and quietly corrupts the
    label space. Anything in NON_CLASS_DIR_NAMES, and any dotfile directory,
    is skipped with a warning instead.
    """

    def find_classes(self, directory):

        entries = sorted(e.name for e in os.scandir(directory) if e.is_dir())

        kept, skipped = [], []
        for name in entries:
            if name.lower() in NON_CLASS_DIR_NAMES or name.startswith("."):
                skipped.append(name)
            else:
                kept.append(name)

        if skipped:
            print("\n" + "!" * 70)
            print(f"WARNING: skipped non-class folders in {directory}:")
            for name in skipped:
                count = count_images_recursive(os.path.join(directory, name))
                print(f"  {name}  ({count} images ignored)")
            print("\nIf one of these is real data, move it out of the split dir")
            print("or rename it. It is NOT being used for training.")
            print("!" * 70)

        if not kept:
            raise SystemExit(
                f"No usable class folders in {directory}. Skipped: {skipped}"
            )

        return kept, {name: i for i, name in enumerate(kept)}


class TransformSubset(Dataset):
    """
    Subset of an ImageFolder with its OWN transform.

    This avoids the common bug of doing
        val_subset.dataset = other_dataset
    which mutates a dataset object shared with the training subset.
    """

    def __init__(self, image_folder, indices, transform):
        self.samples = [image_folder.samples[i] for i in indices]
        self.loader = image_folder.loader
        self.transform = transform
        self.classes = image_folder.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def paths(self):
        return [p for p, _ in self.samples]


def build_transforms(image_size):

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def limit_samples_per_class(image_folder, limit, seed):
    """
    Keep at most `limit` images per class. Used for smoke tests, so the
    whole pipeline can be exercised in a couple of minutes.
    """
    if not limit or limit <= 0:
        return

    rng = random.Random(seed)

    by_class = {}
    for path, target in image_folder.samples:
        by_class.setdefault(target, []).append((path, target))

    kept = []
    for target in sorted(by_class):
        items = by_class[target]
        rng.shuffle(items)
        kept.extend(items[:limit])

    kept.sort(key=lambda item: item[0])

    image_folder.samples = kept
    image_folder.imgs = kept
    image_folder.targets = [target for _, target in kept]

    print(f"  --limit-per-class {limit}: kept {len(kept)} images "
          f"across {len(by_class)} classes.")


# ============================================================
# 6. PRODUCT CATALOG METADATA
# ============================================================
# The catalog is READ from the spreadsheets the user provides in
# --metadata-dir (default <repo>/metadata), not invented here and
# not fetched from any service:
#
#   jewellery_metadata.csv | .xlsx   the catalog itself, one row per
#                                    image, keyed by "Image Path"
#                                    (e.g. train/Bangles/020_0001.jpg)
#   class_index_mapping.csv | .xlsx  optional; cross-checked against
#                                    the folder scan and warned about
#
# Rows are matched to the images on disk by relative path. Matching is
# done on Unicode-normalised paths because the same filename can be
# stored NFC on one filesystem and NFD on another - in this dataset 11
# scraped filenames with accents (buză, délicat, Fußkettchen, ...) match
# only after normalising, and a naive string compare drops them.
#
# The provided sheet defines the output schema: whatever columns it has
# are the columns written out. The synthetic generator below is kept
# only as an offline fallback for images the sheet does not cover, and
# it never touches the network - it uses FIXED_GOLD_RATE_PER_GRAM.
#
# Columns removed by request: Metal Type, Product URL, Stone Clarity
# ============================================================

CATALOG_BASENAMES = ("jewellery_metadata", "metadata", "jewellery_catalog")
CATALOG_EXTENSIONS = (".csv", ".xlsx", ".xlsm", ".xls")
CLASS_MAPPING_BASENAMES = ("class_index_mapping", "class_mapping", "classes")

CATALOG_KEY_COLUMN = "Image Path"
CATALOG_CATEGORY_COLUMN = "Category"

def normalize_class_name(name):
    """
    "Ear Cuff Chains"  -> "earcuffchains"
    "Straps(Pattilu)"  -> "strapspattilu"
    "Maang_Tikka"      -> "maangtikka"
    """
    out = name.lower()
    for character in ["_", "-", " ", "(", ")", ".", "&", "'"]:
        out = out.replace(character, "")
    return out


# Folder names that do not normalise straight onto a CLASS_SPECS key
CLASS_ALIASES = {
    "strapspattilu": "straps",
    "strapspattilus": "straps",
    "pattilu": "straps",
    "pattilus": "straps",
    "anklets": "straps",
    "payals": "straps",
    "earring": "earrings",
    "earcuffchain": "earcuffchains",
    "hairbraidornament": "hairbraidornaments",
    "jadabillai": "hairbraidornaments",
    "nosepins": "nosepin",
    "nosering": "nosepin",
    "noserings": "nosepin",
    "toering": "toerings",
    "waistchains": "waistchain",
    "bangle": "bangles",
    "bracelet": "bracelets",
    "chain": "chains",
    "necklace": "necklaces",
    "ring": "rings",
    "maangtikkas": "maangtikka",
}


CLASS_SPECS = {

    "bangles": {
        "product_type": "Bangle",
        "weight_range": (12, 45),
        "stone_range": (0, 5),
        "shapes": ["Round", "Floral", "Geometric", "Traditional", "Designer"],
        "wear": ["Traditional", "Bridal", "Party Wear", "Daily Wear"],
        "size_type": "Diameter (in)",
        "size_range": (2.2, 2.8),
    },

    "bracelets": {
        "product_type": "Bracelet",
        "weight_range": (8, 35),
        "stone_range": (0, 4),
        "shapes": ["Round", "Oval", "Floral", "Geometric", "Designer"],
        "wear": ["Daily Wear", "Party Wear", "Traditional", "Contemporary"],
        "size_type": "Length (in)",
        "size_range": (6.0, 8.5),
    },

    "chains": {
        "product_type": "Chain",
        "weight_range": (8, 50),
        "stone_range": (0, 3),
        "shapes": ["Round", "Geometric", "Traditional", "Designer"],
        "wear": ["Daily Wear", "Traditional", "Party Wear"],
        "size_type": "Length (in)",
        "size_range": (16, 30),
    },

    "earcuffchains": {
        "product_type": "Ear Cuff Chain",
        "weight_range": (2, 12),
        "stone_range": (0, 2),
        "shapes": ["Curved", "Floral", "Chain", "Traditional", "Designer"],
        "wear": ["Traditional", "Bridal", "Party Wear", "Contemporary"],
        "size_type": "Length (cm)",
        "size_range": (3, 10),
    },

    "earrings": {
        "product_type": "Earrings",
        "weight_range": (2, 25),
        "stone_range": (0, 4),
        "shapes": ["Round", "Oval", "Floral", "Jhumka", "Geometric", "Designer"],
        "wear": ["Traditional", "Bridal", "Party Wear", "Daily Wear"],
        "size_type": "Size",
        "size_fixed": "One Size",
    },

    "hairbraidornaments": {
        "product_type": "Hair Braid Ornament",
        "weight_range": (5, 30),
        "stone_range": (0, 5),
        "shapes": ["Floral", "Peacock", "Leaf", "Temple", "Traditional", "Designer"],
        "wear": ["Traditional", "Bridal", "Festival Wear"],
        "size_type": "Length (in)",
        "size_range": (8, 14),
    },

    "maangtikka": {
        "product_type": "Maang Tikka",
        "weight_range": (5, 20),
        "stone_range": (0, 4),
        "shapes": ["Round", "Floral", "Peacock", "Temple", "Traditional", "Designer"],
        "wear": ["Bridal", "Traditional", "Party Wear"],
        "size_type": "Chain Length (in)",
        "size_range": (10, 16),
    },

    "necklaces": {
        "product_type": "Necklace",
        "weight_range": (20, 100),
        "stone_range": (0, 10),
        "shapes": ["Floral", "Temple", "Peacock", "Traditional", "Geometric",
                   "Designer"],
        "wear": ["Bridal", "Traditional", "Party Wear"],
        "size_type": "Length (in)",
        "size_range": (14, 22),
    },

    "nosepin": {
        "product_type": "Nose Pin",
        "weight_range": (0.5, 3),
        "stone_range": (0, 0.8),
        "shapes": ["Round", "Floral", "Leaf", "Geometric", "Designer"],
        "wear": ["Traditional", "Daily Wear", "Party Wear"],
        "size_type": "Size",
        "size_fixed": "One Size",
    },

    "rings": {
        "product_type": "Ring",
        "weight_range": (3, 12),
        "stone_range": (0, 3),
        "shapes": ["Round", "Oval", "Floral", "Geometric", "Leaf", "Designer"],
        "wear": ["Daily Wear", "Party Wear", "Bridal", "Traditional"],
        "size_type": "Ring Size (US)",
        "size_range": (5, 13),
    },

    "straps": {
        "product_type": "Strap",
        "weight_range": (10, 40),
        "stone_range": (0, 4),
        "shapes": ["Traditional", "Floral", "Chain", "Geometric", "Designer"],
        "wear": ["Traditional", "Bridal", "Festival Wear", "Daily Wear"],
        "size_type": "Length (in)",
        "size_range": (6, 9),
    },

    "toerings": {
        "product_type": "Toe Ring",
        "weight_range": (1, 6),
        "stone_range": (0, 1),
        "shapes": ["Round", "Floral", "Geometric", "Traditional", "Designer"],
        "wear": ["Traditional", "Daily Wear", "Bridal"],
        "size_type": "Size",
        "size_fixed": "Adjustable",
    },

    "waistchain": {
        "product_type": "Waist Chain",
        "weight_range": (20, 80),
        "stone_range": (0, 8),
        "shapes": ["Traditional", "Floral", "Peacock", "Temple", "Designer"],
        "wear": ["Bridal", "Traditional", "Party Wear"],
        "size_type": "Length (in)",
        "size_range": (28, 40),
    },
}


GOLD_PURITY_OPTIONS = ["18K", "20K", "22K", "24K"]

GOLD_COLOR_OPTIONS = ["Yellow Gold", "White Gold", "Rose Gold"]

# These sentinel strings are deliberately NOT "None" / "N/A".
# pandas.read_csv treats "None", "NA" and "N/A" as null by default, so a
# catalog written with those values reads back as NaN and the information
# is lost. Anything used here must stay off pandas' default na_values list.
NO_STONE = "No Stone"
NOT_APPLICABLE = "Not Applicable"

STONE_OPTIONS = [NO_STONE, "Diamond", "Ruby", "Emerald", "Sapphire", "CZ", "Pearl"]

STONE_PRICE_PER_GRAM = {
    NO_STONE: 0, "Diamond": 25000, "Ruby": 7000, "Emerald": 6000,
    "Sapphire": 5000, "CZ": 1000, "Pearl": 3000,
}

STONE_CUT_OPTIONS = [
    "Round Brilliant", "Princess", "Oval",
    "Emerald Cut", "Cushion", "Pear", "Marquise",
]

# Offline rate table. Only used for images the provided catalog does not
# cover. No network call, no API key - a live rate is not worth a hard
# dependency on an external service in a training script.
FIXED_GOLD_RATE_PER_GRAM = {"18K": 6000, "20K": 6500, "22K": 7200, "24K": 7800}

FIXED_RATE_SOURCE = "Fixed / Offline table"

UNMATCHED_METADATA_CLASSES = set()


# ------------------------------------------------------------
# Loading the provided spreadsheets
# ------------------------------------------------------------

def find_spreadsheet(directory, basenames):
    """
    First existing <basename><ext> in `directory`, preferring .csv.

    Returns None when nothing matches, so the caller can decide whether
    that is fatal.
    """
    if not directory or not os.path.isdir(directory):
        return None

    for basename in basenames:
        for extension in CATALOG_EXTENSIONS:
            path = os.path.join(directory, basename + extension)
            if os.path.isfile(path):
                return path

    return None


def read_spreadsheet(path):
    """
    Read a .csv or Excel sheet into a DataFrame.

    keep_default_na=False is deliberate: the catalog uses the strings
    "No Stone" and "Not Applicable", and empty cells must stay empty
    strings. With pandas' default NA handling those become NaN and the
    written-out catalog loses the distinction between "no stone" and
    "unknown".
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(path, keep_default_na=False, encoding="utf-8")

    try:
        return pd.read_excel(path, keep_default_na=False)
    except ImportError as error:
        raise SystemExit(
            f"Reading {os.path.basename(path)} needs an Excel engine:\n"
            f"  {error}\n\n"
            "Either install one:\n"
            "  pip install openpyxl\n"
            "or export the sheet as CSV next to it "
            "(jewellery_metadata.csv) and re-run."
        )


def normalize_lookup_path(path):
    """
    Join key for catalog lookups: forward slashes, NFC-normalised.

    NFC matters. The same accented filename can be stored decomposed
    (NFD) by one filesystem and composed (NFC) by another, so
    'buză'.encode() differs byte-for-byte between the sheet and the
    directory listing while naming the same file. Case is preserved -
    class folders like 'Nose Pin' are matched exactly.
    """
    return unicodedata.normalize("NFC", str(path).replace("\\", "/").strip())


def build_catalog_index(catalog_df):
    """
    Three lookup tables, tried in order of how much they assume:

      1. normalised full relative path      - the real key
      2. (category, normalised basename)    - survives a file moving
                                              between train/ and val/
      3. normalised basename alone          - last resort, only used
                                              when it is unambiguous
    """
    by_path = {}
    by_category_and_name = {}
    basename_counts = {}
    by_basename = {}

    keys = list(catalog_df[CATALOG_KEY_COLUMN])

    if CATALOG_CATEGORY_COLUMN in catalog_df.columns:
        categories = [str(c).strip() for c in catalog_df[CATALOG_CATEGORY_COLUMN]]
    else:
        categories = [None] * len(keys)

    for position, (key, category) in enumerate(zip(keys, categories)):

        normalised = normalize_lookup_path(key)
        by_path.setdefault(normalised, position)

        basename = normalize_lookup_path(os.path.basename(normalised))

        if category is not None:
            by_category_and_name.setdefault((category, basename), position)

        basename_counts[basename] = basename_counts.get(basename, 0) + 1
        by_basename.setdefault(basename, position)

    # A basename shared by two rows cannot identify a row on its own.
    ambiguous = {name for name, count in basename_counts.items() if count > 1}
    for name in ambiguous:
        by_basename.pop(name, None)

    return by_path, by_category_and_name, by_basename


def lookup_catalog_row(index, relative_path, category):
    """
    -> (row position, how it matched) or (None, None).
    """
    by_path, by_category_and_name, by_basename = index

    normalised = normalize_lookup_path(relative_path)

    position = by_path.get(normalised)
    if position is not None:
        return position, "path"

    basename = normalize_lookup_path(os.path.basename(normalised))

    position = by_category_and_name.get((category, basename))
    if position is not None:
        return position, "category+filename"

    position = by_basename.get(basename)
    if position is not None:
        return position, "filename"

    return None, None


def check_provided_class_mapping(metadata_dir, classes):
    """
    Cross-check the provided class_index_mapping against the folder scan.

    A mismatch is not fatal for training - the label space comes from the
    folder scan either way - but it silently breaks any downstream code
    that maps a predicted index back to a name using the provided sheet,
    so it gets a loud warning.
    """
    path = find_spreadsheet(metadata_dir, CLASS_MAPPING_BASENAMES)

    if path is None:
        return None

    mapping_df = read_spreadsheet(path)

    name_column = next(
        (c for c in ("Class_Name", "Class Name", "class_name", "Category")
         if c in mapping_df.columns),
        None,
    )

    if name_column is None:
        print(f"\nWARNING: {os.path.basename(path)} has no Class_Name column "
              f"(found: {list(mapping_df.columns)}). Skipping the cross-check.")
        return None

    index_column = next(
        (c for c in ("Class_Index", "Class Index", "class_index")
         if c in mapping_df.columns),
        None,
    )

    if index_column is not None:
        mapping_df = mapping_df.sort_values(index_column)

    provided = [str(name).strip() for name in mapping_df[name_column]]

    print(f"\nProvided class mapping: {os.path.basename(path)} "
          f"({len(provided)} classes)")

    if provided == list(classes):
        print("  Matches the folder scan exactly (same names, same order).")
        return provided

    print("\n" + "!" * 70)
    print("WARNING: the provided class mapping does not match the folders.")
    print(f"  provided : {provided}")
    print(f"  detected : {list(classes)}")

    missing = [c for c in classes if c not in provided]
    extra = [c for c in provided if c not in classes]
    if missing:
        print(f"  In the folders but not the sheet: {missing}")
    if extra:
        print(f"  In the sheet but not the folders: {extra}")
    if not missing and not extra:
        print("  Same names, DIFFERENT ORDER - so class indices disagree.")

    print("\nTraining uses the folder scan, which is written to")
    print("<results>/metadata/class_index_mapping.csv. Use that file for")
    print("inference, or re-align the provided sheet to it.")
    print("!" * 70)

    return provided


def get_stable_rng(image_path):
    """Same image path always yields the same synthetic attributes."""
    digest = hashlib.sha256(image_path.encode("utf-8")).digest()
    seed_value = int.from_bytes(digest[:8], byteorder="big") % (2 ** 32)
    return random.Random(seed_value)


def resolve_spec(class_name):
    key = normalize_class_name(class_name)
    key = CLASS_ALIASES.get(key, key)

    if key in CLASS_SPECS:
        return CLASS_SPECS[key]

    UNMATCHED_METADATA_CLASSES.add(class_name)

    return {
        "product_type": class_name,
        "weight_range": (5, 30),
        "stone_range": (0, 3),
        "shapes": ["Traditional", "Designer"],
        "wear": ["Traditional", "Party Wear"],
        "size_type": "Size",
        "size_fixed": "One Size",
    }


def generate_metadata(image_path, class_name, gold_rates, gold_rate_source):

    rng = get_stable_rng(image_path)
    spec = resolve_spec(class_name)

    purity = rng.choice(GOLD_PURITY_OPTIONS)
    gold_color = rng.choice(GOLD_COLOR_OPTIONS)
    shape = rng.choice(spec["shapes"])
    wear_type = rng.choice(spec["wear"])
    stone_type = rng.choice(STONE_OPTIONS)

    min_weight, max_weight = spec["weight_range"]
    gross_weight = round(rng.uniform(min_weight, max_weight), 2)

    min_stone, max_stone = spec["stone_range"]

    if stone_type == NO_STONE:
        stone_weight = 0.0
        stone_cut = NOT_APPLICABLE
    else:
        stone_weight = round(rng.uniform(min_stone, max_stone), 2)
        stone_cut = rng.choice(STONE_CUT_OPTIONS)

    # --------------------------------------------------------
    # Size (class-specific)
    # --------------------------------------------------------

    size_type = spec["size_type"]

    if "size_fixed" in spec:
        size_value = spec["size_fixed"]
    else:
        min_size, max_size = spec["size_range"]
        if size_type == "Ring Size (US)":
            steps = int((max_size - min_size) / 0.5)
            size_value = min_size + 0.5 * rng.randint(0, steps)
        else:
            size_value = round(rng.uniform(min_size, max_size), 1)

    # --------------------------------------------------------
    # Confidence score
    # Generated at catalog-build time, BEFORE the model is trained.
    # It is a placeholder for demo / catalog purposes and is not a
    # model output. Real per-image confidences are written after
    # training to reports/model_confidence_validation.csv.
    # --------------------------------------------------------

    confidence_score = round(rng.uniform(0.75, 0.99), 4)

    # --------------------------------------------------------
    # Pricing
    # --------------------------------------------------------

    net_weight = round(max(gross_weight - stone_weight, 0.1), 2)

    gold_rate = gold_rates[purity]
    gold_value = net_weight * gold_rate
    stone_value = stone_weight * STONE_PRICE_PER_GRAM[stone_type]
    making_charge = gold_value * 0.15

    estimated_cost = round((gold_value + stone_value + making_charge) / 100) * 100
    gst_amount = round(estimated_cost * GST_RATE)
    total_price_incl_gst = estimated_cost + gst_amount

    product_type = f"{gold_color} {spec['product_type']}"

    if stone_type == NO_STONE:
        material_sentence = f"It is made from {purity} gold with no stone setting. "
    else:
        material_sentence = (
            f"It is made from {purity} gold with {stone_type.lower()} detailing "
            f"({stone_cut} cut). "
        )

    description = (
        f"{product_type} featuring a {shape.lower()} design. "
        f"This {spec['product_type'].lower()} is suitable for {wear_type.lower()}. "
        f"{material_sentence}"
        f"Available size: {size_value} {size_type}. "
        f"The approximate gross weight is {gross_weight} grams and the "
        f"net gold weight is {net_weight} grams. "
        f"Total price incl. GST: INR {int(total_price_incl_gst):,}."
    )

    return {
        "Image Path": image_path,
        "Category": class_name,
        "Product Type": product_type,
        "Metal Color": gold_color,
        "Gold Purity": purity,
        "Shape": shape,
        "Type of Wear": wear_type,
        "Stone Type": stone_type,
        "Stone Cut": stone_cut,
        "Size Type": size_type,
        "Size": size_value,
        "Gross Weight (g)": gross_weight,
        "Net Weight (g)": net_weight,
        "Stone Weight (g)": stone_weight,
        "Gold Rate Used (INR/g)": gold_rate,
        "Gold Rate Source": gold_rate_source,
        "Estimated Cost (INR)": int(estimated_cost),
        "GST Rate": f"{int(GST_RATE * 100)}%",
        "GST Amount (INR)": int(gst_amount),
        "Total Price incl. GST (INR)": int(total_price_incl_gst),
        "Cost Type": "Estimated / Assumed",
        "Description": description,
        "Confidence Score": confidence_score,
        "Confidence Type": "Synthetic / Simulated",
        "Metadata Source": "Synthetic / Estimated",
    }


def scan_split_images(classes, scan_targets):
    """
    -> [(relative_path, class_name, split_name)] for every image on disk.

    Paths are relative to their split dir, so they survive the
    --copy-to-local relocation and match the provided catalog's keys.
    """
    found = []

    for split_name, split_path in scan_targets.items():

        for class_name in sorted(classes):

            class_dir = os.path.join(split_path, class_name)

            if not os.path.isdir(class_dir):
                continue

            for root, dirs, files in os.walk(class_dir):
                dirs.sort()
                for file_name in sorted(files):

                    if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                        continue

                    absolute_path = os.path.join(root, file_name)

                    relative_path = os.path.join(
                        split_name,
                        os.path.relpath(absolute_path, split_path),
                    ).replace("\\", "/")

                    found.append((relative_path, class_name, split_name))

    return found


def build_metadata_catalog(classes, train_dir, val_dir, metadata_dir,
                           csv_path, summary_path, coverage_path,
                           strict=False):
    """
    Resolve one catalog row per image on disk, taking the values from the
    provided spreadsheet in `metadata_dir`.

    Writes the resolved catalog (only the images this run actually uses),
    a per-class summary and a coverage report into the results folder.
    The provided folder is never written to.
    """

    catalog_path = find_spreadsheet(metadata_dir, CATALOG_BASENAMES)

    if catalog_path is None:
        raise SystemExit(
            f"No catalog spreadsheet found in:\n  {metadata_dir}\n\n"
            "Expected one of "
            + ", ".join(b + ".csv" for b in CATALOG_BASENAMES) +
            " (or .xlsx).\n"
            "Point --metadata-dir at the folder holding it, or pass "
            "--no-metadata to skip the catalog step."
        )

    print("\nReading provided catalog:", catalog_path)

    catalog_df = read_spreadsheet(catalog_path)

    if CATALOG_KEY_COLUMN not in catalog_df.columns:
        raise SystemExit(
            f"{os.path.basename(catalog_path)} has no '{CATALOG_KEY_COLUMN}' "
            f"column.\nColumns found: {list(catalog_df.columns)}\n"
            f"That column is the join key - it must hold paths like "
            f"train/Bangles/020_0001.jpg."
        )

    print(f"  {len(catalog_df)} rows, {len(catalog_df.columns)} columns")

    duplicate_keys = int(catalog_df[CATALOG_KEY_COLUMN].duplicated().sum())
    if duplicate_keys:
        print(f"  NOTE: {duplicate_keys} duplicate '{CATALOG_KEY_COLUMN}' "
              "value(s); the first occurrence of each wins.")

    check_provided_class_mapping(metadata_dir, classes)

    index = build_catalog_index(catalog_df)

    # Scan whichever split dirs actually exist.
    scan_targets = {"train": train_dir}
    if val_dir is not None:
        scan_targets["val"] = val_dir

    print("\nMatching catalog rows to images in:")
    for split_name, split_path in scan_targets.items():
        print(f"  {split_name:<6} {split_path}")

    images = scan_split_images(classes, scan_targets)

    if not images:
        print("\nWARNING: metadata scan found no images. Skipping catalog.")
        return None

    catalog_columns = list(catalog_df.columns)
    catalog_records = catalog_df.to_dict("records")

    rows = []
    match_counts = {"path": 0, "category+filename": 0, "filename": 0}
    unmatched = []

    for relative_path, class_name, split_name in images:

        position, how = lookup_catalog_row(index, relative_path, class_name)

        if position is None:
            unmatched.append((relative_path, class_name, split_name))
            continue

        match_counts[how] += 1

        row = dict(catalog_records[position])

        # Keep the run's own view of where the image is and which split
        # it is in - the sheet can disagree after a re-split, and the
        # folder layout is the ground truth for this run.
        row[CATALOG_KEY_COLUMN] = relative_path
        if CATALOG_CATEGORY_COLUMN in row:
            row[CATALOG_CATEGORY_COLUMN] = class_name
        row["Split"] = split_name
        row["Catalog Match"] = how

        rows.append(row)

    total_images = len(images)
    matched = len(rows)

    print(f"\nCatalog coverage: {matched}/{total_images} images "
          f"({100 * matched / max(total_images, 1):.2f}%)")
    print(f"  matched on full path          : {match_counts['path']}")
    if match_counts["category+filename"]:
        print(f"  matched on category+filename  : "
              f"{match_counts['category+filename']}")
    if match_counts["filename"]:
        print(f"  matched on filename alone     : {match_counts['filename']}")

    # --------------------------------------------------------
    # Images the provided sheet does not cover
    # --------------------------------------------------------

    if unmatched:

        print("\n" + "!" * 70)
        print(f"{len(unmatched)} image(s) have no row in "
              f"{os.path.basename(catalog_path)}:")
        for relative_path, class_name, split_name in unmatched[:10]:
            print(f"  [{split_name}] {relative_path}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more "
                  f"(full list in {os.path.basename(coverage_path)})")

        if strict:
            print("\n--strict-metadata is set, so this is fatal.")
            print("!" * 70)
            raise SystemExit(
                f"{len(unmatched)} image(s) are missing from the provided "
                "catalog. Add them to the sheet, remove the images, or drop "
                "--strict-metadata to fill those rows in offline."
            )

        print("\nFilling those rows in offline from the fixed rate table")
        print("(no network, no API key). They are tagged in the")
        print("'Metadata Source' column so they can be told apart.")
        print("!" * 70)

        for relative_path, class_name, split_name in unmatched:

            generated = generate_metadata(
                relative_path, class_name,
                FIXED_GOLD_RATE_PER_GRAM, FIXED_RATE_SOURCE,
            )
            generated["Split"] = split_name
            generated["Metadata Source"] = "Synthetic / Not in provided catalog"
            generated["Catalog Match"] = "generated"

            # The provided sheet defines the schema: drop generated keys it
            # does not have, and leave its extra columns blank.
            row = {column: generated.get(column, "") for column in catalog_columns}
            row["Split"] = split_name
            row["Catalog Match"] = "generated"

            rows.append(row)

        if UNMATCHED_METADATA_CLASSES:
            print("\nWARNING: these folder names matched no CLASS_SPECS entry,")
            print("so the generated rows for them use a GENERIC spec and their")
            print("weights and prices are rough:")
            for class_name in sorted(UNMATCHED_METADATA_CLASSES):
                print(f"  {class_name}  ->  normalises to "
                      f"'{normalize_class_name(class_name)}'")
            print("Add an entry to CLASS_ALIASES or CLASS_SPECS for each.")

    # --------------------------------------------------------
    # Write out
    # --------------------------------------------------------

    column_order = catalog_columns[:]
    for extra_column in ("Split", "Catalog Match"):
        if extra_column not in column_order:
            column_order.append(extra_column)

    metadata_df = pd.DataFrame(rows).reindex(columns=column_order)
    metadata_df = metadata_df.sort_values(CATALOG_KEY_COLUMN).reset_index(drop=True)
    metadata_df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\nResolved catalog: {len(metadata_df)} rows")
    print("Saved to:", csv_path)

    # --------------------------------------------------------
    # Per-class summary, recomputed from the resolved rows
    # --------------------------------------------------------

    numeric_summary_columns = {
        "Average_Gross_Weight_g": "Gross Weight (g)",
        "Average_Net_Weight_g": "Net Weight (g)",
        "Average_Estimated_Cost_INR": "Estimated Cost (INR)",
        "Average_Total_Price_incl_GST_INR": "Total Price incl. GST (INR)",
        "Average_Confidence_Score": "Confidence Score",
    }

    if CATALOG_CATEGORY_COLUMN in metadata_df.columns:

        summary_frame = metadata_df.copy()

        aggregations = {"Image_Count": (CATALOG_KEY_COLUMN, "count")}

        for output_name, source_column in numeric_summary_columns.items():
            if source_column in summary_frame.columns:
                # Values read back from a sheet can be text; coerce so the
                # mean does not silently become a string concatenation error.
                summary_frame[source_column] = pd.to_numeric(
                    summary_frame[source_column], errors="coerce"
                )
                aggregations[output_name] = (source_column, "mean")

        summary = (
            summary_frame
            .groupby(CATALOG_CATEGORY_COLUMN)
            .agg(**aggregations)
            .round(2)
            .reset_index()
        )

        summary.to_csv(summary_path, index=False, encoding="utf-8")
        print("Summary saved to:", summary_path)

    # --------------------------------------------------------
    # Coverage report
    # --------------------------------------------------------

    with open(coverage_path, "w", encoding="utf-8") as f:
        f.write("PRODUCT CATALOG COVERAGE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Provided catalog   : {catalog_path}\n")
        f.write(f"Catalog rows       : {len(catalog_df)}\n")
        f.write(f"Images on disk     : {total_images}\n")
        f.write(f"Matched from sheet : {matched}\n")
        f.write(f"Generated offline  : {len(unmatched)}\n")
        f.write(f"Gold rate source   : from the provided sheet "
                f"(offline table for generated rows)\n\n")
        f.write("Match breakdown:\n")
        for how, count in match_counts.items():
            f.write(f"  {how:<20} {count}\n")
        if unmatched:
            f.write(f"\nImages with no catalog row ({len(unmatched)}):\n")
            for relative_path, class_name, split_name in unmatched:
                f.write(f"  [{split_name}] {relative_path}\n")

    print("Coverage report saved to:", coverage_path)

    return metadata_df


# ============================================================
# 7. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, memory_fmt, amp_device, use_amp, accum_steps):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Training", leave=False)

    for step, (images, labels) in enumerate(progress):

        images = images.to(device, non_blocking=True).to(memory_format=memory_fmt)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Divide so accumulated gradients average correctly
        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Guard against overshooting OneCycleLR's total_steps
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = outputs.max(1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    return running_loss / max(total, 1), 100 * correct / max(total, 1)


# ============================================================
# 8. VALIDATE
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device, memory_fmt, amp_device, use_amp):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions, all_labels, all_confidences = [], [], []

    for images, labels in tqdm(loader, desc="Validation", leave=False):

        images = images.to(device, non_blocking=True).to(memory_format=memory_fmt)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        probabilities = torch.softmax(outputs.float(), dim=1)
        confidences, predictions = probabilities.max(1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_confidences.extend(confidences.cpu().numpy().tolist())

    return (
        running_loss / max(total, 1),
        100 * correct / max(total, 1),
        all_labels,
        all_predictions,
        all_confidences,
    )


# ============================================================
# 9. PLOT HELPER
# ============================================================

def line_plot(x, series, xlabel, ylabel, title, path):
    plt.figure(figsize=(10, 6))
    for label, values in series:
        plt.plot(x, values, label=label, marker="o", markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# 10. PIPELINE
# ============================================================
# Everything below runs only when this file is executed directly.
#
# The guard is not cosmetic: on Windows and macOS, DataLoader worker
# processes are created with 'spawn', which re-imports this module in
# each worker. Without the guard, every worker would re-run the whole
# training pipeline (and spawn its own workers, and so on).
# ============================================================

if __name__ == "__main__":

    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    # --------------------------------------------------------
    # Colab is still supported, it is just no longer assumed.
    # --------------------------------------------------------

    IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")

    if IN_COLAB:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as error:
            print(f"Colab detected but Drive mount failed: {error}")
    else:
        print("Running locally - no Google Drive mount needed.")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    DATA_ROOT = resolve_data_root(args.data_root)

    RESULTS_ROOT = os.path.abspath(
        args.results_root
        or os.path.join(PROJECT_ROOT, "output", f"{args.model}_{args.image_size}")
    )

    # The folder the user provides the catalog sheets in. Read-only as far
    # as this script is concerned - the resolved catalog is written into
    # the results folder, so a run never overwrites the source data.
    PROVIDED_METADATA_DIR = os.path.abspath(
        args.metadata_dir or os.path.join(PROJECT_ROOT, "metadata")
    )

    if not args.no_metadata and not os.path.isdir(PROVIDED_METADATA_DIR):
        raise SystemExit(
            f"Catalog metadata folder not found:\n  {PROVIDED_METADATA_DIR}\n\n"
            "Pass --metadata-dir with the folder holding "
            "jewellery_metadata.csv (or .xlsx),\n"
            "or pass --no-metadata to skip the catalog step."
        )

    IMAGE_SIZE = args.image_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    LABEL_SMOOTHING = args.label_smoothing
    VAL_SPLIT = args.val_split
    SEED = args.seed
    USE_CLASS_WEIGHTS = not args.no_class_weights
    FRESH_START = not args.resume

    # --------------------------------------------------------
    # Device + batch size
    # --------------------------------------------------------

    device = resolve_device(args.device)

    print("=" * 70)
    print("DEVICE:", device)
    print("Python:", platform.python_version(), "|", platform.system(),
          platform.release())
    print("torch :", torch.__version__, "| CUDA build:", torch.version.cuda)

    gpu_gb = 0.0

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {gpu_name} ({gpu_gb:.1f} GB)")

    USE_AMP = (not args.no_amp) and device.type == "cuda"
    AMP_DEVICE = "cuda" if device.type == "cuda" else "cpu"

    # channels_last is a CUDA/cuDNN optimisation. On CPU it is usually
    # neutral-to-slower, so it is only applied where it pays off.
    MEMORY_FORMAT = (
        torch.channels_last if device.type == "cuda" else torch.contiguous_format
    )

    BATCH_SIZE = args.batch_size or auto_batch_size(gpu_gb)
    if args.batch_size:
        print(f"Batch size set manually: {BATCH_SIZE}")
    else:
        print(f"Batch size auto-selected: {BATCH_SIZE}")

    ACCUM_STEPS = max(1, args.effective_batch_size // BATCH_SIZE)

    NUM_WORKERS = (
        args.workers if args.workers is not None
        else min(4, (os.cpu_count() or 2))
    )

    print(f"Physical batch size : {BATCH_SIZE}")
    print(f"Accumulation steps  : {ACCUM_STEPS}")
    print(f"Effective batch     : {BATCH_SIZE * ACCUM_STEPS}")
    print(f"Mixed precision     : {USE_AMP}")
    print(f"Dataloader workers  : {NUM_WORKERS}")
    print("=" * 70)

    if device.type != "cuda":
        print("\n" + "!" * 70)
        print(f"WARNING: no CUDA GPU. {args.model} at {IMAGE_SIZE}px on "
              f"{device.type.upper()} is")
        print("         roughly 30-60x slower than on a GPU - a full 25-epoch run")
        print("         over ~11k images would take days, not hours.")
        print("")
        print("         Use this to verify the pipeline:")
        print("           --model convnext_tiny --image-size 224 --epochs 1 \\")
        print("             --limit-per-class 8 --batch-size 4 --workers 0")
        print("")
        print("         For the real run, install a CUDA build of torch:")
        print("           pip install --index-url "
              "https://download.pytorch.org/whl/cu124 torch torchvision")
        print("         ...or run this same file on a GPU box / Colab.")
        print("!" * 70)

    # --------------------------------------------------------
    # Dataset layout
    # --------------------------------------------------------

    TRAIN_DIR, VAL_DIR = resolve_split_dirs(DATA_ROOT)

    # Keep the originals so we can fall back to the source if a copy fails
    SOURCE_TRAIN_DIR, SOURCE_VAL_DIR = TRAIN_DIR, VAL_DIR

    print("\nDataset layout detected:")
    print("  Data root:", DATA_ROOT)
    print("  Train dir:", TRAIN_DIR)
    print("  Val dir  :",
          VAL_DIR if VAL_DIR else f"(none - will split {VAL_SPLIT:.0%} from train)")

    train_image_count = count_images_recursive(TRAIN_DIR)

    if train_image_count == 0:

        message = [
            f"No images found under:\n  {TRAIN_DIR}\n",
            f"Contents of the data root ({DATA_ROOT}):",
        ]
        for entry in sorted(os.scandir(DATA_ROOT), key=lambda e: e.name):
            kind = "dir " if entry.is_dir() else "file"
            message.append(f"  [{kind}] {entry.name}")

        siblings = []
        parent = os.path.dirname(DATA_ROOT.rstrip(os.sep))
        if os.path.isdir(parent):
            for entry in sorted(os.scandir(parent), key=lambda e: e.name):
                if entry.is_dir():
                    n = count_images_recursive(entry.path)
                    if n > 0:
                        subs = sorted(
                            d.name for d in os.scandir(entry.path) if d.is_dir()
                        )
                        splits = [
                            s for s in subs
                            if s.lower() in ("train", "val", "valid",
                                             "validation", "test")
                        ]
                        siblings.append((entry.path, n, splits))

        if siblings:
            message.append("\nFolders nearby that DO contain images:")
            for path, n, splits in siblings:
                hint = f"   <-- has split dirs {splits}" if splits else ""
                message.append(f"  {path}  ({n} images){hint}")
            message.append("\nPass the one holding your train/ folder "
                           "with --data-root.")
        else:
            message.append("\nNo image folders found nearby either. "
                           "Check --data-root.")

        raise SystemExit("\n".join(message))

    print(f"\nImages under train dir: {train_image_count}")

    # --------------------------------------------------------
    # Optional local cache.
    # Only useful when the dataset lives on a slow drive. Reading
    # thousands of small files over Drive/SMB is what makes an epoch
    # look like it is hanging.
    # --------------------------------------------------------

    LOCAL_DATA_ROOT = os.path.abspath(args.copy_to_local) if args.copy_to_local else None

    if LOCAL_DATA_ROOT:

        print(f"\nCopying dataset to {LOCAL_DATA_ROOT} "
              "(one-time, speeds up every epoch)...")

        t0 = time.time()
        try:
            TRAIN_DIR = local_copy(TRAIN_DIR, LOCAL_DATA_ROOT, "train",
                                   args.force_recopy)
            VAL_DIR = (
                local_copy(VAL_DIR, LOCAL_DATA_ROOT, "val", args.force_recopy)
                if VAL_DIR else None
            )
            print(f"  Copy finished in {time.time() - t0:.1f}s")
            print("  Train dir now:", TRAIN_DIR)
        except Exception as error:
            print(f"  WARNING: local copy failed ({error})")
            print("           Reading from the source instead (slower, correct).")
            TRAIN_DIR, VAL_DIR = SOURCE_TRAIN_DIR, SOURCE_VAL_DIR

    # --------------------------------------------------------
    # Validate whatever we ended up pointing at
    # --------------------------------------------------------

    print("\nValidating split dirs...")

    validate_split_dir(TRAIN_DIR, "train", LOCAL_DATA_ROOT)

    if VAL_DIR is not None:
        validate_split_dir(VAL_DIR, "val", LOCAL_DATA_ROOT)

    # --------------------------------------------------------
    # Result directories
    # --------------------------------------------------------

    CHECKPOINT_DIR = os.path.join(RESULTS_ROOT, "checkpoints")
    LOG_DIR = os.path.join(RESULTS_ROOT, "logs")
    PLOT_DIR = os.path.join(RESULTS_ROOT, "plots")
    REPORT_DIR = os.path.join(RESULTS_ROOT, "reports")
    METADATA_DIR = os.path.join(RESULTS_ROOT, "metadata")

    for folder in [RESULTS_ROOT, CHECKPOINT_DIR, LOG_DIR,
                   PLOT_DIR, REPORT_DIR, METADATA_DIR]:
        os.makedirs(folder, exist_ok=True)

    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
    TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training.log")
    HISTORY_CSV_PATH = os.path.join(LOG_DIR, "training_history.csv")
    CLASS_MAPPING_PATH = os.path.join(METADATA_DIR, "class_index_mapping.csv")
    METADATA_CSV_PATH = os.path.join(METADATA_DIR, "jewellery_metadata.csv")
    METADATA_SUMMARY_PATH = os.path.join(METADATA_DIR,
                                         "metadata_summary_by_class.csv")
    METADATA_COVERAGE_PATH = os.path.join(METADATA_DIR, "metadata_coverage.txt")
    CONFUSION_MATRIX_PATH = os.path.join(REPORT_DIR, "confusion_matrix.csv")
    CONFUSION_MATRIX_PLOT = os.path.join(PLOT_DIR, "confusion_matrix.png")
    CLASSIFICATION_REPORT_PATH = os.path.join(REPORT_DIR,
                                              "classification_report.txt")
    TRAINING_SUMMARY_PATH = os.path.join(REPORT_DIR, "training_summary.txt")
    MODEL_CONFIDENCE_PATH = os.path.join(REPORT_DIR,
                                         "model_confidence_validation.csv")
    CONFIG_PATH = os.path.join(REPORT_DIR, "training_config.json")
    LOSS_PLOT_PATH = os.path.join(PLOT_DIR, "loss_curve.png")
    ACCURACY_PLOT_PATH = os.path.join(PLOT_DIR, "accuracy_curve.png")
    LR_PLOT_PATH = os.path.join(PLOT_DIR, "learning_rate_curve.png")

    print("\nResults will be written to:", RESULTS_ROOT)

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger = logging.getLogger("JewelleryTraining")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # encoding is explicit: the Windows default (cp1252) cannot write
    # non-ASCII class names and raises mid-epoch.
    _file_handler = logging.FileHandler(TRAINING_LOG_PATH, mode="w",
                                        encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)

    # --------------------------------------------------------
    # Load dataset (classes auto-detected)
    # --------------------------------------------------------

    train_transform, val_transform = build_transforms(IMAGE_SIZE)

    print("\nLoading dataset from:", TRAIN_DIR)

    base_train = CleanImageFolder(root=TRAIN_DIR, transform=train_transform)

    limit_samples_per_class(base_train, args.limit_per_class, SEED)

    CLASSES = base_train.classes
    NUM_CLASSES = len(CLASSES)

    if NUM_CLASSES < 2:
        raise SystemExit(
            f"Found only {NUM_CLASSES} class folder(s) in {TRAIN_DIR}. "
            "Expected one subfolder per class."
        )

    print(f"\nClasses detected: {NUM_CLASSES}")
    for i, class_name in enumerate(CLASSES):
        print(f"{i:02d} -> {class_name}")

    pd.DataFrame({
        "Class_Index": range(NUM_CLASSES),
        "Class_Name": CLASSES,
    }).to_csv(CLASS_MAPPING_PATH, index=False, encoding="utf-8")

    targets_all = np.array([label for _, label in base_train.samples])

    print("\nImages per class:")
    class_counts = {}
    for i, class_name in enumerate(CLASSES):
        count = int((targets_all == i).sum())
        class_counts[class_name] = count
        print(f"  {class_name:<28} {count}")

    print("\nTotal images in train dir:", len(base_train))

    # --------------------------------------------------------
    # Synthetic catalog metadata
    # --------------------------------------------------------

    if not args.no_metadata:
        metadata_df = build_metadata_catalog(
            CLASSES, TRAIN_DIR, VAL_DIR, PROVIDED_METADATA_DIR,
            METADATA_CSV_PATH, METADATA_SUMMARY_PATH, METADATA_COVERAGE_PATH,
            strict=args.strict_metadata,
        )
    else:
        metadata_df = None
        print("\n--no-metadata - skipping the catalog step.")

    # --------------------------------------------------------
    # Train / val split
    # --------------------------------------------------------

    val_folder = None

    if VAL_DIR is not None:

        val_folder = CleanImageFolder(root=VAL_DIR, transform=val_transform)
        limit_samples_per_class(val_folder, args.limit_per_class, SEED)

        if val_folder.classes != CLASSES:

            missing = [c for c in CLASSES if c not in val_folder.classes]
            extra = [c for c in val_folder.classes if c not in CLASSES]

            print("\n" + "!" * 70)
            print("WARNING: val/ does not cover the same classes as train/.")
            if missing:
                print(f"  Missing from val ({len(missing)}): {missing}")
            if extra:
                print(f"  Present only in val ({len(extra)}): {extra}")
            print("\nA val set missing classes cannot measure those classes at")
            print("all, so the predefined val/ folder is being IGNORED. Falling")
            print("back to a stratified split from train/ instead.")
            print("\nTo use your own val/ folder, make sure it has one subfolder")
            print("per class, matching train/ exactly, then re-run.")
            print("!" * 70)

            val_folder = None

    if val_folder is not None:

        train_ds = TransformSubset(base_train, range(len(base_train)),
                                   train_transform)
        val_ds = TransformSubset(val_folder, range(len(val_folder)), val_transform)

        split_note = "predefined train/ and val/ folders"

    else:

        # Stratified split: guarantees every class appears in validation,
        # which plain random_split does not.
        train_idx, val_idx = train_test_split(
            np.arange(len(base_train)),
            test_size=VAL_SPLIT,
            random_state=SEED,
            stratify=targets_all,
        )

        train_ds = TransformSubset(base_train, train_idx, train_transform)
        val_ds = TransformSubset(base_train, val_idx, val_transform)

        split_note = f"stratified {VAL_SPLIT:.0%} split from train dir"

    print("\nSplit:", split_note)
    print("  Training images  :", len(train_ds))
    print("  Validation images:", len(val_ds))

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    if len(train_loader) == 0:
        raise SystemExit(
            f"The training loader is empty: {len(train_ds)} images with "
            f"batch_size={BATCH_SIZE} and drop_last=True yields 0 batches.\n"
            "Lower --batch-size or raise --limit-per-class."
        )

    print(f"\nBatches per epoch: train={len(train_loader)}  val={len(val_loader)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(f"\nCreating {args.model} model...")

    try:
        model = timm.create_model(
            args.model,
            pretrained=not args.no_pretrained,
            num_classes=NUM_CLASSES,
        )
    except Exception as error:
        raise SystemExit(
            f"Could not create model '{args.model}': {error}\n\n"
            "If this is a download failure, either connect to the network or "
            "pass --no-pretrained to start from random weights.\n"
            "Weights are cached under "
            f"{os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')}."
        )

    model = model.to(device, memory_format=MEMORY_FORMAT)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | classes: {NUM_CLASSES} | "
          f"params: {total_params / 1e6:.1f}M | "
          f"pretrained: {not args.no_pretrained}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if USE_CLASS_WEIGHTS:
        counts = np.array([class_counts[c] for c in CLASSES], dtype=np.float64)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (NUM_CLASSES * counts)
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32,
                                           device=device)
        print("\nClass weights (inverse frequency):")
        for c, w in zip(CLASSES, weights):
            print(f"  {c:<28} {w:.3f}")
    else:
        class_weight_tensor = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    # --------------------------------------------------------
    # Optimizer + scheduler
    # --------------------------------------------------------

    # No weight decay on norm layers / biases - standard for ConvNeXt.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )

    WARMUP_EPOCHS = min(args.warmup_epochs, max(EPOCHS - 1, 0))
    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        # small margin for ragged batches
        total_steps=EPOCHS * steps_per_epoch + EPOCHS,
        pct_start=max(WARMUP_EPOCHS / max(EPOCHS, 1), 1e-3),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    # Constructed with device_type "cuda" and enabled=False off CUDA - that is
    # a no-op wrapper, so the training loop needs no branching.
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # --------------------------------------------------------
    # History + fresh start / resume
    # --------------------------------------------------------

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "learning_rate": [], "epoch_time": [],
    }

    start_epoch = 0
    best_accuracy = 0.0

    if FRESH_START:
        print("\nFresh training (default) - removing old checkpoints. "
              "Pass --resume to continue a run.")
        for path in [CHECKPOINT_PATH, BEST_MODEL_PATH, LAST_MODEL_PATH]:
            if os.path.exists(path):
                os.remove(path)

    elif os.path.exists(CHECKPOINT_PATH):
        print("\nLoading checkpoint...")
        # weights_only=False is required: this checkpoint stores python
        # objects (history dict, class list), not just tensors.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,
                                weights_only=False)

        if checkpoint.get("classes") != CLASSES:
            raise SystemExit(
                "Checkpoint was trained on different classes.\n"
                f"checkpoint: {checkpoint.get('classes')}\n"
                f"current   : {CLASSES}\n"
                "Drop --resume to start over."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        history = checkpoint["history"]
        print(f"Resuming from epoch {start_epoch + 1} "
              f"(best acc so far {best_accuracy:.2f}%)")

    else:
        print(f"\n--resume passed but no checkpoint at {CHECKPOINT_PATH} - "
              "starting from scratch.")

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "data_root": DATA_ROOT,
            "train_dir": TRAIN_DIR,
            "val_dir": VAL_DIR,
            "split_note": split_note,
            "results_root": RESULTS_ROOT,
            "provided_metadata_dir": (
                None if args.no_metadata else PROVIDED_METADATA_DIR
            ),
            "strict_metadata": args.strict_metadata,
            "model": args.model,
            "pretrained": not args.no_pretrained,
            "image_size": IMAGE_SIZE,
            "physical_batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": "OneCycleLR (cosine)",
            "optimizer": "AdamW",
            "amp": USE_AMP,
            "class_weights": USE_CLASS_WEIGHTS,
            "num_workers": NUM_WORKERS,
            "limit_per_class": args.limit_per_class,
            "seed": SEED,
            "num_classes": NUM_CLASSES,
            "classes": CLASSES,
            "class_counts": class_counts,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "platform": f"{platform.system()} {platform.release()}",
            "torch": torch.__version__,
            "command": " ".join(sys.argv),
        }, f, indent=4)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    training_start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):

        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 70)

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, MEMORY_FORMAT, AMP_DEVICE, USE_AMP, ACCUM_STEPS,
        )

        val_loss, val_accuracy, val_labels, val_predictions, val_confidences = (
            validate(model, val_loader, criterion, device, MEMORY_FORMAT,
                     AMP_DEVICE, USE_AMP)
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_accuracy:.2f}%")
        print(f"Val Loss       : {val_loss:.4f}")
        print(f"Val Accuracy   : {val_accuracy:.2f}%")
        print(f"Learning Rate  : {current_lr:.8f}")
        print(f"Epoch Time     : {epoch_time:.2f} sec")

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.8f} | {epoch_time:.1f}s"
        )

        pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False,
                                     encoding="utf-8")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved (Val Acc: {val_accuracy:.2f}%)")
            logger.info(f"New best model saved (Val Acc: {val_accuracy:.2f}%)")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
            "best_accuracy": best_accuracy,
            "history": history,
            "classes": CLASSES,
        }, CHECKPOINT_PATH)

    total_training_time = time.time() - training_start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print(f"Total Training Time      : {total_training_time / 60:.2f} minutes")
    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
    print("=" * 70)

    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print("\nLast model saved:", LAST_MODEL_PATH)

    if not history["epoch"]:
        raise SystemExit(
            "No epoch ran, so there is nothing to evaluate. "
            f"start_epoch={start_epoch} and --epochs={EPOCHS}: "
            "raise --epochs above the resumed epoch count."
        )

    # --------------------------------------------------------
    # Final evaluation with the best model
    # --------------------------------------------------------

    print("\nLoading best model for final evaluation...")

    best_model = timm.create_model(args.model, pretrained=False,
                                   num_classes=NUM_CLASSES)
    best_model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    best_model = best_model.to(device, memory_format=MEMORY_FORMAT)

    (
        final_val_loss,
        final_val_accuracy,
        final_labels,
        final_predictions,
        final_confidences,
    ) = validate(best_model, val_loader, criterion, device, MEMORY_FORMAT,
                 AMP_DEVICE, USE_AMP)

    mean_confidence = float(np.mean(final_confidences)) if final_confidences else 0.0

    print("\n" + "=" * 70)
    print("FINAL BEST MODEL RESULTS")
    print(f"Validation Loss            : {final_val_loss:.4f}")
    print(f"Validation Accuracy        : {final_val_accuracy:.2f}%")
    print(f"Mean Prediction Confidence : {mean_confidence:.4f}")
    print("=" * 70)

    # Real per-image confidences.
    # val_loader uses shuffle=False and drop_last=False, so predictions
    # come back in the same order as val_ds.paths().
    val_paths = val_ds.paths()

    pd.DataFrame({
        "Image Path": [
            os.path.join(os.path.basename(os.path.dirname(p)),
                         os.path.basename(p)).replace("\\", "/")
            for p in val_paths
        ],
        "True Label": [CLASSES[i] for i in final_labels],
        "Predicted Label": [CLASSES[i] for i in final_predictions],
        "Model Confidence": [round(float(c), 4) for c in final_confidences],
        "Correct": [t == p for t, p in zip(final_labels, final_predictions)],
    }).to_csv(MODEL_CONFIDENCE_PATH, index=False, encoding="utf-8")

    print("\nReal model confidence scores saved:", MODEL_CONFIDENCE_PATH)

    # --------------------------------------------------------
    # Confusion matrix + classification report
    # --------------------------------------------------------

    cm = confusion_matrix(final_labels, final_predictions,
                          labels=range(NUM_CLASSES))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        CONFUSION_MATRIX_PATH, encoding="utf-8"
    )
    print("Confusion matrix saved:", CONFUSION_MATRIX_PATH)

    report = classification_report(
        final_labels,
        final_predictions,
        labels=range(NUM_CLASSES),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    line_plot(
        history["epoch"],
        [("Training Loss", history["train_loss"]),
         ("Validation Loss", history["val_loss"])],
        "Epoch", "Loss", "Training and Validation Loss", LOSS_PLOT_PATH,
    )

    line_plot(
        history["epoch"],
        [("Training Accuracy", history["train_accuracy"]),
         ("Validation Accuracy", history["val_accuracy"])],
        "Epoch", "Accuracy (%)", "Training and Validation Accuracy",
        ACCURACY_PLOT_PATH,
    )

    line_plot(
        history["epoch"],
        [("Learning Rate", history["learning_rate"])],
        "Epoch", "Learning Rate", "Learning Rate Schedule", LR_PLOT_PATH,
    )

    # Confusion matrix heatmap, row-normalised so small classes stay readable
    cm_normalised = cm.astype(np.float64) / np.clip(
        cm.sum(axis=1, keepdims=True), 1, None
    )

    fig, ax = plt.subplots(figsize=(1.0 * NUM_CLASSES + 3, 1.0 * NUM_CLASSES + 2))
    im = ax.imshow(cm_normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_normalised[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PLOT, dpi=200)
    plt.close(fig)

    print("\nPlots saved to:", PLOT_DIR)

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    best_epoch_number = history["epoch"][int(np.argmax(history["val_accuracy"]))]

    summary = f"""
============================================================
JEWELLERY CLASSIFICATION TRAINING SUMMARY
============================================================

Model                       : {args.model} (pretrained={not args.no_pretrained})
Device                      : {device}
GPU                         : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}
Platform                    : {platform.system()} {platform.release()}
torch                       : {torch.__version__}
Command                     : {' '.join(sys.argv)}

Data root                   : {DATA_ROOT}
Train dir                   : {TRAIN_DIR}
Val dir                     : {VAL_DIR}
Split                       : {split_note}
Catalog metadata source     : {'(skipped)' if args.no_metadata else PROVIDED_METADATA_DIR}
Catalog rows resolved       : {0 if metadata_df is None else len(metadata_df)}

Number of classes           : {NUM_CLASSES}
Classes                     : {CLASSES}
Training images             : {len(train_ds)}
Validation images           : {len(val_ds)}

Image resolution            : {IMAGE_SIZE} x {IMAGE_SIZE}
Physical batch size         : {BATCH_SIZE}
Gradient accumulation       : {ACCUM_STEPS}
Effective batch size        : {BATCH_SIZE * ACCUM_STEPS}
Mixed precision (AMP)       : {USE_AMP}
Epochs                      : {EPOCHS}
Optimizer                   : AdamW (wd={WEIGHT_DECAY})
Scheduler                   : OneCycleLR cosine, max_lr={LEARNING_RATE}
Label smoothing             : {LABEL_SMOOTHING}
Class weights               : {USE_CLASS_WEIGHTS}

Best epoch                  : {best_epoch_number}
Best validation accuracy    : {best_accuracy:.2f}%
Final best-model val loss   : {final_val_loss:.4f}
Final best-model val acc    : {final_val_accuracy:.2f}%
Mean prediction confidence  : {mean_confidence:.4f}
Total training time         : {total_training_time / 60:.2f} minutes

============================================================
RESULT FILES
============================================================

Best model                  : {BEST_MODEL_PATH}
Last model                  : {LAST_MODEL_PATH}
Checkpoint                  : {CHECKPOINT_PATH}
Training history            : {HISTORY_CSV_PATH}
Class mapping               : {CLASS_MAPPING_PATH}
Catalog metadata (resolved) : {METADATA_CSV_PATH}
Catalog summary by class    : {METADATA_SUMMARY_PATH}
Catalog coverage report     : {METADATA_COVERAGE_PATH}
Confusion matrix (csv)      : {CONFUSION_MATRIX_PATH}
Confusion matrix (png)      : {CONFUSION_MATRIX_PLOT}
Classification report       : {CLASSIFICATION_REPORT_PATH}
Model confidences (real)    : {MODEL_CONFIDENCE_PATH}
Loss plot                   : {LOSS_PLOT_PATH}
Accuracy plot               : {ACCURACY_PLOT_PATH}
Learning rate plot          : {LR_PLOT_PATH}
Training config             : {CONFIG_PATH}

============================================================
"""

    print(summary)

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print("=" * 70)
    print("TRAINING PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Results folder: {RESULTS_ROOT}")
