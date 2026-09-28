# Dataset Paths
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")


def resolve_split_dir(split_names, default_name=None):
    """Resolve the local dataset split using the actual folder names present on disk."""
    if not os.path.isdir(DATASET_ROOT):
        return os.path.join(DATASET_ROOT, default_name or split_names[0])

    entries = {entry.name.lower(): entry.name for entry in os.scandir(DATASET_ROOT) if entry.is_dir()}

    for name in split_names:
        key = name.lower()
        if key in entries:
            return os.path.join(DATASET_ROOT, entries[key])

    return os.path.join(DATASET_ROOT, default_name or split_names[0])


TRAIN_DIR = resolve_split_dir(["train", "Train", "training"], "Train")
VAL_DIR = resolve_split_dir(["val", "Val", "valid", "validation"], "val")
TEST_DIR = resolve_split_dir(["test", "Test", "testing"], "Test")

# Image Settings
IMAGE_SIZE = 320

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 0.0001

# Number of Classes
NUM_CLASSES = len([name for name in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, name)) and name.lower() not in {"train", "val", "test", "training", "validation", "testing"}]) if os.path.isdir(DATASET_ROOT) else 13

# Model Save Paths - Use absolute paths
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "outputs", "logs")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "last_model.pth")
LOG_FILE = os.path.join(LOGS_DIR, "training_log.csv")

# Device
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
