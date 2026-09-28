import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = r"E:\Projects\GitHub\GSS\Jewellery_Data"
MODEL_PATH = r"E:\Projects\GitHub\GSS\models\best.pt"
OUTPUT_PATH = r"E:\Projects\GitHub\GSS\Embeddings"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

import torchvision.transforms.functional as F_vision

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# LOAD MODEL
# ==========================================================

from model import get_model

import torchvision.models as models

print(f"Loading custom model weights from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)

# The checkpoint has a standard ConvNeXt classifier with 13 classes
# (e.g., classifier.2.weight is [13, 1024] and classifier.0 is LayerNorm2d).
# This perfectly matches torchvision's convnext_base architecture.
num_classes = state_dict["classifier.2.weight"].shape[0] if "classifier.2.weight" in state_dict else 13

model = models.convnext_base(weights=None, num_classes=num_classes)
model.load_state_dict(state_dict)

# We want to use it as a feature extractor. The standard ConvNeXt model 
# has `features` (the CNN) and `classifier` (which does pooling/norm/linear).
# We can replace the classifier with Identity or just use the pooling layer.
# To extract 1024-d features (before the final classification layer), we can do:
model.classifier[2] = nn.Identity()

# Move to device and set to eval mode
model = model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract embedding (will be size [1, 1024])
        feature = model(image)
        # Flatten
        feature = feature.view(feature.size(0), -1)
        # L2 Normalization
        feature = F.normalize(feature, p=2, dim=1)

    return feature.squeeze().cpu().numpy()

# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ==========================================================
# EXTRACT FEATURES
# ==========================================================

total_images = 0
all_embeddings = []
all_image_paths = []

print("\nStarting Feature Extraction...\n")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in tqdm(files):
        if file.lower().endswith(extensions):
            image_path = os.path.join(root, file)
            try:
                feature = extract_feature(image_path)
                
                all_embeddings.append(feature)
                all_image_paths.append(image_path)

                total_images += 1

            except Exception as e:
                print(f"Error processing {image_path}")
                print(e)

print("\nSaving embeddings and paths...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

if all_embeddings:
    embeddings_array = np.array(all_embeddings)
    np.save(os.path.join(OUTPUT_PATH, "imagenet_image_features.npy"), embeddings_array)
    np.save(os.path.join(OUTPUT_PATH, "image_features.npy"), embeddings_array)
    
    with open(os.path.join(OUTPUT_PATH, "imagenet_image_paths.pkl"), "wb") as f:
        pickle.dump(all_image_paths, f)
    with open(os.path.join(OUTPUT_PATH, "image_paths.pkl"), "wb") as f:
        pickle.dump(all_image_paths, f)
else:
    embeddings_array = np.array([])
    print("No embeddings were generated!")

print("\n====================================")
print(f"Total Images Processed : {total_images}")
if all_embeddings:
    print(f"Embeddings Shape       : {embeddings_array.shape}")
print(f"Saved At               : {OUTPUT_PATH}")
print("L2 Normalization Applied Successfully")
print("Feature Extraction Completed")
print("\nFiles Saved:")
print("imagenet_image_features.npy & image_features.npy")
print("imagenet_image_paths.pkl & image_paths.pkl")
print("====================================")
