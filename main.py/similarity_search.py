import os
import pickle
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# ==========================================================
# ARGUMENTS & PATHS
# ==========================================================

parser = argparse.ArgumentParser(description="Image Similarity Search")
parser.add_argument("--query", type=str, required=False, help="Path to the query image")
args = parser.parse_args()

if args.query:
    QUERY_IMAGE = args.query
else:
    QUERY_IMAGE = input("Please enter the path to the query image: ").strip()
    
# Remove quotes if the user dragged and dropped the file into the terminal
if QUERY_IMAGE.startswith('"') and QUERY_IMAGE.endswith('"'):
    QUERY_IMAGE = QUERY_IMAGE[1:-1]
elif QUERY_IMAGE.startswith("'") and QUERY_IMAGE.endswith("'"):
    QUERY_IMAGE = QUERY_IMAGE[1:-1]

MODEL_PATH = r"E:\Projects\GitHub\GSS\models\best.pt"
EMBEDDINGS_PATH = r"E:\Projects\GitHub\GSS\Embeddings\imagenet_image_features.npy"
IMAGE_PATHS_PATH = r"E:\Projects\GitHub\GSS\Embeddings\imagenet_image_paths.pkl"

NUM_CLASSES = 13
TOP_K = 10

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

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

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

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

import torchvision
import timm

print(f"Loading custom model weights from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
else:
    state_dict = checkpoint
    num_classes = state_dict["head.fc.weight"].shape[0] if "head.fc.weight" in state_dict else NUM_CLASSES

is_torchvision = any(k.startswith('features.') for k in state_dict.keys())

if is_torchvision:
    model = torchvision.models.convnext_base(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.classifier[2] = torch.nn.Identity()
else:
    model = timm.create_model("convnext_base", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.reset_classifier(0)

model.to(device)
model.eval()

print("Model Loaded Successfully")

# ==========================================================
# LOAD DATABASE FEATURES
# ==========================================================

database_features = np.load(EMBEDDINGS_PATH)

with open(IMAGE_PATHS_PATH, "rb") as f:
    image_paths = pickle.load(f)

print("Loaded", len(image_paths), "image features")

# ==========================================================
# EXTRACT QUERY FEATURE
# ==========================================================

def extract_feature(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        feature = model(image)

        feature = feature.view(feature.size(0), -1)

        feature = F.normalize(feature, p=2, dim=1)

    return feature.squeeze().cpu().numpy()

query_feature = extract_feature(QUERY_IMAGE)

# ==========================================================
# COSINE SIMILARITY
# ==========================================================

# Apply L2 normalization to ensure accurate cosine similarity
database_features = database_features / np.linalg.norm(database_features, axis=1, keepdims=True)
query_feature = query_feature / np.linalg.norm(query_feature)

scores = np.dot(database_features, query_feature)

# ==========================================================
# SORT RESULTS
# ==========================================================

sorted_indices = np.argsort(scores)[::-1]

top_indices = sorted_indices[:TOP_K]

print("\nTop Similar Images\n")

for rank, idx in enumerate(top_indices, 1):

    print(f"{rank}. {image_paths[idx]}")
    print(f"   Similarity Score : {scores[idx]:.4f}")

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

plt.figure(figsize=(15, 8))

# Query image
plt.subplot(2, 6, 1)
plt.imshow(Image.open(QUERY_IMAGE))
plt.title("Query")
plt.axis("off")

# Top results
for i, idx in enumerate(top_indices):

    plt.subplot(2, 6, i + 2)

    img = Image.open(image_paths[idx])

    plt.imshow(img)

    plt.title(f"{scores[idx]:.2f}")

    plt.axis("off")

plt.tight_layout()
plt.show()