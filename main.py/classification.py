import os
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F_vision
import timm

# ==========================================
# ARGUMENTS & PATHS
# ==========================================

parser = argparse.ArgumentParser(description="Jewelry Category Classification")
parser.add_argument("--image", type=str, required=False, help="Path to the image to classify")
args = parser.parse_args()

if args.image:
    IMAGE_PATH = args.image
else:
    IMAGE_PATH = input("Please enter the path to the image: ").strip()

if IMAGE_PATH.startswith('"') and IMAGE_PATH.endswith('"'):
    IMAGE_PATH = IMAGE_PATH[1:-1]
elif IMAGE_PATH.startswith("'") and IMAGE_PATH.endswith("'"):
    IMAGE_PATH = IMAGE_PATH[1:-1]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
METADATA_JSON_PATH = os.path.join(BASE_DIR, "metadata.json")

# ==========================================
# SETTINGS & CLASSES
# ==========================================

NUM_CLASSES = 13

CLASS_NAMES = [
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

CONFIDENCE_THRESHOLD = 0.50

# Diameter & Dimension Characteristics Guide
DIAMETER_GUIDE = {
    "Bangles": "Typical Diameter: 2.2 - 2.8 in (55 - 70 mm) | Large rigid circular wrist ornament.",
    "Rings": "Typical Diameter: 0.6 - 0.8 in (15 - 20 mm) | Medium finger ring band.",
    "Toe Rings": "Typical Diameter: 0.4 - 0.6 in (10 - 15 mm) | Small open/adjustable toe ring band.",
    "Nose Pin": "Typical Diameter: 0.2 - 0.4 in (5 - 10 mm) | Micro stud or tiny nose wire hoop."
}

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# IMAGE TRANSFORM (SquarePad + 320x320)
# ==========================================

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
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD MODEL
# ==========================================

model = timm.create_model(
    "convnext_base",
    pretrained=False,
    num_classes=NUM_CLASSES
)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# ==========================================
# CLASSIFICATION & PREDICTION
# ==========================================

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    exit(1)

raw_image = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(raw_image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1).squeeze(0)

# Top predicted classes
top_probs, top_indices = torch.topk(probabilities, k=min(3, NUM_CLASSES))

primary_class = CLASS_NAMES[top_indices[0].item()]
primary_conf = top_probs[0].item()

print("\n==================================================")
print(f"CLASSIFICATION RESULTS FOR: {os.path.basename(IMAGE_PATH)}")
print("==================================================")
print(f"Top Prediction : {primary_class}")
print(f"Confidence     : {round(primary_conf * 100, 2)}%")
print("--------------------------------------------------")

print("Top Probabilities Distribution:")
for rank in range(len(top_indices)):
    c_name = CLASS_NAMES[top_indices[rank].item()]
    c_prob = top_probs[rank].item()
    print(f"  {rank+1}. {c_name:<22} : {round(c_prob * 100, 2)}%")

print("--------------------------------------------------")

# Check if item has diameter/size distinction specifications
if primary_class in DIAMETER_GUIDE:
    print(f"Size / Diameter Profile : {DIAMETER_GUIDE[primary_class]}")

# Try to enrich with catalog metadata if available
if os.path.exists(METADATA_JSON_PATH):
    try:
        with open(METADATA_JSON_PATH, 'r', encoding='utf-8') as f:
            metadata_db = json.load(f)
        
        norm_path = IMAGE_PATH.replace('\\', '/')
        meta = metadata_db.get(norm_path)
        if not meta:
            basename = os.path.basename(norm_path)
            meta = next((v for k, v in metadata_db.items() if os.path.basename(k) == basename), None)
            
        if meta:
            print("\n[Catalog Metadata Found]")
            print(f"  Category Spec  : {meta.get('category', 'N/A')}")
            print(f"  Size / Type    : {meta.get('size', 'N/A')} ({meta.get('size_type', 'N/A')})")
            print(f"  Gross Weight   : {meta.get('gross_weight_g', 'N/A')} g")
            print(f"  Features       : {', '.join(meta.get('features', []))}")
    except Exception as e:
        pass

print("==================================================")

if primary_conf < CONFIDENCE_THRESHOLD:
    print("Status: Rejected (Low confidence score)")
else:
    print("Status: Accepted (High confidence classification)")
print("==================================================\n")