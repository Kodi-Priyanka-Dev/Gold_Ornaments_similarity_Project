import torch
from sentence_transformers import util
from PIL import Image
import os

JEWELRY_CONCEPTS = [
    "a photo of jewelry",
    "a photo of an earring",
    "a photo of a jhumka",
    "a photo of a necklace",
    "a photo of a bangle",
    "a photo of a ring",
    "a photo of a chain",
    "a photo of a bracelet",
    "a photo of bridal jewelry",
    "a photo of gold ornaments",
    "a photo of a person wearing jewelry",
    # Minority categories with no prior representative concept, so text
    # queries naming them (e.g. "toe rings") scored near-tied against
    # non-jewelry concepts and were wrongly rejected before ever reaching
    # category detection.
    "a photo of a toe ring",
    "a photo of a nose pin",
    "a photo of a maang tikka",
    "a photo of a waist chain",
    "a photo of an ear cuff",
    "a photo of a hair braid ornament",
    "a photo of an anklet",
]

NON_JEWELRY_CONCEPTS = [
    "a photo of a car", 
    "a photo of a phone", 
    "a photo of food",
    "a photo of an animal", 
    "a photo of a shoe", 
    "a photo of clothing",
    "a photo of a building", 
    "a photo of furniture", 
    "a photo of a person without jewelry",
    "a photo of a random object", 
    "a blank background", 
    "a selfie",
    "a screenshot of a website",
    "a user interface",
    "text on a screen",
    "a computer monitor"
]

_jewelry_embeddings = None
_non_jewelry_embeddings = None

def get_concept_embeddings(clip_model, device):
    """
    Lazy load and cache the text concept embeddings.
    """
    global _jewelry_embeddings, _non_jewelry_embeddings
    if _jewelry_embeddings is None:
        _jewelry_embeddings = clip_model.encode(JEWELRY_CONCEPTS, convert_to_tensor=True, device=device)
        _non_jewelry_embeddings = clip_model.encode(NON_JEWELRY_CONCEPTS, convert_to_tensor=True, device=device)
    return _jewelry_embeddings, _non_jewelry_embeddings

def validate_jewelry_image(image_path, clip_model, device, threshold_margin=0.0):
    """
    Validates if an image is jewelry using CLIP zero-shot classification.
    Returns: (is_jewelry, max_jewelry_score, max_non_jewelry_score)
    """
    if not os.path.exists(image_path):
        return True, 0.0, 0.0

    try:
        img = Image.open(image_path).convert("RGB")
        img_embedding = clip_model.encode(img, convert_to_tensor=True, device=device)
        
        jewel_emb, non_jewel_emb = get_concept_embeddings(clip_model, device)
        
        # Calculate cosine similarities against both positive and negative concepts
        jewelry_scores = util.cos_sim(img_embedding, jewel_emb)[0]
        non_jewelry_scores = util.cos_sim(img_embedding, non_jewel_emb)[0]
        
        max_jewelry_score = torch.max(jewelry_scores).item()
        max_non_jewelry_score = torch.max(non_jewelry_scores).item()
        
        # Determine if it's jewelry based on which concept group it is closer to
        is_jewelry = (max_jewelry_score - max_non_jewelry_score) >= threshold_margin
        
        return is_jewelry, max_jewelry_score, max_non_jewelry_score
    except Exception as e:
        print(f"Error in CLIP jewelry validation: {e}")
        # Default to True on error to not incorrectly block valid images if something breaks
        return True, 0.0, 0.0

def validate_jewelry_text(query_text, clip_model, device, threshold_margin=0.0):
    """
    Validates if a text query is related to jewelry using CLIP zero-shot classification.
    Returns: (is_jewelry, max_jewelry_score, max_non_jewelry_score)
    """
    if not query_text or query_text.strip() == "":
        return True, 0.0, 0.0
        
    try:
        text_embedding = clip_model.encode(query_text, convert_to_tensor=True, device=device)
        
        jewel_emb, non_jewel_emb = get_concept_embeddings(clip_model, device)
        
        # Calculate cosine similarities against both positive and negative concepts
        jewelry_scores = util.cos_sim(text_embedding, jewel_emb)[0]
        non_jewelry_scores = util.cos_sim(text_embedding, non_jewel_emb)[0]
        
        max_jewelry_score = torch.max(jewelry_scores).item()
        max_non_jewelry_score = torch.max(non_jewelry_scores).item()
        
        is_jewelry = (max_jewelry_score - max_non_jewelry_score) >= threshold_margin
        
        return is_jewelry, max_jewelry_score, max_non_jewelry_score
    except Exception as e:
        print(f"Error in CLIP text validation: {e}")
        return True, 0.0, 0.0
