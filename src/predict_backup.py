"""
EcoSort AI - Prediction Module
================================
Functions for making confidence-aware predictions on single images.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CONFIDENCE_THRESHOLD, SORTING_GUIDANCE,
)


def preprocess_image(image):
    """
    Preprocess a PIL Image for model inference.

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    tensor : torch.Tensor of shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor


def predict(model, image, device=None):
    """
    Run inference on a single PIL image and return confidence-aware results.

    Parameters
    ----------
    model : torch.nn.Module (in eval mode)
    image : PIL.Image.Image
    device : torch.device or None

    Returns
    -------
    result : dict with keys:
        - predicted_class : str
        - confidence : float (0-1)
        - top3 : list of (class_name, score) tuples
        - all_scores : dict mapping class_name -> score
        - is_confident : bool
        - sorting_guidance : dict
        - reliability_message : str or None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)  # shape: (num_classes,)

    # Get all scores
    scores = probabilities.cpu().numpy()
    all_scores = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}

    # Top-3 predictions
    top3_indices = scores.argsort()[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(scores[i])) for i in top3_indices]

    # Primary prediction
    predicted_idx = int(scores.argmax())
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(scores[predicted_idx])

    # Confidence check
    is_confident = confidence >= CONFIDENCE_THRESHOLD

    # Reliability message
    if not is_confident:
        reliability_message = (
            f"⚠️ Low confidence ({confidence:.1%}). "
            f"The model is not certain about this prediction. "
            f"Please verify the waste category before disposing."
        )
    else:
        reliability_message = None

    # Sorting guidance
    guidance = SORTING_GUIDANCE.get(predicted_class, {
        "bin": "Unknown",
        "suggestion": "Unable to determine sorting guidance for this category.",
        "icon": "❓",
    })

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top3": top3,
        "all_scores": all_scores,
        "is_confident": is_confident,
        "sorting_guidance": guidance,
        "reliability_message": reliability_message,
    }
