"""
EcoSort AI - Prediction Module
================================
Functions for making confidence-aware predictions on single images.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, CONFIDENCE_THRESHOLD, SORTING_GUIDANCE,
)


def preprocess_image(image):
    """
    Preprocess a PIL Image for model inference with full aspect-ratio letterboxing.
    Preserves all bounding edges, container shapes, and material features without distortion or center cropping.

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    tensor : torch.Tensor of shape (1, 3, 224, 224)
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    max_dim = max(w, h)
    if max_dim > 0:
        padded = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
        padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    else:
        padded = image

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(padded).unsqueeze(0)  # add batch dimension
    return tensor


def predict(model, image, is_medical=False, device=None):
    """
    Run inference on a single PIL image and return confidence-aware,
    OOD-screened results.

    Parameters
    ----------
    model : torch.nn.Module (in eval mode)
    image : PIL.Image.Image
    is_medical : bool, optional
        Legacy flag for medical packaging.
    device : torch.device or None

    Returns
    -------
    result : dict with keys:
        - predicted_class : str
        - raw_predicted_class : str
        - confidence : float (0-1)
        - margin : float
        - entropy : float
        - energy : float
        - decision_status : str ('CONFIDENT', 'UNCERTAIN', or 'OUT_OF_SCOPE')
        - top3 : list of (class_name, score) tuples
        - all_scores : dict mapping class_name -> score
        - is_confident : bool
        - is_supported : bool
        - is_out_of_scope : bool
        - sorting_guidance : dict
        - reliability_message : str or None
    """
    if is_medical:
        return {
            "predicted_class": "Unrecognized / Out-of-Scope (Medical)",
            "raw_predicted_class": "Medical / Pharmaceutical Packaging",
            "confidence": 1.0,
            "margin": 1.0,
            "entropy": 0.0,
            "energy": -10.0,
            "decision_status": "OUT_OF_SCOPE",
            "top3": [("Medical / Medicine Packaging", 1.0)],
            "all_scores": {"Medical / Medicine Packaging": 1.0},
            "is_confident": False,
            "is_supported": False,
            "is_out_of_scope": True,
            "sorting_guidance": {
                "bin": "Hazardous / Medical Waste — Do Not Recycle",
                "suggestion": (
                    "Medical and pharmaceutical packaging (blister packs, medicine bottles, vials) "
                    "must NOT be placed in standard municipal recycling bins. Please return unused "
                    "medicines to a designated pharmacy take-back scheme or medical waste collection facility."
                ),
                "icon": "⚕️",
            },
            "reliability_message": (
                "⚠️ Unsupported Item: EcoSort AI vision model is trained on municipal waste streams and "
                "cannot provide recycling advice for pharmaceutical or clinical packaging."
            ),
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure RGB and EXIF orientation
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(tensor).squeeze(0)  # shape: (num_classes,)
        probabilities = F.softmax(logits, dim=0)

    # Compute metric values
    scores = probabilities.cpu().numpy()
    top_indices = scores.argsort()[::-1]
    top1_idx = int(top_indices[0])
    top2_idx = int(top_indices[1])

    top_class = CLASS_NAMES[top1_idx]
    confidence = float(scores[top1_idx])
    runner_class = CLASS_NAMES[top2_idx]
    runner_confidence = float(scores[top2_idx])

    margin = float(confidence - runner_confidence)
    # Normalized Shannon entropy across the classes
    entropy = float(-np.sum(scores * np.log(scores + 1e-12)) / np.log(len(CLASS_NAMES)))
    # Energy score with T=1.0: E(x) = -T * logsumexp(logits / T)
    energy = float(-1.0 * torch.logsumexp(logits, dim=0).item())

    # Full dictionary of scores
    all_scores = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
    top3 = [(CLASS_NAMES[i], float(scores[i])) for i in top_indices[:3]]

    # Autonomous 3-tier Decision Rule:
    # 1. OUT_OF_SCOPE: Very low top score (<40%), excessively diffuse entropy (>0.72), or weak energy (>-2.8)
    #    (e.g., electronic boards, clinical packaging, non-municipal industrial waste)
    # 2. CONFIDENT: High probability (>=65%), clear margin (>=18%), low entropy (<=0.55), solid energy (<=-4.0)
    # 3. UNCERTAIN: Intermediate / ambiguous distribution (mandates verification before sorting)
    if confidence < 0.40 or energy > -2.8 or entropy > 0.72:
        decision_status = "OUT_OF_SCOPE"
        predicted_class = "Unrecognized / Out-of-Scope Object"
        is_confident = False
        is_supported = False
        is_out_of_scope = True
        reliability_message = (
            f"🚫 Unrecognized / Out-of-Scope: This object does not exhibit the visual "
            f"features of any of the 9 supported municipal waste categories (Entropy: {entropy:.2f}, "
            f"Energy: {energy:.2f}). To prevent recycling contamination, automated disposal advice is withheld."
        )
        guidance = {
            "bin": "Do Not Recycle in Standard Municipal Bins",
            "suggestion": (
                "This object could not be reliably recognized as municipal waste, or may be a specialized "
                "item (such as electronic waste, medical packaging, chemicals, or household hazard). "
                "Do NOT discard into common blue/green bins. Consult municipal hazardous/special collection services."
            ),
            "icon": "🚫",
        }
    elif confidence >= 0.65 and margin >= 0.18 and entropy <= 0.55 and energy <= -4.0:
        decision_status = "CONFIDENT"
        predicted_class = top_class
        is_confident = True
        is_supported = True
        is_out_of_scope = False
        reliability_message = None
        guidance = SORTING_GUIDANCE.get(top_class, {
            "bin": "Unknown",
            "suggestion": "Unable to determine sorting guidance for this category.",
            "icon": "❓",
        })
    else:
        decision_status = "UNCERTAIN"
        predicted_class = f"{top_class} (Low Certainty)"
        is_confident = False
        is_supported = True
        is_out_of_scope = False
        reliability_message = (
            f"⚠️ Low confidence / Ambiguous identification (Top candidate: {top_class} at {confidence:.1%}, "
            f"Runner-up: {runner_class} at {runner_confidence:.1%}, Margin: {margin:.1%}). "
            f"Automated disposal advice is restricted to prevent recycling contamination."
        )
        guidance = {
            "bin": "Disposal Advice Restricted — Manual Check Required",
            "suggestion": (
                f"The item was classified as potentially {top_class}, but confidence is insufficient ({confidence:.1%}). "
                f"Please manually inspect the item's recycling triangular symbol, resin identification code, "
                f"or local rules before placing it in a sorting bin."
            ),
            "icon": "⚠️",
        }

    return {
        "predicted_class": predicted_class,
        "raw_predicted_class": top_class,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "energy": energy,
        "decision_status": decision_status,
        "top3": top3,
        "all_scores": all_scores,
        "is_confident": is_confident,
        "is_supported": is_supported,
        "is_out_of_scope": is_out_of_scope,
        "sorting_guidance": guidance,
        "reliability_message": reliability_message,
    }

