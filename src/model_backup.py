"""
EcoSort AI - Model Architecture
=================================
Defines the EcoSortNet model using MobileNetV2 with transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, MODEL_PATH


def create_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Create a MobileNetV2 model with a custom classifier head.

    Strategy:
    - Load MobileNetV2 pretrained on ImageNet.
    - Freeze all convolutional (feature-extraction) layers.
    - Replace the classifier with a new head for `num_classes` waste categories.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 9 for RealWaste).
    pretrained : bool
        Whether to load ImageNet pretrained weights.

    Returns
    -------
    model : nn.Module
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v2(weights=weights)

    # Freeze all feature-extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head
    # Original MobileNetV2 classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def save_model(model, path=None):
    """Save model state dict to disk."""
    if path is None:
        path = MODEL_PATH
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path=None, device=None):
    """
    Load a saved EcoSortNet model from disk.

    Parameters
    ----------
    path : str or None
    device : torch.device or None

    Returns
    -------
    model : nn.Module (in eval mode)
    """
    if path is None:
        path = MODEL_PATH
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model
