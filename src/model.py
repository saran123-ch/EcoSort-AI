"""
EcoSort AI - Model Architecture
=================================
Defines the EcoSortNet model using MobileNetV2 with transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, MODEL_PATH


def create_model(num_classes=NUM_CLASSES, pretrained=True, use_batchnorm=True):
    """
    Create a MobileNetV2 model with a custom classifier head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 9 for RealWaste).
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    use_batchnorm : bool
        Whether to include BatchNorm1d in the classifier head.

    Returns
    -------
    model : nn.Module
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v2(weights=weights)

    # Freeze feature layers by default (fine-tuning can unfreeze upper blocks)
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    if use_batchnorm:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
    else:
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
    Load a saved EcoSortNet model from disk with architecture auto-detection.

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

    state_dict = torch.load(path, map_location=device, weights_only=True)
    has_bn = any("classifier.2.weight" in k or "classifier.2.running_mean" in k for k in state_dict.keys())
    model = create_model(pretrained=False, use_batchnorm=has_bn)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {path} (BatchNorm: {has_bn})")
    return model
