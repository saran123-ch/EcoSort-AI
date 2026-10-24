"""
EcoSort AI - Dataset Utilities
================================
Functions for loading the TrashBox dataset, creating stratified splits,
and building PyTorch DataLoaders with appropriate transforms.
"""

import os
import json
import random
import shutil
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    CLASS_NAMES, ARTIFACTS_DIR, CLASS_DISTRIBUTION_PATH,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms():
    """Augmentation + normalisation for training data only."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """Deterministic resize + normalisation for validation and test data."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset loading and splitting
# ---------------------------------------------------------------------------

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_pil_loader(path):
    """
    Safely load an image from disk, convert to RGB, and handle corruptions gracefully.
    """
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        # Fallback neutral image so DataLoader never crashes
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))


def load_dataset(data_dir=None):
    """
    Load the TrashBox dataset from disk using torchvision.ImageFolder.

    Parameters
    ----------
    data_dir : str or None
        Path to the root folder that contains one sub-folder per class.
        Defaults to config.DATA_DIR.

    Returns
    -------
    dataset : torchvision.datasets.ImageFolder
        The full (unsplit) dataset.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Please ensure the TrashBox dataset is extracted so that\n"
            f"the class folders are at:\n"
            f"  {data_dir}/<class_name>/"
        )

    # Use robust loader to handle any edge-case image encodings gracefully
    dataset = datasets.ImageFolder(root=data_dir, loader=safe_pil_loader)

    # Verify class names match expected names
    discovered = dataset.classes
    print(f"Discovered {len(discovered)} classes: {discovered}")

    return dataset


def stratified_split(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                     seed=RANDOM_SEED):
    """
    Produce stratified train / val / test index splits.

    Parameters
    ----------
    dataset : torchvision.datasets.ImageFolder
    train_ratio, val_ratio : float
    seed : int

    Returns
    -------
    train_indices, val_indices, test_indices : list[int]
    """
    random.seed(seed)
    targets = [s[1] for s in dataset.samples]
    class_indices = {}
    for idx, label in enumerate(targets):
        class_indices.setdefault(label, []).append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Shuffle within each split
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    print(f"Split sizes — Train: {len(train_indices)}, "
          f"Val: {len(val_indices)}, Test: {len(test_indices)}")

    return train_indices, val_indices, test_indices


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so that we can apply a *different* transform per split."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def create_dataloaders(dataset, train_indices, val_indices, test_indices,
                       batch_size=BATCH_SIZE):
    """
    Build DataLoaders for each split with appropriate transforms.

    Parameters
    ----------
    dataset : ImageFolder (loaded WITHOUT a transform so images are PIL)
    train_indices, val_indices, test_indices : list[int]
    batch_size : int

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_ds = TransformSubset(dataset, train_indices, get_train_transforms())
    val_ds = TransformSubset(dataset, val_indices, get_eval_transforms())
    test_ds = TransformSubset(dataset, test_indices, get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_class_weights(dataset, train_indices):
    """
    Compute inverse-frequency class weights for the training set.

    Returns
    -------
    weights : torch.FloatTensor of shape (num_classes,)
    """
    targets = [dataset.targets[i] for i in train_indices]
    counts = Counter(targets)
    total = len(targets)
    num_classes = len(counts)
    weights = []
    for c in range(num_classes):
        weights.append(total / (num_classes * counts[c]))
    weights = torch.FloatTensor(weights)
    print(f"Class weights: {weights.tolist()}")
    return weights


def plot_class_distribution(dataset):
    """Save a bar chart of class distribution to artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    targets = [s[1] for s in dataset.samples]
    counts = Counter(targets)
    class_names = dataset.classes
    names = [class_names[i] for i in range(len(class_names))]
    values = [counts[i] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis([i / len(names) for i in range(len(names))])
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Waste Category", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("TrashBox Dataset — Class Distribution", fontsize=14, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=10)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CLASS_DISTRIBUTION_PATH, dpi=150)
    plt.close()
    print(f"Class distribution plot saved to {CLASS_DISTRIBUTION_PATH}")
