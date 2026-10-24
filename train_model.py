"""
EcoSort AI - Training Pipeline
================================
Trains a MobileNetV2-based classifier on the RealWaste dataset.

Usage:
    python train_model.py

The script will:
1. Load and inspect the dataset
2. Create stratified train/val/test splits
3. Train the model with early stopping
4. Save the best model and training history to artifacts/
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    RANDOM_SEED, ARTIFACTS_DIR, MODEL_PATH, HISTORY_PATH,
)
from src.dataset import (
    load_dataset, stratified_split, create_dataloaders,
    compute_class_weights, plot_class_distribution,
)
from src.model import create_model, save_model


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("EcoSort AI — Training Pipeline")
    print("=" * 60)

    # Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Step 1: Load dataset
    print("\n[Step 1/6] Loading dataset...")
    dataset = load_dataset()
    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Step 2: Plot class distribution
    print("\n[Step 2/6] Analysing class distribution...")
    plot_class_distribution(dataset)

    # Step 3: Split dataset
    print("\n[Step 3/6] Creating stratified train/val/test split...")
    train_idx, val_idx, test_idx = stratified_split(dataset)

    # Save split indices for reproducibility
    split_info = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    split_path = os.path.join(ARTIFACTS_DIR, "split_indices_v2.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f)
    print(f"Split indices saved to {split_path}")

    # Step 4: Create data loaders
    print("\n[Step 4/6] Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, train_idx, val_idx, test_idx
    )

    # Step 5: Build model
    print("\n[Step 5/6] Building model...")
    model = create_model(pretrained=True)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Class weights for imbalanced dataset
    class_weights = compute_class_weights(dataset, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # Step 6: Train
    print(f"\n[Step 6/6] Training for up to {NUM_EPOCHS} epochs "
          f"(early stopping patience={EARLY_STOPPING_PATIENCE})...")
    print("-" * 60)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, MODEL_PATH)
            print(f"  → Best model saved (val_loss improved to {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {HISTORY_PATH}")

    print("\n" + "=" * 60)
    print("Training complete! Next step: run evaluate_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
