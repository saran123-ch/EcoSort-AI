"""
EcoSort AI - Evaluation Script
================================
Evaluates the trained model on the held-out test set and generates:
- Per-class precision, recall, F1-score
- Overall accuracy
- Confusion matrix plot
- Learning curves plot
- Saved metrics JSON

Usage:
    python evaluate_model.py
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ARTIFACTS_DIR, MODEL_PATH, METRICS_PATH, HISTORY_PATH,
    CONFUSION_MATRIX_PATH, LEARNING_CURVES_PATH, CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
)
from src.dataset import load_dataset, stratified_split, create_dataloaders
from src.model import load_model


def evaluate_on_test(model, test_loader, device):
    """
    Run the model on the test set and collect predictions.

    Returns
    -------
    all_labels : list[int]
    all_preds : list[int]
    all_confidences : list[float]
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, predicted = probs.max(1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_confidences.extend(max_probs.cpu().numpy().tolist())

    return all_labels, all_preds, all_confidences


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_learning_curves(history, save_path):
    """Generate and save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curves saved to {save_path}")


def analyse_confidence_threshold(all_labels, all_preds, all_confidences):
    """
    Examine the provisional confidence threshold on test predictions.
    Reports how many predictions fall below the threshold and their accuracy.
    """
    total = len(all_labels)
    confident_mask = [c >= CONFIDENCE_THRESHOLD for c in all_confidences]
    uncertain_mask = [not m for m in confident_mask]

    n_confident = sum(confident_mask)
    n_uncertain = sum(uncertain_mask)

    # Accuracy among confident predictions
    if n_confident > 0:
        confident_correct = sum(
            1 for i in range(total) if confident_mask[i] and all_labels[i] == all_preds[i]
        )
        confident_acc = confident_correct / n_confident
    else:
        confident_acc = 0.0

    # Accuracy among uncertain predictions
    if n_uncertain > 0:
        uncertain_correct = sum(
            1 for i in range(total) if uncertain_mask[i] and all_labels[i] == all_preds[i]
        )
        uncertain_acc = uncertain_correct / n_uncertain
    else:
        uncertain_acc = 0.0

    analysis = {
        "threshold": CONFIDENCE_THRESHOLD,
        "total_predictions": total,
        "confident_predictions": n_confident,
        "confident_percentage": round(n_confident / total * 100, 1),
        "confident_accuracy": round(confident_acc * 100, 2),
        "uncertain_predictions": n_uncertain,
        "uncertain_percentage": round(n_uncertain / total * 100, 1),
        "uncertain_accuracy": round(uncertain_acc * 100, 2),
    }

    print(f"\nConfidence Threshold Analysis (threshold = {CONFIDENCE_THRESHOLD}):")
    print(f"  Confident predictions: {n_confident}/{total} "
          f"({analysis['confident_percentage']}%) — "
          f"accuracy: {analysis['confident_accuracy']}%")
    print(f"  Uncertain predictions: {n_uncertain}/{total} "
          f"({analysis['uncertain_percentage']}%) — "
          f"accuracy: {analysis['uncertain_accuracy']}%")

    return analysis


def main():
    print("=" * 60)
    print("EcoSort AI — Evaluation Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at {MODEL_PATH}")
        print("Please run train_model.py first.")
        sys.exit(1)

    # Load model
    print("\n[Step 1/5] Loading trained model...")
    model = load_model(MODEL_PATH, device)

    # Load dataset and recreate the same split
    print("\n[Step 2/5] Loading dataset and test split...")
    dataset = load_dataset()
    split_file = os.path.join(ARTIFACTS_DIR, "split_indices.json")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            saved_split = json.load(f)
        train_idx = saved_split.get("train_indices", [])
        val_idx = saved_split.get("val_indices", [])
        test_idx = saved_split.get("test_indices", [])
        print(f"Loaded verified test split from {split_file}")
    else:
        train_idx, val_idx, test_idx = stratified_split(dataset)
    _, _, test_loader = create_dataloaders(dataset, train_idx, val_idx, test_idx)
    print(f"Test set size: {len(test_idx)}")

    # Evaluate
    print("\n[Step 3/5] Evaluating on test set...")
    all_labels, all_preds, all_confidences = evaluate_on_test(model, test_loader, device)

    # Classification report
    overall_acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(f"\nOverall Test Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"\nClassification Report:\n{report_text}")

    # Confidence threshold analysis
    threshold_analysis = analyse_confidence_threshold(
        all_labels, all_preds, all_confidences
    )

    # Save metrics
    metrics = {
        "overall_accuracy": round(overall_acc * 100, 2),
        "classification_report": report,
        "confidence_threshold_analysis": threshold_analysis,
        "test_set_size": len(test_idx),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Plot confusion matrix
    print("\n[Step 4/5] Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, CONFUSION_MATRIX_PATH)

    # Plot learning curves
    print("\n[Step 5/5] Generating learning curves...")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)
        plot_learning_curves(history, LEARNING_CURVES_PATH)
    else:
        print(f"WARNING: Training history not found at {HISTORY_PATH}")
        print("Learning curves will not be generated.")

    print("\n" + "=" * 60)
    print("Evaluation complete! All results saved in artifacts/")
    print("Next step: run the Streamlit app with 'streamlit run app.py'")
    print("=" * 60)


if __name__ == "__main__":
    main()
