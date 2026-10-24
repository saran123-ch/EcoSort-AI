"""
EcoSort AI — Comprehensive Benchmark Comparison Script
======================================================
Compares:
  1. Baseline Model (ecosort_model_v1_baseline.pth)
  2. Prior Model (ecosort_model.pth)
  3. Enhanced Candidate Model (ecosort_candidate_model.pth)

Evaluates on:
  - Suite A: RealWaste Held-Out Test Set (719 images, 9 classes)
  - Suite B: Real-World Unseen Test Suite (280 images: Cardboard, Glass, Metal, Paper, Plastic, OOD)
"""

import os
import sys
import json
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, ".")
from src.config import CLASS_NAMES, NUM_CLASSES, ARTIFACTS_DIR
from src.dataset import load_dataset, TransformSubset, get_eval_transforms
from src.model import load_model
from src.predict import predict, preprocess_image

device = torch.device("cpu")

print("=" * 70)
print("EcoSort AI — Comprehensive Multi-Model Benchmark")
print("=" * 70)

models_to_test = [
    ("Original Baseline", "artifacts/ecosort_model_v1_baseline.pth"),
    ("Previous Production", "artifacts/ecosort_model.pth"),
    ("Candidate V1", "artifacts/ecosort_candidate_model.pth"),
    ("Fine-Tuned V2", "artifacts/ecosort_model_v2_enhanced.pth"),
    ("V3 Production Champion", "artifacts/ecosort_model_v3_production.pth"),
]

# ---------------------------------------------------------------------------
# SUITE A: RealWaste Held-Out Test Set (719 images, 9 classes)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUITE A: RealWaste Held-Out Test Set (719 images, 9 classes)")
print("=" * 70)

rw_dataset = load_dataset()
with open("artifacts/split_indices.json", "r") as f:
    rw_split = json.load(f)

test_indices = rw_split["test_indices"]
test_subset = TransformSubset(rw_dataset, test_indices, get_eval_transforms())
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

suite_a_results = {}

for label, path in models_to_test:
    if not os.path.exists(path):
        print(f"Skipping {label}: file not found at {path}")
        continue

    model = load_model(path, device)
    all_y, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().tolist()
            all_y.extend(labels.tolist())
            all_preds.extend(preds)

    acc = accuracy_score(all_y, all_preds)
    macro_f1 = f1_score(all_y, all_preds, average="macro", zero_division=0)
    rep = classification_report(all_y, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    suite_a_results[label] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": rep,
    }

    print(f"\n>>> [{label}] on RealWaste Test Set:")
    print(f"    Accuracy: {acc:.2%} | Macro F1: {macro_f1:.4f}")
    print(f"    {'Class':22s} | {'Precision':9s} | {'Recall':9s} | {'F1-score':9s}")
    print("    " + "-" * 55)
    for c in CLASS_NAMES:
        c_stats = rep[c]
        print(f"    {c:22s} | {c_stats['precision']:9.3f} | {c_stats['recall']:9.3f} | {c_stats['f1-score']:9.3f}")

# ---------------------------------------------------------------------------
# SUITE B: Real-World Unseen Test Suite (280 images from TrashBox test suite)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUITE B: Real-World Unseen Test Suite (Genuinely Unseen Camera/Web Photos)")
print("=" * 70)

real_world_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"
rw_classes = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
}

# 35 images each for Cardboard, Glass, Metal, Paper; 70 for Plastic (total 210 in-distribution real-world images)
# 35 medical + 35 e-waste = 70 out-of-scope images (total 280 images)
suite_b_samples = []
for folder, target_cls in rw_classes.items():
    p = os.path.join(real_world_test_dir, folder)
    if os.path.isdir(p):
        files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        limit = 70 if folder == "plastic" else 35
        for f in files[:limit]:
            suite_b_samples.append((os.path.join(p, f), target_cls, False))

# OOD samples
for ood_folder in ["e-waste", "medical"]:
    p = os.path.join(real_world_test_dir, ood_folder)
    if os.path.isdir(p):
        files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for f in files[:35]:
            suite_b_samples.append((os.path.join(p, f), "OUT_OF_SCOPE", True))

print(f"Total Suite B samples: {len(suite_b_samples)} (210 in-domain waste, 70 out-of-scope)")

suite_b_results = {}

for label, path in models_to_test:
    if not os.path.exists(path):
        continue

    model = load_model(path, device)
    
    in_domain_true = []
    in_domain_pred = []
    ood_screened_correctly = 0
    ood_total = 0
    
    for img_path, true_label, is_ood in suite_b_samples:
        try:
            img = Image.open(img_path)
            res = predict(model, img, device=device)
            raw_pred = res["raw_predicted_class"]
            status = res["decision_status"]

            if is_ood:
                ood_total += 1
                # If correctly flagged as OUT_OF_SCOPE or UNCERTAIN, safe
                if status in ("OUT_OF_SCOPE", "UNCERTAIN"):
                    ood_screened_correctly += 1
            else:
                in_domain_true.append(true_label)
                in_domain_pred.append(raw_pred)
        except Exception as e:
            print(f"Error on {img_path}: {e}")

    # Compute in-domain metrics on the 5 classes
    target_5_classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]
    acc_5 = sum(1 for yt, yp in zip(in_domain_true, in_domain_pred) if yt == yp) / len(in_domain_true)
    macro_f1_5 = f1_score(in_domain_true, in_domain_pred, labels=target_5_classes, average="macro", zero_division=0)
    rep_5 = classification_report(in_domain_true, in_domain_pred, labels=target_5_classes, target_names=target_5_classes, output_dict=True, zero_division=0)
    cm_5 = confusion_matrix(in_domain_true, in_domain_pred, labels=target_5_classes).tolist()

    ood_safety_rate = (ood_screened_correctly / ood_total) if ood_total > 0 else 0.0

    suite_b_results[label] = {
        "accuracy": acc_5,
        "macro_f1": macro_f1_5,
        "ood_safety_rate": ood_safety_rate,
        "ood_screened": ood_screened_correctly,
        "ood_total": ood_total,
        "report": rep_5,
        "confusion_matrix": cm_5,
    }

    print(f"\n>>> [{label}] on Real-World Unseen Suite:")
    print(f"    In-Domain Real-World Accuracy: {acc_5:.2%} | Macro F1: {macro_f1_5:.4f}")
    print(f"    OOD Safety Rate (flagged safe): {ood_screened_correctly}/{ood_total} ({ood_safety_rate:.1%})")
    print(f"    {'Class':12s} | {'Precision':9s} | {'Recall':9s} | {'F1-score':9s}")
    print("    " + "-" * 45)
    for c in target_5_classes:
        c_stats = rep_5[c]
        print(f"    {c:12s} | {c_stats['precision']:9.3f} | {c_stats['recall']:9.3f} | {c_stats['f1-score']:9.3f}")

# Save full comparison results
comparison_output = {
    "suite_a_realwaste": suite_a_results,
    "suite_b_real_world": suite_b_results,
}
with open("artifacts/model_comparison_benchmark.json", "w") as f:
    json.dump(comparison_output, f, indent=2)

print("\n" + "=" * 70)
print("Benchmark completed! Results saved to artifacts/model_comparison_benchmark.json")
print("=" * 70)
