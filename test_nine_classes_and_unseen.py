import os
import json
import random
from PIL import Image
from src.config import CLASS_NAMES, DATA_DIR, ARTIFACTS_DIR
from src.model import load_model
from src.predict import predict

# Load split indices to pick test images from held-out test split
split_path = os.path.join(ARTIFACTS_DIR, "split_indices.json")
with open(split_path, "r") as f:
    split_data = json.load(f)
test_indices = set(split_data.get("test_indices", []))

# Find sample images for each of the 9 classes from RealWaste held-out test split
from torchvision.datasets import ImageFolder
from src.dataset import safe_pil_loader
dataset = ImageFolder(root=DATA_DIR, loader=safe_pil_loader)

class_to_test_img = {}
for idx in test_indices:
    path, label_idx = dataset.samples[idx]
    c_name = CLASS_NAMES[label_idx]
    if c_name not in class_to_test_img:
        class_to_test_img[c_name] = path
    if len(class_to_test_img) == len(CLASS_NAMES):
        break

print("Representative Test Images Identified for All 9 Classes:")
for c, p in class_to_test_img.items():
    print(f"  {c}: {os.path.basename(p)}")

# Also look for external unseen images from TrashBox test set if present
trashbox_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"
external_samples = {}
if os.path.exists(trashbox_test_dir):
    for sub in os.listdir(trashbox_test_dir):
        sub_path = os.path.join(trashbox_test_dir, sub)
        if os.path.isdir(sub_path):
            imgs = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                external_samples[sub] = imgs[0]

print(f"\nExternal Unseen Test Categories Found: {len(external_samples)}")
for c, p in list(external_samples.items())[:6]:
    print(f"  {c}: {os.path.basename(p)}")

# Run evaluation on both models
prod_model = load_model(os.path.join(ARTIFACTS_DIR, "ecosort_model.pth"))
cand_model = load_model(os.path.join(ARTIFACTS_DIR, "ecosort_candidate_model.pth"))
base_model = load_model(os.path.join(ARTIFACTS_DIR, "ecosort_model_v1_baseline.pth"))

print("\n" + "="*80)
print("PART 1: 9-CLASS HELD-OUT REPRESENTATIVE EVALUATION")
print("="*80)
print(f"{'Class':<20} | {'Expected':<15} | {'Prod Prediction':<22} | {'Prod Conf':<10} | {'Base Prediction':<20}")
print("-" * 95)

for c_name in CLASS_NAMES:
    p = class_to_test_img.get(c_name)
    if not p:
        continue
    img = Image.open(p)
    res_prod = predict(prod_model, img)
    res_base = predict(base_model, img)
    p_status = f"{res_prod['raw_predicted_class']} ({res_prod['decision_status'][:4]})"
    b_status = f"{res_base['raw_predicted_class']} ({res_base['decision_status'][:4]})"
    conf_str = f"{res_prod['confidence']*100:.1f}%"
    print(f"{c_name:<20} | {c_name:<15} | {p_status:<22} | {conf_str:<10} | {b_status:<20}")

print("\n" + "="*80)
print("PART 2: GENUINELY UNSEEN REAL-WORLD BENCHMARK SAMPLES")
print("="*80)
print(f"{'External Category':<22} | {'Expected EcoSort':<16} | {'Prod Pred (Conf)':<24} | {'Base Pred (Conf)':<24}")
print("-" * 92)

# Map TrashBox classes to EcoSort classes
trashbox_mapping = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "medical": "OOD (Medical)",
    "e-waste": "OOD (E-Waste)",
}

for ext_cat, exp_target in trashbox_mapping.items():
    if ext_cat in external_samples:
        img_path = external_samples[ext_cat]
        img = Image.open(img_path)
        is_med = (ext_cat == "medical_waste")
        res_prod = predict(prod_model, img, is_medical=is_med)
        res_base = predict(base_model, img, is_medical=is_med)
        
        p_desc = f"{res_prod['raw_predicted_class']} ({res_prod['confidence']*100:.1f}%)" if not res_prod['is_out_of_scope'] else f"REJECTED ({res_prod['confidence']*100:.1f}%)"
        b_desc = f"{res_base['raw_predicted_class']} ({res_base['confidence']*100:.1f}%)" if not res_base['is_out_of_scope'] else f"REJECTED ({res_base['confidence']*100:.1f}%)"
        print(f"{ext_cat:<22} | {exp_target:<16} | {p_desc:<24} | {b_desc:<24}")
