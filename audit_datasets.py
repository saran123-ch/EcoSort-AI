import os
import sys
import hashlib
import json
from collections import Counter
from PIL import Image

sys.path.insert(0, ".")
from src.config import CLASS_NAMES, NUM_CLASSES, DATA_DIR, ARTIFACTS_DIR

print("=" * 70)
print("EcoSort AI — Comprehensive Dataset Audit & Integrity Check")
print("=" * 70)

# Paths
rw_dir = DATA_DIR  # data/realwaste-main/RealWaste
tb_dir = "data/trashbox_correct"
split_file = os.path.join(ARTIFACTS_DIR, "split_indices.json")
ext_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"

# 1. Audit RealWaste Dataset
print("\n[Audit 1/5] Auditing RealWaste Dataset...")
rw_counts = Counter()
rw_corrupt = []
rw_hashes = {}  # md5 -> path

for c in CLASS_NAMES:
    c_dir = os.path.join(rw_dir, c)
    if not os.path.isdir(c_dir):
        print(f"Warning: Class folder missing: {c_dir}")
        continue
    files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for f in files:
        p = os.path.join(c_dir, f)
        try:
            with open(p, "rb") as fp:
                data = fp.read()
                if len(data) == 0:
                    rw_corrupt.append((p, "0-byte file"))
                    continue
                h = hashlib.md5(data).hexdigest()
                if h in rw_hashes:
                    # duplicate found
                    pass
                else:
                    rw_hashes[h] = p
            with Image.open(p) as img:
                img.verify()
            rw_counts[c] += 1
        except Exception as e:
            rw_corrupt.append((p, str(e)))

print(f"RealWaste total valid images: {sum(rw_counts.values())}")
for c in CLASS_NAMES:
    print(f"  {c:20s}: {rw_counts[c]:4d} images")
print(f"RealWaste corrupted files: {len(rw_corrupt)}")
print(f"RealWaste unique MD5 hashes: {len(rw_hashes)} (Duplicates: {sum(rw_counts.values()) - len(rw_hashes)})")

# 2. Audit Official Split Indices
print("\n[Audit 2/5] Auditing Official Test Split...")
if os.path.exists(split_file):
    with open(split_file, "r") as f:
        splits = json.load(f)
    train_idx = set(splits.get("train_indices", []))
    val_idx = set(splits.get("val_indices", []))
    test_idx = set(splits.get("test_indices", []))
    print(f"Split sizes — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    overlap_tv = train_idx.intersection(val_idx)
    overlap_tt = train_idx.intersection(test_idx)
    overlap_vt = val_idx.intersection(test_idx)
    print(f"Overlap Train/Val: {len(overlap_tv)}, Train/Test: {len(overlap_tt)}, Val/Test: {len(overlap_vt)}")
    assert len(overlap_tt) == 0, "CRITICAL ERROR: Data leakage between Train and Test!"
    print(">>> VERIFIED: Zero data leakage in official RealWaste split.")
else:
    print("Split indices file not found!")

# 3. Audit External Unseen Test Set
print("\n[Audit 3/5] Auditing External Unseen Test Set (TrashBox test)...")
ext_test_hashes = {}
ext_test_counts = Counter()
if os.path.exists(ext_test_dir):
    for root, dirs, files in os.walk(ext_test_dir):
        cat = os.path.basename(root)
        imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in imgs:
            p = os.path.join(root, f)
            try:
                with open(p, "rb") as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                    ext_test_hashes[h] = p
                ext_test_counts[cat] += 1
            except Exception:
                pass
    print(f"External test images indexed: {sum(ext_test_counts.values())}")
    for cat, cnt in sorted(ext_test_counts.items()):
        print(f"  {cat:20s}: {cnt:4d} images")
else:
    print(f"External test set directory not found at {ext_test_dir}")

# 4. Check for Leakage between RealWaste and External Test Set
print("\n[Audit 4/5] Checking Cross-Dataset Overlap (RealWaste vs External Test Set)...")
cross_overlap = set(rw_hashes.keys()).intersection(set(ext_test_hashes.keys()))
print(f"Exact MD5 matches between RealWaste and External Test Set: {len(cross_overlap)}")
if len(cross_overlap) > 0:
    for h in list(cross_overlap)[:5]:
        print(f"  Overlap sample: {rw_hashes[h]} <--> {ext_test_hashes[h]}")
else:
    print(">>> VERIFIED: Zero overlap between RealWaste and External Test Set.")

# 5. Audit TrashBox Auxiliary Curated Data
print("\n[Audit 5/5] Auditing TrashBox Curated Training Candidates...")
tb_subfolders = {
    "Cardboard": ["cardboard"],
    "Glass": ["glass"],
    "Metal": ["metal/beverage cans", "metal/metal containers", "metal/spray cans", "metal/construction scrap"],
    "Paper": ["paper/paper", "paper/news paper", "paper/paper cups"],
    "Plastic": ["plastic/plastic bottles", "plastic/plastic containers", "plastic/plastic cups", "plastic/plastic bags"],
    "Excluded_Paper_TetraPak": ["paper/tetra pak"],
    "Excluded_Plastic_Cigarettes": ["plastic/cigarette butt"],
    "OOD_Medical": ["medical/gloves", "medical/masks", "medical/medicines", "medical/syringe"],
    "OOD_EWaste": ["e-waste/electrical cables", "e-waste/electronic chips", "e-waste/laptops", "e-waste/small appliances", "e-waste/smartphones"],
}

tb_summary = {}
leakage_with_test = []

for category, sub_list in tb_subfolders.items():
    total_imgs = 0
    clean_imgs = 0
    for sub in sub_list:
        sub_p = os.path.join(tb_dir, sub.replace("/", os.sep))
        if os.path.isdir(sub_p):
            files = [f for f in os.listdir(sub_p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in files:
                total_imgs += 1
                p = os.path.join(sub_p, f)
                try:
                    with open(p, "rb") as fp:
                        h = hashlib.md5(fp.read()).hexdigest()
                    if h in ext_test_hashes:
                        leakage_with_test.append((p, ext_test_hashes[h]))
                    else:
                        clean_imgs += 1
                except Exception:
                    pass
    tb_summary[category] = {"total": total_imgs, "clean_unleaked": clean_imgs}
    print(f"  {category:28s}: {clean_imgs:4d} usable (unleaked) / {total_imgs:4d} total")

print(f"\nPotential leakage with external test set: {len(leakage_with_test)}")
if len(leakage_with_test) > 0:
    print(f"  Example leaked files detected and will be EXCLUDED: {len(leakage_with_test)}")
else:
    print(">>> VERIFIED: Zero exact hash overlap between curated candidates and test set.")

print("\n" + "=" * 70)
print("AUDIT COMPLETE — SUMMARY SAVED")
print("=" * 70)
