"""
EcoSort AI — Production Model Training & Generalization Pipeline (V4)
======================================================================
Solves:
  1. Plastic bottle vs Glass misclassification (expanded PET bottle representation,
     balanced glass prior, smoothed class weights).
  2. Vegetation background shortcut (RandomErasing / Cutout, ColorJitter hue).
  3. Cardboard vs Paper scale/texture confusion (balanced 400 Cardboard cartons vs 400 Paper,
     exclusion of Tetra Pak cartons).
  4. 9-Class preservation (full RealWaste train/val retention for Food Organics, Textile Trash,
     Vegetation, Miscellaneous Trash).
  5. Mid-level texture adaptation (unfreezing MobileNetV2 blocks 12-18).
  6. Strict zero data leakage enforcement against both RealWaste test set (719) and
     TrashBox external test suite (1,793).
"""

import os
import sys
import json
import time
import random
import hashlib
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageFile
from sklearn.metrics import f1_score, accuracy_score, recall_score
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, ".")
from src.config import (
    CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    IMAGE_SIZE, ARTIFACTS_DIR, RANDOM_SEED, DATA_DIR,
)
from src.model import load_model

STARTING_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_model.pth")
PRODUCTION_V4_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_model_v4_realworld.pth")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history_v4.json")
SPLIT_PATH = os.path.join(ARTIFACTS_DIR, "split_indices.json")


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LetterboxTransform:
    def __init__(self, size=IMAGE_SIZE):
        self.size = size

    def __call__(self, img):
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > 0:
            padded = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
            padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
            img = padded
        return img


class WasteDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transforms():
    return transforms.Compose([
        LetterboxTransform(IMAGE_SIZE),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), value=(0.485, 0.456, 0.406)),
    ])


def get_eval_transforms():
    return transforms.Compose([
        LetterboxTransform(IMAGE_SIZE),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def build_curated_dataset():
    realwaste_dir = DATA_DIR
    trashbox_dir = "data/trashbox_correct"
    ext_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"

    print("\n--- AUDITING TEST SETS TO STRICTLY ENFORCE ZERO DATA LEAKAGE ---")
    protected_hashes = set()
    protected_filenames = set()

    # 1. Protect External Test Set
    if os.path.exists(ext_test_dir):
        for root, dirs, files in os.walk(ext_test_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    protected_filenames.add(f.lower())
                    p = os.path.join(root, f)
                    h = compute_file_hash(p)
                    if h:
                        protected_hashes.add(h)
    print(f"External test hashes protected: {len(protected_hashes)} ({len(protected_filenames)} files)")

    # 2. Protect Official RealWaste Test Split
    rw_test_files = set()
    rw_val_indices = set()
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r") as f:
            rw_split = json.load(f)
        from src.dataset import load_dataset
        rw_ds = load_dataset(realwaste_dir)
        for idx in rw_split.get("test_indices", []):
            rw_path = rw_ds.samples[idx][0]
            rw_test_files.add(rw_path.lower())
            h = compute_file_hash(rw_path)
            if h:
                protected_hashes.add(h)
        rw_val_indices = set(rw_split.get("val_indices", []))
    print(f"RealWaste test samples protected: {len(rw_test_files)}")

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    # 3. Collect RealWaste Training samples (strictly excluding test split and val split)
    rw_train_samples = []
    rw_val_samples = []
    for idx, (path, label_idx) in enumerate(rw_ds.samples):
        if path.lower() in rw_test_files:
            continue
        if idx in rw_val_indices:
            rw_val_samples.append((path, label_idx))
        else:
            rw_train_samples.append((path, label_idx))
    print(f"RealWaste Clean Train samples: {len(rw_train_samples)} | Clean Val samples: {len(rw_val_samples)}")

    # 4. Curate Clean, Unleaked Real-World Camera Images from TrashBox
    rng = random.Random(42)

    def collect_clean_files(folder_path, max_count=None):
        out = []
        if os.path.isdir(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if f.lower() not in protected_filenames:
                            full_p = os.path.join(root, f)
                            h = compute_file_hash(full_p)
                            if h and h not in protected_hashes:
                                out.append(full_p)
        rng.shuffle(out)
        if max_count:
            out = out[:max_count]
        return out

    tb_samples = []

    # A. PLASTIC: Major focus to solve the plastic bottle -> glass confusion
    # Include all clean bottles (up to 420), plus containers, cups, bags
    pl_bottles = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic bottles"), 420)
    pl_cont = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic containers"), 250)
    pl_cups = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic cups"), 150)
    pl_bags = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic bags"), 100)
    plastic_add = pl_bottles + pl_cont + pl_cups + pl_bags
    for p in plastic_add:
        tb_samples.append((p, class_to_idx["Plastic"]))
    print(f"TrashBox Plastic added (clean): {len(plastic_add)} (bottles: {len(pl_bottles)}, cont: {len(pl_cont)}, cups: {len(pl_cups)}, bags: {len(pl_bags)})")

    # B. GLASS: 320 clean glass bottles and jars (balanced against plastic bottles)
    gl_add = collect_clean_files(os.path.join(trashbox_dir, "glass"), 320)
    for p in gl_add:
        tb_samples.append((p, class_to_idx["Glass"]))
    print(f"TrashBox Glass added (clean): {len(gl_add)}")

    # C. CARDBOARD: 400 clean corrugated shipping cartons and packaging boxes
    cb_add = collect_clean_files(os.path.join(trashbox_dir, "cardboard"), 400)
    for p in cb_add:
        tb_samples.append((p, class_to_idx["Cardboard"]))
    print(f"TrashBox Cardboard added (clean): {len(cb_add)}")

    # D. PAPER: 400 clean real-world camera images (plain paper + newspaper + paper cups, NO Tetra Pak cartons)
    p_plain = collect_clean_files(os.path.join(trashbox_dir, "paper", "paper"), 200)
    p_news = collect_clean_files(os.path.join(trashbox_dir, "paper", "news paper"), 100)
    p_cups = collect_clean_files(os.path.join(trashbox_dir, "paper", "paper cups"), 100)
    paper_add = p_plain + p_news + p_cups
    for p in paper_add:
        tb_samples.append((p, class_to_idx["Paper"]))
    print(f"TrashBox Paper added (clean): {len(paper_add)} (plain: {len(p_plain)}, news: {len(p_news)}, cups: {len(p_cups)})")

    # E. METAL: 320 clean cans & containers
    m_cans = collect_clean_files(os.path.join(trashbox_dir, "metal", "beverage cans"), 175)
    m_cont = collect_clean_files(os.path.join(trashbox_dir, "metal", "metal containers"), 100)
    m_spray = collect_clean_files(os.path.join(trashbox_dir, "metal", "spray cans"), 45)
    metal_add = m_cans + m_cont + m_spray
    for p in metal_add:
        tb_samples.append((p, class_to_idx["Metal"]))
    print(f"TrashBox Metal added (clean): {len(metal_add)}")

    # F. MISCELLANEOUS TRASH: 200 clean cigarette butts / litter items
    misc_add = collect_clean_files(os.path.join(trashbox_dir, "plastic", "cigarette butt"), 200)
    for p in misc_add:
        tb_samples.append((p, class_to_idx["Miscellaneous Trash"]))
    print(f"TrashBox Miscellaneous Trash added (clean): {len(misc_add)}")

    # Split TrashBox additions into 85% Train / 15% Val for real-world validation
    tb_class_groups = {}
    for item in tb_samples:
        tb_class_groups.setdefault(item[1], []).append(item)

    tb_train = []
    tb_val = []
    for c_idx, grp in tb_class_groups.items():
        rng.shuffle(grp)
        sp = int(0.85 * len(grp))
        tb_train.extend(grp[:sp])
        tb_val.extend(grp[sp:])

    print(f"TrashBox split — Train: {len(tb_train)} | Val: {len(tb_val)}")

    # Combined Training pool
    full_train_samples = rw_train_samples + tb_train
    rng.shuffle(full_train_samples)

    # Combined Validation pool (RealWaste val set + TrashBox val set)
    full_val_samples = rw_val_samples + tb_val
    rng.shuffle(full_val_samples)

    print(f"\nFinal Training Pool: {len(full_train_samples)} samples")
    print(f"Final Validation Pool: {len(full_val_samples)} samples (RW: {len(rw_val_samples)}, TB: {len(tb_val)})")

    return full_train_samples, full_val_samples, rw_val_samples


def evaluate_metrics(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    total = len(all_targets)
    val_loss = running_loss / total
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_by_class = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

    return val_loss, val_acc, val_macro_f1, rec_by_class


def main():
    print("=" * 75)
    print("EcoSort AI — V4 Real-World Production Training Pipeline")
    print("=" * 75)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")

    # Step 1: Build composite dataset
    train_samples, full_val_samples, rw_val_samples = build_curated_dataset()

    # Step 2: Compute balanced, smoothed class weights
    train_targets = [s[1] for s in train_samples]
    counts = Counter(train_targets)
    total_t = len(train_targets)
    print("\nClass Distribution in Combined Training Set:")
    raw_weights = []
    for i, name in enumerate(CLASS_NAMES):
        c_count = counts.get(i, 1)
        # Smoothed square-root inverse weight
        w = (total_t / (NUM_CLASSES * c_count)) ** 0.5
        raw_weights.append(w)
        print(f"  {name:20s}: {c_count:4d} images")

    # Normalize weights so mean is 1.0
    mean_w = sum(raw_weights) / len(raw_weights)
    norm_weights = [w / mean_w for w in raw_weights]
    class_weights = torch.FloatTensor(norm_weights).to(device)
    print(f"\nSmoothed Class Weights (mean=1.0):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {class_weights[i]:.3f}")

    train_loader = DataLoader(
        WasteDataset(train_samples, transform=get_train_transforms()),
        batch_size=48, shuffle=True, drop_last=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        WasteDataset(full_val_samples, transform=get_eval_transforms()),
        batch_size=48, shuffle=False, num_workers=0, pin_memory=False
    )
    rw_val_loader = DataLoader(
        WasteDataset(rw_val_samples, transform=get_eval_transforms()),
        batch_size=48, shuffle=False, num_workers=0, pin_memory=False
    )

    # Step 3: Load starting production model
    print(f"\nLoading starting checkpoint from {STARTING_MODEL_PATH}...")
    model = load_model(STARTING_MODEL_PATH, device)

    # Label-smoothed CrossEntropyLoss with balanced weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)

    # Initial validation evaluation
    init_loss, init_acc, init_f1, init_recs = evaluate_metrics(model, val_loader, criterion, device)
    _, rw_init_acc, rw_init_f1, _ = evaluate_metrics(model, rw_val_loader, criterion, device)
    print(f"\nInitial Full Val -> Loss: {init_loss:.4f} | Acc: {init_acc:.2%} | Macro F1: {init_f1:.4f}")
    print(f"Initial RW Val   -> Acc: {rw_init_acc:.2%} | Macro F1: {rw_init_f1:.4f}")
    print("Initial Per-Class Recall (Combined Val):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {init_recs[i]:.1%}")

    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": [],
        "rw_val_acc": [], "rw_val_macro_f1": [],
    }

    best_combined_score = (init_f1 * 0.6) + (rw_init_f1 * 0.4)
    best_epoch = 0

    # =========================================================================
    # STAGE 1: Classifier Head Alignment (Backbone frozen, 2 epochs)
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 1: Classifier Head Alignment (2 Epochs, Backbone Frozen)")
    print("=" * 75)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_stage1 = optim.AdamW(model.classifier.parameters(), lr=2.0e-4, weight_decay=1e-4)

    for ep in range(1, 3):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_stage1.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt_stage1.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += out.argmax(dim=1).eq(labels).sum().item()
            tr_total += labels.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        vl_loss, vl_acc, vl_f1, vl_recs = evaluate_metrics(model, val_loader, criterion, device)
        _, rw_acc, rw_f1, _ = evaluate_metrics(model, rw_val_loader, criterion, device)
        elapsed = time.time() - t0

        combined_score = (vl_f1 * 0.6) + (rw_f1 * 0.4)
        print(f"Stage 1 - Epoch {ep}/2 | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.1%} | FullVal Acc: {vl_acc:.1%} F1: {vl_f1:.4f} | RW Acc: {rw_acc:.1%} F1: {rw_f1:.4f} | {elapsed:.1f}s")
        print(f"    Plastic Rec: {vl_recs[CLASS_NAMES.index('Plastic')]:.1%} | Paper Rec: {vl_recs[CLASS_NAMES.index('Paper')]:.1%} | Glass Rec: {vl_recs[CLASS_NAMES.index('Glass')]:.1%}")

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)
        history["rw_val_acc"].append(rw_acc)
        history["rw_val_macro_f1"].append(rw_f1)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = ep
            torch.save(model.state_dict(), PRODUCTION_V4_MODEL_PATH)
            print(f"  --> Saved new best checkpoint at Epoch {ep} (Combined Score: {combined_score:.4f})")

    # =========================================================================
    # STAGE 2: Deep Fine-Tuning (Blocks 12-18 + Head, 5 Epochs)
    # =========================================================================
    print("\n" + "=" * 75)
    print("STAGE 2: Deep Material Fine-Tuning (Blocks 12-18 + Head, 5 Epochs)")
    print("=" * 75)

    for idx, block in enumerate(model.features):
        if idx >= 12:
            for p in block.parameters():
                p.requires_grad = True
        else:
            for p in block.parameters():
                p.requires_grad = False

    backbone_params = [p for idx, b in enumerate(model.features) if idx >= 12 for p in b.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    opt_stage2 = optim.AdamW([
        {"params": backbone_params, "lr": 1.2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 7.0e-5, "weight_decay": 1e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_stage2, T_max=5, eta_min=1e-6)

    for ep in range(3, 8):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_stage2.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt_stage2.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += out.argmax(dim=1).eq(labels).sum().item()
            tr_total += labels.size(0)

        scheduler.step()
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total
        vl_loss, vl_acc, vl_f1, vl_recs = evaluate_metrics(model, val_loader, criterion, device)
        _, rw_acc, rw_f1, _ = evaluate_metrics(model, rw_val_loader, criterion, device)
        elapsed = time.time() - t0

        combined_score = (vl_f1 * 0.6) + (rw_f1 * 0.4)
        print(f"Stage 2 - Epoch {ep}/7 | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.1%} | FullVal Acc: {vl_acc:.1%} F1: {vl_f1:.4f} | RW Acc: {rw_acc:.1%} F1: {rw_f1:.4f} | {elapsed:.1f}s")
        print(f"    Plastic Rec: {vl_recs[CLASS_NAMES.index('Plastic')]:.1%} | Paper Rec: {vl_recs[CLASS_NAMES.index('Paper')]:.1%} | Glass Rec: {vl_recs[CLASS_NAMES.index('Glass')]:.1%}")

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)
        history["rw_val_acc"].append(rw_acc)
        history["rw_val_macro_f1"].append(rw_f1)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = ep
            torch.save(model.state_dict(), PRODUCTION_V4_MODEL_PATH)
            print(f"  --> Saved new best checkpoint at Epoch {ep} (Combined Score: {combined_score:.4f})")

    # If no epoch exceeded initial, save model state
    if best_epoch == 0:
        torch.save(model.state_dict(), PRODUCTION_V4_MODEL_PATH)
        print(f"Saved initial state to {PRODUCTION_V4_MODEL_PATH}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} | Best Combined Score: {best_combined_score:.4f}")
    print(f"Checkpoint saved to: {PRODUCTION_V4_MODEL_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    main()
