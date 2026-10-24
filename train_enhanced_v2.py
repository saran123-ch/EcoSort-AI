"""
EcoSort AI — Advanced Transfer Learning & Fine-Tuning Pipeline (V2)
===================================================================
1. Starts from the best production checkpoint (ecosort_model.pth).
2. Curates a balanced composite dataset with RealWaste + clean, unleaked TrashBox real-world images.
3. Strictly enforces zero test data leakage against both RealWaste test split and TrashBox test set.
4. Uses letterbox aspect-ratio preserving augmentations.
5. Implements staged transfer learning with differential learning rates, AdamW, and CosineAnnealing.
6. Evaluates per-epoch validation loss, accuracy, and Macro F1, saving the best checkpoint.
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
from sklearn.metrics import f1_score
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, ".")
from src.config import (
    CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    IMAGE_SIZE, ARTIFACTS_DIR, RANDOM_SEED, DATA_DIR,
)
from src.model import load_model

STARTING_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_model.pth")
ENHANCED_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_model_v2_enhanced.pth")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "enhanced_v2_training_history.json")
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


class WasteImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
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
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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


def build_composite_dataset():
    realwaste_dir = DATA_DIR
    trashbox_dir = "data/trashbox_correct"
    ext_test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"

    print("\n--- Auditing Test Sets to Enforce Zero Data Leakage ---")
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
    print(f"External test hashes protected: {len(protected_hashes)}")

    # 2. Protect Official RealWaste Test Split
    rw_test_files = set()
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
    print(f"RealWaste test samples protected: {len(rw_test_files)}")

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    samples = []

    # 3. Add RealWaste Training & Validation Data (Cleaned of test images)
    rw_added = 0
    for c in CLASS_NAMES:
        c_dir = os.path.join(realwaste_dir, c)
        if os.path.isdir(c_dir):
            for f in os.listdir(c_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_p = os.path.join(c_dir, f)
                    if full_p.lower() not in rw_test_files:
                        samples.append((full_p, class_to_idx[c]))
                        rw_added += 1
    print(f"RealWaste train/val samples added: {rw_added}")

    # 4. Curate Clean, Unleaked Real-World TrashBox Images
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

    # Add Plastic (Bottles, containers, cups, bags - 550 total)
    pl_bottles = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic bottles"), 200)
    pl_cont = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic containers"), 150)
    pl_cups = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic cups"), 100)
    pl_bags = collect_clean_files(os.path.join(trashbox_dir, "plastic", "plastic bags"), 100)
    plastic_add = pl_bottles + pl_cont + pl_cups + pl_bags
    for p in plastic_add:
        samples.append((p, class_to_idx["Plastic"]))
    print(f"TrashBox Plastic added (clean): {len(plastic_add)}")

    # Add Cardboard (350 clean boxes)
    cb_add = collect_clean_files(os.path.join(trashbox_dir, "cardboard"), 350)
    for p in cb_add:
        samples.append((p, class_to_idx["Cardboard"]))
    print(f"TrashBox Cardboard added (clean): {len(cb_add)}")

    # Add Glass (350 clean bottles/jars)
    gl_add = collect_clean_files(os.path.join(trashbox_dir, "glass"), 350)
    for p in gl_add:
        samples.append((p, class_to_idx["Glass"]))
    print(f"TrashBox Glass added (clean): {len(gl_add)}")

    # Add Metal (250 clean cans & containers)
    m_cans = collect_clean_files(os.path.join(trashbox_dir, "metal", "beverage cans"), 150)
    m_cont = collect_clean_files(os.path.join(trashbox_dir, "metal", "metal containers"), 100)
    metal_add = m_cans + m_cont
    for p in metal_add:
        samples.append((p, class_to_idx["Metal"]))
    print(f"TrashBox Metal added (clean): {len(metal_add)}")

    # Add Paper (300 clean sheets & newspapers; NO Tetra Paks to eliminate carton confusion)
    p_plain = collect_clean_files(os.path.join(trashbox_dir, "paper", "paper"), 150)
    p_news = collect_clean_files(os.path.join(trashbox_dir, "paper", "news paper"), 150)
    paper_add = p_plain + p_news
    for p in paper_add:
        samples.append((p, class_to_idx["Paper"]))
    print(f"TrashBox Paper added (clean): {len(paper_add)}")

    rng.shuffle(samples)
    print(f"\nTotal composite training/validation pool: {len(samples)} samples.")
    return samples


def evaluate_val_metrics(model, dataloader, criterion, device):
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
    val_acc = sum(1 for p, t in zip(all_preds, all_targets) if p == t) / total
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, val_macro_f1


def main():
    print("=" * 70)
    print("EcoSort AI — V2 Advanced Fine-Tuning Pipeline")
    print("=" * 70)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Build composite dataset
    samples = build_composite_dataset()

    # Stratified 85% Train / 15% Val split
    rng = random.Random(42)
    class_groups = {}
    for item in samples:
        class_groups.setdefault(item[1], []).append(item)

    train_samples = []
    val_samples = []
    for c_idx, grp in class_groups.items():
        rng.shuffle(grp)
        sp = int(0.85 * len(grp))
        train_samples.extend(grp[:sp])
        val_samples.extend(grp[sp:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    print(f"Split sizes — Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Class sample counts and class-weight balancing
    train_targets = [s[1] for s in train_samples]
    counts = Counter(train_targets)
    total_t = len(train_targets)
    class_weights = torch.FloatTensor([
        total_t / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(device)
    print("\nClass Distribution in Train Split:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {counts.get(i, 0):4d} images (weight: {class_weights[i]:.2f})")

    train_loader = DataLoader(
        WasteImageDataset(train_samples, transform=get_train_transforms()),
        batch_size=32, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        WasteImageDataset(val_samples, transform=get_eval_transforms()),
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )

    # Step 2: Load starting production model
    print(f"\nLoading starting checkpoint from {STARTING_MODEL_PATH}...")
    model = load_model(STARTING_MODEL_PATH, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initial evaluation before fine-tuning
    init_loss, init_acc, init_f1 = evaluate_val_metrics(model, val_loader, criterion, device)
    print(f"Initial Validation Performance -> Loss: {init_loss:.4f} | Acc: {init_acc:.2%} | Macro F1: {init_f1:.4f}")

    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": []
    }

    best_val_f1 = init_f1
    best_epoch = 0

    # =========================================================================
    # STAGE 1: Head Adaptation (Freeze backbone, train classifier head)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Classifier Head Adaptation (2 Epochs, Backbone Frozen)")
    print("=" * 70)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_stage1 = optim.AdamW(model.classifier.parameters(), lr=2.5e-4, weight_decay=1e-4)

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
        vl_loss, vl_acc, vl_f1 = evaluate_val_metrics(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Stage 1 - Epoch {ep}/2 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.1%} F1: {vl_f1:.4f} | Time: {elapsed:.1f}s")
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = ep
            torch.save(model.state_dict(), ENHANCED_MODEL_PATH)
            print(f"  --> Saved new best checkpoint (Val F1: {vl_f1:.4f})")

    # =========================================================================
    # STAGE 2: Deep Fine-Tuning (Unfreeze upper backbone blocks 15-18)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Deep Fine-Tuning (Upper Backbone Blocks 15-18 + Head, 4 Epochs)")
    print("=" * 70)

    for idx, block in enumerate(model.features):
        if idx >= 15:
            for p in block.parameters():
                p.requires_grad = True
        else:
            for p in block.parameters():
                p.requires_grad = False

    backbone_params = [p for idx, b in enumerate(model.features) if idx >= 15 for p in b.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    opt_stage2 = optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 8.0e-5, "weight_decay": 1e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_stage2, T_max=4)

    for ep in range(3, 7):
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
        vl_loss, vl_acc, vl_f1 = evaluate_val_metrics(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Stage 2 - Epoch {ep}/6 | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.1%} F1: {vl_f1:.4f} | Time: {elapsed:.1f}s")
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = ep
            torch.save(model.state_dict(), ENHANCED_MODEL_PATH)
            print(f"  --> Saved new best checkpoint (Val F1: {vl_f1:.4f})")

    # If no improvement over initial, save current model state
    if best_epoch == 0:
        torch.save(model.state_dict(), ENHANCED_MODEL_PATH)
        print(f"Final state saved to {ENHANCED_MODEL_PATH}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE. Best Epoch: {best_epoch} | Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {ENHANCED_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
