"""
EcoSort AI — Candidate Model Training Pipeline (Fine-Tuning)
============================================================
Trains a fine-tuned MobileNetV2 on a composite dataset (RealWaste + curated real-world TrashBox)
with realistic augmentations and unfreezing of upper feature layers.
Zero data leakage is strictly enforced against both the RealWaste test set and the TrashBox test set.
"""

import os
import sys
import json
import time
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, ".")
from src.config import (
    CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    IMAGE_SIZE, ARTIFACTS_DIR, RANDOM_SEED,
)

CANDIDATE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ecosort_candidate_model.pth")
CANDIDATE_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "candidate_training_history.json")
SPLIT_PATH = os.path.join(ARTIFACTS_DIR, "split_indices.json")


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LetterboxTransform:
    """Pad image to square with neutral gray (128) then resize to (IMAGE_SIZE, IMAGE_SIZE)."""
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
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
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


def build_candidate_dataset():
    test_dir = r"C:\Users\Sri saran\Downloads\TrashBox-testandvalid-main\TrashBox-testandvalid-main\TrashBox_testandvalid_set\test"
    realwaste_dir = r"data/realwaste-main/RealWaste"
    trashbox_dir = r"data/trashbox_correct"

    # 1. Strictly preserve all test files from TrashBox test suite
    test_filenames = set()
    if os.path.exists(test_dir):
        for r, d, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_filenames.add(f.lower())
    print(f"Protected external test files count: {len(test_filenames)}")

    # 2. Strictly preserve RealWaste test split indices
    realwaste_test_files = set()
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r") as f:
            rw_split = json.load(f)
        from src.dataset import load_dataset
        rw_dataset = load_dataset(realwaste_dir)
        test_indices = set(rw_split.get("test_indices", []))
        for idx in test_indices:
            realwaste_test_files.add(rw_dataset.samples[idx][0].lower())
    print(f"Protected RealWaste test samples count: {len(realwaste_test_files)}")

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    samples = []

    # 3. Add RealWaste training & validation samples (excluding held-out test samples)
    rw_added = 0
    for c in CLASS_NAMES:
        p = os.path.join(realwaste_dir, c)
        if os.path.isdir(p):
            for f in os.listdir(p):
                full_path = os.path.join(p, f)
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    if full_path.lower() not in realwaste_test_files:
                        samples.append((full_path, class_to_idx[c]))
                        rw_added += 1
    print(f"Added {rw_added} RealWaste train/val samples (zero test leakage).")

    # 4. Curated TrashBox real-world additions
    rng = random.Random(42)

    def collect_files(folder, max_count=None):
        out = []
        if os.path.isdir(folder):
            for r, d, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        if f.lower() not in test_filenames:
                            out.append(os.path.join(r, f))
        rng.shuffle(out)
        if max_count:
            out = out[:max_count]
        return out

    # Cardboard: 350 real-world boxes/cartons
    cb_files = collect_files(os.path.join(trashbox_dir, "cardboard"), max_count=350)
    for p in cb_files:
        samples.append((p, class_to_idx["Cardboard"]))
    print(f"Added {len(cb_files)} Cardboard images from TrashBox.")

    # Glass: 350 real-world glass bottles & jars
    gl_files = collect_files(os.path.join(trashbox_dir, "glass"), max_count=350)
    for p in gl_files:
        samples.append((p, class_to_idx["Glass"]))
    print(f"Added {len(gl_files)} Glass images from TrashBox.")

    # Metal: 150 beverage cans, 100 metal containers, 100 spray cans
    m_cans = collect_files(os.path.join(trashbox_dir, "metal", "beverage cans"), max_count=150)
    m_cont = collect_files(os.path.join(trashbox_dir, "metal", "metal containers"), max_count=100)
    m_spray = collect_files(os.path.join(trashbox_dir, "metal", "spray cans"), max_count=100)
    metal_files = m_cans + m_cont + m_spray
    for p in metal_files:
        samples.append((p, class_to_idx["Metal"]))
    print(f"Added {len(metal_files)} Metal images (cans/containers/sprays) from TrashBox.")

    # Paper: 200 paper sheets, 150 newspapers (avoid tetra pak cartons to eliminate cardboard confusion)
    p_plain = collect_files(os.path.join(trashbox_dir, "paper", "paper"), max_count=200)
    p_news = collect_files(os.path.join(trashbox_dir, "paper", "news paper"), max_count=150)
    paper_files = p_plain + p_news
    for p in paper_files:
        samples.append((p, class_to_idx["Paper"]))
    print(f"Added {len(paper_files)} Paper images (plain/newspaper) from TrashBox.")

    # Plastic: 150 bottles, 150 containers, 100 cups, 100 bags
    pl_bottles = collect_files(os.path.join(trashbox_dir, "plastic", "plastic bottles"), max_count=150)
    pl_cont = collect_files(os.path.join(trashbox_dir, "plastic", "plastic containers"), max_count=150)
    pl_cups = collect_files(os.path.join(trashbox_dir, "plastic", "plastic cups"), max_count=100)
    pl_bags = collect_files(os.path.join(trashbox_dir, "plastic", "plastic bags"), max_count=100)
    plastic_files = pl_bottles + pl_cont + pl_cups + pl_bags
    for p in plastic_files:
        samples.append((p, class_to_idx["Plastic"]))
    print(f"Added {len(plastic_files)} Plastic images (bottles/containers/cups/bags) from TrashBox.")

    # Miscellaneous Trash: 150 cigarette butts / miscellaneous litter
    misc_files = collect_files(os.path.join(trashbox_dir, "plastic", "cigarette butt"), max_count=150)
    for p in misc_files:
        samples.append((p, class_to_idx["Miscellaneous Trash"]))
    print(f"Added {len(misc_files)} Miscellaneous Trash images from TrashBox.")

    rng.shuffle(samples)
    print(f"Total candidate training/val pool: {len(samples)} samples.")
    return samples


def create_candidate_model(pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Strategy: Freeze blocks 0 to 13 (feature primitives)
    # Unfreeze blocks 14 to 18 for fine-tuning high-level material & container representations
    for idx, block in enumerate(model.features):
        if idx < 14:
            for p in block.parameters():
                p.requires_grad = False
        else:
            for p in block.parameters():
                p.requires_grad = True

    # Custom classifier head with regularizing dropout and BatchNorm
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(dataloader)

    for b_idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if (b_idx + 1) % 35 == 0 or (b_idx + 1) == n_batches:
            print(f"    [Train Batch {b_idx + 1:3d}/{n_batches:3d}] Loss: {running_loss / total:.4f} Acc: {correct / total:.1%}", flush=True)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    print("=" * 65)
    print("EcoSort AI — Enhanced Candidate Model Training Pipeline")
    print("=" * 65)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build balanced composite dataset
    print("\n[Step 1/5] Building balanced composite dataset...")
    samples = build_candidate_dataset()

    # Stratified 85% train, 15% val split
    rng = random.Random(42)
    class_groups = {}
    for item in samples:
        class_groups.setdefault(item[1], []).append(item)

    train_samples = []
    val_samples = []
    for c_idx, grp in class_groups.items():
        rng.shuffle(grp)
        split_pt = int(0.85 * len(grp))
        train_samples.extend(grp[:split_pt])
        val_samples.extend(grp[split_pt:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # Compute class weights
    train_targets = [s[1] for s in train_samples]
    counts = Counter(train_targets)
    total_t = len(train_targets)
    class_weights = torch.FloatTensor([
        total_t / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(device)
    print("Class sample counts in training split:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {counts.get(i, 0):4d} (weight: {class_weights[i]:.2f})")

    # DataLoaders
    train_loader = DataLoader(
        WasteImageDataset(train_samples, transform=get_train_transforms()),
        batch_size=32, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        WasteImageDataset(val_samples, transform=get_eval_transforms()),
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )

    # Build model
    print("\n[Step 2/5] Constructing fine-tuned MobileNetV2...")
    model = create_candidate_model(pretrained=True).to(device)

    # Differential learning rates:
    # Upper backbone: 4e-5, Classifier head: 4e-4
    backbone_params = [p for idx, b in enumerate(model.features) if idx >= 14 for p in b.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 4e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 4e-4, "weight_decay": 1e-4},
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    epochs = 6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\n[Step 3/5] Fine-tuning candidate model for {epochs} epochs...", flush=True)
    print("-" * 65)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%} | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CANDIDATE_MODEL_PATH)
            print(f"  --> Saved new best candidate model (Val Acc: {val_acc:.1%}) to {CANDIDATE_MODEL_PATH}")

    # Save training history
    with open(CANDIDATE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Step 4/5] Training finished. Best Val Acc: {best_val_acc:.1%}")
    print(f"Candidate model saved at: {CANDIDATE_MODEL_PATH}")


if __name__ == "__main__":
    main()
