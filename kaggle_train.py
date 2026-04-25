"""
MediScan AI - Training Script
Run this on Kaggle Notebooks with GPU enabled.

STEPS:
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Use in Notebook" → New Notebook (enable GPU in Settings)
3. Paste this entire script and run all cells
4. After training, download: mediscan_best.pth from /kaggle/working/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/chest-xray-pneumonia/chest_xray"
SAVE_PATH  = "/kaggle/working/mediscan_best.pth"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 8
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES    = ["NORMAL", "PNEUMONIA"]

print(f"Using device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_tf)

# Weighted sampler to handle class imbalance
labels     = [s[1] for s in train_ds.samples]
counts     = np.bincount(labels)
weights    = 1.0 / counts[labels]
sampler    = WeightedRandomSampler(weights, len(weights))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=2)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Class map: {train_ds.class_to_idx}")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.densenet121(weights="IMAGENET1K_V1")
    # Freeze early layers, fine-tune last dense block
    for name, param in model.named_parameters():
        if "denseblock4" not in name and "norm5" not in name:
            param.requires_grad = False
    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model.to(DEVICE)

model = build_model()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# ── Loss & Optimizer ──────────────────────────────────────────────────────────
class_weights = torch.tensor([1.0, len(train_ds) / (2 * counts[1])], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out   = model(imgs)
        loss  = criterion(out, labels)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc

# ── Run Training ──────────────────────────────────────────────────────────────
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "val_auc":[]}
best_auc = 0.0

print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Val AUC':>9}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc           = train_epoch(model, train_dl)
    vl_loss, vl_acc, vl_auc  = eval_epoch(model, val_dl)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["val_auc"].append(vl_auc)

    if vl_auc > best_auc:
        best_auc = vl_auc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "auc": best_auc
        }, SAVE_PATH)
        print(f"  Epoch {epoch:>3} | {tr_loss:>11.4f} | {tr_acc:>9.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {vl_auc:>8.4f}  ✓ saved")
    else:
        print(f"  Epoch {epoch:>3} | {tr_loss:>11.4f} | {tr_acc:>9.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {vl_auc:>8.4f}")

# ── Final Test Evaluation ─────────────────────────────────────────────────────
ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
_, test_acc, test_auc = eval_epoch(model, test_dl)
print(f"\nTest Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        out = model(imgs.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# ── Training Curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs_range = range(1, EPOCHS + 1)

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"],   label="Val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train")
axes[1].plot(epochs_range, history["val_acc"],   label="Val")
axes[1].set_title("Accuracy"); axes[1].legend()

axes[2].plot(epochs_range, history["val_auc"], color="green")
axes[2].set_title("Validation AUC-ROC")

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=150)
plt.show()
print(f"\nBest AUC: {best_auc:.4f}")
print(f"Checkpoint saved at: {SAVE_PATH}")
print("\n✅ NEXT STEP: Download mediscan_best.pth from Output panel → upload to HuggingFace Hub")