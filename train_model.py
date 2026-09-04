import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== CONFIG =====
parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18",
                     help="Backbone architecture. resnet18 is the documented baseline.")
parser.add_argument("--seed", type=int, default=None,
                     help="Random seed. Without it, head init, shuffling and augmentation "
                          "all vary run to run, so two runs on identical data can differ "
                          "substantially — set it to compare runs meaningfully.")
parser.add_argument("--model-path", default="slope_model.pth",
                     help="Where to write the checkpoint. Override to avoid clobbering "
                          "the working model during experiments.")
parser.add_argument("--quiet", action="store_true", help="Suppress the per-epoch lines.")
args = parser.parse_args()

DATA_DIR = "slope_dataset"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
UNFREEZE_EPOCH = 10       # epoch at which we unfreeze the backbone for fine-tuning
UNFREEZE_LR = 1e-4        # lower LR for backbone layers after unfreezing
MODEL_PATH = args.model_path

if args.seed is not None:
    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== TRANSFORMS =====
# Aggressive augmentation helps a lot with small datasets.
# We simulate real-world variation: lighting, angle, partial occlusion.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),                          # random crop instead of center crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),                  # occasional grayscale for robustness
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # simulate partial occlusion
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== LOAD DATA =====
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)
# test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_transform)

# num_workers=0 is safest on Windows; increase if on Linux/Mac
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class_names = train_dataset.classes
print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
#  | Test: {len(test_dataset)} insert after
# ===== CLASS WEIGHTS (handles imbalanced datasets) =====
# If you have more stable than unstable images (or vice versa), this corrects for it.
class_counts = [0] * len(class_names)
for _, label in train_dataset.samples:
    class_counts[label] += 1
total = sum(class_counts)
class_weights = torch.tensor([total / (len(class_names) * c) for c in class_counts], dtype=torch.float).to(device)
print(f"Class counts: {dict(zip(class_names, class_counts))}")
print(f"Class weights: {dict(zip(class_names, class_weights.tolist()))}")

# ===== MODEL =====
# ResNet18 is the documented baseline for this project.
# EfficientNet-B0 is available via --arch for secondary comparison, not as the reported result.
DROPOUT_P = 0.3
WEIGHT_DECAY = 5e-4

if args.arch == "efficientnet_b0":
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(DROPOUT_P), nn.Linear(in_features, 2))
    backbone_params = [p for name, p in model.named_parameters() if "classifier" not in name]
    head_params     = list(model.classifier.parameters())
    # Only unfreeze the last feature block during fine-tuning, not the whole backbone —
    # limits capacity added back in, which matters a lot on a small dataset.
    unfreeze_params = list(model.features[-1].parameters())
    print("Using EfficientNet-B0")
else:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(DROPOUT_P), nn.Linear(in_features, 2))
    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    head_params     = list(model.fc.parameters())
    # Only unfreeze layer4 (the last residual block) during fine-tuning, not the whole backbone —
    # limits capacity added back in, which matters a lot on a small dataset.
    unfreeze_params = list(model.layer4.parameters())
    print("Using ResNet18")

# Freeze backbone initially — only train the new head
for p in backbone_params:
    p.requires_grad = False

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Only head params are active at first
optimizer = optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ===== HELPERS =====
def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

# ===== TRAIN LOOP =====
# Best checkpoint is selected by lowest val LOSS, not val accuracy — accuracy on a
# small val set jumps in coarse 1/N increments and is a noisier signal than loss.
best_weights = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_val_acc = 0.0
best_val_preds, best_val_labels = [], []
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    # Unfreeze only the last block at UNFREEZE_EPOCH for limited fine-tuning —
    # keeping the rest of the backbone frozen reduces capacity to memorize the small train set.
    if epoch == UNFREEZE_EPOCH:
        print(f"\n>>> Unfreezing last block at epoch {epoch} with LR={UNFREEZE_LR}")
        for p in unfreeze_params:
            p.requires_grad = True
        optimizer = optim.AdamW([
            {"params": unfreeze_params, "lr": UNFREEZE_LR},
            {"params": head_params,     "lr": LR}
        ], weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

    train_loss, train_acc, _, _ = run_epoch(train_loader, training=True)
    val_loss, val_acc, val_preds, val_labels = run_epoch(val_loader, training=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if not args.quiet:
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        best_val_preds, best_val_labels = val_preds, val_labels
        if not args.quiet:
            print(f"  New best val loss: {best_val_loss:.4f} (val acc: {best_val_acc:.4f})")

# ===== SAVE =====
model.load_state_dict(best_weights)
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "model_arch": model.__class__.__name__
}, MODEL_PATH)

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {MODEL_PATH}")
print("\nRun cross_validate.py for a trustworthy accuracy estimate — a single "
      "fixed split overstates it by roughly 8-10 points.")

# ===== CONFUSION MATRIX (VAL SET, BEST EPOCH) =====
cm = confusion_matrix(best_val_labels, best_val_preds)

print("\nConfusion Matrix (Validation Set, Best Epoch):")
print(cm)

# Machine-readable line so repeated runs can be compared without parsing the matrix.
# The difference between the two recalls is deliberately not printed: it is a
# difference of two noisy ratios and carries the noise of both, so it swings
# widely between runs that changed nothing meaningful. Report the recalls.
stable_rec = cm[0][0] / cm[0].sum() * 100
unstable_rec = cm[1][1] / cm[1].sum() * 100
print(f"RESULT seed={args.seed} val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f} "
      f"stable_recall={stable_rec:.1f} unstable_recall={unstable_rec:.1f}")

if not args.quiet:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.savefig("val_confusion_matrix.png", dpi=150)
    plt.show()
