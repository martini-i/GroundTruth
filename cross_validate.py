"""
cross_validate.py — Grouped, stratified k-fold cross-validation for the slope classifier.

Why this exists: a single fixed train/val split of this dataset is too small to
trust. With ~17 stable validation images, one image moves stable recall by ~6
percentage points, and repeated runs on identical data have produced recall gaps
anywhere from 3.5 to 43.9 pp purely from random seed variation. K-fold validates
every image exactly once and reports a mean with a spread, so a result can be
distinguished from noise.

Grouping: folds are split by `site_id` from labels.csv, so near-duplicate photos
of the same location never land in different folds. Without that, the model would
be scored on an image near-identical to one it trained on, inflating the result.
Rows with no site_id are treated as their own group.

NOTE: the hyperparameters below intentionally mirror train_model.py. If you change
one there, change it here too, or the two stop being comparable.

Usage:
    python cross_validate.py                  # 5 folds, seed 42
    python cross_validate.py --folds 5 --seed 7
    python cross_validate.py --arch efficientnet_b0
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ===== CONFIG (mirrors train_model.py) =====
# load_index/CLASS_NAMES are re-exported so audit_errors.py and learning_curve.py
# can keep reaching them through this module.
from dataset import CLASS_NAMES, load_index  # noqa: F401
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
UNFREEZE_EPOCH = 10
UNFREEZE_LR = 1e-4
DROPOUT_P = 0.3
WEIGHT_DECAY = 5e-4

parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
parser.add_argument("--epochs", type=int, default=EPOCHS)

# Only read the command line when run directly. Imported (e.g. by
# learning_curve.py, which has its own flags), fall back to defaults — otherwise
# this parser sees the caller's arguments and aborts on ones it doesn't know.
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SlopeDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_model():
    if args.arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(DROPOUT_P), nn.Linear(in_f, 2))
        backbone = [p for n, p in model.named_parameters() if "classifier" not in n]
        head = list(model.classifier.parameters())
        unfreeze = list(model.features[-1].parameters())
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(nn.Dropout(DROPOUT_P), nn.Linear(model.fc.in_features, 2))
        backbone = [p for n, p in model.named_parameters() if "fc" not in n]
        head = list(model.fc.parameters())
        unfreeze = list(model.layer4.parameters())
    for p in backbone:
        p.requires_grad = False
    return model.to(device), head, unfreeze


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            preds = out.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all += preds.cpu().tolist()
            labels_all += labels.cpu().tolist()
    return total_loss / total, correct / total, preds_all, labels_all


def train_one_fold(tr_idx, va_idx, paths, labels, fold):
    tr = SlopeDataset([paths[i] for i in tr_idx], labels[tr_idx], train_transform)
    va = SlopeDataset([paths[i] for i in va_idx], labels[va_idx], eval_transform)
    tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    counts = Counter(labels[tr_idx].tolist())
    weights = torch.tensor(
        [len(tr_idx) / (2 * counts[i]) for i in range(2)], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model, head, unfreeze = build_model()
    optimizer = optim.AdamW(head, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss, best = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        if epoch == UNFREEZE_EPOCH:
            for p in unfreeze:
                p.requires_grad = True
            optimizer = optim.AdamW(
                [{"params": unfreeze, "lr": UNFREEZE_LR}, {"params": head, "lr": LR}],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - UNFREEZE_EPOCH)
        run_epoch(model, tr_loader, criterion, optimizer)
        vl, va_acc, preds, labs = run_epoch(model, va_loader, criterion)
        scheduler.step()
        if vl < best_loss:
            best_loss, best = vl, (va_acc, preds, labs)
        print(f"  fold {fold} epoch {epoch:02d}/{args.epochs} val_loss={vl:.4f}", flush=True)
    return best_loss, best


def main():
    paths, labels, groups = load_index()
    print(f"{len(paths)} images | {len(set(groups))} groups | "
          f"stable={int((labels == 0).sum())} unstable={int((labels == 1).sum())}")
    print(f"folds={args.folds} seed={args.seed} arch={args.arch} epochs={args.epochs}\n")

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(paths, labels, groups), 1):
        # Assert the grouping actually held — a shared group across folds would
        # silently inflate every number that follows.
        overlap = set(groups[tr_idx]) & set(groups[va_idx])
        assert not overlap, f"fold {fold} leaks groups: {overlap}"
        best_loss, (acc, preds, labs) = train_one_fold(tr_idx, va_idx, paths, labels, fold)
        cm = confusion_matrix(labs, preds, labels=[0, 1])
        s_rec = cm[0][0] / cm[0].sum() * 100 if cm[0].sum() else float("nan")
        u_rec = cm[1][1] / cm[1].sum() * 100 if cm[1].sum() else float("nan")
        results.append({"fold": fold, "n_val": len(va_idx), "loss": best_loss,
                        "acc": acc * 100, "s": s_rec, "u": u_rec})
        print(f"FOLD {fold}: n={len(va_idx)} acc={acc*100:.1f} "
              f"stable={s_rec:.1f} unstable={u_rec:.1f}\n", flush=True)

    print("=" * 66)
    print(f"{'fold':>4} {'n':>4} {'acc':>7} {'stable':>8} {'unstable':>9}")
    for r in results:
        print(f"{r['fold']:>4} {r['n_val']:>4} {r['acc']:>6.1f}% {r['s']:>7.1f}% "
              f"{r['u']:>8.1f}%")
    print("-" * 66)
    # The difference between the two recalls is deliberately not reported. It is
    # a difference of two noisy ratios, so it carries the noise of both: on this
    # dataset the seed alone moved it across 3.5-43.9 pp with the data unchanged.
    # Tracking it run to run reads as signal and is not. Report the recalls.
    for key, name in [("acc", "accuracy"), ("s", "stable recall"),
                      ("u", "unstable recall")]:
        vals = [r[key] for r in results]
        mean, sd = float(np.mean(vals)), float(np.std(vals))
        print(f"{name:>16}: {mean:5.1f} +/- {sd:4.1f}   (range {min(vals):.1f} - {max(vals):.1f})")
    print("=" * 66)
    print("Report the mean +/- spread, not a single fold.")


if __name__ == "__main__":
    main()
