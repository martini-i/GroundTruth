"""
audit_errors.py — Which images does the model get wrong, and how consistently?

Runs grouped k-fold cross-validation across several seeds. Every image is
validated once per seed, so each one accumulates a record of how often it was
misclassified when held out. Images wrong under every seed are not bad luck —
they are either mislabelled, genuinely ambiguous, or unlike anything else in
the set.

This exists to distinguish two very different situations:

  * persistent errors are borderline judgement calls -> the label is not fully
    determined by the photo, and accuracy is near a task-imposed ceiling
  * persistent errors are clean, obvious cases       -> the representation is
    the problem, and a different backbone is worth trying

Writes audit_errors.csv (per-image error rate, mean P(unstable), label) so the
worst offenders can be reviewed directly.

Usage:
    python audit_errors.py                    # 3 seeds x 5 folds
    python audit_errors.py --seeds 1 2 3 4 5
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import cross_validate as cv

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--out", default="audit_errors.csv")
args = parser.parse_args()


def val_probs(model, loader, device):
    """P(unstable) for each validation image, in loader order."""
    model.eval()
    out = []
    with torch.no_grad():
        for images, _ in loader:
            p = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            out += p.cpu().tolist()
    return out


def fit_fold(tr_idx, va_idx, paths, labels):
    """Same recipe as cross_validate.train_one_fold, but returns probabilities."""
    device = cv.device
    tr = cv.SlopeDataset([paths[i] for i in tr_idx], labels[tr_idx], cv.train_transform)
    va = cv.SlopeDataset([paths[i] for i in va_idx], labels[va_idx], cv.eval_transform)
    tr_loader = DataLoader(tr, batch_size=cv.BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=cv.BATCH_SIZE, shuffle=False, num_workers=0)

    counts = np.bincount(labels[tr_idx], minlength=2)
    weights = torch.tensor([len(tr_idx) / (2 * c) for c in counts],
                           dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model, head, unfreeze = cv.build_model()
    optimizer = optim.AdamW(head, lr=cv.LR, weight_decay=cv.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss, best_probs = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        if epoch == cv.UNFREEZE_EPOCH:
            for p in unfreeze:
                p.requires_grad = True
            optimizer = optim.AdamW(
                [{"params": unfreeze, "lr": cv.UNFREEZE_LR}, {"params": head, "lr": cv.LR}],
                weight_decay=cv.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - cv.UNFREEZE_EPOCH)
        cv.run_epoch(model, tr_loader, criterion, optimizer)
        vl, _, _, _ = cv.run_epoch(model, va_loader, criterion)
        scheduler.step()
        if vl < best_loss:
            best_loss = vl
            best_probs = val_probs(model, va_loader, device)
    return best_probs


def main():
    paths, labels, groups = cv.load_index()
    cv.args.epochs = args.epochs
    print(f"{len(paths)} images | seeds={args.seeds} folds={args.folds}\n", flush=True)

    wrong = defaultdict(int)
    seen = defaultdict(int)
    probs = defaultdict(list)

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(paths, labels, groups), 1):
            p_unstable = fit_fold(tr_idx, va_idx, paths, labels)
            for j, i in enumerate(va_idx):
                p = p_unstable[j]
                pred = 1 if p >= 0.5 else 0
                seen[i] += 1
                probs[i].append(p)
                if pred != labels[i]:
                    wrong[i] += 1
            print(f"  seed {seed} fold {fold} done", flush=True)

    rows = []
    for i, path in enumerate(paths):
        n = seen[i]
        rows.append({
            "filename": path.name,
            "true_label": cv.CLASS_NAMES[labels[i]],
            "times_validated": n,
            "times_wrong": wrong[i],
            "error_rate": round(wrong[i] / n, 3) if n else "",
            "mean_p_unstable": round(float(np.mean(probs[i])), 3) if probs[i] else "",
        })
    rows.sort(key=lambda r: (-r["error_rate"], r["filename"]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    always = [r for r in rows if r["error_rate"] == 1.0]
    mostly = [r for r in rows if 0.5 <= r["error_rate"] < 1.0]
    print(f"\n{'='*70}")
    print(f"wrong EVERY time ({len(always)}):")
    for r in always:
        print(f"  {r['filename']:34s} true={r['true_label']:8s} "
              f"mean_p_unstable={r['mean_p_unstable']}")
    print(f"\nwrong most of the time ({len(mostly)}):")
    for r in mostly:
        print(f"  {r['filename']:34s} true={r['true_label']:8s} "
              f"err={r['error_rate']} mean_p_unstable={r['mean_p_unstable']}")
    by_class = defaultdict(lambda: [0, 0])
    for r in rows:
        by_class[r["true_label"]][0] += r["times_wrong"]
        by_class[r["true_label"]][1] += r["times_validated"]
    print(f"\nerror rate by class:")
    for k, (w_, n_) in by_class.items():
        print(f"  {k:9s} {w_}/{n_} = {w_/n_:.1%}")
    print(f"\nfull table -> {args.out}")


if __name__ == "__main__":
    main()
