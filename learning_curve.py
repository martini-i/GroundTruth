"""
learning_curve.py — Does this model need more DATA, or is data not the bottleneck?

Runs the same grouped k-fold cross-validation at several training-set sizes and
reports accuracy at each. Read the resulting curve like this:

  * still climbing steeply at 100%  -> more images is the highest-value work
  * clearly flattening at 100%      -> more images buys little; the limit is
                                       label quality, task ambiguity, or the
                                       modelling approach itself

Only the TRAINING portion of each fold is subsampled. The validation fold is
always kept whole, so every point on the curve is scored against the same amount
of held-out data and the numbers are comparable across fractions.

Subsampling is group-aware: it drops whole sites rather than individual photos,
so a site never ends up half-included. Class balance is preserved within each
fraction.

Usage:
    python learning_curve.py                          # 25/50/75/100%, 5 folds
    python learning_curve.py --fractions 0.5 1.0 --folds 5 --seed 42
"""

import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

import cross_validate as cv

parser = argparse.ArgumentParser()
parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()


def subsample(tr_idx, labels, groups, frac, rng):
    """Keep `frac` of the training fold, dropping whole groups and holding class balance."""
    if frac >= 1.0:
        return tr_idx

    # Bucket groups by the class they belong to so both classes shrink together.
    by_class = defaultdict(list)
    seen = {}
    for i in tr_idx:
        g = groups[i]
        if g not in seen:
            seen[g] = labels[i]
            by_class[labels[i]].append(g)

    keep_groups = set()
    for cls, gs in by_class.items():
        gs = sorted(gs)
        rng.shuffle(gs)
        n = max(1, round(len(gs) * frac))
        keep_groups.update(gs[:n])

    return np.array([i for i in tr_idx if groups[i] in keep_groups])


def main():
    paths, labels, groups = cv.load_index()
    print(f"{len(paths)} images | {len(set(groups))} groups | "
          f"stable={int((labels==0).sum())} unstable={int((labels==1).sum())}")
    print(f"fractions={args.fractions} folds={args.folds} seed={args.seed}\n", flush=True)

    curve = []
    for frac in args.fractions:
        # Re-seed per fraction so each point is reproducible on its own.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)

        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=args.seed)
        accs, sizes = [], []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(paths, labels, groups), 1):
            sub = subsample(tr_idx, labels, groups, frac, rng)
            assert not (set(groups[sub]) & set(groups[va_idx])), "group leak"
            cv.args.epochs = args.epochs
            _, (acc, preds, labs) = cv.train_one_fold(sub, va_idx, paths, labels, fold)
            cm = confusion_matrix(labs, preds, labels=[0, 1])
            accs.append(acc * 100)
            sizes.append(len(sub))
            print(f"  frac={frac:.2f} fold {fold}: train_n={len(sub)} acc={acc*100:.1f}",
                  flush=True)

        mean, sd = float(np.mean(accs)), float(np.std(accs))
        curve.append((frac, int(np.mean(sizes)), mean, sd))
        print(f"FRACTION {frac:.2f}: train_n~{int(np.mean(sizes))} "
              f"acc={mean:.1f} +/- {sd:.1f}\n", flush=True)

    print("=" * 58)
    print(f"{'frac':>6} {'train n':>8} {'accuracy':>18}")
    for frac, n, mean, sd in curve:
        bar = "#" * int(round((mean - 45) / 1.2)) if mean > 45 else ""
        print(f"{frac:>6.2f} {n:>8} {mean:>9.1f} +/- {sd:<4.1f} {bar}")
    print("=" * 58)
    if len(curve) >= 2:
        delta = curve[-1][2] - curve[-2][2]
        print(f"gain over the last step: {delta:+.1f} points "
              f"({curve[-2][1]} -> {curve[-1][1]} training images)")
        print("Compare that gain against the +/- spread before concluding anything.")


if __name__ == "__main__":
    main()
