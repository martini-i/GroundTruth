"""
train_probe.py — Fit the deployed classifier: a calibrated linear probe on
frozen CLIP image features.

Replaces fine-tuning as the production model: grouped cross-validation put the
frozen CLIP probe well ahead of the fine-tuned ResNet18 baseline while training
three orders of magnitude fewer parameters. Current measured figures are in
metrics.json, written by frozen_features.py.

Two details that are easy to get wrong:

1. **No crop_sky here.** Images in slope_dataset/ are already sky-cropped on
   disk (crop_dataset_sky.py, and each image is cropped when integrated).
   Cropping again would fit the probe on doubly-cropped images while inference
   crops a raw upload exactly once — a train/inference mismatch.

2. **Probabilities are calibrated.** A plain logistic regression on separable
   CLIP features is wildly overconfident — most held-out predictions land
   above 95% or below 5%, and the most confident "stable" bin still contained a
   sizeable share of unstable images. Calibration is fitted on grouped held-out
   folds (never on the same data the base model saw) so the reported confidence
   tracks reality.

Writes slope_probe.joblib for model_utils.py to load.

Usage:
    python train_probe.py
"""

from pathlib import Path

import joblib
import numpy as np
import probe as P
from dataset import CLASS_NAMES, CLIP_ID, CLIP_TTA, clip_features, load_index

OUT = Path("slope_probe.joblib")
SEED = 42


def main():
    paths, labels, groups = load_index()
    print(f"{len(paths)} images | stable={int((labels==0).sum())} unstable={int((labels==1).sum())}")

    feats = clip_features(paths)
    print(f"features {feats.shape}")

    # One implementation of the fit, shared with the evaluator (probe.py), so
    # the measurement and the deployed model cannot disagree about it.
    calibrated = P.fit_probe(feats, labels, groups)

    p = calibrated.predict_proba(feats)[:, 1]
    joblib.dump({
        "calibrated": calibrated,
        "class_names": CLASS_NAMES,
        "clip_id": CLIP_ID,
        # Recorded, not assumed: model_utils reads this back rather than
        # keeping its own copy, so inference cannot apply a different
        # augmentation from the one the probe was fitted on.
        "tta": CLIP_TTA,
        "n_train": len(paths),
        "seed": SEED,
    }, OUT)

    extreme = ((p > 0.95) | (p < 0.05)).mean()
    print(f"\nfit on {len(paths)} images")
    print(f"  mean confidence in predicted class: {np.maximum(p, 1-p).mean():.1%}")
    print(f"  predictions above 95% / below 5%:   {extreme:.0%}")
    print(f"  (honest accuracy estimate comes from frozen_features.py, not from here)")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
