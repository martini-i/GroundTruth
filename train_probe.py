"""
train_probe.py — Fit the deployed classifier: a calibrated linear probe on
frozen CLIP image features.

Replaces fine-tuning as the production model. Grouped 5-fold cross-validation
put the frozen CLIP probe at 81.0% +/- 1.1 (10 seeds) vs 66.5% +/- 9.6 for the fine-tuned
ResNet18 (FINDINGS.md section 5), training 513 parameters instead of ~8.4M.

Two details that are easy to get wrong:

1. **No crop_sky here.** Images in slope_dataset/ are already sky-cropped on
   disk (crop_dataset_sky.py, and each image is cropped when integrated).
   Cropping again would fit the probe on doubly-cropped images while inference
   crops a raw upload exactly once — a train/inference mismatch.

2. **Probabilities are calibrated.** A plain logistic regression on separable
   CLIP features is wildly overconfident: 68% of held-out predictions land
   above 95% or below 5%, and the most confident "stable" bin was actually
   18.5% unstable. Calibration is fitted on grouped held-out folds (never on
   the same data the base model saw) so the reported confidence tracks reality.

Writes slope_probe.joblib for model_utils.py to load.

Usage:
    python train_probe.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import CLASS_NAMES, CLIP_ID, clip_features, load_index

OUT = Path("slope_probe.joblib")
SEED = 42


def main():
    paths, labels, groups = load_index()
    print(f"{len(paths)} images | stable={int((labels==0).sum())} unstable={int((labels==1).sum())}")

    feats = clip_features(paths)
    print(f"features {feats.shape}")

    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0),
    )

    # Calibrate on grouped held-out folds so same-site photos never sit on both
    # sides of a calibration split.
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                       random_state=SEED).split(feats, labels, groups))
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=splits)
    calibrated.fit(feats, labels)

    # A separate uncalibrated fit on all data supplies the linear direction used
    # for the heatmap. Calibration is monotonic, so it cannot change which
    # regions push a prediction toward unstable — only the reported number.
    scaler = StandardScaler().fit(feats)
    linear = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    linear.fit(scaler.transform(feats), labels)

    p = calibrated.predict_proba(feats)[:, 1]
    joblib.dump({
        "calibrated": calibrated,
        "scaler": scaler,
        "clf": linear,
        "class_names": CLASS_NAMES,
        "clip_id": CLIP_ID,
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
