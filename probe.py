"""
probe.py — the probe itself, and the rules for fitting and scoring it.

Everything that fits a classifier on frozen features goes through here:
train_probe.py, frozen_features.py, audit_errors.py and experiments.py. Before
this module they each carried their own copy of the calibration splits, the
pipeline and the metrics — five implementations of `inner_splits` alone — which
is the same hazard dataset.py was written to remove one level down. A probe
fitted differently in the evaluator than in the deployed script reports an
accuracy the app cannot deliver, and nothing errors.

The hyperparameters live here too, so the measurement and the deployed model
cannot disagree about them.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inverse regularisation strength. NOT the sklearn default of 1.0: with ~768
# features and fewer than 200 images the default is badly under-regularised.
# Nested selection inside the training folds picked 0.01 in 13 of 15 folds, and
# at ten seeds it gains ~0.7 accuracy and ~0.9 AUC while narrowing the seed
# spread. Re-check with `python experiments.py tune` after any change to the
# backbone or a large change in dataset size.
PROBE_C = 0.01

# sigmoid over isotonic deliberately. Isotonic scored marginally higher overall
# but cost ~3 points of unstable recall at the deployed threshold, which is the
# error this project treats as expensive; it also degenerates toward step
# functions at this sample size.
CALIBRATION = "sigmoid"

MAX_CALIBRATION_FOLDS = 5


def inner_splits(y, groups, max_splits=MAX_CALIBRATION_FOLDS):
    """
    Grouped calibration folds inside one training fold.

    Calibration must be fitted on data the base estimator did not see, and the
    split has to respect site groups for the same reason the outer one does:
    two photographs of one cutting are not independent samples. The fold count
    drops when the minority class has too few groups to fill max_splits, which
    happens on small training portions.
    """
    per_class = min(len(set(groups[y == c])) for c in (0, 1))
    n = max(2, min(max_splits, per_class))
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0)
                .split(np.zeros(len(y)), y, groups))


def fit_probe(X, y, groups, C=PROBE_C, method=CALIBRATION):
    """Standardise, fit balanced logistic regression, calibrate on grouped folds."""
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=C),
    )
    model = CalibratedClassifierCV(base, method=method, cv=inner_splits(y, groups))
    model.fit(X, y)
    return model


def score(y_true, p_unstable, flag, high):
    """
    Every metric the project reports, from one set of predictions.

    Two operating points, deliberately. Accuracy at 0.5 is what compares across
    backbones and against the fine-tuned baseline; the flag threshold is what a
    user actually experiences, and the two are not interchangeable. AUC is
    threshold-free, and Brier asks whether the calibrated probabilities mean
    what they say.
    """
    y = np.asarray(y_true)
    p = np.asarray(p_unstable)

    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    flagged = (p >= flag).astype(int)
    fcm = confusion_matrix(y, flagged, labels=[0, 1])
    tp, fp = fcm[1][1], fcm[0][1]

    return {
        "acc": (pred == y).mean() * 100,
        "stable_recall": cm[0][0] / max(cm[0].sum(), 1) * 100,
        "unstable_recall": cm[1][1] / max(cm[1].sum(), 1) * 100,
        "auc": roc_auc_score(y, p) * 100 if len(set(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "flag_unstable_recall": tp / max(fcm[1].sum(), 1) * 100,
        "flag_precision": tp / max(tp + fp, 1) * 100,
        "flag_stable_recall": fcm[0][0] / max(fcm[0].sum(), 1) * 100,
        "borderline_rate": ((p >= flag) & (p < high)).mean() * 100,
    }


def grouped_cv(X, y, groups, seeds, folds=5, predict=None):
    """
    Repeated grouped k-fold, yielding (train_idx, val_idx) and asserting no leak.

    A single seed is a lottery ticket: fold assignment alone moves the mean by
    several points on a dataset this size, so callers pass several.
    """
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, va in splitter.split(X, y, groups):
            assert not (set(groups[tr]) & set(groups[va])), "group leak"
            yield seed, tr, va
