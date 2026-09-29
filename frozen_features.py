"""
frozen_features.py — Is fine-tuning the problem, or is the representation?

Fine-tuning adapts millions of parameters to roughly a hundred training images.
This instead keeps a pretrained backbone frozen, turns each image into one
feature vector, and fits a small calibrated classifier on top. Features are
extracted once and cached, so evaluating a backbone takes seconds rather than
the tens of minutes a fine-tuning cross-validation run costs.

Backbones:
    resnet18   ImageNet supervised, 512-d   — same features train_model.py starts from
    dinov2     DINOv2 ViT-S/14, 384-d       — self-supervised, far broader pretraining
    clip       CLIP ViT-B/32, 512-d         — image-text pretraining

Uses the same grouped, stratified folds as cross_validate.py, so numbers are
directly comparable with the fine-tuned baseline it measures.

The deployed backbone's result is written to metrics.json, which the backend
serves and both clients display — the only place a performance figure is quoted.

Reading the result:
  frozen ~= or > fine-tuning        -> fine-tuning was overfitting; use a probe
  dinov2/clip clearly best          -> the ImageNet representation was the ceiling
  everything lands near baseline    -> the signal may not be separable at this size

Usage:
    python frozen_features.py                          # all backbones, 5 folds
    python frozen_features.py --backbones dinov2
    python frozen_features.py --no-cache               # force re-extraction
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image

import cross_validate as cv
import model_utils
from dataset import index_key

warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = Path(".feature_cache")

# The deployed bands, imported rather than restated: if the app changes where it
# flags, the measurement has to follow it or it stops describing the app.
FLAG_THRESHOLD = model_utils.UNSTABLE_THRESHOLD
HIGH_THRESHOLD = model_utils.UNSTABLE_HIGH_CONFIDENCE

# Publishing to metrics.json requires this exact seed set — see main().
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

parser = argparse.ArgumentParser()
parser.add_argument("--backbones", nargs="+", default=["resnet18", "dinov2", "clip"])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="fold seeds to average over; one seed is a lottery ticket")
parser.add_argument("--no-cache", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_resnet18(paths):
    from torchvision import models, transforms
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Identity()          # keep the 512-d pooled features
    m.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            out.append(m(x).squeeze(0).cpu().numpy())
            if i % 40 == 0:
                print(f"    {i}/{len(paths)}", flush=True)
    return np.stack(out)


def extract_hf(paths, model_id, pool):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    if pool == "clip":
        # AutoModel resolves CLIP to the vision tower, whose get_image_features
        # returns a model output rather than a tensor. Load CLIPModel directly.
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(model_id).eval().to(device)
    else:
        model = AutoModel.from_pretrained(model_id).eval().to(device)
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            img = Image.open(p).convert("RGB")
            inputs = proc(images=img, return_tensors="pt").to(device)
            if pool == "clip":
                # transformers 5.x returns a model output here, not a tensor, so
                # take the pooled vision output and apply CLIP's own projection
                # to land in the canonical 512-d image-embedding space.
                vis = model.vision_model(pixel_values=inputs["pixel_values"])
                feat = model.visual_projection(vis.pooler_output).squeeze(0)
            else:
                # DINOv2: the CLS token is the standard image-level descriptor.
                feat = model(**inputs).last_hidden_state[:, 0].squeeze(0)
            out.append(feat.cpu().numpy())
            if i % 40 == 0:
                print(f"    {i}/{len(paths)}", flush=True)
    return np.stack(out)


EXTRACTORS = {
    "resnet18": lambda p: extract_resnet18(p),
    "dinov2": lambda p: extract_hf(p, "facebook/dinov2-small", "cls"),
    "clip": lambda p: extract_hf(p, "openai/clip-vit-base-patch32", "clip"),
}


def get_features(name, paths):
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{name}_{len(paths)}_{index_key(paths)}.npy"
    if cache.exists() and not args.no_cache:
        feats = np.load(cache)
        print(f"  {name}: loaded cached features {feats.shape}")
        return feats
    print(f"  {name}: extracting features for {len(paths)} images...", flush=True)
    feats = EXTRACTORS[name](paths)
    np.save(cache, feats)
    print(f"  {name}: extracted {feats.shape} -> cached")
    return feats


def _inner_splits(y, g, max_splits=5):
    """
    Grouped calibration folds inside one training fold.

    Calibration has to be fitted on data the base model did not see, and the
    split has to respect site groups for the same reason the outer one does.
    The fold count is reduced if the minority class has too few groups to fill
    max_splits, which happens on small training portions.
    """
    per_class = min(len(set(g[y == c])) for c in (0, 1))
    n = max(2, min(max_splits, per_class))
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True,
                                     random_state=0).split(np.zeros(len(y)), y, g))


def evaluate(feats, labels, groups):
    """
    Repeated grouped CV of the *deployed* pipeline: every seed in args.seeds.

    Two things this deliberately does that a plain accuracy sweep does not.

    1. It calibrates inside the fold, exactly as train_probe.py does when
       fitting the shipped probe. Scoring a raw LogisticRegression measured a
       model nobody uses: uncalibrated CLIP-feature probes are wildly
       overconfident, so every probability-dependent number was wrong.

    2. It reports the deployed operating point. The app flags at
       UNSTABLE_THRESHOLD, not at 0.5, because a missed unstable slope costs
       more than a false alarm — so recall at 0.5 is not the recall anyone
       experiences. Accuracy at 0.5 is kept as well, since that is the figure
       comparable across backbones and against the fine-tuned baseline.

    A single seed is a lottery ticket: fold assignment alone moves the mean by
    several points on a dataset this size, so the default averages over ten.
    The +/- reported by main() is the spread of per-seed means, i.e. uncertainty
    on the estimate, not the wider within-seed fold spread.

    Returns a dict of metric name -> list of per-fold values, plus the per-seed
    accuracy means under "seed_means".
    """
    out = defaultdict(list)
    for seed in args.seeds:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=seed)
        seed_accs = []
        for tr, va in splitter.split(feats, labels, groups):
            assert not (set(groups[tr]) & set(groups[va])), "group leak"

            base = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0),
            )
            model = CalibratedClassifierCV(
                base, method="sigmoid",
                cv=_inner_splits(labels[tr], groups[tr]),
            )
            model.fit(feats[tr], labels[tr])

            y = labels[va]
            p = model.predict_proba(feats[va])[:, 1]

            # Argmax point: comparable across backbones and with the baseline.
            pred = (p >= 0.5).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            acc = (pred == y).mean() * 100
            out["acc"].append(acc); seed_accs.append(acc)
            out["stable_recall"].append(cm[0][0] / max(cm[0].sum(), 1) * 100)
            out["unstable_recall"].append(cm[1][1] / max(cm[1].sum(), 1) * 100)

            # Deployed point: what a user of the app actually gets.
            flagged = (p >= FLAG_THRESHOLD).astype(int)
            fcm = confusion_matrix(y, flagged, labels=[0, 1])
            tp, fp = fcm[1][1], fcm[0][1]
            out["flag_unstable_recall"].append(tp / max(fcm[1].sum(), 1) * 100)
            out["flag_stable_recall"].append(fcm[0][0] / max(fcm[0].sum(), 1) * 100)
            out["flag_precision"].append(tp / max(tp + fp, 1) * 100)
            out["borderline_rate"].append(
                ((p >= FLAG_THRESHOLD) & (p < HIGH_THRESHOLD)).mean() * 100
            )

            # Threshold-free ranking quality, and whether the calibrated
            # probabilities mean what they say.
            out["auc"].append(roc_auc_score(y, p) * 100 if len(set(y)) > 1 else np.nan)
            out["brier"].append(brier_score_loss(y, p))
        out["seed_means"].append(np.mean(seed_accs))
    return out


def main():
    paths, labels, groups = cv.load_index()
    print(f"{len(paths)} images | {len(set(groups))} groups | "
          f"folds={args.folds} seeds={len(args.seeds)}\n")

    results = {}
    for name in args.backbones:
        feats = get_features(name, paths)
        m = evaluate(feats, labels, groups)
        m["dim"] = feats.shape[1]
        results[name] = m

        seed_means = m["seed_means"]
        # +/- is the spread of per-seed means: uncertainty on the estimate. The
        # within-seed fold spread is larger and answers a different question.
        print(f"  {name}: acc={np.mean(seed_means):.1f} +/- {np.std(seed_means):.1f}  "
              f"(seeds {min(seed_means):.1f}-{max(seed_means):.1f})  "
              f"stable={np.mean(m['stable_recall']):.1f} "
              f"unstable={np.mean(m['unstable_recall']):.1f}  "
              f"auc={np.nanmean(m['auc']):.1f} brier={np.mean(m['brier']):.3f}")
        print(f"  {'':>{len(name)}}  at the deployed {FLAG_THRESHOLD:.2f} flag: "
              f"unstable recall={np.mean(m['flag_unstable_recall']):.1f} "
              f"precision={np.mean(m['flag_precision']):.1f} "
              f"stable recall={np.mean(m['flag_stable_recall']):.1f} "
              f"borderline={np.mean(m['borderline_rate']):.1f}%\n", flush=True)

    # Write the deployed backbone's result to metrics.json. Everything that
    # quotes a figure to a user reads that file, so it cannot go stale silently
    # the way a hardcoded string does.
    #
    # Only a full-strength run may write it. A quick exploratory run with a
    # couple of seeds would otherwise overwrite the public figure with a
    # lower-confidence estimate, silently and with no indication in the app.
    full_strength = args.seeds == DEFAULT_SEEDS and not args.no_cache
    clip = results.get("clip")
    if clip is not None and not full_strength:
        print()
        print(f"not writing metrics.json: needs the default {len(DEFAULT_SEEDS)} "
              f"seeds (got {len(args.seeds)}). Re-run without --seeds to publish.")
    if clip is not None and full_strength:
        import json, datetime
        avg = lambda k: round(float(np.nanmean(clip[k])), 1)
        Path("metrics.json").write_text(json.dumps({
            "backbone": "clip",
            # Argmax point: comparable with the fine-tuned baseline and across
            # backbones.
            "accuracy": round(float(np.mean(clip["seed_means"])), 1),
            "spread": round(float(np.std(clip["seed_means"])), 1),
            "stable_recall": avg("stable_recall"),
            "unstable_recall": avg("unstable_recall"),
            "auc": avg("auc"),
            "brier": round(float(np.mean(clip["brier"])), 3),
            # The operating point the app actually ships. Reported separately
            # because it is a different question from accuracy: it trades
            # stable recall away to catch more unstable slopes.
            "deployed": {
                "flag_threshold": FLAG_THRESHOLD,
                "high_threshold": HIGH_THRESHOLD,
                "unstable_recall": avg("flag_unstable_recall"),
                "unstable_precision": avg("flag_precision"),
                "stable_recall": avg("flag_stable_recall"),
                "borderline_rate": avg("borderline_rate"),
            },
            "n_images": len(paths),
            "n_groups": len(set(groups)),
            "folds": args.folds,
            "seeds": args.seeds,
            "measured": datetime.date.today().isoformat(),
        }, indent=2) + "\n", encoding="utf-8")
        print("\nwrote metrics.json")

    print("=" * 78)
    print(f"{'backbone':>10} {'dim':>6} {'accuracy':>16} {'stable':>9} {'unstable':>10} {'AUC':>8}")
    for name, m in results.items():
        sm = m["seed_means"]
        print(f"{name:>10} {m['dim']:>6} {np.mean(sm):>9.1f} +/- {np.std(sm):<4.1f} "
              f"{np.mean(m['stable_recall']):>8.1f}% {np.mean(m['unstable_recall']):>9.1f}% "
              f"{np.nanmean(m['auc']):>7.1f}")
    print("-" * 78)
    print("accuracy/recall above are at 0.5. The app flags at "
          f"{FLAG_THRESHOLD:.2f}; see the per-backbone lines for that point.")
    print("=" * 78)


if __name__ == "__main__":
    main()
