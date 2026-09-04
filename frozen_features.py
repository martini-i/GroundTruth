"""
frozen_features.py — Is fine-tuning the problem, or is the representation?

Fine-tuning adapts ~8.4M parameters to ~135 training images. This instead keeps
a pretrained backbone frozen, turns each image into one feature vector, and fits
a small classifier on top (~1k parameters). Features are extracted once and
cached, so evaluating a backbone takes seconds rather than the ~45 minutes a
fine-tuning cross-validation run costs.

Backbones:
    resnet18   ImageNet supervised, 512-d   — same features train_model.py starts from
    dinov2     DINOv2 ViT-S/14, 384-d       — self-supervised, far broader pretraining
    clip       CLIP ViT-B/32, 512-d         — image-text pretraining

Uses the same grouped, stratified folds as cross_validate.py, so numbers are
directly comparable with the fine-tuning baseline in FINDINGS.md.

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
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from PIL import Image

import cross_validate as cv
from dataset import index_key

warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = Path(".feature_cache")

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


def evaluate(feats, labels, groups):
    """
    Repeated grouped CV: every seed in args.seeds, pooled.

    A single seed is a lottery ticket. Measured on this dataset, ten seeds span
    79.3-83.8 for the same model and data — 4.5 points purely from fold
    assignment. Reporting one seed is the same mistake as reporting one held-out
    split, which FINDINGS section 2 already caught one level down; the fix is the
    same, so the default here is ten seeds rather than one.

    Returns per-fold values pooled across seeds, plus the per-seed means, whose
    spread is the uncertainty on the *estimate* (the within-seed fold spread is
    not).
    """
    accs, s_recs, u_recs, seed_means = [], [], [], []
    for seed in args.seeds:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=seed)
        seed_accs = []
        for tr, va in splitter.split(feats, labels, groups):
            assert not (set(groups[tr]) & set(groups[va])), "group leak"
            scaler = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
            clf.fit(scaler.transform(feats[tr]), labels[tr])
            pred = clf.predict(scaler.transform(feats[va]))
            cm = confusion_matrix(labels[va], pred, labels=[0, 1])
            acc = (pred == labels[va]).mean() * 100
            accs.append(acc); seed_accs.append(acc)
            s_recs.append(cm[0][0] / cm[0].sum() * 100)
            u_recs.append(cm[1][1] / cm[1].sum() * 100)
        seed_means.append(np.mean(seed_accs))
    return accs, s_recs, u_recs, seed_means


def main():
    paths, labels, groups = cv.load_index()
    print(f"{len(paths)} images | {len(set(groups))} groups | "
          f"folds={args.folds} seeds={len(args.seeds)}\n")

    rows = []
    for name in args.backbones:
        feats = get_features(name, paths)
        accs, s_recs, u_recs, seed_means = evaluate(feats, labels, groups)
        # +/- is the spread of per-seed means: uncertainty on the estimate. The
        # within-seed fold spread is larger and answers a different question.
        rows.append((name, feats.shape[1], np.mean(seed_means), np.std(seed_means),
                     np.mean(s_recs), np.mean(u_recs)))
        print(f"  {name}: acc={np.mean(seed_means):.1f} +/- {np.std(seed_means):.1f}  "
              f"(seeds {min(seed_means):.1f}-{max(seed_means):.1f})  "
              f"stable={np.mean(s_recs):.1f} unstable={np.mean(u_recs):.1f}\n", flush=True)

    # Write the deployed backbone's result to metrics.json. Everything that
    # quotes an accuracy to a user reads that file, so the figure cannot go stale
    # silently the way a hardcoded string does — it read 82% +/- 7 in the app long
    # after the real number had moved twice.
    #
    # Only a full-strength run may write it. A quick exploratory run with a
    # couple of seeds would otherwise overwrite the public figure with a
    # lower-confidence estimate, silently and with no indication in the app.
    full_strength = args.seeds == DEFAULT_SEEDS and not args.no_cache
    clip_row = next((r for r in rows if r[0] == "clip"), None)
    if clip_row is not None and not full_strength:
        print()
        print(f"not writing metrics.json: needs the default {len(DEFAULT_SEEDS)} "
              f"seeds (got {len(args.seeds)}). Re-run without --seeds to publish.")
    if clip_row is not None and full_strength:
        import json, datetime
        name, dim, acc, sd, s_rec, u_rec = clip_row
        Path("metrics.json").write_text(json.dumps({
            "backbone": name,
            "accuracy": round(float(acc), 1),
            "spread": round(float(sd), 1),
            "stable_recall": round(float(s_rec), 1),
            "unstable_recall": round(float(u_rec), 1),
            "n_images": len(paths),
            "n_groups": len(set(groups)),
            "folds": args.folds,
            "seeds": args.seeds,
            "measured": datetime.date.today().isoformat(),
        }, indent=2) + "\n", encoding="utf-8")
        print("\nwrote metrics.json")

    print("=" * 74)
    print(f"{'backbone':>10} {'dim':>6} {'accuracy':>16} {'stable':>9} {'unstable':>10}")
    for name, dim, a, sd, s, u in rows:
        print(f"{name:>10} {dim:>6} {a:>9.1f} +/- {sd:<4.1f} {s:>8.1f}% {u:>9.1f}%")
    print("-" * 74)
    print(f"{'fine-tuned':>10} {'-':>6} {66.5:>9.1f} +/- {9.6:<4.1f} {64.3:>8.1f}% {68.3:>9.1f}%"
          "   <- baseline (FINDINGS.md §2)")
    print("=" * 74)


if __name__ == "__main__":
    main()
