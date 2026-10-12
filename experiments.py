"""
experiments.py — the diagnostic experiments, in one place.

Three questions that are not part of the headline measurement but decide what
to believe about it. They share the dataset loader, the probe and the metrics
(probe.py), so they cannot drift from each other or from the deployed model.

    shortcut   How much of the accuracy is scene type rather than stability?
    tiles      Does local evidence beat whole-image scene type?
    tune       Are the probe's hyperparameters and threshold the right ones?

Usage:
    python experiments.py shortcut
    python experiments.py tiles --backbone clip-large
    python experiments.py tune --seeds 1 2 3 4 5
"""

import argparse
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

import model_utils as mu
import probe as P
from dataset import CACHE_DIR, CLIP_ID, clip_features, index_key, load_index

FLAG, HIGH = mu.UNSTABLE_THRESHOLD, mu.UNSTABLE_HIGH_CONFIDENCE

# Filename prefixes that denote a built structure. Everything else is natural
# ground. Deliberately coarse: the question is whether a crude scene-type split
# already explains the score, so a crude split is the right instrument.
#
# Scene type comes from the filename, NOT from primary_indicator, because that
# field is a deterministic function of the label and so cannot straddle the
# stable/unstable boundary by construction. The prefixes can.
ENGINEERED_PREFIXES = {"engineered", "cut", "slope"}


def scene_of(path):
    parts = path.name.split("_")
    return "engineered" if len(parts) > 1 and parts[1] in ENGINEERED_PREFIXES else "natural"


def load_cached(stem, paths):
    f = CACHE_DIR / f"{stem}_{len(paths)}_{index_key(paths)}.npy"
    if not f.exists():
        raise SystemExit(f"no cached features at {f}\n"
                         f"  run: python frozen_features.py --backbones <name>")
    return np.load(f)


# ===================================================================== shortcut

def cmd_shortcut(args):
    """How much of the accuracy is obtainable from scene type alone?"""
    paths, labels, groups = load_index()
    scenes = np.array([scene_of(p) for p in paths])
    feats = clip_features(paths)
    print(f"{len(paths)} images | features {feats.shape}\n")

    print("=" * 72)
    print("1. SCENE-TYPE BASELINE  (predict the majority label per scene type)")
    print("-" * 72)
    print(f"{'':>12} {'stable':>8} {'unstable':>9} {'majority':>10} {'correct':>9}")
    correct = 0
    for s in ("engineered", "natural"):
        m = scenes == s
        ns, nu = int((labels[m] == 0).sum()), int((labels[m] == 1).sum())
        correct += max(ns, nu)
        print(f"{s:>12} {ns:>8} {nu:>9} {'stable' if ns >= nu else 'unstable':>10} {max(ns, nu):>9}")
    base_acc = correct / len(paths) * 100
    print(f"\n  scene-type-only accuracy: {base_acc:.1f}%   ({correct}/{len(paths)})")
    print("  Obtainable with no knowledge of slope stability whatsoever.")

    cell_hits, cell_n, accs = Counter(), Counter(), []
    for _, tr, va in P.grouped_cv(feats, labels, groups, args.seeds, args.folds):
        m = P.fit_probe(feats[tr], labels[tr], groups[tr])
        pred = (m.predict_proba(feats[va])[:, 1] >= 0.5).astype(int)
        accs.append((pred == labels[va]).mean() * 100)
        for i, idx in enumerate(va):
            cell = (scenes[idx], "stable" if labels[idx] == 0 else "unstable")
            cell_n[cell] += 1
            cell_hits[cell] += int(pred[i] == labels[idx])

    print()
    print("=" * 72)
    print("2. PER-CELL ACCURACY  (the off-diagonal cells are the real test)")
    print("-" * 72)
    print(f"{'scene':>12} {'label':>10} {'n':>6} {'accuracy':>10}   shortcut agrees?")
    for scene in ("engineered", "natural"):
        for lab in ("stable", "unstable"):
            c = (scene, lab)
            if not cell_n[c]:
                continue
            says = "stable" if scene == "engineered" else "unstable"
            flag = "yes" if says == lab else "NO  <-- tests the hypothesis"
            print(f"{scene:>12} {lab:>10} {cell_n[c]:>6} "
                  f"{cell_hits[c] / cell_n[c] * 100:>9.1f}%   {flag}")
    print(f"\n  overall {np.mean(accs):.1f}%  vs scene-type baseline {base_acc:.1f}%"
          f"   margin {np.mean(accs) - base_acc:+.1f}")

    sc_y = (scenes == "engineered").astype(int)
    sc = [((P.fit_probe(feats[tr], sc_y[tr], groups[tr])
            .predict_proba(feats[va])[:, 1] >= 0.5).astype(int) == sc_y[va]).mean() * 100
          for _, tr, va in P.grouped_cv(feats, sc_y, groups, args.seeds, args.folds)]
    print()
    print("=" * 72)
    print(f"3. SCENE TYPE ITSELF IS {np.mean(sc):.1f}% +/- {np.std(sc):.1f} DECODABLE")
    print("   The higher this is, the more freely available the shortcut is.")
    print("=" * 72)


# ======================================================================== tiles

def tile_boxes(w, h, n):
    """n x n windows, each half the image on a side, so neighbours overlap 50%."""
    tw, th = w // 2, h // 2
    xs = [round(i * (w - tw) / max(n - 1, 1)) for i in range(n)]
    ys = [round(i * (h - th) / max(n - 1, 1)) for i in range(n)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def extract_tiles(paths, backbone, grid):
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"tiles_{backbone}_g{grid}_{len(paths)}_{index_key(paths)}.npy"
    if cache.exists():
        out = np.load(cache)
        print(f"  cached tile features {out.shape}")
        return out
    import torch
    from transformers import AutoImageProcessor, CLIPModel
    mid = {"clip": "openai/clip-vit-base-patch32",
           "clip-large": "openai/clip-vit-large-patch14"}[backbone]
    proc, model = AutoImageProcessor.from_pretrained(mid), CLIPModel.from_pretrained(mid).eval()
    print(f"  extracting {grid * grid} tiles x {len(paths)} images...", flush=True)
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            img = Image.open(p).convert("RGB")
            crops = [img.crop(b) for b in tile_boxes(img.width, img.height, grid)]
            inputs = proc(images=crops, return_tensors="pt")
            vis = model.vision_model(pixel_values=inputs["pixel_values"])
            out.append(model.visual_projection(vis.pooler_output).numpy())
            if i % 25 == 0:
                print(f"    {i}/{len(paths)}", flush=True)
    out = np.stack(out)
    np.save(cache, out)
    return out


def cmd_tiles(args):
    """
    Does building the descriptor from local tiles beat the whole frame?

    A tile has no scene context, so it cannot read "engineered cutting versus
    natural cliff" and is forced onto local evidence. tile-max is naive
    multiple-instance learning and carries real label noise -- a stable tile
    inside an unstable photograph trains as unstable -- which is inherent at
    this scale, not a bug.
    """
    paths, labels, groups = load_index()
    tiles = extract_tiles(paths, args.backbone, args.grid)
    glob = clip_features(paths)
    print(f"{len(paths)} images | tiles {tiles.shape}\n")

    res = {v: [] for v in ("global", "tile-mean", "tile-max")}
    tmean = tiles.mean(axis=1)
    T = tiles.shape[1]
    for _, tr, va in P.grouped_cv(tiles, labels, groups, args.seeds, args.folds):
        y = labels[va]
        m = P.fit_probe(glob[tr], labels[tr], groups[tr])
        res["global"].append(P.score(y, m.predict_proba(glob[va])[:, 1], FLAG, HIGH))

        m = P.fit_probe(tmean[tr], labels[tr], groups[tr])
        res["tile-mean"].append(P.score(y, m.predict_proba(tmean[va])[:, 1], FLAG, HIGH))

        m = P.fit_probe(tiles[tr].reshape(-1, tiles.shape[2]),
                        np.repeat(labels[tr], T), np.repeat(groups[tr], T))
        pt = m.predict_proba(tiles[va].reshape(-1, tiles.shape[2]))[:, 1]
        res["tile-max"].append(P.score(y, pt.reshape(len(va), T).max(axis=1), FLAG, HIGH))

    print("=" * 84)
    print(f"{'variant':>12} {'acc':>13} {'stable':>8} {'unstable':>9} {'AUC':>7} {'Brier':>7}")
    print("-" * 84)
    for v, r in res.items():
        g = lambda k: np.nanmean([x[k] for x in r])
        print(f"{v:>12} {g('acc'):>8.1f} +/- {np.std([x['acc'] for x in r]):<3.1f} "
              f"{g('stable_recall'):>7.1f} {g('unstable_recall'):>8.1f} "
              f"{g('auc'):>7.1f} {g('brier'):>7.3f}")
    print("=" * 84)


# ========================================================================= tune

C_GRID = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
RECALL_TARGETS = [0.90, 0.95, 0.98]


def select_C(X, y, g, method):
    """Choose C inside the training fold. Selecting on the outer fold is leakage."""
    best, best_auc = C_GRID[0], -1.0
    for C in C_GRID:
        aucs = [roc_auc_score(y[va], P.fit_probe(X[tr], y[tr], g[tr], C=C, method=method)
                              .predict_proba(X[va])[:, 1])
                for tr, va in P.inner_splits(y, g) if len(set(y[va])) > 1]
        if aucs and np.mean(aucs) > best_auc:
            best, best_auc = C, float(np.mean(aucs))
    return best


def cmd_tune(args):
    paths, labels, groups = load_index()
    X = clip_features(paths)
    print(f"{len(paths)} images | {CLIP_ID} {X.shape} | seeds={args.seeds}\n")

    def run(method, C):
        accs, aucs, chosen, op, oy = [], [], [], [], []
        for _, tr, va in P.grouped_cv(X, labels, groups, args.seeds, args.folds):
            c = C if C is not None else select_C(X[tr], labels[tr], groups[tr], method)
            chosen.append(c)
            p = P.fit_probe(X[tr], labels[tr], groups[tr], C=c,
                            method=method).predict_proba(X[va])[:, 1]
            accs.append(((p >= 0.5).astype(int) == labels[va]).mean() * 100)
            aucs.append(roc_auc_score(labels[va], p) if len(set(labels[va])) > 1 else np.nan)
            op.append(p); oy.append(labels[va])
        return (np.mean(accs), np.std(accs), np.nanmean(aucs) * 100, chosen,
                np.concatenate(op), np.concatenate(oy))

    print("=" * 76)
    print("REGULARISATION AND CALIBRATION   (C selected inside the training fold)")
    print("-" * 76)
    print(f"{'config':>32} {'accuracy':>14} {'AUC':>7}")
    for method in ("sigmoid", "isotonic"):
        for C in (P.PROBE_C, None):
            acc, sd, auc, chosen, _, _ = run(method, C)
            extra = ""
            if C is None:
                v, c = np.unique(chosen, return_counts=True)
                extra = "  picked " + ", ".join(f"{a:g}x{b}" for a, b in zip(v, c))
            tag = f"C={C if C else 'selected'}, {method}"
            print(f"{tag:>32} {acc:>8.1f} +/- {sd:<4.1f} {auc:>7.1f}{extra}")
    print(f"\n  deployed: C={P.PROBE_C}, {P.CALIBRATION}")

    _, _, _, _, p, y = run(P.CALIBRATION, P.PROBE_C)
    fpr, tpr, thr = roc_curve(y, p)
    print()
    print("=" * 76)
    print("WHAT OPERATING POINT DOES A RECALL TARGET IMPLY?")
    print("-" * 76)
    print(f"{'target unstable recall':>24} {'threshold':>10} {'precision':>10} {'stable recall':>14}")
    for t in RECALL_TARGETS:
        cut = thr[int(np.argmax(tpr >= t))]
        f = p >= cut
        tp, fp_ = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        print(f"{t * 100:>23.0f}% {cut:>10.3f} {tp / max(tp + fp_, 1) * 100:>9.1f}% "
              f"{float(((~f) & (y == 0)).sum()) / max((y == 0).sum(), 1) * 100:>13.1f}%")
    f = p >= FLAG
    tp, fp_ = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    print(f"{'deployed':>23}  {FLAG:>10.3f} {tp / max(tp + fp_, 1) * 100:>9.1f}% "
          f"{float(((~f) & (y == 0)).sum()) / max((y == 0).sum(), 1) * 100:>13.1f}%"
          f"   at {tp / max((y == 1).sum(), 1) * 100:.1f}% recall")
    print("=" * 76)


# ========================================================================= main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, extra in (("shortcut", cmd_shortcut, None),
                            ("tiles", cmd_tiles, "tiles"),
                            ("tune", cmd_tune, None)):
        p = sub.add_parser(name, help=fn.__doc__.strip().split("\n")[0] if fn.__doc__ else "")
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
        if extra == "tiles":
            p.add_argument("--backbone", default="clip", choices=["clip", "clip-large"])
            p.add_argument("--grid", type=int, default=3)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
