"""
dupes.py — near-duplicate audit over the curated dataset.

Site leakage is the failure this guards against. Two photographs of the same
cutting, taken a few metres apart, are not independent samples: if one lands in
train and the other in validation, the reported accuracy is measuring memory
rather than generalisation. labels.csv carries a site_id column and
cross_validate.py asserts on it, but both rely on somebody *noticing* the
relationship in the first place.

CLIP cosine similarity finds those pairs. On this dataset a threshold of 0.96
recovered the one true byte-identical duplicate with no false positives, which
is what the constant below is set from. Treat it as a review aid, not a verdict:
the output is a suggested site_id link for a human to confirm.

Images in slope_dataset/ are stored already sky-cropped, so they are embedded
as they sit on disk. Anything compared against them must be cropped the same
way first — embedding one cropped and one uncropped compares different framings
and quietly lowers every similarity.
"""

from pathlib import Path

import numpy as np

from dataset import CACHE_DIR, dataset_paths, index_key


# Above this cosine, two images are near-certainly the same scene. Measured on
# the 179-image dataset: with the one true duplicate removed, the highest
# similarity between any two distinct images is 0.9518, so 0.96 sits clearly
# above the noise floor and does not fire on merely similar terrain.
NEAR_DUPLICATE = 0.96

# Between this and NEAR_DUPLICATE, a pair is worth a human glance — but treat
# the band as weak evidence. At 0.93 it surfaces 8 pairs of which 3 straddle the
# stable/unstable boundary; at 0.90 it surfaces 32 of which 9 do. CLIP puts
# stable_rock_011 and unstable_cliff_016 at 0.9453, higher than most same-label
# pairs, because in this range the similarity is reporting "both are rock faces"
# rather than "both are the same place" — the same semantic shortcut FINDINGS.md
# documents in the classifier itself. Set here so the list stays short enough to
# actually review.
SAME_SITE_HINT = 0.93


def _embed_paths(paths: list[Path]) -> np.ndarray:
    import model_utils
    from PIL import Image

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        emb, _ = model_utils._clip_embedding(img)
        out.append(emb.cpu().numpy().squeeze(0))
    return np.stack(out)


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


_index: dict | None = None


def build_index(refresh: bool = False) -> dict:
    """Load or compute unit-norm CLIP embeddings for the curated dataset."""
    global _index
    if _index is not None and not refresh:
        return _index

    paths = dataset_paths()
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"dupe_index_{index_key(paths)}.npy"

    if cache.exists() and not refresh:
        feats = np.load(cache)
    else:
        feats = _embed_paths(paths)
        np.save(cache, feats)

    _index = {"paths": paths, "feats": _normalise(feats)}
    return _index


def audit_dataset(threshold: float = NEAR_DUPLICATE) -> list[tuple[str, str, float]]:
    """
    Every within-dataset pair above `threshold`, for auditing site_id coverage.

    Run this after adding images: any pair it reports should either share a
    site_id in labels.csv or be a genuine duplicate to delete.
    """
    idx = build_index()
    feats, paths = idx["feats"], idx["paths"]
    sims = feats @ feats.T
    pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if sims[i, j] >= threshold:
                pairs.append((paths[i].name, paths[j].name, round(float(sims[i, j]), 4)))
    return sorted(pairs, key=lambda t: -t[2])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Audit the dataset for near-duplicate pairs.")
    ap.add_argument("--threshold", type=float, default=NEAR_DUPLICATE)
    ap.add_argument("--refresh", action="store_true", help="recompute embeddings")
    args = ap.parse_args()

    build_index(refresh=args.refresh)
    found = audit_dataset(args.threshold)
    if not found:
        print(f"No pairs at or above cosine {args.threshold}.")
    else:
        print(f"{len(found)} pair(s) at or above cosine {args.threshold}:\n")
        for a, b, s in found:
            print(f"  {s:.4f}  {a}  <->  {b}")
        print("\nEach pair should share a site_id in labels.csv, or one should be removed.")
