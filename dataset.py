"""
dataset.py — one place to enumerate the curated dataset and turn it into CLIP features.

Path listing, the labels join, the feature-cache key and CLIP extraction were
each implemented three times over (cross_validate, train_probe, frozen_features,
dupes). Two of those wrote the same cache files, so any drift between them would
have silently reintroduced train/serve skew — the class of bug that has already
cost this project twice. One implementation removes that risk.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ["stable", "unstable"]
DATASET_DIR = Path("slope_dataset")
LABELS_CSV = Path("labels.csv")
CACHE_DIR = Path(".feature_cache")
CLIP_ID = "openai/clip-vit-base-patch32"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dataset_paths() -> list[Path]:
    """Every curated image, in a stable order so cached features stay aligned."""
    return sorted(
        (p for p in DATASET_DIR.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.as_posix(),
    )


def load_index():
    """
    Pair every image on disk with its label and site group from labels.csv.

    Raises rather than skipping on a missing row: a silent skip means the
    evaluation quietly runs on fewer images than the dataset claims.
    """
    meta = {r["filename"]: r for r in csv.DictReader(LABELS_CSV.open(encoding="utf-8"))}
    paths, labels, groups = [], [], []
    missing = []
    for p in dataset_paths():
        row = meta.get(p.name)
        if row is None:
            missing.append(p.name)
            continue
        paths.append(p)
        labels.append(CLASS_NAMES.index(row["label"]))
        # No site_id means the image shares a location with nothing else, so it
        # becomes its own group rather than being lumped in with other blanks.
        groups.append(row["site_id"] or f"__solo__{p.name}")
    if missing:
        raise SystemExit(f"{len(missing)} image(s) missing from labels.csv: {missing[:5]}")
    return paths, np.array(labels), np.array(groups)


def index_key(paths) -> str:
    """
    Cache key derived from the file list *and the file contents*.

    Two failure modes, both silent:

    1. Keying on the count alone returns feature rows belonging to a previous set
       of images whenever the dataset changes without changing size. They still
       line up positionally against the new labels, so nothing errors and the
       model is fitted on the wrong data.

    2. Keying on paths alone misses an *in-place* edit. crop_dataset_sky.py
       rewrites images under their existing names, so the path list is unchanged
       and a stale cache survives a re-crop — measured: after re-cropping four
       images the cache was reused and the reported accuracy was computed on the
       pre-crop pixels.

    Size and mtime are enough to catch both without hashing megabytes of JPEG.
    """
    parts = []
    for p in paths:
        st = Path(p).stat()
        parts.append(f"{Path(p).as_posix()}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def clip_features(paths, use_cache: bool = True, progress_every: int = 40) -> np.ndarray:
    """
    CLIP image embeddings for `paths`, cached on disk.

    Images are read as they sit on disk. They are already sky-cropped, and
    cropping again here would fit on doubly-cropped images while inference crops
    a raw upload exactly once.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"clip_{len(paths)}_{index_key(paths)}.npy"
    if use_cache and cache.exists():
        return np.load(cache)

    from transformers import AutoImageProcessor, CLIPModel

    proc = AutoImageProcessor.from_pretrained(CLIP_ID)
    model = CLIPModel.from_pretrained(CLIP_ID).eval()
    feats = []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            inputs = proc(images=Image.open(p).convert("RGB"), return_tensors="pt")
            vis = model.vision_model(pixel_values=inputs["pixel_values"])
            feats.append(model.visual_projection(vis.pooler_output).squeeze(0).numpy())
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    feats = np.stack(feats)
    np.save(cache, feats)
    return feats
