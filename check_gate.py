"""
check_gate.py — does the input gate refuse anything it shouldn't?

model_utils.check_assessable() refuses images the classifier cannot speak to:
too dark, too small, indoors, all sky, no recognisable ground. That is only
useful if it never fires on a real slope photograph, so this runs it over every
image in slope_dataset/ and reports anything it rejects.

**A refusal here is a bug in the gate, not a bad image.** The dataset is the
definition of what the model is expected to handle. Any hit means a threshold in
model_utils needs loosening.

It also prints the distribution of every statistic the gate measures, with the
threshold alongside, so the margins are visible. A check whose dataset minimum
sits just above its cutoff is one unusual photograph away from firing.

Caveat worth remembering when reading the output: images in slope_dataset/ are
stored already sky-cropped, so their visible_fraction is near 1.0 and the
no_ground check is effectively untested here. It is exercised at inference, on
raw uploads. Point --paths at some uncropped originals to test it properly.

Usage:
    python check_gate.py                       # the whole curated dataset
    python check_gate.py --paths a.jpg b.jpg   # arbitrary images
    python check_gate.py --expect-refusal --paths carpet.jpg ceiling.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import model_utils as mu
from dataset import dataset_paths

# Each stat the gate records, the threshold it is compared against, and whether
# the check fires when the value is low or high. Kept here rather than derived,
# so a stat added to check_assessable without being reported shows up as missing.
CHECKS = [
    ("width",             None,                     None),
    ("height",            None,                     None),
    ("aspect_ratio",      mu.MAX_ASPECT_RATIO,      "high"),
    ("mean_luminance",    mu.MIN_MEAN_LUMINANCE,    "low"),
    ("luminance_std",     mu.MIN_LUMINANCE_STD,     "low"),
    ("visible_fraction",  mu.VISIBLE_MIN_FRACTION,  "low"),
    ("indoor_fraction",   mu.INDOOR_MAX_FRACTION,   "high"),
    ("terrain_fraction",  mu.TERRAIN_MIN_FRACTION,  "low"),
    ("person_fraction",   mu.PERSON_MAX_FRACTION,   "high"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", type=Path,
                    help="images to check instead of the curated dataset")
    ap.add_argument("--expect-refusal", action="store_true",
                    help="invert the verdict: these images SHOULD be refused, so a "
                         "pass is the failure. Use it on known-bad photographs.")
    args = ap.parse_args()

    paths = args.paths if args.paths else dataset_paths()
    if not paths:
        print("no images to check")
        return 1

    print(f"{len(paths)} image(s) | "
          f"{'expecting refusals' if args.expect_refusal else 'expecting all to pass'}\n",
          flush=True)

    refused, collected = [], {name: [] for name, _, _ in CHECKS}
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            print(f"  SKIP  {Path(p).name}: {exc}")
            continue

        result = mu.check_assessable(img)
        for name in collected:
            if name in result["stats"]:
                collected[name].append(result["stats"][name])
        if not result["ok"]:
            refused.append((Path(p).name, result["reason"], result["stats"]))
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    print()
    print("=" * 76)
    print(f"{'stat':>18} {'min':>10} {'median':>10} {'max':>10} {'threshold':>12}")
    print("-" * 76)
    for name, threshold, direction in CHECKS:
        vals = collected[name]
        if not vals:
            print(f"{name:>18} {'(not recorded — check_assessable returned early)':>52}")
            continue
        arr = np.asarray(vals, dtype=float)
        thr = "—" if threshold is None else f"{threshold:g} ({direction})"
        print(f"{name:>18} {arr.min():>10.3g} {np.median(arr):>10.3g} "
              f"{arr.max():>10.3g} {thr:>12}")
    print("=" * 76)

    if args.expect_refusal:
        passed = [Path(p).name for p in paths
                  if Path(p).name not in {r[0] for r in refused}]
        print(f"\nrefused {len(refused)}/{len(paths)} (wanted all)")
        for name, reason, _ in refused:
            print(f"  ok    {name:38s} {reason}")
        for name in passed:
            print(f"  MISS  {name:38s} slipped through the gate")
        return 1 if passed else 0

    if not refused:
        print("\n  OK  the gate refuses nothing in the dataset.")
        return 0

    print(f"\n  FAIL  the gate refuses {len(refused)} dataset image(s). "
          "Loosen the threshold named on each line:")
    for name, reason, stats in refused:
        relevant = {k: v for k, v in stats.items() if k in collected}
        print(f"        {name:38s} {reason:20s} {relevant}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
