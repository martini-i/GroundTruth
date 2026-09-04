"""
verify_dataset.py — check the invariants labels.csv and slope_dataset/ must hold.

Run this after any change to the dataset, before trusting an evaluation number.
The three failures it catches have all actually happened in this project:

  * a filename in labels.csv with no file on disk, or the reverse — the loaders
    silently work off whichever side they read, so the two drift apart without
    anything erroring;
  * a duplicate filename, which double-counts one image;
  * a site_id appearing in more than one split, which puts photographs of the
    same location on both sides of the evaluation boundary and inflates
    accuracy. cross_validate.py asserts on this at runtime, but by then the
    dataset is already wrong.

Exits non-zero on failure so it can gate a script.
"""

import csv
import sys
from pathlib import Path

LABELS_CSV = Path("labels.csv")
DATASET_DIR = Path("slope_dataset")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def verify(labels_csv: Path = LABELS_CSV, dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Return a list of problems; empty means the dataset is consistent."""
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    names = [r["filename"] for r in rows]
    on_disk = {p.name: p for p in dataset_dir.rglob("*")
               if p.suffix.lower() in IMAGE_SUFFIXES}

    problems = []

    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        problems.append(f"duplicate filenames in labels.csv: {dupes}")

    missing = sorted(set(names) - set(on_disk))
    if missing:
        problems.append(f"in labels.csv but not on disk: {missing}")

    extra = sorted(set(on_disk) - set(names))
    if extra:
        problems.append(f"on disk but not in labels.csv: {extra}")

    # The directory a file sits in is the authority on its split and label; a row
    # that disagrees means the image is being trained on under the wrong header.
    mismatched = []
    for r in rows:
        p = on_disk.get(r["filename"])
        if p is None:
            continue
        parts = p.parts
        if len(parts) >= 3 and (parts[-3], parts[-2]) != (r["split"], r["label"]):
            mismatched.append(f"{r['filename']} (csv: {r['split']}/{r['label']}, "
                              f"disk: {parts[-3]}/{parts[-2]})")
    if mismatched:
        problems.append(f"split/label disagrees with directory: {mismatched}")

    sites: dict[str, set[str]] = {}
    for r in rows:
        if r.get("site_id"):
            sites.setdefault(r["site_id"], set()).add(r["split"])
    spanning = {s: sorted(v) for s, v in sites.items() if len(v) > 1}
    if spanning:
        problems.append(f"site_id spanning splits: {spanning}")

    return problems


def main() -> None:
    problems = verify()
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        sys.exit(1)

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    counts = Counter((r["split"], r["label"]) for r in rows)
    print(f"  OK  {len(rows)} rows, 1:1 with disk, no site_id spanning splits.")
    for key in sorted(counts):
        print(f"        {key[0]}/{key[1]}: {counts[key]}")


if __name__ == "__main__":
    main()
