"""
crop_dataset_sky.py — apply model_utils.crop_sky() to dataset images in place.

Why this has to exist: crop_sky runs at BOTH dataset-prep and inference time.
Images in slope_dataset/ are stored already cropped, and train_probe.py
deliberately does not crop them again. A newly added image that has not been
through here is therefore framed differently from every image the probe was
fitted on — the classic train/serve skew, and it fails silently.

So: run this on every image you add, before training anything.

    python crop_dataset_sky.py slope_dataset/train/unstable/new_photo.jpg
    python crop_dataset_sky.py --dry-run slope_dataset/train/unstable/*.jpg
    python crop_dataset_sky.py --all          # only when crop_sky's logic changes

Cropping is close to idempotent but not exactly — measured over a 25-image
sample of the already-cropped dataset, 24 were untouched by a second pass and
one lost a further 11% of its width. That is why --all is opt-in rather than the
default: re-running over the whole dataset is a real edit, not a no-op, and it
invalidates the fitted probe (refit with train_probe.py afterwards).

The dataset is git-tracked, so a bad run is recoverable with git checkout.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

from model_utils import crop_sky

DATA_DIR = Path("slope_dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def all_dataset_images() -> list[Path]:
    return sorted(p for p in DATA_DIR.glob("*/*/*") if p.suffix.lower() in EXTENSIONS)


def crop_one(path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Crop one image in place. Returns (changed, human-readable summary)."""
    img = Image.open(path).convert("RGB")
    out = crop_sky(img)

    # Compare BOTH dimensions. The original version of this script only checked
    # height, dating from when crop_sky trimmed the top alone. crop_sky now
    # takes a 2D content bounding box and trims the sides too, so a height-only
    # test silently discards every width-only crop it computes.
    if (out.width, out.height) == (img.width, img.height):
        return False, f"unchanged  {img.width}x{img.height}"

    kept = (out.width * out.height) / max(img.width * img.height, 1)
    summary = (f"cropped    {img.width}x{img.height} -> {out.width}x{out.height} "
               f"(kept {kept:.0%} of area)")
    if not dry_run:
        out.save(path)
    return True, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="images to crop (typically the ones you just added)")
    ap.add_argument("--all", action="store_true",
                    help="re-crop the entire dataset; only needed when crop_sky changes")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing anything")
    args = ap.parse_args()

    if args.all and args.paths:
        sys.exit("Pass either explicit paths or --all, not both.")
    if not args.all and not args.paths:
        ap.print_usage()
        sys.exit("\nNothing to do. Pass the image(s) you added, or --all to redo the dataset.")

    targets = all_dataset_images() if args.all else args.paths

    missing = [p for p in targets if not p.exists()]
    if missing:
        sys.exit(f"No such file(s): {[str(p) for p in missing]}")
    bad = [p for p in targets if p.suffix.lower() not in EXTENSIONS]
    if bad:
        sys.exit(f"Not image file(s): {[str(p) for p in bad]}")

    if args.all and not args.dry_run:
        print(f"Re-cropping all {len(targets)} dataset images in place.")
        print("This edits the training data and invalidates slope_probe.joblib.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            sys.exit("Aborted.")

    changed = 0
    for i, p in enumerate(targets, 1):
        was_changed, summary = crop_one(p, dry_run=args.dry_run)
        changed += was_changed
        if was_changed or not args.all:
            try:
                label = p.relative_to(DATA_DIR)
            except ValueError:
                label = p
            print(f"[{i}/{len(targets)}] {summary}  {label}")

    verb = "would be cropped" if args.dry_run else "cropped and overwritten"
    print(f"\nDone. {changed}/{len(targets)} {verb}.")
    if changed and not args.dry_run:
        print("Refit the deployed model so it matches the data:")
        print("  python train_probe.py")


if __name__ == "__main__":
    main()
