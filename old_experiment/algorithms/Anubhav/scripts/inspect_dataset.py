"""Report what a dataset root actually contains, before trying to train on it.

Run this on every new dataset. It answers the one question that decides whether
a download is usable for segmentation at all:

**Does it ship masks, or only pre-cropped conjunctiva?**

A set of cropped conjunctiva patches — which is what most published anaemia
datasets distribute — has no input/target pair a segmenter can learn from: the
input a segmenter needs is the *whole eye*, and the crop has thrown that away.
Those datasets are still valuable, just for the regression stage, through
`anemia.data.load_eyes_defy_anemia` / `load_cp_anemic`.

The distinction is invisible on a download page and easy to lose an afternoon
to, so it is measured here rather than assumed.

    python scripts/inspect_dataset.py datasets/eyes_defy

Abbreviations
-------------
GT     ground truth, the reference mask a model is trained against
px     pixels
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_segmenter import find_pairs, subject_of  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def describe_mask(path: Path) -> str:
    """Is this file a mask, or a picture of cropped tissue?

    A mask is near-binary: almost every pixel is 0 or the label value. A cropped
    photograph has a broad intensity histogram. Counting distinct values
    separates the two without needing to look at a single file by hand.
    """

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "unreadable"
    distinct = len(np.unique(image))
    if distinct <= 2:
        return "binary mask"
    if distinct <= 16:
        return f"label map ({distinct} values)"
    zero_fraction = float((image == 0).mean())
    if zero_fraction > 0.25:
        return f"cutout on black ({100 * zero_fraction:.0f}% background) — mask recoverable"
    return f"photograph ({distinct} grey levels) — not a mask"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--show", type=int, default=8, help="Example paths to print")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    tables = [p for p in files if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]

    print(f"=== {root} ===")
    print(f"{len(files)} files, {len(images)} images, {len(tables)} tables\n")

    print("directory shape (top levels):")
    depths = Counter()
    for p in images:
        rel = p.relative_to(root).parent
        depths[str(rel)] += 1
    for name, count in sorted(depths.items(), key=lambda kv: -kv[1])[: args.show]:
        print(f"  {count:5d}  {name}/")
    if len(depths) > args.show:
        print(f"  ... and {len(depths) - args.show} more directories")

    if tables:
        print("\nlabel tables (haemoglobin likely here):")
        for t in tables[: args.show]:
            print(f"  {t.relative_to(root)}")

    pairs = find_pairs(root)
    print(f"\nimage/mask pairs discovered: {len(pairs)}")

    if pairs:
        subjects = {subject_of(image) for image, _ in pairs}
        print(f"distinct subjects (for grouped splitting): {len(subjects)}")
        print("\nsample pairs, with what the 'mask' side really is:")
        for image_path, mask_path in pairs[: args.show]:
            print(f"  {image_path.relative_to(root)}")
            print(f"    -> {mask_path.relative_to(root)}  [{describe_mask(mask_path)}]")
        print("\nIf the mask side says 'photograph', these are cropped patches,")
        print("not masks — usable for regression, not for training a segmenter.")
    else:
        print("\nNo pairs found. Either the layout is unusual, or this dataset")
        print("ships cropped conjunctiva rather than masks. Sample images:")
        for p in images[: args.show]:
            print(f"  {p.relative_to(root)}  [{describe_mask(p)}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
