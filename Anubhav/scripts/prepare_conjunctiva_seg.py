"""Convert the Eye Conjunctiva Segmentation Dataset into images/ + masks/.

Source: https://data.mendeley.com/datasets/yxwjgcndg2/1 (CC BY 4.0)
547 smartphone captures, each with masks from **two independent annotators**.

Three things about this dataset decide how it is prepared.

**The masks are RGB with values {0, 128}, not {0, 255}.** A conventional
`> 127` threshold therefore reads every mask as empty — and silently, since an
empty prediction against an empty target scores a perfect Dice. Measured here
with `> 0`.

**The two annotators agree far less than the description suggests.** Their
inter-annotator **Dice is 0.736 mean / 0.745 median**, with 546 of 547 pairs
below 0.90. The "~99.9% agreement" quoted for this dataset is *pixel accuracy*,
which is uninformative when the conjunctiva occupies a median **0.76%** of the
frame: predicting "background everywhere" already scores 99.24%. The real human
ceiling for this task is Dice ≈ 0.74, and no model should be tuned past it.

**The default combines them by intersection.** Where the annotators disagree,
one of them included tissue the other did not, and this project has measured
that contamination costs more than omission — every step of widening the mask
degraded haemoglobin prediction, monotonically
(`experiments/13_learned_extractor/`). The intersection is the region both
experts agree is conjunctiva, which is the conservative and better-motivated
target. `--annotator` selects a different policy.

Images are downscaled on the way out. The originals are ~12 megapixels and
training runs at 320px, so full-resolution files would dominate the epoch time
without changing anything the network sees.

Abbreviations
-------------
Dice     overlap score, 2|A∩B| / (|A|+|B|); 1.0 is a perfect match
px       pixels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SOURCE = Path("datasets/conjunctiva_seg/Eye Conjunctiva Segmentation Dataset")
MIN_COVERAGE = 0.0005  # below this the target is too small to learn from at 320px


def load_binary(path: Path, reduce_flag: int) -> np.ndarray | None:
    """Read a mask as a boolean array. Foreground is any non-zero value."""

    raw = cv2.imread(str(path), reduce_flag)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    return raw > 0


def combine(m1, m2, policy: str):
    if policy == "1":
        return m1
    if policy == "2":
        return m2
    if policy == "union":
        return np.logical_or(m1, m2)
    return np.logical_and(m1, m2)  # intersection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path,
                        default=Path("datasets/conjunctiva_seg_prepared"))
    parser.add_argument("--annotator", default="intersection",
                        choices=["1", "2", "union", "intersection"],
                        help="how to combine the two annotations "
                             "(default: intersection — see module docstring)")
    parser.add_argument("--max-size", type=int, default=768,
                        help="longest edge of the written image; training runs "
                             "at 320px so the originals are wasteful")
    args = parser.parse_args()

    images = args.source / "Images"
    ann1 = args.source / "Masks Annotator 1"
    ann2 = args.source / "Masks Annotator 2"
    for d in (images, ann1, ann2):
        if not d.is_dir():
            print(f"missing {d}", file=sys.stderr)
            return 1

    out_images = args.out / "images"
    out_masks = args.out / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    kept, rejected, agreements = [], [], []

    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = image_path.stem

        # Decode at 1/4 resolution: 12 MP originals, and the output is 768px.
        photo = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_4)
        m1 = load_binary(ann1 / image_path.name, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        m2 = load_binary(ann2 / image_path.name, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if photo is None or m1 is None or m2 is None:
            rejected.append({"stem": stem, "reason": "unreadable"})
            continue

        for name, m in (("m1", m1), ("m2", m2)):
            if m.shape != photo.shape[:2]:
                resized = cv2.resize(m.astype(np.uint8), (photo.shape[1], photo.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0
                if name == "m1":
                    m1 = resized
                else:
                    m2 = resized

        inter = np.logical_and(m1, m2).sum()
        union = m1.sum() + m2.sum()
        agreement = 1.0 if union == 0 else 2.0 * inter / union
        agreements.append(agreement)

        mask = combine(m1, m2, args.annotator)
        coverage = float(mask.mean())
        if coverage < MIN_COVERAGE:
            rejected.append({"stem": stem, "reason": "mask too small",
                             "coverage": round(coverage, 6)})
            continue

        scale = args.max_size / max(photo.shape[:2])
        if scale < 1.0:
            size = (int(round(photo.shape[1] * scale)), int(round(photo.shape[0] * scale)))
            photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), size,
                              interpolation=cv2.INTER_NEAREST) > 0

        # One capture per subject, so `<stem>_0` keeps `subject_of` correct.
        cv2.imwrite(str(out_images / f"{stem}_0.png"), photo)
        cv2.imwrite(str(out_masks / f"{stem}_0.png"), mask.astype(np.uint8) * 255)
        kept.append({"stem": stem, "coverage": round(coverage, 5),
                     "annotator_dice": round(agreement, 4)})

    agreements = np.array(agreements)
    reasons: dict[str, int] = {}
    for entry in rejected:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1

    print(f"wrote {len(kept)} image/mask pairs to {args.out}")
    print(f"combination policy: {args.annotator}")
    if len(agreements):
        print(f"inter-annotator Dice over {len(agreements)} pairs: "
              f"mean {agreements.mean():.4f}, median {np.median(agreements):.4f}")
        print(f"  --> this is the human ceiling; a model should not be tuned past it")
    if rejected:
        print(f"rejected {len(rejected)}: " +
              ", ".join(f"{v} {k}" for k, v in reasons.items()))

    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps({
        "source": str(args.source), "annotator_policy": args.annotator,
        "max_size": args.max_size,
        "inter_annotator_dice_mean": float(agreements.mean()) if len(agreements) else None,
        "inter_annotator_dice_median": float(np.median(agreements)) if len(agreements) else None,
        "kept": kept, "rejected": rejected,
    }, indent=2), encoding="utf-8")
    print(f"manifest: {manifest}")
    print(f"\ntrain with:  python scripts/train_segmenter.py --data {args.out} \\")
    print(f"               --init runs/segmenter/unet.pt --out runs/segmenter/unet_v2.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
