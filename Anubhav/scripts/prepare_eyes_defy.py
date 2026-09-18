"""Convert Eyes-Defy-Anemia into the standard images/ + masks/ layout.

Each dataset gets a small preparation script like this one, so
`train_segmenter.py` stays generic and every dataset-specific decision is
visible in one auditable place rather than buried in the training loop.

What this dataset ships
-----------------------
Per subject: a full-resolution photograph (`<stem>.jpg`, ~2988x3984) and up to
three annotation images (`<stem>_palpebral.png`, `_forniceal.png`,
`_forniceal_palpebral.png`). An annotation is the **same frame downscaled ~3.7x
with everything outside the annotated region blacked out**, so a mask is
recoverable as the non-black pixels. Verified: on foreground pixels the
annotation and the downscaled original differ by a mean of 0.06-0.60 out of
255, i.e. resampling noise only.

The catch, and why this script exists
-------------------------------------
**The annotations are not uniformly traced.** Roughly 45% of `_palpebral`
annotations are plain axis-aligned *rectangles* — a bounding box around the
lid, frequently including sclera, iris and pupil — while the rest are genuine
hand-traced crescents. Training on the mixture would teach the network to
predict boxes containing an eye, which is worse than the classical extractor it
is meant to replace.

A rectangle is detectable without opening a file: it fills its own bounding box,
so `extent = area / bbox_area` is ~1.0, against a median of 0.19 for the traced
crescents. This script keeps only the traced ones.

`_forniceal_palpebral` is discarded entirely — 95% of those are rectangles.

Abbreviations
-------------
extent   mask area divided by the area of its bounding box
bbox     bounding box
fg       foreground, the annotated pixels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Above this, the annotation fills its bounding box and is a rectangle, not a
# traced region. The two populations are well separated — traced contours have
# a median extent of 0.19 — so the exact cut is not delicate.
MAX_EXTENT = 0.95

# Below this share of the frame an annotation is too small to be a usable
# target at training resolution.
MIN_COVERAGE = 0.002


def build(source: Path, out: Path, variant: str, max_extent: float) -> dict:
    images_dir, masks_dir = out / "images", out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    kept, rejected = [], []
    counters: dict[str, int] = {}

    for photo in sorted(source.rglob("*.jpg")):
        annotation = photo.with_name(photo.stem + variant + ".png")
        if not annotation.exists():
            rejected.append({"image": str(photo), "reason": "no annotation"})
            continue

        cut = cv2.imread(str(annotation))
        base = cv2.imread(str(photo))
        if cut is None or base is None:
            rejected.append({"image": str(photo), "reason": "unreadable"})
            continue

        mask = (cut.any(axis=2)).astype(np.uint8)
        area = int(mask.sum())
        if area < 200:
            rejected.append({"image": str(photo), "reason": "empty annotation"})
            continue

        ys, xs = np.nonzero(mask)
        bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        extent = area / float(bbox)
        coverage = area / float(mask.size)

        if extent > max_extent:
            rejected.append({"image": str(photo), "reason": "rectangle",
                             "extent": round(extent, 3)})
            continue
        if coverage < MIN_COVERAGE:
            rejected.append({"image": str(photo), "reason": "too small",
                             "coverage": round(coverage, 5)})
            continue

        # The annotation defines the resolution: it is the frame downscaled, so
        # resizing the photograph to match keeps mask and image registered.
        small = cv2.resize(base, (cut.shape[1], cut.shape[0]), interpolation=cv2.INTER_AREA)

        # `<country>-<subject>_<n>` keeps the subject recoverable by
        # `train_segmenter.subject_of`, which strips the trailing index.
        subject = f"{photo.parent.parent.name}-{photo.parent.name}"
        index = counters.get(subject, 0)
        counters[subject] = index + 1
        stem = f"{subject}_{index}"

        cv2.imwrite(str(images_dir / f"{stem}.png"), small)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask * 255)
        kept.append({"stem": stem, "source": str(photo),
                     "extent": round(extent, 3), "coverage": round(coverage, 4)})

    return {"kept": kept, "rejected": rejected, "subjects": len(counters)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path,
                        default=Path("datasets/eyes_defy"),
                        help="unzipped Eyes-Defy-Anemia root")
    parser.add_argument("--out", type=Path,
                        default=Path("datasets/eyes_defy_prepared"))
    parser.add_argument("--variant", default="_palpebral",
                        choices=["_palpebral", "_forniceal", "_forniceal_palpebral"],
                        help="which annotation to use; _palpebral is the "
                             "palpebral conjunctiva this project measures")
    parser.add_argument("--max-extent", type=float, default=MAX_EXTENT,
                        help="reject annotations that fill this much of their "
                             "bounding box (they are rectangles, not contours)")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"No such directory: {args.source}", file=sys.stderr)
        return 1

    report = build(args.source, args.out, args.variant, args.max_extent)
    kept, rejected = report["kept"], report["rejected"]

    reasons: dict[str, int] = {}
    for entry in rejected:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1

    print(f"wrote {len(kept)} image/mask pairs from {report['subjects']} subjects "
          f"to {args.out}")
    print(f"rejected {len(rejected)}:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {reason}")

    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(
        {"source": str(args.source), "variant": args.variant,
         "max_extent": args.max_extent, **report}, indent=2), encoding="utf-8")
    print(f"\nmanifest (every keep/reject decision): {manifest}")
    print(f"train with:  python scripts/train_segmenter.py --data {args.out} "
          f"--out runs/segmenter/unet.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
