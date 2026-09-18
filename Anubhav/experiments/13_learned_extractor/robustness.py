"""Which extractor survives a capture taken differently?

The comparison in `compare_bundles.py` scores masks on the 52 local captures —
one camera, one operator, one setting. A model that is *more robust to varied
capture conditions* looks identical on that test, because those conditions are
not in it. Accuracy on a narrow slice is not robustness, and the two can be
measured apart.

This perturbs each capture in ways a real user would produce — the phone held at
an angle, held closer or further, a dimmer room, a shakier hand — and measures
how far each extractor's mask moves from the one it produced on the clean image.

The metric is **self-consistency**: IoU between the mask on the perturbed image
and the mask on the clean image, after undoing the geometric transform. A robust
extractor finds the same tissue regardless of how the photograph was taken. This
needs no ground truth, which is what makes it usable on a cohort that has none.

Also reported is the **failure rate** — how often a perturbation makes the
extractor return nothing at all, which is the failure mode that matters in
deployment because it costs the patient a retake.

Run: python experiments/13_learned_extractor/robustness.py

Abbreviations
-------------
IoU    intersection over union, an overlap measure between two masks
px     pixels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

WORK = (512, 512)
MIN_PIXELS = 300


def perturbations():
    """Each entry returns (transform image, map a mask back to clean geometry).

    The inverse map matters: a rotated capture yields a rotated mask, and
    comparing it to the clean mask without undoing the rotation would measure
    the rotation rather than the extractor.
    """

    def rotate(deg):
        def apply(img):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        def invert(mask):
            h, w = mask.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
            return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return apply, invert

    def zoom(factor):
        def apply(img):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, factor)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        def invert(mask):
            h, w = mask.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.0 / factor)
            return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return apply, invert

    def photometric(fn):
        return fn, (lambda mask: mask)

    return {
        "rotate 10deg": rotate(10),
        "rotate 20deg": rotate(20),
        "rotate -15deg": rotate(-15),
        "zoom in 1.3x": zoom(1.3),
        "zoom out 0.75x": zoom(0.75),
        "dim 60%": photometric(lambda i: np.clip(i.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)),
        "bright 140%": photometric(lambda i: np.clip(i.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)),
        "motion blur": photometric(
            lambda i: cv2.filter2D(i, -1, np.eye(9, dtype=np.float32) / 9.0)),
        "jpeg quality 30": photometric(
            lambda i: cv2.imdecode(cv2.imencode(".jpg", i, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1)),
        "warm cast": photometric(
            lambda i: np.clip(i.astype(np.float32) * np.array([1.15, 1.0, 0.85], np.float32),
                              0, 255).astype(np.uint8)),
    }


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="+", type=Path,
                    default=[ROOT / "runs/segmenter/unet.pt",
                             ROOT / "runs/segmenter/unet_v3.pt"])
    args = ap.parse_args()

    models = {p.stem: LearnedSegmenter(p) for p in args.bundles}
    models["classical"] = get_extractor("refined")

    perts = perturbations()
    scores = {m: {k: [] for k in perts} for m in models}
    failures = {m: {k: 0 for k in perts} for m in models}
    clean_failures = {m: 0 for m in models}

    samples = list(load_local_cohort(ROOT))
    print(f"{len(samples)} captures x {len(perts)} perturbations x {len(models)} extractors\n")

    for sample in samples:
        base = gray_world_white_balance(resize(read_rgb(sample.image_path), WORK))
        clean = {}
        for name, seg in models.items():
            mask, _ = seg(base)
            if np.count_nonzero(mask) < MIN_PIXELS:
                clean_failures[name] += 1
                continue
            clean[name] = mask > 0

        for label, (apply, invert) in perts.items():
            moved = gray_world_white_balance(apply(resize(read_rgb(sample.image_path), WORK)))
            for name, seg in models.items():
                if name not in clean:
                    continue
                mask, _ = seg(moved)
                if np.count_nonzero(mask) < MIN_PIXELS:
                    failures[name][label] += 1
                    scores[name][label].append(0.0)
                    continue
                scores[name][label].append(iou(invert(mask) > 0, clean[name]))

    names = list(models)
    print(f"{'perturbation':<18}" + "".join(f"{n:>16}" for n in names))
    print("-" * (18 + 16 * len(names)))
    summary = {}
    for label in perts:
        row = f"{label:<18}"
        for n in names:
            v = np.array(scores[n][label])
            fails = failures[n][label]
            row += f"{np.median(v):>11.3f}{'(' + str(fails) + ')':>5}"
            summary.setdefault(n, {})[label] = {"median_iou": float(np.median(v)),
                                                "failures": fails}
        print(row)
    print("-" * (18 + 16 * len(names)))

    print(f"{'MEDIAN OVERALL':<18}", end="")
    overall = {}
    for n in names:
        allv = np.concatenate([np.array(scores[n][k]) for k in perts])
        totalf = sum(failures[n].values())
        overall[n] = {"median_iou": float(np.median(allv)),
                      "total_failures": int(totalf),
                      "clean_failures": clean_failures[n]}
        print(f"{np.median(allv):>11.3f}{'(' + str(totalf) + ')':>5}", end="")
    print("\n\n(N) = perturbed captures where the extractor returned nothing.")
    print("IoU is against the SAME extractor's mask on the clean image, with the")
    print("geometric transform undone — so it measures self-consistency, not accuracy.")
    print(f"\nclean-image failures: " +
          ", ".join(f"{n} {clean_failures[n]}" for n in names))

    out = Path(__file__).resolve().parent / "robustness.json"
    out.write_text(json.dumps({"n_captures": len(samples), "per_perturbation": summary,
                               "overall": overall}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
