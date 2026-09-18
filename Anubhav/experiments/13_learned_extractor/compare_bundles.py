"""Compare two trained segmenter bundles on the local cohort.

The local 52 captures have no ground-truth masks, so Dice cannot be computed
here and a validation Dice from whichever dataset a bundle was trained on does
not transfer as a claim about this cohort. What can be measured is what the
masks *do*: where they land, how large they are, whether they agree with each
other, and — the part that actually matters — whether the haemoglobin model
built on top of them predicts any better.

That last point is the whole reason for the comparison. A segmenter is not the
deliverable; it is an instrument for measuring colour, and the only test that
counts is whether the measurement improves.

    python experiments/13_learned_extractor/compare_bundles.py \\
        --old runs/segmenter/unet.pt --new runs/segmenter/unet_v2.pt

Abbreviations
-------------
IoU      intersection over union, an overlap measure between two masks
LOO      leave-one-out cross-validation, here leaving out one patient
px       pixels
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

from head_to_head import WORK, leave_one_out, score  # noqa: E402

ON_TISSUE = 0.0      # experiment 12's cut: conjunctiva ~ +9, sclera ~ -4
MIN_PIXELS = 500
PERMUTATIONS = 20000


def chroma(rgb, mask):
    px = rgb[mask > 0].astype(np.float64)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    return np.array([(R / total).mean(), (G / total).mean()])


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def permutation_p(X, y, observed):
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(y)
        hits += r2_of(leave_one_out(X, shuffled), shuffled) >= observed
    return hits / PERMUTATIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", type=Path, default=ROOT / "runs/segmenter/unet.pt")
    ap.add_argument("--new", type=Path, default=ROOT / "runs/segmenter/unet_v2.pt")
    args = ap.parse_args()

    models = {}
    for label, path in (("old", args.old), ("new", args.new)):
        seg = LearnedSegmenter(path)
        models[label] = seg
        prov, met = seg.provenance, seg.metrics
        print(f"{label}: {path.name}")
        print(f"  trained {prov.get('trained_at')}  on {prov.get('datasets')}")
        print(f"  {prov.get('train_images')} train / {prov.get('val_images')} val images"
              f"   val Dice {met.get('val_dice', float('nan')):.4f}")
        if prov.get("warm_started_from"):
            print(f"  warm-started from {prov['warm_started_from']}")
    models["classical"] = get_extractor("refined")
    print()

    per_patient: dict[str, dict] = {}
    stats: dict[str, dict] = {k: {"redness": [], "area": [], "empty": 0}
                              for k in models}
    ious, disagreements = [], []

    for sample in load_local_cohort(ROOT):
        rgb = gray_world_white_balance(resize(read_rgb(sample.image_path), WORK))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        redness = (lab[:, :, 1] - 128.0) - 0.5 * (lab[:, :, 2] - 128.0)

        masks = {}
        entry = per_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex,
            "f": {k: [] for k in models}})

        for label, seg in models.items():
            mask, _ = seg(rgb)
            masks[label] = mask
            if np.count_nonzero(mask) < MIN_PIXELS:
                stats[label]["empty"] += 1
                continue
            sel = mask > 0
            stats[label]["redness"].append(float(redness[sel].mean()))
            stats[label]["area"].append(float(sel.mean()))
            entry["f"][label].append(chroma(rgb, mask))

        a, b = masks["old"] > 0, masks["new"] > 0
        if a.any() and b.any():
            ious.append(float((a & b).sum()) / float((a | b).sum()))
            disagreements.append((float(b.sum() - a.sum()) / max(a.sum(), 1)))

    # ---------------- where the masks land ----------------
    print(f"{'extractor':<12}{'median redness':>16}{'min':>8}{'max':>8}"
          f"{'off-tissue':>13}{'median area':>13}{'empty':>7}")
    print("-" * 78)
    placement = {}
    for label in models:
        vals = np.array(stats[label]["redness"])
        if not len(vals):
            print(f"{label:<12}  no usable masks")
            continue
        off = int((vals < ON_TISSUE).sum())
        placement[label] = {"median_redness": float(np.median(vals)),
                            "min": float(vals.min()), "max": float(vals.max()),
                            "off_tissue": off, "n": len(vals),
                            "median_area": float(np.median(stats[label]["area"])),
                            "empty": stats[label]["empty"]}
        print(f"{label:<12}{np.median(vals):>16.2f}{vals.min():>8.2f}{vals.max():>8.2f}"
              f"{off:>8}/{len(vals):<4}{100*np.median(stats[label]['area']):>12.2f}%"
              f"{stats[label]['empty']:>7}")
    print("-" * 78)
    print(f"old vs new mask IoU: median {np.median(ious):.3f}, mean {np.mean(ious):.3f}")
    print(f"new mask area relative to old: median {100*np.median(disagreements):+.1f}%")

    # ---------------- does the haemoglobin model improve? ----------------
    patients = sorted(per_patient, key=lambda p: int(p.split(":")[1]))
    usable = [p for p in patients if all(per_patient[p]["f"][k] for k in models)]
    y = np.array([per_patient[p]["hb"] for p in usable])
    meta = [(per_patient[p]["age"], per_patient[p]["sex"]) for p in usable]

    print(f"\ndownstream haemoglobin model, {len(usable)} patients usable by all "
          f"extractors, chroma features, LOO")
    print(f"{'extractor':<12}{'MAE':>9}{'R²':>9}{'F1':>8}{'null mean':>11}{'p':>8}")
    print("-" * 57)

    baseline = score(np.array([np.delete(y, i).mean() for i in range(len(y))]), y, meta)
    downstream = {}
    for label in models:
        X = np.array([np.mean(per_patient[p]["f"][label], axis=0) for p in usable])
        m = score(leave_one_out(X, y), y, meta)
        rng = np.random.default_rng(0)
        null = np.array([r2_of(leave_one_out(X, s), s)
                         for s in (rng.permutation(y) for _ in range(2000))])
        p_value = float((null >= m["r2"]).mean())
        downstream[label] = {**m, "null_mean": float(null.mean()), "p": p_value}
        print(f"{label:<12}{m['mae']:>9.3f}{m['r2']:>9.3f}{m['f1']:>8.3f}"
              f"{null.mean():>11.3f}{p_value:>8.3f}")
    print("-" * 57)
    print(f"{'baseline':<12}{baseline['mae']:>9.3f}{baseline['r2']:>9.3f}"
          f"{baseline['f1']:>8.3f}")

    out = Path(__file__).resolve().parent / "compare_bundles.json"
    out.write_text(json.dumps({
        "old": str(args.old), "new": str(args.new),
        "n_patients": len(usable), "placement": placement,
        "old_vs_new_iou_median": float(np.median(ious)),
        "downstream": downstream, "baseline": baseline,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
