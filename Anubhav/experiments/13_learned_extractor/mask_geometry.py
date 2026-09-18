"""Is the learned extractor's thin mask costing signal, or protecting it?

The U-Net's masks cover a median 1.8% of frame against the classical chain's
10.7%. Two opposite worries follow, and they make opposite predictions:

  * **Too few pixels.** A colour mean over a thin sliver is noisier than one
    over a broad region. If that dominates, widening the mask should help.
  * **Contamination is worse than omission.** The feature is a *mean* over the
    mask. Including sclera shifts that mean by an amount set by how much sclera
    got in, which varies per capture for reasons unrelated to haemoglobin — so
    contamination injects noise that is correlated with framing, not biology.
    Omitting conjunctiva only reduces the sample, and any consistent offset from
    always measuring the same part of the tissue is absorbed by the intercept.
    If that dominates, widening should hurt and tightening should be roughly
    free.

This sweeps mask geometry and scores each variant by the same leave-one-patient-
out protocol used everywhere else, so the two predictions can be separated.

Also tested: **probability-weighted pooling**, which avoids the threshold
entirely by weighting each pixel by the network's confidence. Boundary pixels
that are half eyelash contribute half.

Run: python experiments/13_learned_extractor/mask_geometry.py

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
px     pixels
"""

from __future__ import annotations

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

from head_to_head import WORK, leave_one_out, score  # noqa: E402

BUNDLE = ROOT / "runs/segmenter/unet.pt"
REPRESENTATION = "chroma"  # the best-performing representation for C


def features(rgb, weights):
    """Chromaticity, pooled with per-pixel weights.

    A binary mask is the special case where every weight is 0 or 1, so the hard
    and soft variants go through identical code and differ only in `weights`.
    """

    w = weights.reshape(-1).astype(np.float64)
    total_w = w.sum()
    if total_w < 1.0:
        return None
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    return np.array([float((w * (R / total)).sum() / total_w),
                     float((w * (G / total)).sum() / total_w)])


def variants(probability, threshold=0.5):
    """Every mask geometry under test, as a weight map."""

    hard = (probability >= threshold).astype(np.uint8)
    out = {}

    for radius in (-6, -4, -2, 0, 2, 4, 8, 16):
        if radius == 0:
            m = hard
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * abs(radius) + 1, 2 * abs(radius) + 1))
            m = cv2.dilate(hard, k) if radius > 0 else cv2.erode(hard, k)
        label = ("baseline (p>=0.5)" if radius == 0
                 else f"{'dilate' if radius > 0 else 'erode'} {abs(radius)}px")
        out[label] = m.astype(np.float64)

    # Threshold sweep: a higher cut keeps only confident pixels.
    for t in (0.3, 0.7, 0.9):
        out[f"threshold p>={t}"] = (probability >= t).astype(np.float64)

    # Soft pooling — no threshold at all.
    out["probability-weighted"] = probability.astype(np.float64)
    out["probability^2-weighted"] = (probability ** 2).astype(np.float64)

    return out


def main() -> None:
    learned = get_extractor("unet", model_dir=BUNDLE)

    per_patient: dict[str, dict] = {}
    areas: dict[str, list] = {}

    for sample in load_local_cohort(ROOT):
        rgb = gray_world_white_balance(resize(read_rgb(sample.image_path), WORK))
        probability = learned.probability(rgb)

        entry = per_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "f": {}})

        for label, weights in variants(probability).items():
            f = features(rgb, weights)
            if f is None:
                continue
            entry["f"].setdefault(label, []).append(f)
            areas.setdefault(label, []).append(float((weights > 0).mean()))

    labels = list(areas)
    # A patient counts only if every variant produced a usable measurement, so
    # the variants are compared on identical patients.
    patients = sorted((p for p, e in per_patient.items()
                       if all(lbl in e["f"] for lbl in labels)),
                      key=lambda p: int(p.split(":")[1]))
    y = np.array([per_patient[p]["hb"] for p in patients])
    meta = [(per_patient[p]["age"], per_patient[p]["sex"]) for p in patients]

    baseline = score(np.array([np.delete(y, i).mean() for i in range(len(y))]), y, meta)
    print(f"{len(patients)} patients, {REPRESENTATION} features, leave-one-patient-out\n")
    print(f"{'mask variant':<24}{'area%':>8}{'MAE':>9}{'R²':>9}{'F1':>8}{'vs baseline':>13}")
    print("-" * 71)

    results = {}
    for label in labels:
        X = np.array([np.mean(per_patient[p]["f"][label], axis=0) for p in patients])
        m = score(leave_one_out(X, y), y, meta)
        results[label] = {**m, "mean_area": float(np.mean(areas[label]))}
        delta = baseline["mae"] - m["mae"]
        print(f"{label:<24}{100*np.mean(areas[label]):>8.2f}{m['mae']:>9.3f}"
              f"{m['r2']:>9.3f}{m['f1']:>8.3f}{delta:>+13.3f}")

    print("-" * 71)
    print(f"{'baseline (predict mean)':<24}{'-':>8}{baseline['mae']:>9.3f}"
          f"{baseline['r2']:>9.3f}{baseline['f1']:>8.3f}")

    best = min(results, key=lambda k: results[k]["mae"])
    print(f"\nbest by MAE: {best}  (MAE {results[best]['mae']:.3f}, "
          f"R² {results[best]['r2']:+.3f})")

    out = Path(__file__).resolve().parent / "mask_geometry.json"
    out.write_text(json.dumps({"n_patients": len(patients),
                               "representation": REPRESENTATION,
                               "baseline": baseline, "variants": results,
                               "best_by_mae": best}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
