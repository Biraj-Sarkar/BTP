"""Is a mean over the mask enough, or does the colour distribution carry more?

Every model in this project reduces the conjunctiva to **means** — mean
chromaticity, mean log-ratio, mean a*. That discards everything about the
distribution's shape: how spread the colour is, whether the tissue is uniform or
graded, whether a few contaminating pixels dragged the average.

Three families are tested against that baseline, all kept to <= 3 features
because n = 26:

  * **Robust centre.** Median and trimmed mean instead of mean. If residual
    contamination is what limits the features — as the mask-geometry sweep
    suggests — a robust centre should beat the mean *using the same mask*.
  * **Spread.** Adding an IQR or standard deviation term. Pallor may change how
    uniform the tissue looks, not only how red it is on average.
  * **Shape.** Quartiles of the distribution, which capture centre and spread
    together without assuming which matters.

Scored by the same leave-one-patient-out protocol as everything else, on
pipeline C's masks.

Run: python experiments/13_learned_extractor/feature_ablation.py

Abbreviations
-------------
IQR    interquartile range, p75 - p25
LOO    leave-one-out cross-validation, here leaving out one patient
k      number of free feature parameters
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

from head_to_head import WORK, apply_model, leave_one_out, ridge, score  # noqa: E402

BUNDLE = ROOT / "runs/segmenter/unet.pt"
PERMUTATIONS = 20000


def trimmed_mean(values, lo=10, hi=90):
    a, b = np.percentile(values, [lo, hi])
    kept = values[(values >= a) & (values <= b)]
    return float(kept.mean()) if kept.size else float(values.mean())


def feature_sets(rgb, mask):
    """Every candidate feature set, from one image and its mask."""

    selected = mask > 0
    pixels = rgb[selected].astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    r_chroma, g_chroma = R / total, G / total

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    redness = ((lab[:, :, 1] - 128.0) - 0.5 * (lab[:, :, 2] - 128.0))[selected].astype(np.float64)
    log_rg = np.log(np.maximum(R, 1) / np.maximum(G, 1))
    log_rb = np.log(np.maximum(R, 1) / np.maximum(B, 1))

    q25_r, q50_r, q75_r = np.percentile(r_chroma, [25, 50, 75])
    q25_red, q50_red, q75_red = np.percentile(redness, [25, 50, 75])

    return {
        # --- the current baseline ---
        "chroma mean (current)":      np.array([r_chroma.mean(), g_chroma.mean()]),
        "erythema mean":              np.array([log_rg.mean(), log_rb.mean()]),
        "redness mean":               np.array([redness.mean()]),
        # --- robust centre, same mask ---
        "chroma median":              np.array([q50_r, float(np.median(g_chroma))]),
        "chroma trimmed mean":        np.array([trimmed_mean(r_chroma), trimmed_mean(g_chroma)]),
        "erythema median":            np.array([float(np.median(log_rg)), float(np.median(log_rb))]),
        "redness median":             np.array([q50_red]),
        # --- adding spread ---
        "chroma mean + IQR":          np.array([r_chroma.mean(), g_chroma.mean(), q75_r - q25_r]),
        "chroma mean + std":          np.array([r_chroma.mean(), g_chroma.mean(), float(r_chroma.std())]),
        "redness mean + std":         np.array([redness.mean(), float(redness.std())]),
        # --- distribution shape ---
        "chroma quartiles":           np.array([q25_r, q50_r, q75_r]),
        "redness quartiles":          np.array([q25_red, q50_red, q75_red]),
        "redness p25 + p75":          np.array([q25_red, q75_red]),
    }


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def noise_floor(X, y, observed):
    rng = np.random.default_rng(0)
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        shuffled = rng.permutation(y)
        null[i] = r2_of(leave_one_out(X, shuffled), shuffled)
    return float(null.mean()), float((null >= observed).mean())


def main() -> None:
    learned = get_extractor("unet", model_dir=BUNDLE)
    per_patient: dict[str, dict] = {}

    for sample in load_local_cohort(ROOT):
        rgb = gray_world_white_balance(resize(read_rgb(sample.image_path), WORK))
        mask, _ = learned(rgb)
        if np.count_nonzero(mask) < 500:
            continue
        entry = per_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "f": {}})
        for label, vector in feature_sets(rgb, mask).items():
            entry["f"].setdefault(label, []).append(vector)

    labels = list(next(iter(per_patient.values()))["f"])
    patients = sorted((p for p, e in per_patient.items() if all(l in e["f"] for l in labels)),
                      key=lambda p: int(p.split(":")[1]))
    y = np.array([per_patient[p]["hb"] for p in patients])
    meta = [(per_patient[p]["age"], per_patient[p]["sex"]) for p in patients]
    n = len(y)

    baseline = score(np.array([np.delete(y, i).mean() for i in range(n)]), y, meta)
    print(f"{n} patients, pipeline C masks, leave-one-patient-out\n")
    print(f"{'feature set':<26}{'k':>3}{'MAE':>9}{'R²':>9}{'F1':>8}{'vs current':>12}")
    print("-" * 67)

    results = {}
    current_mae = None
    for label in labels:
        X = np.array([np.mean(per_patient[p]["f"][label], axis=0) for p in patients])
        m = score(leave_one_out(X, y), y, meta)
        results[label] = {**m, "k": int(X.shape[1])}
        if "current" in label:
            current_mae = m["mae"]

    for label in labels:
        m = results[label]
        delta = "" if current_mae is None else f"{current_mae - m['mae']:+12.3f}"
        print(f"{label:<26}{m['k']:>3}{m['mae']:>9.3f}{m['r2']:>9.3f}{m['f1']:>8.3f}{delta}")

    print("-" * 67)
    print(f"{'baseline (predict mean)':<26}{'-':>3}{baseline['mae']:>9.3f}"
          f"{baseline['r2']:>9.3f}{baseline['f1']:>8.3f}")

    ranked = sorted(results, key=lambda k: results[k]["mae"])
    print(f"\nbest by MAE: {ranked[0]}  (MAE {results[ranked[0]]['mae']:.3f}, "
          f"R² {results[ranked[0]]['r2']:+.3f})")

    # Noise floor for the top three and the current baseline, so the winner is
    # not quoted without knowing what chance would score with that many
    # parameters.
    print(f"\nnoise floor ({PERMUTATIONS:,} permutations, held out):")
    print(f"{'feature set':<26}{'k':>3}{'R²':>9}{'null mean':>11}{'p':>8}")
    print("-" * 57)
    floors = {}
    for label in dict.fromkeys(ranked[:3] + ["chroma mean (current)"]):
        X = np.array([np.mean(per_patient[p]["f"][label], axis=0) for p in patients])
        null_mean, p_value = noise_floor(X, y, results[label]["r2"])
        floors[label] = {"null_mean": null_mean, "p": p_value}
        print(f"{label:<26}{results[label]['k']:>3}{results[label]['r2']:>9.3f}"
              f"{null_mean:>11.3f}{p_value:>8.3f}")

    out = Path(__file__).resolve().parent / "feature_ablation.json"
    out.write_text(json.dumps({"n_patients": n, "baseline": baseline,
                               "results": results, "noise_floors": floors,
                               "ranked_by_mae": ranked}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
