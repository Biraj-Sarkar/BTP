"""Does the learned extractor beat the classical one at predicting haemoglobin?

The obvious comparison is wrong. Running `anemia fit-linear` with each extractor
gives R² held-out +0.007 for the classical chain and +0.048 for the U-Net — but
the U-Net's thinner masks fail the quality gate more often, so those two numbers
are computed on **different patients** (26 vs 21). Their baselines differ too
(−0.082 vs −0.103), which is the tell. A model scored on an easier subset looks
better without being better.

This script fixes that: features are extracted with both extractors, the
patient set is restricted to those usable under **both**, and everything is
scored on that matched set. A permutation noise floor accompanies each R²,
because with k=2 free parameters on ~20 patients a fitted R² of +0.2 is not
obviously distinguishable from fitting noise.

Run: python experiments/13_learned_extractor/learned_vs_classical.py

Abbreviations
-------------
LOO      leave-one-out cross-validation, here leaving out one patient
R²       coefficient of determination; 0 means "no better than the mean"
k        number of free feature parameters in the model
IoU      intersection over union, an overlap measure between two masks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

ALPHA = 1.0
MIN_MASK_PIXELS = 500
PERMUTATIONS = 20000
SEED = 0


def ridge(X, y, alpha=ALPHA):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(y), 1)), (X - mu) / sd])
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    return beta[0], beta[1:], mu, sd


def apply_model(X, b0, b, mu, sd):
    return b0 + ((X - mu) / sd) @ b


def leave_one_out(X, y):
    out = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.ones(len(y), bool)
        keep[i] = False
        out[i] = apply_model(X[i:i + 1], *ridge(X[keep], y[keep]))[0]
    return out


def r2(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def erythema(rgb, mask):
    """log(R/G) and log(R/B), averaged over the mask — the deployed features."""

    pixels = rgb[mask > 0].astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return np.array([
        np.log(np.maximum(R, 1) / np.maximum(G, 1)).mean(),
        np.log(np.maximum(R, 1) / np.maximum(B, 1)).mean(),
    ])


def noise_floor(X, y, observed, rng, held_out=False):
    """What would this many parameters score on shuffled labels?

    An in-sample R² has no meaningful zero — a fitted model with an intercept
    essentially cannot score below the full-sample mean. The honest reference is
    the distribution obtained by destroying the feature/label relationship and
    refitting, which is what this measures.
    """

    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        shuffled = rng.permutation(y)
        if held_out:
            null[i] = r2(leave_one_out(X, shuffled), shuffled)
        else:
            null[i] = r2(apply_model(X, *ridge(X, shuffled)), shuffled)
    return {
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p": float((null >= observed).mean()),
    }


def main() -> int:
    extractors = {"refined": get_extractor("refined"),
                  "unet": get_extractor("unet", model_dir=ROOT / "runs/segmenter/unet.pt")}

    samples = load_local_cohort(ROOT)
    print(f"cohort: {len(samples)} captures")

    # Per patient, per extractor: the mean feature vector over usable eyes.
    per_patient: dict[str, dict] = {}
    ious = []

    for sample in samples:
        rgb = gray_world_white_balance(read_rgb(sample.image_path))
        entry = per_patient.setdefault(sample.patient_id,
                                       {"hb": sample.hb, "refined": [], "unet": []})
        masks = {}
        for name, extractor in extractors.items():
            mask, _ = extractor(rgb)
            masks[name] = mask
            if np.count_nonzero(mask) >= MIN_MASK_PIXELS:
                entry[name].append(erythema(rgb, mask))

        a, u = masks["refined"] > 0, masks["unet"] > 0
        if a.any() and u.any():
            ious.append(float((a & u).sum()) / float((a | u).sum()))

    matched = sorted(p for p, e in per_patient.items() if e["refined"] and e["unet"])
    only_refined = sorted(p for p, e in per_patient.items() if e["refined"] and not e["unet"])
    print(f"patients usable under both: {len(matched)}")
    print(f"patients lost to the U-Net's thinner masks: {len(only_refined)} {only_refined}")
    print(f"mask IoU between the two: median {np.median(ious):.3f}\n")

    y = np.array([per_patient[p]["hb"] for p in matched])
    rng = np.random.default_rng(SEED)
    results = {}

    print(f"{'extractor':12s} {'in-sample R2':>13s} {'null mean':>10s} {'p':>7s}"
          f" {'LOO R2':>9s} {'LOO p':>7s} {'MAE':>7s}")
    for name in extractors:
        X = np.array([np.mean(per_patient[p][name], axis=0) for p in matched])
        fitted = apply_model(X, *ridge(X, y))
        loo = leave_one_out(X, y)

        r2_fit, r2_loo = r2(fitted, y), r2(loo, y)
        floor_fit = noise_floor(X, y, r2_fit, np.random.default_rng(SEED))
        floor_loo = noise_floor(X, y, r2_loo, np.random.default_rng(SEED), held_out=True)

        results[name] = {"n": len(matched), "r2_in_sample": r2_fit, "r2_loo": r2_loo,
                         "mae_loo": float(np.abs(loo - y).mean()),
                         "in_sample_floor": floor_fit, "loo_floor": floor_loo}
        print(f"{name:12s} {r2_fit:13.3f} {floor_fit['null_mean']:10.3f} "
              f"{floor_fit['p']:7.3f} {r2_loo:9.3f} {floor_loo['p']:7.3f} "
              f"{results[name]['mae_loo']:7.3f}")

    baseline_loo = float(1 - (len(matched) / (len(matched) - 1)) ** 2)
    print(f"\nLOO baseline R² (predict the mean of the other n-1): {baseline_loo:.4f}")
    print(f"in-sample baseline R² (full-sample mean): 0.0000 by definition")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(
        {"matched_patients": matched, "lost_to_unet": only_refined,
         "mask_iou_median": float(np.median(ious)),
         "loo_baseline_r2": baseline_loo,
         "permutations": PERMUTATIONS, "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
