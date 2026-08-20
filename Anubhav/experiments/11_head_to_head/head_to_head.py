"""Head-to-head: both pipelines, every feature representation, full metrics.

This is the deployment decision. Two pipelines, each producing a model, scored
on identical patients with identical protocol.

Feature representations tested (all small, because n = 26):

    means        mean R, G, B                      3 features
    chroma       r, g chromaticity = R/(R+G+B)     2 features, intensity-invariant
    erythema     log(R/G), log(R/B)                2 features, dermatology standard
    lab          mean a*, b*                        2 features
    a_only       mean a* alone                      1 feature, minimum variance

Chromaticity and erythema are included because they are *designed* to be
invariant to illumination intensity: dividing by the total removes any overall
brightness scaling. They are the physically motivated choices for measuring
tissue colour, and were not tested in earlier experiments.

Two evaluations are reported and they must not be confused:

  * EXPLORATORY — every representation scored by leave-one-out. Useful for
    understanding, but picking the winner from this table is selection on the
    test set and the winning number is optimistic.

  * NESTED — an inner loop chooses the representation using only training
    patients, the outer loop scores on a patient never used for that choice.
    This is the honest estimate of "what would we get if we let the pipeline
    pick its own features".

    python experiments/11_head_to_head/head_to_head.py
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

from anemia.data import anemia_threshold, is_anemic, load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402

import cielab_pipeline as cp  # noqa: E402

ALPHA = 1.0
WORK = (512, 512)
REPRESENTATIONS = ["means", "chroma", "erythema", "lab", "a_only"]


# ----------------------------------------------------------------- model

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


# --------------------------------------------------------------- metrics

def score(pred, true, meta):
    error = pred - true
    ss_res = float((error ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    truth = np.array([is_anemic(h, a, s) for h, (a, s) in zip(true, meta)])
    flag = np.array([p < anemia_threshold(a, s) for p, (a, s) in zip(pred, meta)])
    tp = int(np.count_nonzero(flag & truth)); fp = int(np.count_nonzero(flag & ~truth))
    tn = int(np.count_nonzero(~flag & ~truth)); fn = int(np.count_nonzero(~flag & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "mse": float((error ** 2).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mae": float(np.abs(error).mean()),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(error.mean()),
        "pearson_r": float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 1e-9 else float("nan"),
        "accuracy": (tp + tn) / len(true),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# -------------------------------------------------------------- features

def representations(rgb_image, mask):
    """Every feature representation, from one image and its mask."""

    selected = mask > 0
    pixels = rgb_image[selected].astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = np.maximum(R + G + B, 1e-6)

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = (lab[:, :, 1][selected] - 128.0).astype(np.float64)
    b_star = (lab[:, :, 2][selected] - 128.0).astype(np.float64)

    return {
        "means": np.array([R.mean(), G.mean(), B.mean()]),
        # chromaticity: divides out overall intensity, so a brighter capture of
        # the same tissue gives the same numbers
        "chroma": np.array([(R / total).mean(), (G / total).mean()]),
        # erythema index: the dermatology standard for redness, log-ratio based
        "erythema": np.array([np.log(np.maximum(R, 1) / np.maximum(G, 1)).mean(),
                              np.log(np.maximum(R, 1) / np.maximum(B, 1)).mean()]),
        "lab": np.array([a_star.mean(), b_star.mean()]),
        "a_only": np.array([a_star.mean()]),
    }


def collect():
    data: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
        rgb = resize(read_rgb(sample.image_path), WORK)

        balanced = gray_world_white_balance(rgb)
        mask_a, _ = heuristic_segmenter(balanced)

        norm_rgb, _ = cp.cielab_normalization(rgb)
        mask_b, _ = cp.segment_conjunctiva(norm_rgb)

        if not (mask_a > 0).any() or not (mask_b > 0).any():
            continue

        entry = data.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex,
            "A": [], "B": []})
        entry["A"].append(representations(balanced, mask_a))
        entry["B"].append(representations(norm_rgb, mask_b))
    return data


def stack(data, patients, pipeline, key):
    return np.array([np.mean([f[key] for f in data[p][pipeline]], axis=0)
                     for p in patients])


# ------------------------------------------------------------------ main

def main() -> None:
    data = collect()
    patients = sorted(data, key=lambda p: int(p.split(":")[1]))
    y = np.array([data[p]["hb"] for p in patients])
    meta = [(data[p]["age"], data[p]["sex"]) for p in patients]
    n = len(y)

    baseline = np.array([np.delete(y, i).mean() for i in range(n)])
    base = score(baseline, y, meta)

    print(f"{n} patients, both pipelines, leave-one-patient-out\n")

    # ---------- exploratory: every representation, both pipelines ----------
    print("=" * 78)
    print("1. EXPLORATORY — every feature representation, both pipelines")
    print("   (selecting the winner from this table is optimistic; see section 3)")
    print("=" * 78)
    print(f"\n{'representation':<14}{'k':>3}{'A: MAE':>9}{'A: R²':>9}{'A: F1':>8}"
          f"{'B: MAE':>9}{'B: R²':>9}{'B: F1':>8}")
    print("-" * 78)

    results: dict[str, dict] = {}
    for key in REPRESENTATIONS:
        row = {}
        for pipeline in ("A", "B"):
            X = stack(data, patients, pipeline, key)
            row[pipeline] = score(leave_one_out(X, y), y, meta)
            row[f"{pipeline}_k"] = X.shape[1]
        results[key] = row
        print(f"{key:<14}{row['A_k']:>3}{row['A']['mae']:>9.3f}{row['A']['r2']:>9.3f}"
              f"{row['A']['f1']:>8.3f}{row['B']['mae']:>9.3f}{row['B']['r2']:>9.3f}"
              f"{row['B']['f1']:>8.3f}")
    print("-" * 78)
    print(f"{'baseline':<14}{'-':>3}{base['mae']:>9.3f}{base['r2']:>9.3f}{base['f1']:>8.3f}"
          f"{base['mae']:>9.3f}{base['r2']:>9.3f}{base['f1']:>8.3f}")

    best_a = min(REPRESENTATIONS, key=lambda k: results[k]["A"]["mae"])
    best_b = min(REPRESENTATIONS, key=lambda k: results[k]["B"]["mae"])
    print(f"\nbest for A: {best_a}   best for B: {best_b}")

    # ---------- full metric table for the best of each ----------
    print("\n" + "=" * 78)
    print("2. FULL METRIC COMPARISON — best representation for each pipeline")
    print("=" * 78)
    mA, mB = results[best_a]["A"], results[best_b]["B"]
    print(f"\n{'metric':<18}{'Pipeline A':>16}{'Pipeline B':>16}{'baseline':>14}")
    print(f"{'':18}{'gray-world/' + best_a:>16}{'CIELAB/' + best_b:>16}{'':>14}")
    print("-" * 64)
    for key, label in [("mse", "MSE (g/dL)²"), ("rmse", "RMSE (g/dL)"), ("mae", "MAE (g/dL)"),
                       ("r2", "R²"), ("bias", "bias (g/dL)"), ("pearson_r", "Pearson r"),
                       ("accuracy", "Accuracy"), ("precision", "Precision"),
                       ("recall", "Recall"), ("specificity", "Specificity"), ("f1", "F1")]:
        print(f"{label:<18}{mA[key]:>16.3f}{mB[key]:>16.3f}{base.get(key, float('nan')):>14.3f}")
    print("-" * 64)
    print(f"{'confusion':<18}"
          f"{'TP%d FP%d FN%d TN%d' % (mA['tp'], mA['fp'], mA['fn'], mA['tn']):>16}"
          f"{'TP%d FP%d FN%d TN%d' % (mB['tp'], mB['fp'], mB['fn'], mB['tn']):>16}")

    # ---------- fitted on all patients (the deployed model) ----------
    print("\n" + "=" * 78)
    print("2b. FITTED ON ALL 26 PATIENTS — the deployed model, scored in-sample")
    print("=" * 78)
    print("\nThis is the model that ships: every patient contributes to the fit.")
    print("Scored on the same patients, so it describes the FIT, not what to")
    print("expect from a new patient (section 2 has the held-out numbers).\n")
    fitted = {}
    for pipeline, key in (("A", best_a), ("B", best_b)):
        X = stack(data, patients, pipeline, key)
        b0, b, mu, sd = ridge(X, y)
        fitted[pipeline] = score(apply_model(X, b0, b, mu, sd), y, meta)
    fA, fB = fitted["A"], fitted["B"]
    print(f"{'metric':<18}{'Pipeline A':>16}{'Pipeline B':>16}{'baseline':>14}")
    print("-" * 64)
    for key, label in [("mse", "MSE (g/dL)2"), ("rmse", "RMSE (g/dL)"), ("mae", "MAE (g/dL)"),
                       ("r2", "R2"), ("bias", "bias (g/dL)"), ("pearson_r", "Pearson r"),
                       ("accuracy", "Accuracy"), ("precision", "Precision"),
                       ("recall", "Recall"), ("specificity", "Specificity"), ("f1", "F1")]:
        print(f"{label:<18}{fA[key]:>16.3f}{fB[key]:>16.3f}{base.get(key, float('nan')):>14.3f}")
    print("-" * 64)
    print(f"{'confusion':<18}"
          f"{'TP%d FP%d FN%d TN%d' % (fA['tp'], fA['fp'], fA['fn'], fA['tn']):>16}"
          f"{'TP%d FP%d FN%d TN%d' % (fB['tp'], fB['fp'], fB['fn'], fB['tn']):>16}")

    # ---------- nested CV: the honest number ----------
    print("\n" + "=" * 78)
    print("3. NESTED CV — honest estimate when the pipeline picks its own features")
    print("=" * 78)
    print("\nInner loop selects the representation using training patients only;")
    print("outer loop scores a patient that selection never saw.\n")
    nested = {}
    for pipeline in ("A", "B"):
        Xs = {k: stack(data, patients, pipeline, k) for k in REPRESENTATIONS}
        pred = np.zeros(n)
        chosen = []
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            inner_idx = np.where(keep)[0]
            best_key, best_mae = None, np.inf
            for key in REPRESENTATIONS:
                X, errs = Xs[key], []
                for j in inner_idx:                    # inner leave-one-out
                    k2 = keep.copy(); k2[j] = False
                    p = apply_model(X[j:j + 1], *ridge(X[k2], y[k2]))[0]
                    errs.append(abs(p - y[j]))
                if np.mean(errs) < best_mae:
                    best_mae, best_key = np.mean(errs), key
            chosen.append(best_key)
            X = Xs[best_key]
            pred[i] = apply_model(X[i:i + 1], *ridge(X[keep], y[keep]))[0]
        m = score(pred, y, meta)
        nested[pipeline] = m
        counts = {k: chosen.count(k) for k in REPRESENTATIONS if chosen.count(k)}
        print(f"  Pipeline {pipeline}: MAE {m['mae']:.3f}  R² {m['r2']:+.3f}  F1 {m['f1']:.3f}")
        print(f"     representation chosen: {counts}")
    print(f"\n  baseline:    MAE {base['mae']:.3f}  R² {base['r2']:+.3f}  F1 {base['f1']:.3f}")

    # ---------- paired test between the pipelines ----------
    print("\n" + "=" * 78)
    print("4. IS THE DIFFERENCE BETWEEN PIPELINES REAL?")
    print("=" * 78)
    predA = leave_one_out(stack(data, patients, "A", best_a), y)
    predB = leave_one_out(stack(data, patients, "B", best_b), y)
    errA, errB = np.abs(predA - y), np.abs(predB - y)
    diff = errB - errA                       # positive => A better for that patient

    rng = np.random.default_rng(0)
    boot = [np.mean(diff[rng.integers(0, n, n)]) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = abs(diff.mean())
    perm = [abs((diff * rng.choice([-1, 1], n)).mean()) for _ in range(20000)]
    p_value = (np.count_nonzero(np.array(perm) >= observed) + 1) / (len(perm) + 1)

    wins = int(np.count_nonzero(diff > 0))
    print(f"\n  A has lower error on {wins} of {n} patients ({100 * wins / n:.0f}%)")
    print(f"  Mean advantage of A: {diff.mean():+.4f} g/dL")
    print(f"     bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"     paired permutation test p = {p_value:.3f}")
    print(f"\n  => {'A is measurably better' if p_value < 0.05 else 'NOT statistically separable at n=26'}")

    Path(Path(__file__).resolve().parent / "head_to_head.json").write_text(json.dumps({
        "exploratory": {k: {"A": results[k]["A"], "B": results[k]["B"]} for k in REPRESENTATIONS},
        "best": {"A": best_a, "B": best_b},
        "full_comparison_heldout": {"A": mA, "B": mB, "baseline": base},
        "fitted_on_all": {"A": fitted["A"], "B": fitted["B"], "baseline": base},
        "nested": nested,
        "paired": {"mean_advantage_A": float(diff.mean()), "ci95": [float(lo), float(hi)],
                   "wins_A": wins, "n": n, "p_value": float(p_value)},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {Path(__file__).resolve().parent / 'head_to_head.json'}")


if __name__ == "__main__":
    main()
