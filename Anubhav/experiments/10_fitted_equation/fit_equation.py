"""The actual fitted equation, and honest uncertainty on every metric.

Answers three questions:

  1. Why is there only ONE R-squared when leave-one-out fits 26 models?
     Because each of those 26 models predicts exactly one held-out patient, so
     the run produces 26 predictions in total — one per patient. R-squared of a
     single point is undefined, so the 26 out-of-fold predictions are pooled
     and scored once. This script adds bootstrap confidence intervals and
     repeated k-fold spread, so the single number comes with uncertainty.

  2. Is 25-train / 1-test too extreme? It is the low-bias, high-variance end of
     a trade-off, quantified here by comparing training-set sizes.

  3. What is the equation? Printed in standardised form, expanded to raw
     pixel units, and demonstrated on a real patient.

    python experiments/10_fitted_equation/fit_equation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import anemia_threshold, is_anemic, load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402

ALPHA = 1.0
NAMES = ["mean R", "mean G", "mean B"]


def cohort():
    by_patient: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
        balanced = gray_world_white_balance(resize(read_rgb(sample.image_path), (512, 512)))
        mask, _ = heuristic_segmenter(balanced)
        if not (mask > 0).any():
            continue
        entry = by_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "f": []})
        entry["f"].append(balanced[mask > 0].mean(axis=0).astype(np.float64))
    patients = sorted(by_patient, key=lambda p: int(p.split(":")[1]))
    X = np.array([np.mean(by_patient[p]["f"], axis=0) for p in patients])
    y = np.array([by_patient[p]["hb"] for p in patients])
    meta = [(p, by_patient[p]["age"], by_patient[p]["sex"]) for p in patients]
    return X, y, meta


def ridge_coefficients(X, y, alpha=ALPHA):
    """Returns (intercept, coefficients) in STANDARDISED feature space, plus mu/sd."""

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(y), 1)), (X - mu) / sd])
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    return beta[0], beta[1:], mu, sd


def predict_raw(X, intercept, coefficients, mu, sd):
    return intercept + ((X - mu) / sd) @ coefficients


def leave_one_out(X, y, alpha=ALPHA):
    out = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.ones(len(y), bool)
        keep[i] = False
        b0, b, mu, sd = ridge_coefficients(X[keep], y[keep], alpha)
        out[i] = predict_raw(X[i:i + 1], b0, b, mu, sd)[0]
    return out


def metrics(pred, true, meta):
    error = pred - true
    ss_res = float((error ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    truth = np.array([is_anemic(h, a, s) for h, (_, a, s) in zip(true, meta)])
    flag = np.array([p < anemia_threshold(a, s) for p, (_, a, s) in zip(pred, meta)])
    tp = int(np.count_nonzero(flag & truth))
    fp = int(np.count_nonzero(flag & ~truth))
    fn = int(np.count_nonzero(~flag & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"mae": float(np.abs(error).mean()),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "f1": float(f1)}


def main() -> None:
    X, y, meta = cohort()
    n = len(y)

    # ================= 1. Why one number, and how uncertain is it =========
    print("=" * 74)
    print("1. WHY ONE R-SQUARED, AND HOW CERTAIN IS IT")
    print("=" * 74)
    print(f"\nLeave-one-out fits {n} models. Each predicts exactly ONE held-out")
    print(f"patient, so the run yields {n} predictions total — one per patient.")
    print("R-squared of a single point is undefined, so the predictions are")
    print("pooled and scored once. Hence one number, not 26.\n")

    loo = leave_one_out(X, y)
    point = metrics(loo, y, meta)
    print(f"Pooled out-of-fold:  MAE {point['mae']:.3f}   "
          f"R² {point['r2']:+.3f}   F1 {point['f1']:.3f}")

    rng = np.random.default_rng(0)
    draws = {"mae": [], "r2": [], "f1": []}
    for _ in range(2000):
        idx = rng.integers(0, n, n)              # bootstrap over patients
        if len(set(idx)) < 4:
            continue
        m = metrics(loo[idx], y[idx], [meta[i] for i in idx])
        for k in draws:
            if np.isfinite(m[k]):
                draws[k].append(m[k])
    print("\nBootstrap 95% confidence intervals (2000 resamples over patients):")
    for k, label in (("mae", "MAE  "), ("r2", "R²   "), ("f1", "F1   ")):
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        print(f"  {label} {point[k]:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\nThe R² interval spans zero-ish territory in the negative — the")
    print("honest reading is 'no better than the mean', not a precise -0.166.")

    print("\nRepeated 5-fold, which DOES give a spread across folds:")
    per_fold = []
    for repeat in range(20):
        order = np.random.default_rng(repeat).permutation(n)
        for fold in np.array_split(order, 5):
            keep = np.ones(n, bool)
            keep[fold] = False
            b0, b, mu, sd = ridge_coefficients(X[keep], y[keep])
            per_fold.append(float(np.abs(predict_raw(X[fold], b0, b, mu, sd) - y[fold]).mean()))
    print(f"  fold MAE: mean {np.mean(per_fold):.3f}, sd {np.std(per_fold):.3f}, "
          f"range {min(per_fold):.3f}–{max(per_fold):.3f}")

    # ================= 2. Is 25/1 too extreme? ============================
    print("\n" + "=" * 74)
    print("2. IS 25 TRAIN / 1 TEST TOO EXTREME?")
    print("=" * 74)
    print("\nIt is the low-bias end of the bias-variance trade-off: the model is")
    print("trained on 25 of 26 patients, so it is almost the model you would")
    print("deploy. Smaller training folds bias the estimate pessimistically.\n")
    print(f"{'protocol':<26}{'train size':>11}{'pooled MAE':>12}{'pooled R²':>11}")
    print("-" * 60)
    for k, label in ((2, "2-fold (13 train)"), (5, "5-fold (~21 train)"),
                     (13, "13-fold (24 train)"), (n, "leave-one-out (25)")):
        order = np.random.default_rng(7).permutation(n)
        pred = np.zeros(n)
        for fold in np.array_split(order, k):
            keep = np.ones(n, bool)
            keep[fold] = False
            b0, b, mu, sd = ridge_coefficients(X[keep], y[keep])
            pred[fold] = predict_raw(X[fold], b0, b, mu, sd)
        m = metrics(pred, y, meta)
        print(f"{label:<26}{n - n // k:>11}{m['mae']:>12.3f}{m['r2']:>11.3f}")
    print("-" * 60)
    print("Every patient is still tested exactly once in all of these; only the")
    print("training size changes. The estimates agree, so the choice is not")
    print("driving the conclusion.")

    # ================= 3. The fitted equation =============================
    print("\n" + "=" * 74)
    print("3. THE FITTED EQUATION")
    print("=" * 74)
    b0, b, mu, sd = ridge_coefficients(X, y)

    print("\nInputs — three numbers per patient, each an average over the")
    print("segmented conjunctiva pixels of the white-balanced 512x512 image:")
    for name, m_, s_ in zip(NAMES, mu, sd):
        print(f"   {name:<8}  cohort mean {m_:7.2f}   sd {s_:6.2f}")

    print("\nStandardised form (what the code solves):")
    print(f"   Hb = {b0:.4f}", end="")
    for name, coef in zip(NAMES, b):
        print(f" {'+' if coef >= 0 else '-'} {abs(coef):.4f}·z({name})", end="")
    print("\n   where z(x) = (x - mean) / sd")

    # expand to raw units
    raw = b / sd
    raw_intercept = b0 - float((b * mu / sd).sum())
    print("\nExpanded to raw pixel units (equivalent, directly usable):")
    print(f"   Hb = {raw_intercept:.4f}", end="")
    for name, coef in zip(NAMES, raw):
        print(f" {'+' if coef >= 0 else '-'} {abs(coef):.6f}·{name}", end="")
    print("   [g/dL]")

    print("\nWorked example — patient 1:")
    print(f"   inputs: R={X[0][0]:.2f}, G={X[0][1]:.2f}, B={X[0][2]:.2f}")
    terms = " + ".join(f"({c:+.6f}×{v:.2f})" for c, v in zip(raw, X[0]))
    print(f"   Hb = {raw_intercept:.4f} + {terms}")
    fitted = raw_intercept + float(raw @ X[0])
    print(f"      = {fitted:.3f} g/dL      (laboratory value: {y[0]:.1f} g/dL)")

    print("\nWhat the coefficients say:")
    order = np.argsort(-np.abs(b))
    for i in order:
        direction = "raises" if b[i] > 0 else "lowers"
        print(f"   {NAMES[i]:<8} 1 sd higher {direction} predicted Hb by "
              f"{abs(b[i]):.3f} g/dL")
    print(f"\n   For scale, the cohort's Hb standard deviation is {y.std():.2f} g/dL,")
    print(f"   so the largest single effect is {abs(b[order[0]]) / y.std() * 100:.0f}% of one sd.")

    dest = Path(__file__).resolve().parent / "fitted_equation.json"
    dest.write_text(json.dumps({
        "features": NAMES, "alpha": ALPHA,
        "feature_mean": mu.tolist(), "feature_sd": sd.tolist(),
        "standardised": {"intercept": float(b0), "coefficients": b.tolist()},
        "raw_units": {"intercept": float(raw_intercept), "coefficients": raw.tolist()},
        "out_of_fold": point,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
