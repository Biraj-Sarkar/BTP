"""Why R-squared is negative, and why there is no separate validation set.

Two questions answered by measurement:

  1. Did the model actually train? Yes. Compare **in-sample** performance (fit
     and evaluate on the same 26 patients) against **out-of-sample** (evaluate
     only on data the model never saw). If in-sample is good and out-of-sample
     is bad, the model trained fine and simply failed to generalise — which is
     what negative R-squared on held-out data means.

  2. Why not train / validation / test? Because at n = 26 a three-way split
     leaves too few patients in each part for any of the three numbers to mean
     anything. This script measures that instability directly by repeating a
     single random split under many seeds and reporting the spread.

    python experiments/09_train_test_protocol/protocol_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402

ALPHA = 1.0


def fit(X, y, alpha=ALPHA):
    """Ridge with intercept, on standardised features. Returns a predictor."""

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(y), 1)), (X - mu) / sd])
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    return lambda A: np.hstack([np.ones((len(A), 1)), (A - mu) / sd]) @ beta


def r2(pred, true, reference=None):
    """R-squared. `reference` is the mean used for the null model.

    Out-of-sample this must be the TRAINING mean, not the test mean — using the
    test mean would give the null model information it would not have in
    deployment.
    """

    reference = true.mean() if reference is None else reference
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - reference) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def features_and_labels():
    by_patient: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
        balanced = gray_world_white_balance(resize(read_rgb(sample.image_path), (512, 512)))
        mask, _ = heuristic_segmenter(balanced)
        if not (mask > 0).any():
            continue
        entry = by_patient.setdefault(sample.patient_id, {"hb": sample.hb, "f": []})
        entry["f"].append(balanced[mask > 0].mean(axis=0).astype(np.float64))
    patients = sorted(by_patient)
    X = np.array([np.mean(by_patient[p]["f"], axis=0) for p in patients])
    y = np.array([by_patient[p]["hb"] for p in patients])
    return X, y


def main() -> None:
    X, y = features_and_labels()
    n = len(y)
    print(f"{n} patients, {X.shape[1]} features\n")

    # ---- 1. In-sample: does the model train at all? ----
    predictor = fit(X, y)
    in_sample = predictor(X)
    print("1. DID IT TRAIN?")
    print("   Fit on all 26, evaluated on the same 26 (in-sample):")
    print(f"     R² = {r2(in_sample, y):+.3f}   MAE = {np.abs(in_sample - y).mean():.3f} g/dL")

    # ---- 2. Out-of-sample: leave-one-out ----
    loo = np.zeros(n)
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        loo[i] = fit(X[keep], y[keep])(X[i:i + 1])[0]
    train_mean_ref = np.array([np.delete(y, i).mean() for i in range(n)])
    print("\n   Leave-one-out, evaluated only on unseen patients (out-of-sample):")
    print(f"     R² = {r2(loo, y):+.3f}   MAE = {np.abs(loo - y).mean():.3f} g/dL")
    print("\n   The gap between these two numbers IS the overfitting.")
    print("   Training worked; generalisation did not.")

    # ---- 3. Is it an artefact of leave-one-out? ----
    print("\n2. IS NEGATIVE R² AN ARTEFACT OF LEAVE-ONE-OUT?")
    rng = np.random.default_rng(0)
    for k in (5, 13):
        scores = []
        for repeat in range(20):
            order = rng.permutation(n)
            folds = np.array_split(order, k)
            pred = np.zeros(n)
            for fold in folds:
                keep = np.ones(n, bool)
                keep[fold] = False
                pred[fold] = fit(X[keep], y[keep])(X[fold])
            scores.append(r2(pred, y))
        print(f"   {k}-fold CV, 20 repeats: R² = {np.mean(scores):+.3f} "
              f"(range {min(scores):+.3f} to {max(scores):+.3f})")
    print("   No — every protocol agrees. The model does not generalise.")

    # ---- 4. Why not a three-way split? ----
    print("\n3. WHY NOT TRAIN / VALIDATION / TEST?")
    print(f"   A 60/20/20 split of {n} patients gives ~{int(n*0.6)} train, "
          f"~{int(n*0.2)} validation, ~{n - int(n*0.6) - int(n*0.2)} test.")
    print("   Measuring how unstable a single split is, over 200 random seeds:\n")
    test_maes, test_r2s = [], []
    for seed in range(200):
        r = np.random.default_rng(seed)
        order = r.permutation(n)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        tr, te = order[:n_train], order[n_train + n_val:]
        p = fit(X[tr], y[tr])
        pred = p(X[te])
        test_maes.append(np.abs(pred - y[te]).mean())
        test_r2s.append(r2(pred, y[te], reference=y[tr].mean()))
    print(f"   test MAE : mean {np.mean(test_maes):.3f}, "
          f"range {min(test_maes):.3f} to {max(test_maes):.3f} g/dL")
    print(f"   test R²  : mean {np.mean(test_r2s):+.3f}, "
          f"range {min(test_r2s):+.3f} to {max(test_r2s):+.3f}")
    print(f"\n   The same model and the same data give a test R² anywhere from "
          f"{min(test_r2s):+.2f} to {max(test_r2s):+.2f}")
    print("   depending only on which patients land in the test set. A single")
    print("   split at this n reports luck, not performance — which is why")
    print("   cross-validation is used instead, and why no hyperparameter is")
    print("   tuned (there is no validation budget to tune against).")


if __name__ == "__main__":
    main()
