"""Full metric suite for both the regression and the screening decision.

Two evaluations are produced, and they must never be confused:

  A. SYNTHETIC — the ResNet-18 checkpoint trained on generated phantoms,
     scored on out-of-fold phantom predictions. Real numbers, zero clinical
     meaning: the phantoms are flat-coloured shapes whose redness is a
     deterministic function of Hb, so the task is trivially learnable.
     Included because it verifies the metric code and shows what the pipeline
     reports when a model genuinely fits.

  B. REAL — the 26-patient local cohort with laboratory Hb. No CNN has been
     trained on real data, so the model here is the ridge regression over
     channel means from experiment 05, evaluated leave-one-patient-out. It is
     the only honest predictor currently available for real images.

Every regression metric is paired with the predict-the-mean baseline, and every
classification metric with a majority-class baseline, because on small skewed
cohorts both baselines are deceptively strong.

    python experiments/06_metrics_evaluation/evaluate.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import anemia_threshold, is_anemic, load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402


# --------------------------------------------------------------------------
# Metric definitions, written out rather than imported, so the arithmetic is
# visible and checkable against the report.
# --------------------------------------------------------------------------

def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Error metrics for a continuous prediction, in g/dL unless noted."""

    error = pred - true                       # signed: positive = over-estimate
    abs_error = np.abs(error)
    squared = error ** 2

    ss_res = float(squared.sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())

    metrics = {
        "n": int(pred.size),
        # Mean Squared Error: average of squared errors. Units are (g/dL)^2,
        # which is why it is rarely quoted directly in clinical work — it
        # punishes large errors disproportionately, useful as a loss, awkward
        # to interpret.
        "mse": float(squared.mean()),
        # Root Mean Squared Error: sqrt(MSE), back in g/dL. Still dominated by
        # the worst cases.
        "rmse": float(np.sqrt(squared.mean())),
        # Mean Absolute Error: average |error|. The number a clinician can read
        # directly as "typically wrong by this much".
        "mae": float(abs_error.mean()),
        # Median absolute error: robust to a single catastrophic case.
        "median_ae": float(np.median(abs_error)),
        # Bias: mean signed error. Near zero means errors cancel; a large
        # positive value means the model systematically over-estimates Hb,
        # which for a screening tool means systematically missing anaemia.
        "bias": float(error.mean()),
        # Coefficient of determination: fraction of variance explained.
        # Negative means worse than predicting the mean.
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "max_abs_error": float(abs_error.max()),
    }
    if pred.size > 1:
        sd = float(error.std(ddof=1))
        # Bland-Altman limits of agreement: the interval containing ~95% of
        # errors. This is the standard way a new device is compared against a
        # laboratory reference.
        metrics["loa_lower"] = metrics["bias"] - 1.96 * sd
        metrics["loa_upper"] = metrics["bias"] + 1.96 * sd
        if pred.std() > 1e-9 and true.std() > 1e-9:
            metrics["pearson_r"] = float(np.corrcoef(pred, true)[0, 1])
    return metrics


def classification_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Screening metrics. Positive class = ANAEMIC (the case we must catch)."""

    tp = int(np.count_nonzero(pred & true))
    tn = int(np.count_nonzero(~pred & ~true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    # Recall / sensitivity: of all truly anaemic patients, what fraction did we
    # catch? The metric that matters most here - a false negative sends an
    # unwell patient home.
    recall = ratio(tp, tp + fn)
    # Precision / PPV: of everyone we flagged, what fraction really was
    # anaemic? Low precision means wasted confirmatory blood tests.
    precision = ratio(tp, tp + fp)
    # F1: harmonic mean of precision and recall. The harmonic mean is used so
    # that a model cannot score well by sacrificing one for the other.
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 and np.isfinite(precision + recall) else 0.0)

    return {
        "n": int(pred.size),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": ratio(tp + tn, pred.size),
        "precision_ppv": precision,
        "recall_sensitivity": recall,
        "specificity": ratio(tn, tn + fp),
        "npv": ratio(tn, tn + fn),
        "f1": float(f1),
        # Balanced accuracy: mean of sensitivity and specificity. Immune to
        # class imbalance, unlike plain accuracy.
        "balanced_accuracy": float(np.nanmean([recall, ratio(tn, tn + fp)])),
        "prevalence": ratio(int(np.count_nonzero(true)), pred.size),
    }


# --------------------------------------------------------------------------

def evaluate_synthetic() -> dict | None:
    path = ROOT.parent.parent / "Desktop/BTP-AnemiaDetection/runs/smoke/out_of_fold_predictions.csv"
    if not path.exists():
        path = Path.home() / "Desktop/BTP-AnemiaDetection/runs/smoke/out_of_fold_predictions.csv"
    if not path.exists():
        return None

    rows = list(csv.DictReader(path.open()))
    true_hb = np.array([float(r["actual_hb"]) for r in rows])
    pred_hb = np.array([float(r["predicted_hb"]) for r in rows])
    true_an = np.array([r["actual_anemic"] == "True" for r in rows])
    pred_an = np.array([r["predicted_anemic"] == "True" for r in rows])

    baseline_hb = np.full_like(true_hb, true_hb.mean())
    majority = np.full_like(true_an, np.count_nonzero(true_an) > len(true_an) / 2)

    return {
        "regression": regression_metrics(pred_hb, true_hb),
        "regression_baseline": regression_metrics(baseline_hb, true_hb),
        "classification": classification_metrics(pred_an, true_an),
        "classification_baseline": classification_metrics(majority, true_an),
    }


def evaluate_real() -> dict:
    """Ridge over channel means, leave-one-patient-out. See experiment 05."""

    samples = load_local_cohort(ROOT)
    by_patient: dict[str, dict] = {}
    for sample in samples:
        rgb = resize(read_rgb(sample.image_path), (512, 512))
        balanced = gray_world_white_balance(rgb)
        mask, _ = heuristic_segmenter(balanced)
        if not (mask > 0).any():
            continue
        entry = by_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "feats": []})
        entry["feats"].append(balanced[mask > 0].mean(axis=0).astype(np.float64))

    patients = sorted(by_patient)
    X = np.array([np.mean(by_patient[p]["feats"], axis=0) for p in patients])
    y = np.array([by_patient[p]["hb"] for p in patients])

    predictions = np.zeros_like(y)
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        mu, sd = X[keep].mean(axis=0), X[keep].std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = np.hstack([np.ones((keep.sum(), 1)), (X[keep] - mu) / sd])
        penalty = np.eye(Z.shape[1]); penalty[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y[keep])
        predictions[i] = np.hstack([[1.0], (X[i] - mu) / sd]) @ beta

    # Thresholds are per patient (age/sex), so the classification is derived
    # exactly as the deployed app would derive it.
    true_an = np.array([is_anemic(by_patient[p]["hb"], by_patient[p]["age"],
                                  by_patient[p]["sex"]) for p in patients])
    pred_an = np.array([predictions[i] < anemia_threshold(by_patient[p]["age"],
                                                          by_patient[p]["sex"])
                        for i, p in enumerate(patients)])

    baseline_hb = np.array([np.delete(y, i).mean() for i in range(len(y))])
    baseline_an = np.array([baseline_hb[i] < anemia_threshold(by_patient[p]["age"],
                                                              by_patient[p]["sex"])
                            for i, p in enumerate(patients)])

    return {
        "regression": regression_metrics(predictions, y),
        "regression_baseline": regression_metrics(baseline_hb, y),
        "classification": classification_metrics(pred_an, true_an),
        "classification_baseline": classification_metrics(baseline_an, true_an),
    }


def show(title: str, result: dict, caveat: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{caveat}\n{'=' * 74}")

    r, rb = result["regression"], result["regression_baseline"]
    print(f"\nREGRESSION  (n = {r['n']})")
    print(f"  {'metric':<22}{'model':>12}{'baseline':>12}   interpretation")
    rows = [
        ("MSE  (g/dL)^2", "mse", "lower is better"),
        ("RMSE (g/dL)", "rmse", "typical error, outlier-sensitive"),
        ("MAE  (g/dL)", "mae", "typical error, readable"),
        ("Median AE (g/dL)", "median_ae", "robust typical error"),
        ("Bias (g/dL)", "bias", "+ve = over-estimates Hb = misses anaemia"),
        ("R^2", "r2", "1 = perfect, 0 = no better than mean"),
        ("Max |error| (g/dL)", "max_abs_error", "worst single case"),
        ("Pearson r", "pearson_r", "linear agreement with lab"),
        ("LoA lower (g/dL)", "loa_lower", "95% of errors fall"),
        ("LoA upper (g/dL)", "loa_upper", "  between these"),
    ]
    for label, key, note in rows:
        m = r.get(key, float("nan")); b = rb.get(key, float("nan"))
        print(f"  {label:<22}{m:>12.3f}{b:>12.3f}   {note}")

    c, cb = result["classification"], result["classification_baseline"]
    print(f"\nSCREENING  (n = {c['n']}, prevalence {c['prevalence']:.2f}, positive class = anaemic)")
    print(f"  confusion:  TP {c['tp']}   FN {c['fn']}   FP {c['fp']}   TN {c['tn']}")
    print(f"  {'metric':<22}{'model':>12}{'baseline':>12}")
    for label, key in [("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
                       ("Precision (PPV)", "precision_ppv"), ("Recall (sensitivity)", "recall_sensitivity"),
                       ("Specificity", "specificity"), ("NPV", "npv"), ("F1", "f1")]:
        print(f"  {label:<22}{c.get(key, float('nan')):>12.3f}{cb.get(key, float('nan')):>12.3f}")


def main() -> None:
    out = {}

    synthetic = evaluate_synthetic()
    if synthetic:
        show("A. SYNTHETIC PHANTOMS — ResNet-18 checkpoint, out-of-fold",
             synthetic,
             "NOT a clinical result. Phantom colour is a deterministic function of Hb.")
        out["synthetic"] = synthetic

    real = evaluate_real()
    show("B. REAL COHORT — 26 patients, ridge over channel means, leave-one-out",
         real,
         "The only honest predictor available for real images today.")
    out["real"] = real

    dest = Path(__file__).resolve().parent / "metrics.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
