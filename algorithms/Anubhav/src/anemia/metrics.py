"""Regression, segmentation, and screening metrics.

Reported alongside a baseline on purpose. On a small, mean-heavy cohort a
regressor that ignores the image and always predicts the training mean already
achieves a respectable MAE, so an MAE quoted on its own says nothing about
whether the model learned anything. `regression_metrics` therefore always
carries `mae_vs_baseline`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def dice_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    pred = (np.asarray(pred_mask) > 0).astype(np.float32)
    target = (np.asarray(target_mask) > 0).astype(np.float32)
    intersection = float((pred * target).sum())
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    pred = (np.asarray(pred_mask) > 0).astype(np.float32)
    target = (np.asarray(target_mask) > 0).astype(np.float32)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum() - intersection)
    return float((intersection + smooth) / (union + smooth))


def regression_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    baseline: Optional[float] = None,
) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if pred.size == 0:
        return {"count": 0}

    errors = pred - true
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    metrics = {
        "count": int(pred.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }

    if pred.size > 1:
        # Bland-Altman limits of agreement — the standard way a non-invasive
        # device is compared against a lab reference in the clinical literature.
        metrics["loa_lower"] = float(np.mean(errors) - 1.96 * np.std(errors, ddof=1))
        metrics["loa_upper"] = float(np.mean(errors) + 1.96 * np.std(errors, ddof=1))
        if np.std(pred) > 1e-9 and np.std(true) > 1e-9:
            metrics["pearson_r"] = float(np.corrcoef(pred, true)[0, 1])

    if baseline is not None:
        baseline_mae = float(np.mean(np.abs(baseline - true)))
        metrics["mae_baseline"] = baseline_mae
        metrics["mae_vs_baseline"] = baseline_mae - metrics["mae"]

    return metrics


def screening_metrics(
    predicted_anemic: Sequence[bool],
    actual_anemic: Sequence[bool],
) -> Dict[str, float]:
    """Sensitivity-first summary.

    For a screening tool a missed anemic patient costs far more than a false
    alarm, so sensitivity and NPV matter more than headline accuracy — which is
    also the metric most inflated by an imbalanced cohort.
    """

    pred = np.asarray(predicted_anemic, dtype=bool)
    true = np.asarray(actual_anemic, dtype=bool)
    if pred.size == 0:
        return {"count": 0}

    tp = int(np.count_nonzero(pred & true))
    tn = int(np.count_nonzero(~pred & ~true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "count": int(pred.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, pred.size),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
    }


def inverse_frequency_weights(
    targets: Sequence[float],
    bins: int = 10,
    strength: float = 1.0,
) -> np.ndarray:
    """Deep-imbalanced-regression weights over binned continuous labels.

    Rare Hb ranges (severe anemia) get up-weighted so the network cannot settle
    on predicting the cohort mean. `strength` interpolates between uniform (0.0)
    and full inverse frequency (1.0); weights are normalised to mean 1 so the
    effective learning rate does not move when `bins` changes.
    """

    values = np.asarray(targets, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    if values.min() == values.max() or strength <= 0:
        return np.ones_like(values, dtype=np.float32)

    histogram, edges = np.histogram(values, bins=bins)
    bin_ids = np.clip(np.digitize(values, edges[1:-1], right=True), 0, bins - 1)
    counts = np.maximum(histogram[bin_ids], 1).astype(np.float64)

    weights = (1.0 / counts) ** strength
    weights *= weights.size / weights.sum()
    return weights.astype(np.float32)


def aggregate_folds(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean +/- std across CV folds — the number that belongs in the report."""

    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    summary: Dict[str, float] = {"folds": len(fold_metrics)}
    for key in keys:
        values = [float(m[key]) for m in fold_metrics if key in m and np.isfinite(float(m[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary
