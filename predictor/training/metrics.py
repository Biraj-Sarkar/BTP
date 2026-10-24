"""Evaluation metrics for regression and clinical anemia screening."""

from __future__ import annotations

from typing import Dict, Any, Sequence, Optional
import numpy as np
from scipy import stats


def compute_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    anemia_cutoff: float = 12.0,
) -> Dict[str, Any]:
    """Calculates comprehensive continuous regression and clinical agreement metrics.
    
    All continuous metrics are reported in physical units (g/dL).
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_t) == 0:
        return {}

    errors = y_p - y_t
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    bias = float(np.mean(errors))
    std_diff = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    # Bland-Altman 95% Limits of Agreement
    loa_lower = float(bias - 1.96 * std_diff)
    loa_upper = float(bias + 1.96 * std_diff)

    # R-squared (Coefficient of Determination)
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(sq_errors))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

    # Pearson & Spearman correlation
    if len(y_t) > 2 and np.std(y_p) > 1e-8 and np.std(y_t) > 1e-8:
        pearson_r, pearson_p = stats.pearsonr(y_t, y_p)
        spearman_rho, spearman_p = stats.spearmanr(y_t, y_p)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0

    # Dummy baseline: MAE of predicting cohort mean
    baseline_mae = float(np.mean(np.abs(y_t - np.mean(y_t))))
    mae_vs_baseline = baseline_mae - mae  # Positive indicates improvement over dummy mean

    # Binary clinical screening metrics (WHO threshold)
    actual_anemic = (y_t < anemia_cutoff)
    pred_anemic = (y_p < anemia_cutoff)

    tp = int(np.sum(actual_anemic & pred_anemic))
    tn = int(np.sum((~actual_anemic) & (~pred_anemic)))
    fp = int(np.sum((~actual_anemic) & pred_anemic))
    fn = int(np.sum(actual_anemic & (~pred_anemic)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_t))
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "count": len(y_t),
        "mae_g_dl": round(mae, 3),
        "rmse_g_dl": round(rmse, 3),
        "bias_g_dl": round(bias, 3),
        "r2": round(r2, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 5),
        "spearman_rho": round(float(spearman_rho), 4),
        "loa_lower_g_dl": round(loa_lower, 3),
        "loa_upper_g_dl": round(loa_upper, 3),
        "baseline_mae_g_dl": round(baseline_mae, 3),
        "mae_improvement_vs_baseline": round(mae_vs_baseline, 3),
        "screening": {
            "cutoff_g_dl": anemia_cutoff,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "ppv": round(ppv, 3),
            "npv": round(npv, 3),
            "accuracy": round(accuracy, 3),
            "f1_score": round(f1, 3),
        },
    }

