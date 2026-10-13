from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "pearson_r": float(pearsonr(actual, predicted).statistic) if len(actual) > 1 else float("nan"),
        "spearman_r": float(spearmanr(actual, predicted).statistic) if len(actual) > 1 else float("nan"),
    }


def bland_altman_plot(actual, predicted, output_path: str | Path) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    means = (actual + predicted) / 2
    differences = predicted - actual
    bias, std = float(differences.mean()), float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    limits = (bias - 1.96 * std, bias + 1.96 * std)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        # Metrics remain usable in the lightweight training environment; the
        # optional figure is simply omitted when matplotlib is unavailable.
        print("matplotlib is not installed; skipping Bland-Altman figure")
        return {"bias": bias, "lower_limit": limits[0], "upper_limit": limits[1]}
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, differences, alpha=0.7)
    ax.axhline(bias, color="tab:red", label=f"Bias: {bias:.2f}")
    ax.axhline(limits[0], color="tab:gray", linestyle="--", label=f"95% limits: [{limits[0]:.2f}, {limits[1]:.2f}]")
    ax.axhline(limits[1], color="tab:gray", linestyle="--")
    ax.set(xlabel="Mean of laboratory and predicted Hb (g/dL)", ylabel="Predicted - laboratory Hb (g/dL)", title="Bland-Altman analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {"bias": bias, "lower_limit": limits[0], "upper_limit": limits[1]}
