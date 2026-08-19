"""Diagnostic plots for the linear model on the real cohort.

Three panels, because each exposes a different thing and a single scatter can
hide two of them:

  A. Predicted vs actual — the honest headline. Perfect prediction lies on the
     diagonal; a model with no signal collapses onto a horizontal band near the
     cohort mean, which is what happens here.
  B. Bland-Altman — the clinical standard. Plots error against the average of
     the two measurements, revealing bias and whether error grows with the
     value.
  C. Residuals vs predicted — reveals systematic structure the other two miss.

    python experiments/06_metrics_evaluation/plot_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import anemia_threshold, load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402

INK, ACCENT, MUTED = "#1A1A1A", "#8C2F39", "#5A5A5A"


def leave_one_out_predictions():
    by_patient: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
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
        penalty = np.eye(Z.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y[keep])
        predictions[i] = np.hstack([[1.0], (X[i] - mu) / sd]) @ beta

    thresholds = np.array([anemia_threshold(by_patient[p]["age"], by_patient[p]["sex"])
                           for p in patients])
    return predictions, y, thresholds


def main() -> None:
    pred, true, thresholds = leave_one_out_predictions()
    error = pred - true
    mean_hb = true.mean()

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # ---- A. predicted vs actual ----
    ax = axes[0]
    low = min(true.min(), pred.min()) - 0.5
    high = max(true.max(), pred.max()) + 0.5
    ax.plot([low, high], [low, high], ls="--", lw=1.2, color=MUTED, label="perfect prediction")
    ax.axhline(mean_hb, ls=":", lw=1.2, color=ACCENT, label=f"cohort mean ({mean_hb:.1f})")
    anaemic = true < thresholds
    ax.scatter(true[~anaemic], pred[~anaemic], s=52, color="#4C72B0",
               edgecolor="white", linewidth=0.8, label="not anaemic", zorder=3)
    ax.scatter(true[anaemic], pred[anaemic], s=52, color=ACCENT,
               edgecolor="white", linewidth=0.8, label="anaemic", zorder=3)
    ax.set_xlabel("laboratory Hb (g/dL)")
    ax.set_ylabel("predicted Hb (g/dL)")
    r = np.corrcoef(pred, true)[0, 1]
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    ax.set_title(f"A. Predicted vs actual\nr = {r:+.2f},  R² = {1 - ss_res / ss_tot:+.3f}",
                 fontsize=10)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # ---- B. Bland-Altman ----
    ax = axes[1]
    average = (pred + true) / 2
    bias = error.mean()
    sd = error.std(ddof=1)
    ax.axhline(bias, color=ACCENT, lw=1.3, label=f"bias {bias:+.2f}")
    ax.axhline(bias + 1.96 * sd, ls="--", color=MUTED, lw=1.1,
               label=f"95% limits {bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f}")
    ax.axhline(bias - 1.96 * sd, ls="--", color=MUTED, lw=1.1)
    ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=0)
    ax.scatter(average, error, s=48, color="#4C72B0", edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.set_xlabel("mean of predicted and laboratory Hb (g/dL)")
    ax.set_ylabel("predicted − laboratory (g/dL)")
    ax.set_title("B. Bland–Altman agreement", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # ---- C. residuals vs predicted ----
    ax = axes[2]
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.scatter(pred, error, s=48, color="#4C72B0", edgecolor="white",
               linewidth=0.8, zorder=3)
    slope, intercept = np.polyfit(pred, error, 1)
    xs = np.linspace(pred.min(), pred.max(), 10)
    ax.plot(xs, slope * xs + intercept, ls="--", lw=1.2, color=ACCENT,
            label=f"trend (slope {slope:+.2f})")
    ax.set_xlabel("predicted Hb (g/dL)")
    ax.set_ylabel("residual (g/dL)")
    ax.set_title("C. Residual structure", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9)

    for ax in axes:
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)

    figure.suptitle(
        "Ridge regression on channel means, 26 patients, leave-one-patient-out "
        "— predictions collapse toward the cohort mean",
        fontsize=11)
    figure.tight_layout()
    out = Path(__file__).resolve().parent / "linear_model_diagnostics.png"
    figure.savefig(out, dpi=150)
    print(f"Wrote {out}")
    print(f"  prediction spread: {pred.std():.3f} g/dL   actual spread: {true.std():.3f} g/dL")
    print(f"  ratio {pred.std() / true.std():.3f}  (a model with no signal tends to 0)")


if __name__ == "__main__":
    main()
