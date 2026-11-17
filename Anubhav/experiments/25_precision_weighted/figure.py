"""Figure for the precision-weighted rescoring (report Section 8.11).

Reads run.py's outputs here; scores nothing.

    python experiments/25_precision_weighted/figure.py

Left: how the four scores move as the mask threshold rises, averaged over the
models whose probabilities reach 1 (29 of 30; one Attention U-Net never rises
above 0.53, see RESULTS.md). The x-axis is stretched near 0 and 1 (logit scale)
because the action is between 0.9 and 0.999. Precision climbs and recall falls;
Dice is flat around 0.5 and only falls late; F0.5 keeps rising to about 0.97.
Right: test F0.5 of each architecture at the default threshold of 0.5 (grey)
and at its own validation-chosen threshold (blue). Small dots are single splits;
the larger dot and bar are the mean and its 95% interval.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_figure.png"

LABEL = {"attention": "Attention\nU-Net", "unetplusplus": "UNet++", "unet": "U-Net",
         "fpn": "FPN", "linknet": "LinkNet", "pspnet": "PSPNet"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OLD, NEW = "#9a9993", "#2a78d6"   # same pair as experiment 24's figure
NUDGE, EPS, BETA = 0.17, 1e-6, 0.5


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, width=0.6, length=2.5)


def curves(counts: dict, thresholds: np.ndarray) -> dict:
    """Mean per-image score at each threshold, then mean over models."""

    per_model = defaultdict(list)
    for key, c in counts.items():
        if not key.endswith("_test"):
            continue
        tp, fp, fn = (c[..., k].astype(float) for k in range(3))
        if ((tp + fp)[:, -1] == 0).all():      # never reaches the top threshold
            continue
        b2 = BETA ** 2
        per_model["F0.5"].append((((1 + b2) * tp + EPS) / ((1 + b2) * tp + b2 * fn + fp + EPS)).mean(0))
        per_model["Dice"].append(((2 * tp + EPS) / (2 * tp + fp + fn + EPS)).mean(0))
        per_model["Precision"].append(((tp + EPS) / (tp + fp + EPS)).mean(0))
        per_model["Recall"].append(((tp + EPS) / (tp + fn + EPS)).mean(0))
    n = len(per_model["F0.5"])
    return {k: np.mean(v, axis=0) for k, v in per_model.items()}, n


def main() -> None:
    res = json.load(open(HERE / "results.json"))
    saved = np.load(HERE / "pixel_counts.npz")
    thresholds = saved["thresholds"]
    counts = {k: saved[k] for k in saved.files if k != "thresholds"}
    mean_curves, n_models = curves(counts, thresholds)

    by = defaultdict(dict)
    for r in csv.DictReader(open(HERE / "per_split.csv")):
        by[r["arch"]][int(r["seed"])] = r
    s = res["summary"]
    order = sorted(s, key=lambda a: -s[a]["f05_chosen"]["mean"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                      gridspec_kw={"width_ratios": [1, 1.05]})

    # Left: the trade-off as the threshold rises.
    look = {"F0.5": dict(color=NEW, lw=2.0, ls="-"), "Dice": dict(color=OLD, lw=1.6, ls="-"),
            "Precision": dict(color=INK_2, lw=1.0, ls="--"),
            "Recall": dict(color=INK_2, lw=1.0, ls=":")}
    x = thresholds
    for name, kw in look.items():
        y = mean_curves[name]
        left.plot(x, y, **kw, zorder=3)
        if name == "Recall":     # leaves the panel at the bottom; label it near the start
            i = int(np.flatnonzero(np.isclose(x, 0.15))[0])
            left.text(x[i], y[i] + 0.012, name, color=INK_2, fontsize=6.5, va="bottom", ha="left")
        else:
            left.text(x[-1] + 0.0002, y[-1], f" {name}", color=INK_2, fontsize=6.5,
                      va="center", ha="left")
    chosen = float(np.median([t for a in s for t in s[a]["thresholds"]]))
    for t, label in ((0.5, "default"), (chosen, "chosen")):   # chosen = median over 30 models
        left.axvline(t, color=INK_2, lw=0.6, ls="-", zorder=1)
        left.text(t, 0.995, f" {label}", transform=left.get_xaxis_transform(), color=INK_2,
                  fontsize=6.3, rotation=90, va="top", ha="right")
    left.set_xscale("logit")
    ticks = [0.1, 0.5, 0.9, 0.97, 0.99, 0.999]
    left.set_xticks(ticks, [f"{t:g}" for t in ticks])
    left.xaxis.set_minor_formatter(plt.NullFormatter())
    left.set_xlim(0.05, 0.999)
    left.set_ylim(0.55, 1.0)
    left.set_xlabel("Mask threshold (probability)", color=INK_2)
    left.set_ylabel("Test score", color=INK_2)
    left.set_title(f"Higher threshold: precision up, recall down ({n_models} models)",
                   color=INK, fontsize=8, loc="left")
    style(left)

    # Right: F0.5 at 0.5 and at each model's chosen threshold.
    xs = list(range(len(order)))
    for offset, key, colour, name in ((-NUDGE, "f05_default", OLD, "Threshold 0.5"),
                                      (+NUDGE, "f05_chosen", NEW, "Threshold chosen on validation")):
        for i, a in enumerate(order):
            pts = [float(r[key]) for r in by[a].values()]
            right.scatter([i + offset - 0.07] * len(pts), pts, s=6, color=colour, alpha=0.55,
                          linewidths=0, zorder=2)
        m = [s[a][key] for a in order]
        right.errorbar([i + offset + 0.03 for i in xs], [d["mean"] for d in m],
                       yerr=[[d["mean"] - d["ci_low"] for d in m], [d["ci_high"] - d["mean"] for d in m]],
                       fmt="o", color=colour, markersize=4, elinewidth=1.5, capsize=0, zorder=3,
                       label=name)
    right.set_xticks(xs, [LABEL[a] for a in order], fontsize=6.5)
    right.set_ylabel("Test F0.5", color=INK_2)
    right.set_title("F0.5 per architecture", color=INK, fontsize=8, loc="left")
    right.legend(loc="lower left", frameon=False, fontsize=6.3, labelcolor=INK_2,
                 handletextpad=0.3, borderaxespad=0.2)
    style(right)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
