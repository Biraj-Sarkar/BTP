"""Figure for the repeated-split comparison (report Section 8.7).

Reads the outputs of run.py; trains and scores nothing.

    python experiments/21_repeated_splits/figure.py

Left: held-out Dice of every architecture on each of the five splits. Each grey
line is one split, so a split that is easy for one architecture is visibly easy
for all of them; the blue dot and bar are the mean and its 95% interval.
Right: each architecture's Dice minus Attention U-Net's on the same split, which
removes that shared split effect. The bar is the corrected resampled interval
used for the test, not a plain t interval.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_figure.png"

LABEL = {"attention": "Attention\nU-Net", "unetplusplus": "UNet++", "unet": "U-Net",
         "fpn": "FPN", "linknet": "LinkNet", "pspnet": "PSPNet"}
REFERENCE = "attention"

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SPLIT = "#b9b8b2"
ACCENT = "#2a78d6"
NUDGE = 0.12  # mean and interval sit beside the per-split points, not on top of them


def load():
    rows = list(csv.DictReader(open(HERE / "per_split.csv")))
    dice = defaultdict(dict)
    for r in rows:
        dice[r["arch"]][int(r["seed"])] = float(r["dice"])
    results = json.load(open(HERE / "results.json"))
    return dice, results


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, width=0.6, length=2.5)


def main() -> None:
    dice, results = load()
    summary = results["summary"]
    seeds = results["complete_seeds"]
    order = sorted(dice, key=lambda a: -summary[a]["dice"]["mean"])
    x = range(len(order))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7),
                                      gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: Dice per split.
    for seed in seeds:
        left.plot(x, [dice[a][seed] for a in order], color=SPLIT, linewidth=0.8,
                  marker="o", markersize=2.6, zorder=2)
    means = [summary[a]["dice"]["mean"] for a in order]
    lo = [summary[a]["dice"]["mean"] - summary[a]["dice"]["ci_low"] for a in order]
    hi = [summary[a]["dice"]["ci_high"] - summary[a]["dice"]["mean"] for a in order]
    left.errorbar([i + NUDGE for i in x], means, yerr=[lo, hi], fmt="o", color=ACCENT, markersize=4.5,
                  elinewidth=1.6, capsize=0, zorder=3)
    left.set_xticks(list(x), [LABEL[a] for a in order])
    left.set_ylabel("Held-out Dice", color=INK_2)
    left.set_title("Dice on each of five splits", color=INK, fontsize=8, loc="left")
    left.text(0.99, 0.97, "grey line = one split\nblue = mean, 95% CI",
              transform=left.transAxes, ha="right", va="top", color=INK_2, fontsize=6.5)
    style(left)

    # Right: paired difference from the reference, per split.
    others = [a for a in order if a != REFERENCE]
    family = next(v for k, v in results["families"].items() if "Attention" in k)
    by_arch = {c["a"]: c for c in family if c["b"] == REFERENCE}
    right.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    for i, a in enumerate(others):
        diffs = [dice[a][s] - dice[REFERENCE][s] for s in seeds]
        right.scatter([i - NUDGE] * len(diffs), diffs, s=9, color=SPLIT, zorder=2, linewidths=0)
        c = by_arch[a]
        right.errorbar([i + NUDGE], [c["mean_diff"]],
                       yerr=[[c["mean_diff"] - c["ci"][0]], [c["ci"][1] - c["mean_diff"]]],
                       fmt="o", color=ACCENT, markersize=4.5, elinewidth=1.6, capsize=0, zorder=3)
    right.set_xticks(range(len(others)), [LABEL[a] for a in others])
    right.set_ylabel("Dice minus Attention U-Net", color=INK_2)
    right.set_title("Same split, paired difference", color=INK, fontsize=8, loc="left")
    right.text(0.02, 0.04, "grey dot = one split\nblue = mean, corrected 95% CI",
               transform=right.transAxes, ha="left", va="bottom", color=INK_2, fontsize=6.5)
    style(right)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
