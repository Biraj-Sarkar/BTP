"""Figure for the person-safe split comparison (report Section 8.10).

Reads the outputs of run.py here and of experiment 21; trains and scores nothing.

    python experiments/24_grouped_splits/figure.py

Left: held-out Dice of every architecture under experiment 21's splits (grey)
and under the person-safe splits of this experiment (blue). Small dots are single
splits; the larger dot and bar are the mean and its 95% interval. The order of the
architectures is the same; the blue dots spread further because each test set is
now a few runs of consecutive photographs rather than a random sprinkle.
Right: the leak estimate per architecture, (Mendeley - Eyes-defy) in experiment 21
minus the same gap here, with its 95% interval. Positive means Mendeley lost more
than the Eyes-defy control when repeat shots were kept on one side of the split.
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
EXP21 = HERE.parent / "21_repeated_splits"
OUT = HERE / "report_figure.png"

LABEL = {"attention": "Attention\nU-Net", "unetplusplus": "UNet++", "unet": "U-Net",
         "fpn": "FPN", "linknet": "LinkNet", "pspnet": "PSPNet"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OLD = "#9a9993"      # experiment 21: the reference, kept recessive
NEW = "#2a78d6"      # experiment 24
NUDGE = 0.17


def per_split(folder: Path) -> dict:
    dice = defaultdict(dict)
    for r in csv.DictReader(open(folder / "per_split.csv")):
        dice[r["arch"]][int(r["seed"])] = float(r["dice"])
    return dice


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, width=0.6, length=2.5)


def interval(summary: dict, arch: str):
    d = summary[arch]["dice"]
    return d["mean"], d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]


def main() -> None:
    old, new = per_split(EXP21), per_split(HERE)
    res21 = json.load(open(EXP21 / "results.json"))
    res24 = json.load(open(HERE / "results.json"))
    order = sorted(new, key=lambda a: -res24["summary"][a]["dice"]["mean"])
    x = list(range(len(order)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                      gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: Dice per split and mean, both experiments.
    for offset, dice, summary, colour, name in (
            (-NUDGE, old, res21["summary"], OLD, "Exp. 21: shots of one subject on both sides"),
            (+NUDGE, new, res24["summary"], NEW, "Exp. 24: person-safe splits")):
        for i, a in enumerate(order):
            pts = list(dice[a].values())
            left.scatter([i + offset - 0.07] * len(pts), pts, s=6, color=colour, alpha=0.55,
                         linewidths=0, zorder=2)
        m = [interval(summary, a) for a in order]
        left.errorbar([i + offset + 0.03 for i in x], [v[0] for v in m],
                      yerr=[[v[1] for v in m], [v[2] for v in m]], fmt="o", color=colour,
                      markersize=4, elinewidth=1.5, capsize=0, zorder=3, label=name)
    left.set_xticks(x, [LABEL[a] for a in order])
    left.set_ylabel("Held-out Dice", color=INK_2)
    left.set_title("Dice under both split designs", color=INK, fontsize=8, loc="left")
    left.legend(loc="lower left", frameon=False, fontsize=6.3, labelcolor=INK_2,
                handletextpad=0.3, borderaxespad=0.2)
    style(left)

    # Right: the leak estimate with its interval.
    leak = res24["leak"]
    right.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    for i, a in enumerate(order):
        est = leak[a]["leak_estimate"]
        v, (lo, hi) = est["value"], est["ci"]
        right.errorbar([i], [v], yerr=[[v - lo], [hi - v]], fmt="o", color=NEW, markersize=4,
                       elinewidth=1.5, capsize=0, zorder=3)
    right.set_xticks(x, [LABEL[a] for a in order])
    right.set_ylabel("Leak estimate (Dice)", color=INK_2)
    right.set_title("How much did shared subjects inflate Mendeley?", color=INK, fontsize=8,
                    loc="left")
    right.text(0.02, 0.97, "Mendeley's drop from exp. 21 minus the\nEyes-defy control's; "
               "dot = mean, bar = 95% CI", transform=right.transAxes, ha="left", va="top",
               color=INK_2, fontsize=6.3)
    style(right)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
