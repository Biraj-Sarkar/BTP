"""Figure for the colour-correction pre-checks (experiment 26).

Reads run.py's outputs here; measures nothing.

    python experiments/26_correction_prechecks/figure.py

Left: how far each estimate of the light is from the device's white plastic
(angle between colours, median and interquartile range). The brightest pixels
are taken above the lower lid, so the plastic itself is not among them.
Middle: sclera yellowness against age, with the light removed (sclera b* minus
plastic b*), one line per site.
Right: the two Italian capture batches differ in haemoglobin and in the colour
of the light (plastic log R/G); within the dated batch the plastic does not
follow haemoglobin.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_figure.png"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE = "#9a9993", "#2a78d6"     # same pair as experiments 24 and 25
PLASTIC_MIN = 3000


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=6.5, width=0.6, length=2.5)


def angle(u, v):
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main() -> None:
    res = json.load(open(HERE / "results.json"))
    df = pd.read_csv(HERE / "measurements.csv")
    grading = json.load(open(HERE / "sclera_grading.json"))
    df = df[~df.patient.isin(grading["fail"]) & (df.plastic_px >= PLASTIC_MIN)].copy()
    dated = df.file.str.contains(r"20\d{6}", regex=True)
    df["batch"] = np.where(df.site == "india", "india", np.where(dated, "dated", "numbered"))
    rgb = lambda name: df[[f"{name}_{c}" for c in "RGB"]].to_numpy(float)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # Left: angle to the plastic.
    ref = rgb("plastic")
    arms = [("No\ncorrection", np.ones_like(ref)), ("Gray-\nworld", rgb("gray_world")),
            ("Max-\nRGB", rgb("max_rgb")), ("Sclera", rgb("sclera")),
            ("Brightest\nabove lid", rgb("bright_eye"))]
    for k, (site, colour, off) in enumerate((("india", BLUE, -0.13), ("italy", GREY, 0.13))):
        m = (df.site == site).to_numpy()
        for j, (_, est) in enumerate(arms):
            a = angle(est[m], ref[m])
            q1, med, q3 = np.percentile(a, [25, 50, 75])
            a1.plot([j + off] * 2, [q1, q3], color=colour, lw=1.6, solid_capstyle="butt")
            a1.plot(j + off, med, "o", color=colour, ms=3.6, label=site.capitalize() if j == 0 else None)
    a1.set_xticks(range(len(arms)), [n for n, _ in arms], fontsize=6.0)
    a1.set_ylabel("Angle to the plastic (degrees)", color=INK_2)
    a1.set_ylim(0, None)
    a1.set_title("Estimates of the light", color=INK, fontsize=8, loc="left")
    a1.legend(frameon=False, fontsize=6.3, labelcolor=INK_2, loc="upper right", handletextpad=0.2)
    style(a1)

    # Middle: sclera b* minus plastic b* against age.
    df["b_rel"] = df.sclera_b - df.plastic_b
    for site, colour, marker in (("india", BLUE, "o"), ("italy", GREY, "s")):
        d = df[df.site == site]
        a2.scatter(d.age, d.b_rel, s=6, color=colour, alpha=0.6, linewidths=0, marker=marker)
        slope, icpt = np.polyfit(d.age, d.b_rel, 1)
        xs = np.array([d.age.min(), d.age.max()])
        a2.plot(xs, icpt + slope * xs, color=colour, lw=1.4)
        o = res["sclera"]["per_site"][site]["ols_b_rel"]["per_decade"]
        a2.plot([], [], color=colour, lw=1.4, marker=marker, ms=3,
                label=f"{site.capitalize()}: {o['coef']:+.2f} per decade")
    a2.legend(frameon=False, fontsize=6.0, labelcolor=INK_2, loc="lower right", handletextpad=0.3,
              handlelength=1.6)
    a2.set_xlim(15, 92)
    a2.set_xlabel("Age (years)", color=INK_2)
    a2.set_ylabel("Sclera b* minus plastic b*", color=INK_2)
    a2.set_title("The sclera yellows with age", color=INK, fontsize=8, loc="left")
    style(a2)

    # Right: Italian batches.
    it = df[df.site == "italy"].copy()
    it["pl_rg"] = np.log(it.plastic_R / it.plastic_G)
    for batch, colour, marker, name in (("numbered", BLUE, "o", "Patients 1–31"),
                                        ("dated", GREY, "s", "Patients 32–123")):
        d = it[it.batch == batch]
        a3.scatter(d.hb, d.pl_rg, s=7, color=colour, alpha=0.7, linewidths=0, marker=marker, label=name)
    a3.set_xlabel("Haemoglobin (g/dL)", color=INK_2)
    a3.set_ylabel("Plastic log(R/G)", color=INK_2)
    a3.set_title("Italy: two batches, two lights", color=INK, fontsize=8, loc="left")
    a3.legend(frameon=False, fontsize=6.3, labelcolor=INK_2, loc="upper left", handletextpad=0.2)
    style(a3)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
