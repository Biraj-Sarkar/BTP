"""Figure for experiment 22: one conjunctiva after each resize, at the same zoom.

Reads summary.json and ceiling_per_image.csv from measure.py and one photograph;
trains and scores nothing.

    python experiments/22_resolution/figure.py

Top: a 360x240-pixel window of the native photograph, centred on the thickest
point of the annotated conjunctiva, then the same window after each resize was
undone, so every panel shows only what survived. The image is the Mendeley
capture whose mask thickness is closest to the median.
Bottom: share of detail energy kept in each band of feature size, all 547
Mendeley images (median).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as m  # noqa: E402

OUT = HERE / "report_figure.png"
WINDOW = (360, 240)   # native px, width x height
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# One accent for the app's current path, greys for the reference grids,
# a second hue for the alternative crop. Order matches the legend.
VARIANTS = [("prepared_768", "Training copy (576x768)", "#b9b8b2", "-"),
            ("app_512", "App frame (512x512)", "#2a78d6", "-"),
            ("seg_320", "Segmenter grid (320x320)", "#52514e", "--"),
            ("cnn_224_app", "Neural input, cut from 512 frame", "#2a78d6", "--"),
            ("cnn_224_native", "Neural input, cut from original", "#c2571a", "-")]


def pick_stem() -> str:
    rows = [r for r in csv.DictReader(open(HERE / "ceiling_per_image.csv")) if r["dataset"] == "mendeley"]
    t = np.array([float(r["thickness_px"]) for r in rows])
    return rows[int(np.argmin(np.abs(t - np.median(t))))]["stem"]


def rebuilt_versions(stem: str):
    item = {"dataset": "mendeley", "stem": stem}
    mask = m.native_mask(item)
    image = m.native_image(item)
    height, width = mask.shape

    # Window centred on the deepest point inside the mask
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(depth), depth.shape)
    w, h = WINDOW
    # Stay inside the neural crop's box: its edge borders black padding, which
    # blurs into the outermost rows and is not conjunctiva.
    left, top, right, bottom = m.tight_bbox(mask.astype(np.uint8), 0.08)
    lo_x, hi_x = left + 16, max(left + 16, right - 16 - w)
    lo_y, hi_y = top + 16, max(top + 16, bottom - 16 - h)
    x0, y0 = int(np.clip(cx - w // 2, lo_x, hi_x)), int(np.clip(cy - h // 2, lo_y, hi_y))
    win = (slice(y0, y0 + h), slice(x0, x0 + w))

    out = {"Original\nphotograph": image[win]}
    for name, size, label in (("app_512", (512, 512), "App frame\n512x512"),
                              ("seg_320", (320, 320), "Segmenter grid\n320x320")):
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        out[label] = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)[win]

    # Neural input as the app builds it: from the 512 frame
    work = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    work_m = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    _, _, (l5, t5, r5, b5) = m.cnn_crop(work, work_m)
    small = cv2.resize(m.pad_to_square(work[t5:b5, l5:r5]), (224, 224), interpolation=cv2.INTER_AREA)
    sx, sy = width / 512, height / 512
    X0, Y0, X1, Y1 = (int(round(v)) for v in (l5 * sx, t5 * sy, r5 * sx, b5 * sy))
    canvas = cv2.resize(work, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas[Y0:Y1, X0:X1] = m.unpad_resize(small, b5 - t5, r5 - l5, X1 - X0, Y1 - Y0)
    out["Neural input\n(cut from 512 frame)"] = canvas[win]

    # Neural input cut from the original photograph
    left, top, right, bottom = m.tight_bbox(mask.astype(np.uint8), 0.08)
    small = cv2.resize(m.pad_to_square(image[top:bottom, left:right]), (224, 224),
                       interpolation=cv2.INTER_AREA)
    canvas = image.copy()
    canvas[top:bottom, left:right] = m.unpad_resize(small, bottom - top, right - left,
                                                    right - left, bottom - top)
    out["Neural input\n(cut from original)"] = canvas[win]
    return out


def main() -> None:
    summary = json.load(open(HERE / "summary.json"))
    stem = pick_stem()
    panels = rebuilt_versions(stem)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig = plt.figure(figsize=(7.2, 3.6))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 2.1], hspace=0.38, wspace=0.05)
    for i, (label, patch) in enumerate(panels.items()):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(patch)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(label, fontsize=6.5, color=INK)

    ax = fig.add_subplot(grid[1, :])
    bands = list(summary["structure"]["app_512"]["band_kept_median"].keys())
    x = np.arange(len(bands))
    for key, label, colour, style in VARIANTS:
        kept = [summary["structure"][key]["band_kept_median"][b] for b in bands]
        ax.plot(x, kept, style, color=colour, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_xticks(x, [f"{b} features" for b in bands])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Detail energy kept (median)", color=INK_2)
    ax.set_xlabel("Feature size in the original 12 MP photograph", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, width=0.6, length=2.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.text(0.01, 0.985, f"Mendeley image {stem}, 360x240 px window of the original",
             fontsize=6.5, color=INK_2, va="top")
    fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT} (image {stem})")


if __name__ == "__main__":
    main()
