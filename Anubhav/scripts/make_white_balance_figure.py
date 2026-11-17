"""One photograph under four colour corrections, for a slide or the progress report.

    python scripts/make_white_balance_figure.py                  # India example, slide
    python scripts/make_white_balance_figure.py --example italy
    python scripts/make_white_balance_figure.py --target report  # for the progress report

Writes `figures/white_balance_comparison_<example>.png` (16:9, for a slide), or
with `--target report` `figures/white_balance_comparison_<example>_report.png`:
the same panels drawn at the report's full text width (168 mm) so the text stays
readable in print, without the footnote, which the report's caption replaces.

Columns:
- No correction, gray-world and white-patch (the default) come from
  `preprocess.prepare()` itself, with only `PreprocessConfig.white_balance`
  changed, so they show exactly what the pipeline does.
- Sclera as white is a proposed correction, not in the pipeline. Its sclera
  pixels come from experiment 26's own detector (`sclera_mask` in
  `experiments/26_correction_prechecks/run.py`), on the same uncorrected 512x512
  frame, and the crop is cut exactly as `prepare()` cuts it.

The deployed U-Net draws the mask. It is shown raw pixels in every case, so the
mask is identical in all four columns and only colour differs.

The example is typical rather than flattering: among the photographs experiment
26 could measure against the capture device's white plastic, it is the one whose
four angles (one per column) lie closest to that site's medians.

The angle under each column is how far the white plastic is from neutral grey
after that correction. The plastic records only the light, so a correction that
removed the light well leaves it grey. The plastic colour comes from experiment
26's measurements of the same 512x512 frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.data import load_dataset  # noqa: E402
from anemia.imaging import (  # noqa: E402
    apply_mask,
    below_lower_lid,
    pad_to_square,
    read_rgb,
    resize,
    tight_bbox,
    white_patch_estimate,
    white_patch_pixels,
)
from anemia.preprocess import prepare  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

DATA = ROOT / "datasets/eyes_defy/dataset anemia"
EXP26 = ROOT / "experiments/26_correction_prechecks"
EXAMPLES = {   # chosen by the rule in the docstring (closest to the site medians)
    "india": ("eyesdefy-india:14", "20200204_154215.jpg"),
    "italy": ("eyesdefy-italy:118", "T_87_20190614_081929.jpg"),
}
INK, INK_2, BLUE = "#0b0b0b", "#52514e", "#2a78d6"

# Sizes in points at the size the figure is shown. The slide is 13.33 x 7.5 in;
# the report figure is drawn at the report's text width, so nothing is scaled.
TARGETS = {
    "slide": dict(figsize=(13.33, 7.5), dpi=180, ratios=[3.0, 0.95, 0.7], hspace=0.34,
                  wspace=0.10, right=0.985, top=0.86, bottom=0.07, title=13, how=9, panel=8, label=9,
                  value=9.5, legend=7.5, title_y=1.16, how_y=1.135, line_y0=0.86, line_step=0.30,
                  outline_lw=1.1, lid_lw=1.2, sub_wspace=0.12, panel_pad=3, footnote=True,
                  gains="Gains R / G / B",
                  wp_legend="blue: pixels used   dashed: nothing below is used"),
    # The numbers start above their own row (line_y0 > 1): one hspace has to
    # leave room for the swatch titles, and would otherwise also open a gap here.
    "report": dict(figsize=(6.61, 3.2), dpi=300, ratios=[3.0, 0.95, 0.5], hspace=0.40,
                   wspace=0.16, right=0.972, top=0.85, bottom=0.01, title=7.5, how=5.8, panel=5.0,
                   label=5.8, value=6.0, legend=5.0, title_y=1.19, how_y=1.165,
                   line_y0=1.75, line_step=0.50, outline_lw=0.6, lid_lw=0.7, sub_wspace=0.14, panel_pad=1.5,
                   footnote=False, gains="Gains R/G/B",
                   wp_legend="blue: pixels used\ndashed: nothing below is used"),
}


def load_exp26():
    spec = importlib.util.spec_from_file_location("exp26_run", EXP26 / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_like_prepare(frame: np.ndarray, mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """The crop `prepare()` cuts from a corrected frame (same steps, same order)."""

    masked = apply_mask(frame, mask)
    left, top, right, bottom = tight_bbox(mask, cfg.bbox_pad_ratio)
    cropped = masked[top:bottom, left:right]
    if cropped.size == 0:
        cropped = masked
    return resize(pad_to_square(cropped), cfg.model_size)


def angle_from_grey(rgb: np.ndarray) -> float:
    rgb = np.asarray(rgb, float)
    cos = rgb.sum() / (np.linalg.norm(rgb) * np.sqrt(3))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def swatch(rgb: np.ndarray, size=(60, 60)) -> np.ndarray:
    return np.tile(np.clip(np.round(rgb), 0, 255).astype(np.uint8), (*size, 1))


def outline(ax, pixels: np.ndarray, lw: float) -> None:
    grown = cv2.dilate(pixels.astype(np.uint8), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = c[:, 0, :]
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=BLUE, lw=lw)


def legend_box(ax, text: str, size: float) -> None:
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=size,
            color="white", bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--example", choices=sorted(EXAMPLES), default="india")
    parser.add_argument("--target", choices=sorted(TARGETS), default="slide")
    args = parser.parse_args()
    st = TARGETS[args.target]
    patient, filename = EXAMPLES[args.example]

    sample = next(s for s in load_dataset(DATA)
                  if s.patient_id == patient and Path(s.image_path).name == filename)
    row = pd.read_csv(EXP26 / "measurements.csv").query(
        "patient == @patient and file == @filename").iloc[0]
    plastic_raw = row[["plastic_R", "plastic_G", "plastic_B"]].to_numpy(float)

    rgb = read_rgb(sample.image_path)
    segmenter = get_extractor("unet", None)
    quality = QualityConfig(enforce=False)
    base = prepare(rgb, segmenter, PreprocessConfig(white_balance="none"), quality)
    working, mask = base.balanced, base.mask

    columns = []

    # The three corrections the pipeline offers, straight from prepare().
    for method, name, how in [
        ("none", "No correction", "the camera's own colour"),
        ("gray_world", "Gray-world", "the frame's average\nis taken as grey"),
        ("white_patch", "White-patch (default)", "the brightest 1% above the\nlower lid is taken as white"),
    ]:
        cfg = PreprocessConfig(white_balance=method)
        prepared = prepare(rgb, segmenter, cfg, quality)
        assert np.array_equal(prepared.mask, mask), "the mask must not depend on the correction"
        assert np.array_equal(crop_like_prepare(prepared.balanced, mask, cfg), prepared.crop)
        if method == "gray_world":
            means = working.reshape(-1, 3).astype(float).mean(0)
            gains = means.mean() / means
        elif method == "white_patch":
            white = white_patch_estimate(working, mask)
            gains = white.mean() / white
        else:
            gains = np.ones(3)
        columns.append({"name": name, "how": how, "frame": prepared.balanced,
                        "crop": prepared.crop, "gains": gains})

    # The proposed sclera correction, with experiment 26's detector.
    exp26 = load_exp26()
    sclera = exp26.sclera_mask(working, mask > 0)
    white = working[sclera].reshape(-1, 3).astype(float).mean(0)
    gains = white.mean() / white
    frame = np.clip(working.astype(np.float32) * gains.astype(np.float32), 0, 255).astype(np.uint8)
    columns.append({"name": "Sclera (proposed)", "how": "the brightest part of the sclera\nis taken as white",
                    "frame": frame, "crop": crop_like_prepare(frame, mask, PreprocessConfig()),
                    "gains": gains})

    for col in columns:
        conj = col["frame"][mask > 0].reshape(-1, 3).astype(float).mean(0)
        col["conj"] = conj
        col["log_rg"] = float(np.log(conj[0] / conj[1]))
        col["plastic"] = plastic_raw * col["gains"]
        col["plastic_angle"] = angle_from_grey(col["plastic"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=st["figsize"], facecolor="white")
    grid = fig.add_gridspec(3, len(columns), height_ratios=st["ratios"], hspace=st["hspace"],
                            wspace=st["wspace"], left=0.02, right=st["right"], top=st["top"],
                            bottom=st["bottom"])

    for j, col in enumerate(columns):
        # Row 1: the corrected frame, with where the light was read.
        ax = fig.add_subplot(grid[0, j])
        ax.imshow(col["frame"])
        ax.set_axis_off()
        ax.set_anchor("SW")
        # Subtitles hang from a fixed line, so one- and two-line ones align.
        ax.text(0, st["how_y"], col["how"], transform=ax.transAxes, fontsize=st["how"],
                color=INK_2, va="top", linespacing=1.3)
        ax.text(0, st["title_y"], col["name"], transform=ax.transAxes, fontsize=st["title"],
                color=INK, va="bottom", fontweight="bold")
        if col["name"].startswith("White-patch"):
            outline(ax, white_patch_pixels(working, mask), st["outline_lw"])
            below = below_lower_lid(mask)
            edge = np.argmax(below, axis=0).astype(float)
            edge[~below.any(axis=0)] = np.nan
            ax.plot(np.arange(edge.size), edge, color=BLUE, lw=st["lid_lw"], ls=(0, (4, 3)))
            legend_box(ax, st["wp_legend"], st["legend"])
        elif col["name"].startswith("Sclera"):
            outline(ax, sclera, st["outline_lw"])
            legend_box(ax, "blue: pixels used", st["legend"])

        # Row 2: the crop the regressor reads, and two flat swatches.
        sub = grid[1, j].subgridspec(1, 3, wspace=st["sub_wspace"], width_ratios=[1.15, 1, 1])
        panels = [
            (col["crop"], "Regressor's\ninput"),
            (swatch(col["conj"]), "Conjunctiva,\nmean colour"),
            # Same brightness in every column, so only the colour is compared.
            (swatch(col["plastic"] * (200.0 / col["plastic"].mean())), "White plastic,\nshould be grey"),
        ]
        for k, (image, title) in enumerate(panels):
            ax = fig.add_subplot(sub[k])
            ax.imshow(image)
            ax.set_axis_off()
            ax.set_anchor("NW")
            ax.set_title(title, fontsize=st["panel"], color=INK_2, loc="left", pad=st["panel_pad"])

        # Row 3: numbers.
        ax = fig.add_subplot(grid[2, j])
        ax.set_axis_off()
        g = col["gains"]
        lines = [
            ("Plastic off grey", f"{col['plastic_angle']:.1f}°"),
            ("Conjunctiva log(R/G)", f"{col['log_rg']:.3f}"),
            (st["gains"], f"{g[0]:.2f} / {g[1]:.2f} / {g[2]:.2f}"),
        ]
        for k, (label, value) in enumerate(lines):
            y = st["line_y0"] - k * st["line_step"]
            ax.text(0.0, y, label, fontsize=st["label"], color=INK_2, va="top",
                    transform=ax.transAxes)
            ax.text(0.95, y, value, fontsize=st["value"], color=INK, va="top", ha="right",
                    transform=ax.transAxes, fontweight="bold" if k == 0 else "normal")

    if st["footnote"]:
        site = "India" if args.example == "india" else "Italy"
        fig.text(0.02, 0.018,
                 f"One Eyes-defy-anemia photograph ({site}), typical of its site rather than chosen for "
                 "effect. Same mask in all four (deployed U-Net). 'Plastic off grey': how far the capture "
                 "device's white plastic is from neutral after the correction; smaller means the light "
                 "was removed better.",
                 fontsize=7.5, color=INK_2, wrap=True)

    suffix = "" if args.target == "slide" else f"_{args.target}"
    out = ROOT / f"figures/white_balance_comparison_{args.example}{suffix}.png"
    fig.savefig(out, dpi=st["dpi"], facecolor="white")
    print(f"wrote {out}")
    for col in columns:
        print(f"  {col['name']:<22} plastic off grey {col['plastic_angle']:.2f} deg   "
              f"log(R/G) {col['log_rg']:.3f}   gains {np.round(col['gains'], 3)}")


if __name__ == "__main__":
    main()
