"""Does a wider mask help, and do the conjunctiva's visible vessels carry signal?

Two questions, both on the 217 external patients, where the colour signal is
known to be present.

**1. Coverage.** The extractor keeps pixels above probability 0.5. Experiment 13
swept mask size and found widening hurt monotonically — but that was measured on
the 26 local patients, later shown to lack the colour signal entirely
(experiment 16), so the conclusion needs redoing on a cohort that can answer it.
Here the probability threshold is swept (0.3 to 0.9) and the 0.5 mask is eroded
and dilated.

**2. Vessels.** A mean colour throws away all structure. The conjunctiva's
vasculature is visible and physiological, so texture features are tested
alongside colour: how variable the tissue is, how strong its edges are, and how
much of it reads as vessel rather than background.

A caution carried into the interpretation: experiment 16 found that pulling the
eyelid down harder engorges the vessels and reddens the tissue. Vessel prominence
may therefore track **capture technique** as much as haemoglobin, so any gain
here needs that check before it is believed.

Measured at 1024px (vessels are a few pixels wide and vanish at 512), with **no
colour correction**, which experiment 18 found better within and across sites.
Model inputs always include sex, the best configuration from experiment 17.
Ten-fold cross-validation, 5 random splits, ridge regression.

Run: python experiments/19_coverage_and_vessels/sweep.py

Abbreviations
-------------
Hb     haemoglobin (g/dL)
MAE    mean absolute error, held out
px     pixels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))
sys.path.insert(0, str(ROOT / "experiments" / "17_screening_categories"))

from anemia.data import anemia_threshold, load_dataset  # noqa: E402
from anemia.imaging import read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

from categories import folds  # noqa: E402
from head_to_head import apply_model, ridge  # noqa: E402
from sex_feature import run_per_sex_cut  # noqa: E402

WORK = (1024, 1024)
REPEATS = 5
FOLDS = 10
MIN_PIXELS = 800

# mask variants: (name, threshold, morphology pixels — negative erodes)
VARIANTS = [("p>=0.9", 0.9, 0), ("p>=0.7", 0.7, 0), ("p>=0.5 (deployed)", 0.5, 0),
            ("p>=0.3", 0.3, 0), ("erode 8px", 0.5, -8), ("dilate 8px", 0.5, 8),
            ("dilate 16px", 0.5, 16), ("dilate 32px", 0.5, 32)]


def morph(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(pixels) * 2 + 1,) * 2)
    return cv2.dilate(mask, k) if pixels > 0 else cv2.erode(mask, k)


def colour_features(rgb, sel):
    px = rgb[sel].astype(np.float64)
    total = np.maximum(px.sum(axis=1), 1e-6)
    return {"r": float((px[:, 0] / total).mean()), "g": float((px[:, 1] / total).mean())}


def vessel_features(rgb, sel):
    """Structure inside the mask: variability, edges, and vessel-like darkness.

    Vessels are thin, darker than the tissue around them, and show most contrast
    in the green channel, which haemoglobin absorbs strongly. `vessel_fraction`
    is the share of mask pixels meaningfully darker than their local background —
    a simple, explainable stand-in for a vesselness filter.
    """

    green = rgb[:, :, 1].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    redness = (lab[:, :, 1] - 128.0) - 0.5 * (lab[:, :, 2] - 128.0)

    background = cv2.GaussianBlur(green, (0, 0), 6)
    relief = background - green                      # positive where darker than surroundings
    edges = cv2.magnitude(cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3))

    inside = relief[sel]
    return {"green_sd": float(green[sel].std()),
            "redness_sd": float(redness[sel].std()),
            "edge_strength": float(edges[sel].mean()),
            "vessel_fraction": float((inside > 4.0).mean()),
            "vessel_depth": float(inside[inside > 4.0].mean()) if (inside > 4.0).any() else 0.0}


def measure() -> pd.DataFrame:
    cache = HERE / "features.csv"
    if cache.exists():
        return pd.read_csv(cache)
    seg: LearnedSegmenter = get_extractor("unet")
    rows = []
    for s in load_dataset(ROOT / "datasets/eyes_defy/dataset anemia"):
        rgb = resize(read_rgb(s.image_path), WORK)     # no colour correction
        probability = seg.probability(rgb)
        row = {"patient": s.patient_id, "hb": s.hb, "age": s.age_years, "sex": s.sex,
               "cutoff": anemia_threshold(s.age_years, s.sex),
               "male": 1.0 if s.sex == "M" else 0.0}
        usable = True
        for name, threshold, pixels in VARIANTS:
            mask = morph((probability >= threshold).astype(np.uint8) * 255, pixels)
            sel = mask > 0
            if sel.sum() < MIN_PIXELS:
                usable = False
                break
            row[f"{name}|coverage"] = float(sel.mean())
            for key, value in {**colour_features(rgb, sel), **vessel_features(rgb, sel)}.items():
                row[f"{name}|{key}"] = value
        if usable:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def cv_r2_mae(df, cols, seed):
    X = df[cols].to_numpy(dtype=float); y = df["hb"].to_numpy()
    rng = np.random.default_rng(seed)
    pred = np.zeros(len(y))
    for test in folds(len(y), FOLDS, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = apply_model(X[test], *ridge(X[train], y[train]))
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot, float(np.abs(y - pred).mean())


def evaluate(df, cols):
    scores = [cv_r2_mae(df, cols, s) for s in range(REPEATS)]
    r2 = [s[0] for s in scores]; mae = [s[1] for s in scores]
    return float(np.mean(r2)), float(np.min(r2)), float(np.max(r2)), float(np.mean(mae))


def main() -> None:
    df = measure()
    print(f"{len(df)} patients, measured at {WORK[0]}px, no colour correction\n")

    results = {"coverage": {}, "vessels": {}}
    print("1. COVERAGE — colour + sex, mask varied")
    print(f"   {'mask':<20}{'coverage':>10}{'R²':>22}{'MAE':>8}")
    for name, _, _ in VARIANTS:
        cols = [f"{name}|r", f"{name}|g", "male"]
        m, lo, hi, mae = evaluate(df, cols)
        cov = df[f"{name}|coverage"].median()
        results["coverage"][name] = {"coverage": float(cov), "r2": m, "r2_min": lo,
                                     "r2_max": hi, "mae": mae}
        print(f"   {name:<20}{100 * cov:>9.2f}%{f'{m:.3f} [{lo:.3f}–{hi:.3f}]':>22}{mae:>8.3f}")

    print("\n2. VESSELS — added to colour + sex, on the deployed mask")
    base = "p>=0.5 (deployed)"
    sets = {
        "colour + sex (reference)": [f"{base}|r", f"{base}|g", "male"],
        "+ variability (sd)": [f"{base}|r", f"{base}|g", "male", f"{base}|green_sd", f"{base}|redness_sd"],
        "+ edge strength": [f"{base}|r", f"{base}|g", "male", f"{base}|edge_strength"],
        "+ vessel fraction & depth": [f"{base}|r", f"{base}|g", "male", f"{base}|vessel_fraction", f"{base}|vessel_depth"],
        "+ all structure": [f"{base}|r", f"{base}|g", "male", f"{base}|green_sd", f"{base}|redness_sd",
                            f"{base}|edge_strength", f"{base}|vessel_fraction", f"{base}|vessel_depth"],
        "structure only, no colour": ["male", f"{base}|green_sd", f"{base}|redness_sd",
                                      f"{base}|edge_strength", f"{base}|vessel_fraction", f"{base}|vessel_depth"],
    }
    print(f"   {'features':<28}{'k':>3}{'R²':>22}{'MAE':>8}")
    for label, cols in sets.items():
        m, lo, hi, mae = evaluate(df, cols)
        results["vessels"][label] = {"k": len(cols), "r2": m, "r2_min": lo, "r2_max": hi, "mae": mae}
        print(f"   {label:<28}{len(cols):>3}{f'{m:.3f} [{lo:.3f}–{hi:.3f}]':>22}{mae:>8.3f}")

    print("\n   correlation of each structure feature with Hb, and with mask coverage:")
    for key in ("green_sd", "redness_sd", "edge_strength", "vessel_fraction", "vessel_depth"):
        col = df[f"{base}|{key}"]
        print(f"   {key:<18} vs Hb {np.corrcoef(col, df.hb)[0, 1]:+.3f}   "
              f"vs coverage {np.corrcoef(col, df[f'{base}|coverage'])[0, 1]:+.3f}")

    print("\n3. SCREENING OUTCOME — per-sex category cut, best colour vs best structure set")
    for label, cols in (("colour + sex", [f"{base}|r", f"{base}|g", "male"]),
                        ("colour + sex + all structure", sets["+ all structure"])):
        d = df.rename(columns={c: c.split("|")[-1] for c in cols if "|" in c}).copy()
        names = [c.split("|")[-1] for c in cols]
        out = []
        for seed in range(REPEATS):
            _, est, cats, _ = run_per_sex_cut(d, seed, tuple(names))
            y = d["hb"].to_numpy(); anaemic = y < d["cutoff"].to_numpy()
            missed = (cats == "Unlikely") & anaemic
            out.append((1 - missed.sum() / anaemic.sum(), float((cats == "Possibly").mean())))
        caught = float(np.mean([o[0] for o in out])); possibly = float(np.mean([o[1] for o in out]))
        results.setdefault("screening", {})[label] = {"caught": caught, "possibly": possibly}
        print(f"   {label:<30} caught {100 * caught:.0f}%   'Possibly' {100 * possibly:.0f}%")

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
