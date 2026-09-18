"""Measure the conjunctiva against the sclera beside it, not against absolute colour.

Experiment 14 found the fatal problem: a model fitted on one cohort scores
R² −1.9 on another, because chromaticity is invariant to *how much light there
was* but not to *which camera measured it*. The two cohorts' features are more
than three local standard deviations apart, so a fitted line is extrapolating
far outside its training range.

The idea tested here: stop asking "how red is the conjunctiva" and start asking
"how red is it **compared to the sclera two millimetres away in the same
photograph**". Both tissues pass through the same lens, the same sensor, the
same white balance and the same illumination, so dividing one by the other
should cancel most of what the camera did. Sclera is the natural reference —
approximately neutral, always adjacent, and present in every capture.

The sclera detector is `legacy_bright_neutral_mask`, which this project
originally evaluated as a *conjunctiva* extractor and rejected precisely
because it selects bright, low-chroma pixels — that is, sclera. What made it a
bad extractor makes it the right reference detector.

Two things are measured, and the second is the one that matters:

  1. **Within-cohort accuracy** — do relative features predict at all?
  2. **Cohort separation and transfer** — do they survive the move between
     cameras, which absolute features do not?

Run: python experiments/15_sclera_relative/sclera_relative.py

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
sd     standard deviation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))
sys.path.insert(0, str(ROOT / "experiments" / "14_pooled_regression"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor, legacy_bright_neutral_mask  # noqa: E402

from head_to_head import WORK, apply_model, leave_one_out, ridge, score  # noqa: E402
from pooled import EXTERNAL, european_float  # noqa: E402

BUNDLE = ROOT / "runs/segmenter/unet.pt"
MIN_PIXELS = 500
SEED = 0


def stats(rgb, mask):
    """Mean channel values and chromaticity inside a mask."""

    px = rgb[mask > 0].astype(np.float64)
    R, G, B = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    total = max(R + G + B, 1e-6)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = float((lab[:, :, 1][mask > 0] - 128.0).mean())
    b = float((lab[:, :, 2][mask > 0] - 128.0).mean())
    return {"R": R, "G": G, "B": B, "r": R / total, "g": G / total,
            "a": a, "b": b, "redness": a - 0.5 * b}


def feature_sets(conj, scl):
    """Absolute features, and the same measurements made relative to sclera.

    A ratio cancels any per-capture gain the camera applied to a channel; a
    chromaticity difference cancels a per-capture shift. Which of the two forms
    survives better is an empirical question, so both are tested.
    """

    eps = 1e-6
    return {
        "absolute chroma (current)": np.array([conj["r"], conj["g"]]),
        "absolute redness": np.array([conj["redness"]]),
        "ratio RGB": np.array([conj["R"] / max(scl["R"], eps),
                               conj["G"] / max(scl["G"], eps)]),
        "log-ratio RGB": np.array([np.log(max(conj["R"], eps) / max(scl["R"], eps)),
                                   np.log(max(conj["G"], eps) / max(scl["G"], eps))]),
        "chroma difference": np.array([conj["r"] - scl["r"], conj["g"] - scl["g"]]),
        "redness difference": np.array([conj["redness"] - scl["redness"]]),
        "a* difference": np.array([conj["a"] - scl["a"]]),
    }


def measure(path, extractor):
    rgb = gray_world_white_balance(resize(read_rgb(path), WORK))
    conj_mask, _ = extractor(rgb)
    if np.count_nonzero(conj_mask) < MIN_PIXELS:
        return None

    # Sclera: bright and low-chroma, explicitly excluding the conjunctiva so the
    # reference cannot contain the tissue being measured against it.
    scl_mask = legacy_bright_neutral_mask(rgb)
    scl_mask = cv2.bitwise_and(scl_mask, cv2.bitwise_not(conj_mask))
    if np.count_nonzero(scl_mask) < MIN_PIXELS:
        return None

    return feature_sets(stats(rgb, conj_mask), stats(rgb, scl_mask))


def load_local(extractor):
    per: dict[str, dict] = {}
    for s in load_local_cohort(ROOT):
        f = measure(s.image_path, extractor)
        if f is None:
            continue
        e = per.setdefault(s.patient_id, {"hb": s.hb, "age": s.age_years,
                                          "sex": s.sex, "f": []})
        e["f"].append(f)
    keys = list(next(iter(per.values()))["f"][0])
    return [{"x": {k: np.mean([f[k] for f in e["f"]], axis=0) for k in keys},
             "hb": e["hb"], "age": e["age"], "sex": e["sex"]}
            for _, e in sorted(per.items(), key=lambda kv: int(kv[0].split(":")[1]))], keys


def load_external(extractor):
    import pandas as pd
    rows, skipped = [], 0
    for country in ("India", "Italy"):
        table = EXTERNAL / country / f"{country}.xlsx"
        if not table.exists():
            continue
        df = pd.read_excel(table, engine="openpyxl")
        labels = {int(r["Number"]): r for _, r in df.iterrows()
                  if pd.notna(r.get("Number")) and pd.notna(r.get("Hgb"))}
        for folder in sorted((EXTERNAL / country).iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            row = labels.get(int(folder.name))
            photos = sorted(folder.glob("*.jpg"))
            if row is None or not photos:
                continue
            hb = european_float(row["Hgb"])
            if hb is None or not 3.0 <= hb <= 22.0:
                continue
            f = measure(photos[0], extractor)
            if f is None:
                skipped += 1
                continue
            sex = str(row.get("Gender", "")).strip().upper()[:1] or None
            rows.append({"x": f, "hb": hb, "age": european_float(row.get("Age")),
                         "sex": sex})
    if skipped:
        print(f"  {skipped} external subjects skipped (no usable conjunctiva or sclera)")
    return rows


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    extractor = get_extractor("unet", model_dir=BUNDLE)
    print("measuring local cohort ...")
    local, keys = load_local(extractor)
    print(f"  {len(local)} patients")
    print("measuring external cohort ...")
    external = load_external(extractor)
    print(f"  {len(external)} subjects\n")

    yl = np.array([r["hb"] for r in local])
    ye = np.array([r["hb"] for r in external])
    ml = [(r["age"], r["sex"]) for r in local]
    me = [(r["age"], r["sex"]) for r in external]

    print(f"{'feature set':<26}{'k':>3}{'sep':>7}{'local R²':>10}{'ext R²':>9}"
          f"{'ext->loc':>10}{'loc->ext':>10}")
    print("-" * 75)

    results = {}
    for key in keys:
        Xl = np.array([np.atleast_1d(r["x"][key]) for r in local])
        Xe = np.array([np.atleast_1d(r["x"][key]) for r in external])

        # Cohort separation: how far apart the two clouds sit, in units of their
        # own pooled spread. This is the quantity that decides whether a model
        # fitted on one is extrapolating on the other.
        pooled_sd = np.sqrt((Xl.var(axis=0) + Xe.var(axis=0)) / 2.0)
        pooled_sd[pooled_sd < 1e-12] = 1.0
        separation = float(np.max(np.abs(Xe.mean(axis=0) - Xl.mean(axis=0)) / pooled_sd))

        local_r2 = r2_of(leave_one_out(Xl, yl), yl)
        ext_r2 = r2_of(leave_one_out(Xe, ye), ye)
        e2l = r2_of(apply_model(Xl, *ridge(Xe, ye)), yl)
        l2e = r2_of(apply_model(Xe, *ridge(Xl, yl)), ye)

        results[key] = {"k": int(Xl.shape[1]), "separation": separation,
                        "local_r2": local_r2, "external_r2": ext_r2,
                        "external_to_local_r2": e2l, "local_to_external_r2": l2e,
                        "external_to_local_mae": float(np.abs(
                            apply_model(Xl, *ridge(Xe, ye)) - yl).mean())}
        print(f"{key:<26}{Xl.shape[1]:>3}{separation:>7.2f}{local_r2:>10.3f}"
              f"{ext_r2:>9.3f}{e2l:>10.3f}{l2e:>10.3f}")

    print("-" * 75)
    print("sep = |mean difference| between cohorts, in pooled standard deviations.")
    print("      Below ~1 the clouds overlap; above ~2 a fitted model extrapolates.")
    print("ext->loc / loc->ext = fit on one cohort, score on the other.")

    best = min(results, key=lambda k: results[k]["separation"])
    best_transfer = max(results, key=lambda k: results[k]["external_to_local_r2"])
    print(f"\nmost portable features:   {best} (separation {results[best]['separation']:.2f})")
    print(f"best external->local:     {best_transfer} "
          f"(R² {results[best_transfer]['external_to_local_r2']:+.3f}, "
          f"MAE {results[best_transfer]['external_to_local_mae']:.3f})")

    out = Path(__file__).resolve().parent / "sclera_relative.json"
    out.write_text(json.dumps({"n_local": len(local), "n_external": len(external),
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
