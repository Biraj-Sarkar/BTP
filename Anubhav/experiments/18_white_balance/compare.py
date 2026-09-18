"""Gray-world white balance versus no colour correction, measured properly.

Gray-world was adopted early in the project on evidence from the 26-patient local
cohort, since shown to be anomalous (experiment 16). Later results point the
other way: across 217 external patients red chromaticity tracks haemoglobin at
r = +0.67 without correction and +0.47 with it, and one severely anaemic patient
is missed because gray-world over-corrects her sclera-dominated close-up
(experiment 17). The argument *for* gray-world is portability: uncorrected colour
depends on the camera and lighting, which is what gray-world is meant to absorb.

So both sides of the trade-off are measured:

  1. **Within a site** — India (95) and Italy (122) separately, and both together.
     Held-out 10-fold cross-validation, repeated over 5 random splits, with a
     patient-level bootstrap confidence interval on the difference.
  2. **Across sites** — fit on India, test on Italy, and the reverse. This is the
     job gray-world exists to do.
  3. **The screening outcome** — the best app configuration from experiment 17
     (colour + sex, per-sex category cut), under each correction.

**Only the colour reading changes.** Every patient gets one mask — the learned
extractor run on the gray-world image, as deployed — and each arm reads colour
inside that same mask from either the raw or the balanced pixels. Any difference
is therefore due to colour correction, not to where the mask landed.

Run: python experiments/18_white_balance/compare.py

Abbreviations
-------------
Hb     haemoglobin (g/dL)
MAE    mean absolute error of the held-out estimate
CI     confidence interval (95%, patient-level bootstrap)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))
sys.path.insert(0, str(ROOT / "experiments" / "17_screening_categories"))

from anemia.data import anemia_threshold, load_dataset  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

from categories import folds  # noqa: E402
from head_to_head import apply_model, ridge  # noqa: E402
from sex_feature import metrics, run_per_sex_cut  # noqa: E402

REPEATS = 5
FOLDS = 10
BOOTSTRAP = 2000
ARMS = ("raw", "gw")
LABEL = {"raw": "no correction", "gw": "gray-world"}


def chroma(img, mask):
    px = img[mask].astype(np.float64)
    px = px / np.maximum(px.sum(axis=1, keepdims=True), 1e-6)
    return px[:, 0].mean(), px[:, 1].mean()


def measure() -> pd.DataFrame:
    cache = HERE / "features.csv"
    if cache.exists():
        return pd.read_csv(cache)
    unet = get_extractor("unet")
    rows = []
    for s in load_dataset(ROOT / "datasets/eyes_defy/dataset anemia"):
        raw = resize(read_rgb(s.image_path), (512, 512))
        gw = gray_world_white_balance(raw)
        mask = unet(gw)[0] > 0
        if mask.sum() < 500:
            continue
        rr, rg = chroma(raw, mask)
        gr, gg = chroma(gw, mask)
        rows.append({"patient": s.patient_id, "site": s.source.split("/")[-1], "hb": s.hb,
                     "age": s.age_years, "sex": s.sex,
                     "cutoff": anemia_threshold(s.age_years, s.sex),
                     "raw_r": rr, "raw_g": rg, "gw_r": gr, "gw_g": gg})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def cv_predict(X, y, seed):
    rng = np.random.default_rng(seed)
    pred = np.zeros(len(y))
    for test in folds(len(y), FOLDS, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = apply_model(X[test], *ridge(X[train], y[train]))
    return pred


def r2(y, p):
    return 1 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def within(df, label):
    y = df["hb"].to_numpy()
    out = {"n": len(df)}
    preds = {}
    for arm in ARMS:
        X = df[[f"{arm}_r", f"{arm}_g"]].to_numpy()
        reps = [cv_predict(X, y, s) for s in range(REPEATS)]
        preds[arm] = reps[0]
        out[arm] = {"mae": [float(np.abs(y - p).mean()) for p in reps],
                    "r2": [r2(y, p) for p in reps],
                    "corr_r": float(np.corrcoef(df[f"{arm}_r"], y)[0, 1])}
    # paired bootstrap on the same folds (split 0): gray-world minus no correction
    rng = np.random.default_rng(0)
    e_raw, e_gw = np.abs(y - preds["raw"]), np.abs(y - preds["gw"])
    d_mae, d_r2 = [], []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        d_mae.append(e_gw[i].mean() - e_raw[i].mean())
        d_r2.append(r2(y[i], preds["gw"][i]) - r2(y[i], preds["raw"][i]))
    out["gw_minus_raw_mae_ci"] = [float(np.percentile(d_mae, 2.5)), float(np.percentile(d_mae, 97.5))]
    out["gw_minus_raw_r2_ci"] = [float(np.percentile(d_r2, 2.5)), float(np.percentile(d_r2, 97.5))]
    return out


def across(train, test):
    out = {}
    y_tr, y_te = train["hb"].to_numpy(), test["hb"].to_numpy()
    for arm in ARMS:
        cols = [f"{arm}_r", f"{arm}_g"]
        p = apply_model(test[cols].to_numpy(), *ridge(train[cols].to_numpy(), y_tr))
        out[arm] = {"mae": float(np.abs(y_te - p).mean()), "r2": r2(y_te, p)}
    base = np.full(len(y_te), y_tr.mean())
    out["baseline"] = {"mae": float(np.abs(y_te - base).mean()), "r2": r2(y_te, base)}
    return out


def app_outcome(df):
    out = {}
    d = df.copy()
    d["male"] = (d["sex"] == "M").astype(float)
    for arm in ARMS:
        d["r"], d["g"] = d[f"{arm}_r"], d[f"{arm}_g"]
        reps = [metrics(d, r[1], r[2]) for r in (run_per_sex_cut(d, s, ("r", "g", "male"))
                                                  for s in range(REPEATS))]
        out[arm] = {k: float(np.mean([r[k] for r in reps]))
                    for k in ("mae", "r2", "caught", "missed_women", "missed_men", "possibly",
                              "likely_ppv", "unlikely_npv")}
        out[arm]["severe_missed_split0"] = reps[0]["severe_missed"]
    return out


def mr(values):
    v = np.array(values)
    return f"{v.mean():.3f} [{v.min():.3f}–{v.max():.3f}]"


def main() -> None:
    df = measure()
    sites = {"india": df[df.site == "india"], "italy": df[df.site == "italy"], "both": df}
    print(f"{len(df)} patients: India {len(sites['india'])}, Italy {len(sites['italy'])}\n")

    results = {"within": {}, "across": {}, "app": {}}
    print("1. WITHIN A SITE — held out, mean [range over 5 splits]")
    print(f"   {'site':<7}{'':<16}{'MAE':>24}{'R²':>24}{'r(red, Hb)':>12}")
    for name, sub in sites.items():
        w = within(sub, name)
        results["within"][name] = w
        for arm in ARMS:
            print(f"   {name:<7}{LABEL[arm]:<16}{mr(w[arm]['mae']):>24}{mr(w[arm]['r2']):>24}"
                  f"{w[arm]['corr_r']:>+12.3f}")
        lo, hi = w["gw_minus_raw_mae_ci"]; rlo, rhi = w["gw_minus_raw_r2_ci"]
        print(f"   {'':<7}gray-world minus none:  MAE CI [{lo:+.3f}, {hi:+.3f}]   "
              f"R² CI [{rlo:+.3f}, {rhi:+.3f}]\n")

    print("2. ACROSS SITES — fit on one site, test on the other")
    for a, b in (("india", "italy"), ("italy", "india")):
        x = across(sites[a], sites[b])
        results["across"][f"{a}->{b}"] = x
        print(f"   {a} -> {b}:  no correction MAE {x['raw']['mae']:.3f} R² {x['raw']['r2']:+.3f}  |  "
              f"gray-world MAE {x['gw']['mae']:.3f} R² {x['gw']['r2']:+.3f}  |  "
              f"baseline MAE {x['baseline']['mae']:.3f} R² {x['baseline']['r2']:+.3f}")

    print("\n3. SCREENING OUTCOME — colour + sex, per-sex category cut, both sites")
    app = app_outcome(df)
    results["app"] = app
    rows = [("held-out MAE", "mae", False), ("held-out R²", "r2", False),
            ("anaemic caught", "caught", True), ("anaemic women told unlikely", "missed_women", True),
            ("anaemic men told unlikely", "missed_men", True), ("share in 'Possibly'", "possibly", True),
            ("'Likely' truly anaemic", "likely_ppv", True), ("'Unlikely' truly not", "unlikely_npv", True)]
    print(f"   {'':<30}{'no correction':>15}{'gray-world':>13}")
    for lab, key, pct in rows:
        f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.3f}")
        print(f"   {lab:<30}{f(app['raw'][key]):>15}{f(app['gw'][key]):>13}")
    print(f"   {'severe (<9) missed, split 0':<30}{str(app['raw']['severe_missed_split0']):>15}"
          f"{str(app['gw']['severe_missed_split0']):>13}")

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
