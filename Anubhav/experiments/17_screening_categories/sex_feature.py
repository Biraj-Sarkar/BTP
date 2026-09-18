"""Does giving the model the patient's sex fix the misses on anaemic women?

`categories.py` found anaemic women told "unlikely" three times as often as
anaemic men (17% vs 5%). The proposed mechanism: estimates shrink toward one
pooled average (12.8 g/dL) that sits above the women's WHO cutoff (12.0) and below
the men's (13.0). If the model knows sex, it can shrink toward the right average
for each group.

Arms, all under the identical nested protocol and category rule from
`categories.py` (only the model inputs change):

    colour                 r and g chromaticity inside the mask (current)
    colour + sex           adds a male/female indicator: shifts estimates by sex
    colour + sex, interact also lets colour's effect differ between men and women
    sex only, no photo     the demographic alone — how much comes from the photo?
    no information         everyone gets the training anaemia rate

The sex-only arm matters: men have higher haemoglobin on average, so part of any
gain from adding sex is demographic, not the photograph. The photo is earning its
place only if colour + sex clearly beats sex alone.

Everything is repeated over 5 random fold splits and reported as mean [min–max].

Run: python experiments/17_screening_categories/sex_feature.py

Abbreviations
-------------
Hb     haemoglobin (g/dL)
MAE    mean absolute error of the haemoglobin estimate, held out
PPV    share of "Likely" patients who truly are anaemic
NPV    share of "Unlikely" patients who truly are not anaemic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from categories import (  # noqa: E402
    OUTER_FOLDS, P_HIGH, REPEATS, TARGET_SENSITIVITY, apply_model, categorise, features, folds,
    oof_residuals, prob_below, ridge, run,
)


def run_per_sex_cut(df, seed, cols):
    """As `categories.run`, but the lower category cut is chosen separately for
    women and men on training patients, so each group reaches the target on its
    own instead of the pooled rate hiding a gap between them."""

    X = df[list(cols)].to_numpy(dtype=float); y = df["hb"].to_numpy()
    cut = df["cutoff"].to_numpy(); sex = df["sex"].to_numpy()
    rng = np.random.default_rng(seed)
    est_out = np.zeros(len(y)); cat_out = np.empty(len(y), dtype=object)
    for test in folds(len(y), OUTER_FOLDS, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        resid, pred_tr = oof_residuals(X[train], y[train], rng)
        model = ridge(X[train], y[train])
        est_te = apply_model(X[test], *model)
        p_tr = prob_below(pred_tr, resid, cut[train])
        p_te = prob_below(est_te, resid, cut[test])
        est_out[test] = est_te
        for group in ("F", "M"):
            tr = (sex[train] == group) & (y[train] < cut[train])
            p_low = float(np.quantile(p_tr[tr], 1 - TARGET_SENSITIVITY)) if tr.any() else 0.0
            te = sex[test] == group
            cat_out[test[te]] = categorise(p_te[te], min(p_low, P_HIGH))
    return None, est_out, cat_out, None

ARMS = [
    ("colour (current)", ("r", "g"), True),
    ("colour, per-sex cut", ("r", "g"), "per_sex"),
    ("colour + sex, per-sex cut", ("r", "g", "male"), "per_sex"),
    ("colour + sex", ("r", "g", "male"), True),
    ("colour + sex, interaction", ("r", "g", "male", "r_male", "g_male"), True),
    ("sex only, no photo", ("male",), True),
    ("no information", ("r", "g"), False),
]


def metrics(df, est, cats):
    y = df["hb"].to_numpy(); anaemic = y < df["cutoff"].to_numpy()
    female = (df["sex"] == "F").to_numpy(); male = ~female
    missed = (cats == "Unlikely") & anaemic
    ss_res = float(((y - est) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(y - est).mean()),
        "r2": 1 - ss_res / ss_tot,
        "caught": 1 - missed.sum() / anaemic.sum(),
        "missed_women": missed[female].sum() / anaemic[female].sum(),
        "missed_men": missed[male].sum() / anaemic[male].sum(),
        "missed_women_n": int(missed[female].sum()),
        "missed_men_n": int(missed[male].sum()),
        "possibly": float((cats == "Possibly").mean()),
        "likely_ppv": ((cats == "Likely") & anaemic).sum() / max((cats == "Likely").sum(), 1),
        "unlikely_npv": ((cats == "Unlikely") & ~anaemic).sum() / max((cats == "Unlikely").sum(), 1),
        "est_anaemic_women": float(est[female & anaemic].mean()),
        "true_anaemic_women": float(y[female & anaemic].mean()),
        "severe_missed": sorted(np.round(y[missed & (y < 9.0)], 1).tolist()),
    }


def fmt(values, pct=False, digits=3):
    v = np.array(values, dtype=float)
    if pct:
        return f"{100 * v.mean():.0f}% [{100 * v.min():.0f}–{100 * v.max():.0f}]"
    return f"{v.mean():.{digits}f} [{v.min():.{digits}f}–{v.max():.{digits}f}]"


def main() -> None:
    df = features("unet").copy()
    df["male"] = (df["sex"] == "M").astype(float)
    df["r_male"] = df["r"] * df["male"]
    df["g_male"] = df["g"] * df["male"]
    anaemic = df.hb < df.cutoff
    print(f"{len(df)} patients | anaemic: {int((anaemic & (df.sex == 'F')).sum())} women, "
          f"{int((anaemic & (df.sex == 'M')).sum())} men | Hb mean women "
          f"{df[df.sex == 'F'].hb.mean():.2f}, men {df[df.sex == 'M'].hb.mean():.2f}\n")

    results = {}
    for label, cols, informative in ARMS:
        runner = (lambda s: run_per_sex_cut(df, s, cols)) if informative == "per_sex" \
            else (lambda s: run(df, s, informative, cols))
        reps = [metrics(df, r[1], r[2]) for r in (runner(s) for s in range(REPEATS))]
        results[label] = reps

    rows = [
        ("held-out MAE (g/dL)", "mae", False, 3),
        ("held-out R²", "r2", False, 3),
        ("anaemic caught (top two)", "caught", True, 0),
        ("anaemic WOMEN told 'unlikely'", "missed_women", True, 0),
        ("anaemic MEN told 'unlikely'", "missed_men", True, 0),
        ("share in 'Possibly'", "possibly", True, 0),
        ("'Likely' truly anaemic", "likely_ppv", True, 0),
        ("'Unlikely' truly not anaemic", "unlikely_npv", True, 0),
        ("mean estimate, anaemic women", "est_anaemic_women", False, 2),
    ]
    for label, _, _ in ARMS:
        print(f"=== {label}")
        for name, key, pct, digits in rows:
            print(f"  {name:<32}{fmt([r[key] for r in results[label]], pct, digits)}")
        print(f"  {'severe (<9) missed, split 0':<32}{results[label][0]['severe_missed']}")
        print()
    print(f"true mean Hb of anaemic women: {results[ARMS[0][0]][0]['true_anaemic_women']:.2f}")

    (HERE / "sex_feature.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {HERE / 'sex_feature.json'}")


if __name__ == "__main__":
    main()
