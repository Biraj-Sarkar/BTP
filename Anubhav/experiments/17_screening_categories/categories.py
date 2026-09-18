"""Would a three-category screening message work? Tested on the 217 external patients.

Instead of a number or a yes/no verdict, the app would say one of:

    Likely anaemic       — get a blood test
    Possibly anaemic     — consider a blood test
    Anaemia unlikely from this photo

Each patient's chance of anaemia is read off the model's own track record — the
actual distribution of its errors on other patients — rather than an assumed
bell curve, so the method works unchanged for any model, including a network:

    P(anaemic) = fraction of past errors e for which  estimate + e  <  WHO cutoff

Categories: "Likely" when P >= 0.5 (more likely than not). The lower cut is chosen
so that the top two categories together catch a target share of anaemic patients
(default 90%) — a screening choice that favours catching cases over avoiding tests.

**Nothing is tuned on the patients being scored.** An outer 10-fold loop holds each
patient out; inside it, the model is fitted, its errors measured by an inner
10-fold loop, and the lower cut chosen — all on the other patients only.

Three questions, per extractor:
  1. In each category, what share of patients are actually anaemic?
  2. How many anaemic patients land in "Unlikely" — the misses that matter?
  3. How many land in "Possibly" — a model that puts everyone there never commits?

A no-information reference (every patient gets the training anaemia rate as their
probability) shows what a useless model looks like on the same tables.

Run: python experiments/17_screening_categories/categories.py

Abbreviations
-------------
Hb     haemoglobin (g/dL)
WHO    World Health Organization; its cutoff depends on age, sex, pregnancy
PPV    positive predictive value — share of flagged patients who truly are anaemic
NPV    negative predictive value — share of "unlikely" patients who truly are not
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

from anemia.data import anemia_threshold, load_dataset  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.linear_model import extract_features  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

from head_to_head import apply_model, ridge  # noqa: E402

TARGET_SENSITIVITY = 0.90
P_HIGH = 0.50
OUTER_FOLDS = 10
INNER_FOLDS = 10
REPEATS = 5
CATS = ("Likely", "Possibly", "Unlikely")


def features(extractor_name: str) -> pd.DataFrame:
    cache = HERE / f"features_{extractor_name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    seg = get_extractor(extractor_name)
    rows = []
    for s in load_dataset(ROOT / "datasets/eyes_defy/dataset anemia"):
        rgb = gray_world_white_balance(resize(read_rgb(s.image_path), (512, 512)))
        mask, _ = seg(rgb)
        if np.count_nonzero(mask) < 500:
            continue
        r, g = extract_features(rgb, mask, "chroma")
        rows.append({"patient": s.patient_id, "hb": s.hb, "age": s.age_years, "sex": s.sex,
                     "cutoff": anemia_threshold(s.age_years, s.sex), "r": r, "g": g})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def folds(n, k, rng):
    order = rng.permutation(n)
    return [order[i::k] for i in range(k)]


def prob_below(estimates, residuals, cutoffs):
    """P(true Hb < cutoff) using the empirical distribution of past errors."""

    return np.array([np.mean(est + residuals < cut) for est, cut in zip(estimates, cutoffs)])


def oof_residuals(X, y, rng):
    pred = np.zeros(len(y))
    for test in folds(len(y), INNER_FOLDS, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = apply_model(X[test], *ridge(X[train], y[train]))
    return y - pred, pred


def categorise(p, p_low, p_high=P_HIGH):
    return np.where(p >= p_high, "Likely", np.where(p >= p_low, "Possibly", "Unlikely"))


def run(df: pd.DataFrame, seed: int, informative: bool = True, cols=("r", "g")):
    X = df[list(cols)].to_numpy(dtype=float); y = df["hb"].to_numpy(); cut = df["cutoff"].to_numpy()
    rng = np.random.default_rng(seed)
    p_out = np.zeros(len(y)); est_out = np.zeros(len(y)); cat_out = np.empty(len(y), dtype=object)
    chosen = []
    for test in folds(len(y), OUTER_FOLDS, rng):
        train = np.setdiff1d(np.arange(len(y)), test)
        Xtr, ytr, ctr = X[train], y[train], cut[train]
        if informative:
            resid, pred_tr = oof_residuals(Xtr, ytr, rng)
            model = ridge(Xtr, ytr)
            est_te = apply_model(X[test], *model)
            p_tr = prob_below(pred_tr, resid, ctr)
            p_te = prob_below(est_te, resid, cut[test])
        else:
            rate = float(np.mean(ytr < ctr))
            est_te = np.full(len(test), ytr.mean())
            p_tr = np.full(len(train), rate); p_te = np.full(len(test), rate)
        # lower cut: the value that keeps TARGET_SENSITIVITY of anaemic training patients above it
        anaemic_tr = ytr < ctr
        p_low = float(np.quantile(p_tr[anaemic_tr], 1 - TARGET_SENSITIVITY)) if anaemic_tr.any() else 0.0
        p_low = min(p_low, P_HIGH)
        chosen.append(p_low)
        p_out[test], est_out[test] = p_te, est_te
        cat_out[test] = categorise(p_te, p_low)
    return p_out, est_out, cat_out, np.array(chosen)


def tables(df, cats):
    y = df["hb"].to_numpy(); anaemic = y < df["cutoff"].to_numpy(); n = len(y)
    t1 = []
    for c in CATS:
        m = cats == c
        t1.append({"category": c, "patients": int(m.sum()), "share": m.mean(),
                   "anaemic": int((m & anaemic).sum()),
                   "anaemic_rate": (m & anaemic).sum() / m.sum() if m.any() else np.nan,
                   "mean_hb": y[m].mean() if m.any() else np.nan})
    missed = (cats == "Unlikely") & anaemic
    depth = df["cutoff"].to_numpy()[missed] - y[missed]
    t2 = {"anaemic_total": int(anaemic.sum()), "missed": int(missed.sum()),
          "missed_share": missed.sum() / anaemic.sum(),
          "caught_top_two": 1 - missed.sum() / anaemic.sum(),
          "missed_hb": sorted(np.round(y[missed], 1).tolist()),
          "missed_below_cutoff_by": sorted(np.round(depth, 1).tolist()),
          "missed_moderate_or_severe": int((missed & (y < df["cutoff"].to_numpy() - 2.0)).sum())}
    t3 = {"possibly_share": (cats == "Possibly").mean(),
          "decided_share": (cats != "Possibly").mean(),
          "unlikely_npv": ((cats == "Unlikely") & ~anaemic).sum() / max((cats == "Unlikely").sum(), 1),
          "likely_ppv": ((cats == "Likely") & anaemic).sum() / max((cats == "Likely").sum(), 1)}
    return t1, t2, t3


def main() -> None:
    results = {}
    arms = [("U-Net (deployed)", "unet", True), ("Classical", "refined", True),
            ("No information", "unet", False)]
    for label, name, informative in arms:
        df = features(name)
        reps = [run(df, seed, informative) for seed in range(REPEATS)]
        p, est, cats, chosen = reps[0]
        t1, t2, t3 = tables(df, cats)
        spread = [tables(df, r[2]) for r in reps]
        results[label] = {"n": len(df), "prevalence": float(np.mean(df.hb < df.cutoff)),
                          "p_low_mean": float(chosen.mean()), "table1": t1, "table2": t2,
                          "table3": t3,
                          "repeats": {"caught_top_two": [s[1]["caught_top_two"] for s in spread],
                                      "possibly_share": [s[2]["possibly_share"] for s in spread]}}
        if label.startswith("U-Net"):
            out = df.assign(p_anaemic=p, estimate=est, category=cats)
            out.to_csv(HERE / "unet_patient_categories.csv", index=False)

    for label, r in results.items():
        print(f"\n{'=' * 78}\n{label}  —  {r['n']} patients, {100 * r['prevalence']:.0f}% anaemic "
              f"by WHO cutoff; lower cut chosen per fold, mean P = {r['p_low_mean']:.2f}\n{'=' * 78}")
        print("TABLE 1  What the app would say, and who is actually anaemic")
        print(f"  {'category':<10}{'patients':>10}{'share':>8}{'anaemic':>9}{'rate':>8}{'mean Hb':>9}")
        for row in r["table1"]:
            print(f"  {row['category']:<10}{row['patients']:>10}{100 * row['share']:>7.0f}%"
                  f"{row['anaemic']:>9}{100 * np.nan_to_num(row['anaemic_rate']):>7.0f}%"
                  f"{np.nan_to_num(row['mean_hb']):>9.1f}")
        t2 = r["table2"]
        print("TABLE 2  Anaemic patients told 'unlikely'")
        print(f"  {t2['missed']} of {t2['anaemic_total']} anaemic patients "
              f"({100 * t2['missed_share']:.0f}%) — top two categories catch {100 * t2['caught_top_two']:.0f}%")
        print(f"  their Hb: {t2['missed_hb']}")
        print(f"  g/dL below their own cutoff: {t2['missed_below_cutoff_by']}"
              f"   (>2 g/dL below: {t2['missed_moderate_or_severe']})")
        t3 = r["table3"]
        print("TABLE 3  Does the model commit?")
        print(f"  'Possibly' {100 * t3['possibly_share']:.0f}%  |  given a clear answer "
              f"{100 * t3['decided_share']:.0f}%  |  'Likely' truly anaemic {100 * t3['likely_ppv']:.0f}%"
              f"  |  'Unlikely' truly not {100 * t3['unlikely_npv']:.0f}%")
        rp = r["repeats"]
        print(f"  over {REPEATS} random fold splits: caught "
              f"{100 * min(rp['caught_top_two']):.0f}–{100 * max(rp['caught_top_two']):.0f}%, "
              f"'Possibly' {100 * min(rp['possibly_share']):.0f}–{100 * max(rp['possibly_share']):.0f}%")

    (HERE / "categories.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {HERE / 'categories.json'} and unet_patient_categories.csv")


if __name__ == "__main__":
    main()
