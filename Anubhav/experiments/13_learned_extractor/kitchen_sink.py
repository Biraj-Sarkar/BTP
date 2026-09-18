"""Throw every colour statistic in and let L1 decide. Does it beat two features?

The argument for this is sound: regularisation exists so that useless inputs get
shrunk, and L1 in particular drives them to exactly zero. If mean R, G, B throws
away information that the median or the spread would keep, a penalised model
given all of them should find it and discard the rest.

The argument against is the sample size. At n = 26 with ~30 candidate features,
two things bite:

  * **Selection instability.** These features are near-collinear — experiment 08
    measured max |r| = 0.993 — and L1 responds to a correlated group by keeping
    one member essentially at random. A different fold keeps a different one.
    The stability count below measures exactly that.
  * **Selection leakage.** If the penalty strength, or the feature set, is chosen
    using all 26 patients and then scored by leave-one-out, the held-out patient
    already influenced the model. **Every fold here re-runs the whole procedure
    on its own 25 patients**, penalty selection included, so the reported number
    is what a new patient would actually see.

Compared: L1 (lasso), L2 (ridge), elastic net, all on the full feature set with
nested penalty selection, against the current two-feature model under identical
code.

Run: python experiments/13_learned_extractor/kitchen_sink.py

Abbreviations
-------------
L1 / lasso     penalty on the sum of |coefficients|; drives some to exactly zero
L2 / ridge     penalty on the sum of squared coefficients; shrinks, never zeroes
LOO            leave-one-out cross-validation, here leaving out one patient
CV             cross-validation
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

from head_to_head import WORK, leave_one_out as project_loo, score  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BUNDLE = ROOT / "runs/segmenter/unet.pt"
PERMUTATIONS = 1000  # each costs one full cross-validation pass
SEED = 0


def all_features(rgb, mask):
    """Every colour statistic that could plausibly carry signal, with names."""

    selected = mask > 0
    px = rgb[selected].astype(np.float64)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    r_c, g_c = R / total, G / total

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = (lab[:, :, 1] - 128.0)[selected].astype(np.float64)
    b_star = (lab[:, :, 2] - 128.0)[selected].astype(np.float64)
    L_star = lab[:, :, 0][selected].astype(np.float64)
    redness = a_star - 0.5 * b_star
    log_rg = np.log(np.maximum(R, 1) / np.maximum(G, 1))
    log_rb = np.log(np.maximum(R, 1) / np.maximum(B, 1))

    out: dict[str, float] = {}
    for name, v in [("R", R), ("G", G), ("B", B), ("r_chroma", r_c), ("g_chroma", g_c),
                    ("redness", redness), ("a*", a_star), ("b*", b_star), ("L*", L_star),
                    ("log_RG", log_rg), ("log_RB", log_rb)]:
        out[f"{name} mean"] = float(v.mean())
        out[f"{name} median"] = float(np.median(v))
        out[f"{name} std"] = float(v.std())
    # Distribution tails for the two most physically motivated channels.
    for name, v in [("redness", redness), ("r_chroma", r_c)]:
        p25, p75 = np.percentile(v, [25, 75])
        out[f"{name} p25"] = float(p25)
        out[f"{name} p75"] = float(p75)
    out["mask area"] = float(selected.mean())
    return out


ALPHAS = np.logspace(-4, 2, 60)

MODELS = {
    "L1 lasso": lambda rs: LassoCV(alphas=ALPHAS, cv=5, random_state=rs, max_iter=20000),
    "L2 ridge": lambda rs: RidgeCV(alphas=np.logspace(-3, 3, 60)),
    "elastic net": lambda rs: ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.0],
                                           alphas=ALPHAS, cv=5, random_state=rs,
                                           max_iter=20000),
}


def nested_loo(X, y, make_model, folds=None):
    """Cross-validation where the penalty is re-chosen inside every fold.

    `folds=None` is leave-one-out. On a large cohort that is unaffordable inside
    a permutation test — 1,000 shuffles x 217 folds is ~217,000 penalised fits —
    so an integer `folds` runs k-fold instead. The observed statistic and its
    null distribution must always be computed the *same* way, so whichever is
    used here is used for both.

    Returns predictions and how often each feature survived selection.
    """

    n = len(y)
    predictions = np.zeros(n)
    kept = np.zeros(X.shape[1], dtype=int)

    if folds is None:
        splits = [(np.arange(n) != i) for i in range(n)]
    else:
        order = np.random.default_rng(SEED).permutation(n)
        splits = [~np.isin(np.arange(n), order[block])
                  for block in np.array_split(order, folds)]

    for train in splits:
        Xtr, ytr = X[train], y[train]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        model = make_model(SEED).fit((Xtr - mu) / sd, ytr)
        predictions[~train] = model.predict((X[~train] - mu) / sd)
        kept += np.abs(getattr(model, "coef_", np.zeros(X.shape[1]))) > 1e-8

    return predictions, kept


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_local_rows(learned):
    per_patient: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
        rgb = gray_world_white_balance(resize(read_rgb(sample.image_path), WORK))
        mask, _ = learned(rgb)
        if np.count_nonzero(mask) < 500:
            continue
        entry = per_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "f": []})
        entry["f"].append(all_features(rgb, mask))

    names = list(next(iter(per_patient.values()))["f"][0])
    return [{"x": np.array([np.mean([f[k] for f in e["f"]]) for k in names]),
             "hb": e["hb"], "age": e["age"], "sex": e["sex"]}
            for _, e in sorted(per_patient.items(), key=lambda kv: int(kv[0].split(":")[1]))], names


def load_external_rows(learned):
    """The 217 Eyes-Defy subjects, measured with the same 38 statistics."""

    import pandas as pd
    sys.path.insert(0, str(ROOT / "experiments" / "14_pooled_regression"))
    from pooled import EXTERNAL, european_float

    rows, names = [], None
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
            rgb = gray_world_white_balance(resize(read_rgb(photos[0]), WORK))
            mask, _ = learned(rgb)
            if np.count_nonzero(mask) < 500:
                continue
            f = all_features(rgb, mask)
            names = names or list(f)
            sex = str(row.get("Gender", "")).strip().upper()[:1] or None
            rows.append({"x": np.array([f[k] for k in names]), "hb": hb,
                         "age": european_float(row.get("Age")), "sex": sex})
    return rows, names


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", default="local", choices=["local", "external", "pooled"])
    ap.add_argument("--folds", type=int, default=0,
                    help="outer CV folds; 0 = leave-one-out (default for small n)")
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = ap.parse_args()

    learned = get_extractor("unet", model_dir=BUNDLE)
    print(f"measuring cohort '{args.cohort}' ...")
    if args.cohort == "local":
        rows, names = load_local_rows(learned)
    elif args.cohort == "external":
        rows, names = load_external_rows(learned)
    else:
        ext, names = load_external_rows(learned)
        loc, _ = load_local_rows(learned)
        rows = ext + loc

    y = np.array([r["hb"] for r in rows])
    meta = [(r["age"], r["sex"]) for r in rows]
    X = np.array([r["x"] for r in rows])
    n, k = X.shape

    # LOO is affordable as a point estimate at any size here, but not inside a
    # permutation test on a large cohort, so k-fold takes over there. Observed
    # and null always use the same procedure.
    folds = args.folds or None
    perm_folds = folds if folds else (10 if n > 60 else None)

    print(f"{n} patients, {k} candidate features, pipeline C masks")
    print(f"outer CV: {'leave-one-out' if folds is None else str(folds) + '-fold'}"
          f"; permutation test uses "
          f"{'leave-one-out' if perm_folds is None else str(perm_folds) + '-fold'}\n")

    # How collinear is this really?
    corr = np.corrcoef(X, rowvar=False)
    off = np.abs(corr - np.eye(k))
    print(f"feature collinearity: max |r| = {off.max():.3f}, "
          f"{int((off > 0.95).sum() // 2)} pairs above 0.95\n")

    baseline = score(np.array([np.delete(y, i).mean() for i in range(n)]), y, meta)

    # The incumbent, through identical code.
    current_idx = [names.index("r_chroma mean"), names.index("g_chroma mean")]
    results = {}

    print(f"{'model':<26}{'k used':>8}{'MAE':>9}{'R²':>9}{'F1':>8}")
    print("-" * 60)

    # Two versions of the incumbent, and the difference between them matters.
    # The published number uses the project's fixed alpha = 1.0; the second
    # re-chooses alpha inside every fold, exactly as the penalised models below
    # must. Only the second is comparable to them.
    pred_fixed = project_loo(X[:, current_idx], y)
    m_fixed = score(pred_fixed, y, meta)
    results["current 2, alpha=1 fixed"] = {**m_fixed, "k_used": 2.0}
    print(f"{'current 2, alpha=1 fixed':<26}{2:>8}{m_fixed['mae']:>9.3f}"
          f"{m_fixed['r2']:>9.3f}{m_fixed['f1']:>8.3f}")

    pred_cur, _ = nested_loo(X[:, current_idx], y, MODELS["L2 ridge"], folds)
    m_cur = score(pred_cur, y, meta)
    results["current 2, alpha nested"] = {**m_cur, "k_used": 2.0}
    print(f"{'current 2, alpha nested':<26}{2:>8}{m_cur['mae']:>9.3f}"
          f"{m_cur['r2']:>9.3f}{m_cur['f1']:>8.3f}")

    stability = {}
    for label, make in MODELS.items():
        pred, kept = nested_loo(X, y, make, folds)
        m = score(pred, y, meta)
        k_used = float(np.mean([(kept > 0).sum()])) if label == "L2 ridge" else float(
            (kept > 0).sum())
        results[f"all {k} features, {label}"] = {**m, "k_used": k_used}
        stability[label] = {names[i]: int(kept[i]) for i in np.argsort(-kept)[:12]}
        shown = f"{int(kept.max() and (kept > 0).sum())}" if label != "L2 ridge" else "all"
        print(f"{'all %d features, %s' % (k, label):<26}{shown:>8}{m['mae']:>9.3f}"
              f"{m['r2']:>9.3f}{m['f1']:>8.3f}")

    print("-" * 60)
    print(f"{'baseline (predict mean)':<26}{'-':>8}{baseline['mae']:>9.3f}"
          f"{baseline['r2']:>9.3f}{baseline['f1']:>8.3f}")

    # ---- selection stability: how often did L1 keep each feature? ----
    print(f"\nL1 selection stability — times kept across the {n} folds:")
    for name, count in list(stability["L1 lasso"].items())[:12]:
        bar = "#" * int(20 * count / n)
        print(f"  {name:<20}{count:>3}/{n}  {bar}")
    always = sum(1 for c in stability["L1 lasso"].values() if c == n)
    print(f"\n  features kept in every fold: {always}")

    # ---- noise floor for the full-feature L1 model ----
    # The observed statistic must be produced by the SAME procedure as the null.
    # On a large cohort the permutation loop drops to k-fold for affordability,
    # so the observed value is recomputed that way rather than reusing the
    # leave-one-out number from the table above — comparing a LOO observation
    # against a k-fold null would be meaningless.
    if perm_folds == folds:
        obs = results[f"all {k} features, L1 lasso"]["r2"]
    else:
        obs_pred, _ = nested_loo(X, y, MODELS["L1 lasso"], perm_folds)
        obs = r2_of(obs_pred, y)
        print(f"\n  (observed R² recomputed at {perm_folds}-fold for the permutation "
              f"test: {obs:+.3f}, vs {results[f'all {k} features, L1 lasso']['r2']:+.3f} "
              f"leave-one-out)")

    rng = np.random.default_rng(SEED)
    null = np.empty(args.permutations)
    for i in range(args.permutations):
        shuffled = rng.permutation(y)
        pred, _ = nested_loo(X, shuffled, MODELS["L1 lasso"], perm_folds)
        null[i] = r2_of(pred, shuffled)
    p_value = float((null >= obs).mean())
    print(f"\nnoise floor for all-{k}-feature L1 ({args.permutations:,} permutations, "
          f"{'leave-one-out' if perm_folds is None else str(perm_folds) + '-fold'}):")
    print(f"  observed R² {obs:+.3f}   null mean {null.mean():+.3f}   p = {p_value:.3f}")

    out = Path(__file__).resolve().parent / "kitchen_sink.json"
    out.write_text(json.dumps({
        "n_patients": n, "n_features": k, "feature_names": names,
        "max_abs_correlation": float(off.max()),
        "baseline": baseline, "results": results,
        "l1_stability": stability["L1 lasso"],
        "l1_noise_floor": {"observed_r2": obs, "null_mean": float(null.mean()),
                           "p": p_value, "permutations": PERMUTATIONS},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
