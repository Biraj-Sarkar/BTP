"""Many sclera-relative features, penalised. Does portability survive richness?

Two results point in opposite directions and neither tested the combination:

  * Experiment 13 — at n = 217, **more features help**. A penalised model over
    38 absolute colour statistics reaches R² 0.421 against the two-feature
    model's 0.350, with nine features stable across all 217 folds.
  * Experiment 15 — absolute features **do not transfer** between cameras
    (R² −2.010), while a single sclera-relative feature does (+0.032) at the
    cost of most of the within-cohort accuracy (0.221 vs 0.346).

So: does a *penalised model over many relative features* recover the accuracy
that sclera-referencing gave up, while keeping the portability that made it
worth giving up? That is the only configuration that would be deployable across
cameras, and nothing has tested it.

Every feature here is a comparison between the conjunctiva and the sclera in the
**same photograph**. Differences are used for quantities that can be negative
(a\*, b\*, redness); ratios and log-ratios for quantities that cannot (channel
means, chromaticity, lightness). Both forms are supplied because which one
cancels the camera better is an empirical question.

Run: python experiments/15_sclera_relative/penalised_relative.py

Abbreviations
-------------
L1 / lasso     penalty on the sum of |coefficients|; drives some to exactly zero
L2 / ridge     penalty on the sum of squared coefficients; shrinks, never zeroes
LOO            leave-one-out cross-validation, here leaving out one patient
sd             standard deviation
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
sys.path.insert(0, str(ROOT / "experiments" / "13_learned_extractor"))
sys.path.insert(0, str(ROOT / "experiments" / "14_pooled_regression"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor, legacy_bright_neutral_mask  # noqa: E402

from head_to_head import WORK  # noqa: E402
from kitchen_sink import MODELS, nested_loo  # noqa: E402
from pooled import EXTERNAL, european_float  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BUNDLE = ROOT / "runs/segmenter/unet.pt"
MIN_PIXELS = 500
PERMUTATIONS = 500
SEED = 0
EPS = 1e-6


def region_stats(rgb, mask):
    """Every base statistic for one region, as a flat dict."""

    sel = mask > 0
    px = rgb[sel].astype(np.float64)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    total = np.maximum(R + G + B, EPS)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = (lab[:, :, 1][sel] - 128.0).astype(np.float64)
    b = (lab[:, :, 2][sel] - 128.0).astype(np.float64)
    L = lab[:, :, 0][sel].astype(np.float64)

    out = {}
    for name, v in [("R", R), ("G", G), ("B", B), ("r", R / total), ("g", G / total),
                    ("L", L), ("a", a), ("b", b), ("redness", a - 0.5 * b)]:
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_median"] = float(np.median(v))
        out[f"{name}_std"] = float(v.std())
    return out


# Quantities that are strictly positive can be divided; ones that cross zero
# cannot, because a ratio flips sign and explodes near the origin.
POSITIVE = ("R", "G", "B", "r", "g", "L")
SIGNED = ("a", "b", "redness")


def relative_features(conj, scl):
    """Conjunctiva measured against sclera in the same frame."""

    out = {}
    for base in POSITIVE:
        for stat in ("mean", "median"):
            c, s = conj[f"{base}_{stat}"], scl[f"{base}_{stat}"]
            out[f"{base}_{stat} ratio"] = c / max(abs(s), EPS)
            out[f"{base}_{stat} logratio"] = float(np.log(max(c, EPS) / max(abs(s), EPS)))
            out[f"{base}_{stat} diff"] = c - s
        # Spread ratio: how much more variable the tissue is than its reference.
        out[f"{base}_std ratio"] = conj[f"{base}_std"] / max(scl[f"{base}_std"], EPS)
    for base in SIGNED:
        for stat in ("mean", "median"):
            out[f"{base}_{stat} diff"] = conj[f"{base}_{stat}"] - scl[f"{base}_{stat}"]
        out[f"{base}_std ratio"] = conj[f"{base}_std"] / max(scl[f"{base}_std"], EPS)
    return out


def measure(path, extractor):
    rgb = gray_world_white_balance(resize(read_rgb(path), WORK))
    conj, _ = extractor(rgb)
    if np.count_nonzero(conj) < MIN_PIXELS:
        return None
    scl = cv2.bitwise_and(legacy_bright_neutral_mask(rgb), cv2.bitwise_not(conj))
    if np.count_nonzero(scl) < MIN_PIXELS:
        return None
    return relative_features(region_stats(rgb, conj), region_stats(rgb, scl))


def load_local(extractor):
    per: dict[str, dict] = {}
    for s in load_local_cohort(ROOT):
        f = measure(s.image_path, extractor)
        if f is None:
            continue
        per.setdefault(s.patient_id, {"hb": s.hb, "f": []})["f"].append(f)
    names = list(next(iter(per.values()))["f"][0])
    rows = [{"x": np.array([np.mean([f[k] for f in e["f"]]) for k in names]), "hb": e["hb"]}
            for _, e in sorted(per.items(), key=lambda kv: int(kv[0].split(":")[1]))]
    return rows, names


def load_external(extractor, names):
    import pandas as pd
    rows = []
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
                continue
            rows.append({"x": np.array([f[k] for k in names]), "hb": hb})
    return rows


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_predict(Xtr, ytr, Xte, make_model):
    """Fit on one cohort, predict another. Standardisation comes from training."""

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    model = make_model(SEED).fit((Xtr - mu) / sd, ytr)
    return model.predict((Xte - mu) / sd), model


def main() -> None:
    extractor = get_extractor("unet", model_dir=BUNDLE)
    print("measuring local cohort ...")
    local, names = load_local(extractor)
    print(f"  {len(local)} patients, {len(names)} relative features")
    print("measuring external cohort ...")
    external = load_external(extractor, names)
    print(f"  {len(external)} subjects\n")

    Xl = np.array([r["x"] for r in local]);  yl = np.array([r["hb"] for r in local])
    Xe = np.array([r["x"] for r in external]); ye = np.array([r["hb"] for r in external])

    pooled_sd = np.sqrt((Xl.var(axis=0) + Xe.var(axis=0)) / 2.0)
    pooled_sd[pooled_sd < 1e-12] = 1.0
    seps = np.abs(Xe.mean(axis=0) - Xl.mean(axis=0)) / pooled_sd
    print(f"cohort separation across {len(names)} relative features: "
          f"median {np.median(seps):.2f}, max {seps.max():.2f}")
    print(f"  features overlapping (separation < 1): {(seps < 1).sum()} of {len(names)}\n")

    print(f"{'model':<24}{'ext R² (10-fold)':>18}{'ext->loc':>10}{'loc->ext':>10}"
          f"{'ext->loc MAE':>14}")
    print("-" * 76)

    results = {}
    for label, make in MODELS.items():
        within, kept = nested_loo(Xe, ye, make, 10)
        within_r2 = r2_of(within, ye)

        pred_l, model_e = fit_predict(Xe, ye, Xl, make)
        pred_e, _ = fit_predict(Xl, yl, Xe, make)
        e2l, l2e = r2_of(pred_l, yl), r2_of(pred_e, ye)
        mae = float(np.abs(pred_l - yl).mean())

        results[label] = {"external_r2_10fold": within_r2, "external_to_local_r2": e2l,
                          "local_to_external_r2": l2e, "external_to_local_mae": mae,
                          "features_kept": int((kept > 0).sum()),
                          "stable_features": [names[i] for i in np.argsort(-kept)[:10]
                                              if kept[i] == 10]}
        print(f"{label:<24}{within_r2:>18.3f}{e2l:>10.3f}{l2e:>10.3f}{mae:>14.3f}")

    print("-" * 76)
    print(f"{'baseline (local mean)':<24}{'':>18}{'':>10}{'':>10}"
          f"{float(np.abs(yl - yl.mean()).mean()):>14.3f}")

    print("\nreference points from earlier experiments, same cohorts:")
    print("  38 absolute features, ridge (exp 13):  ext R² 0.421")
    print("  2 absolute chroma       (exp 15):      ext R² 0.346   ext->loc -2.010")
    print("  1 relative a* difference (exp 15):     ext R² 0.221   ext->loc +0.032")

    best = max(results, key=lambda k: results[k]["external_to_local_r2"])
    print(f"\nbest transfer among relative models: {best} "
          f"(ext->loc R² {results[best]['external_to_local_r2']:+.3f}, "
          f"MAE {results[best]['external_to_local_mae']:.3f})")

    # Permutation test on the best transfer result: shuffle the target in the
    # TEST cohort, so the null is "these predictions carry no information about
    # this cohort's haemoglobin".
    rng = np.random.default_rng(SEED)
    pred_l, _ = fit_predict(Xe, ye, Xl, MODELS[best])
    obs = r2_of(pred_l, yl)
    null = np.array([r2_of(pred_l, rng.permutation(yl)) for _ in range(PERMUTATIONS)])
    p = float((null >= obs).mean())
    print(f"  permutation test ({PERMUTATIONS} shuffles of the local targets): "
          f"p = {p:.3f}")

    out = Path(__file__).resolve().parent / "penalised_relative.json"
    out.write_text(json.dumps({
        "n_local": len(local), "n_external": len(external),
        "n_features": len(names), "feature_names": names,
        "separation_median": float(np.median(seps)),
        "separation_overlapping": int((seps < 1).sum()),
        "results": results, "best_transfer": best,
        "best_transfer_permutation_p": p,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
