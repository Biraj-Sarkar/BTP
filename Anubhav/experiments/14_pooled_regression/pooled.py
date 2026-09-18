"""Train the haemoglobin model on 218 more patients. Does it survive the move?

Every predictive number in this project rests on 26 patients, and experiment 09
showed what that means: a single split reports test R² anywhere from −2.43 to
+0.78 on identical data. Eyes-Defy-Anemia carries haemoglobin, age and sex for
**218 further subjects**, which is the first chance to (a) fit on a sample where
the statistics mean something and (b) run a real **external validation** — fit
on one population and camera, test on a completely different one.

Note what is *not* used here: the dataset's own annotation files. Features come
from running **our** extractor on the original photographs, exactly as the
serving path would. That keeps the local and external cohorts measured by the
same instrument, which is the whole point — and it sidesteps the fact that ~45%
of this dataset's annotations are bounding boxes rather than traced contours.

Five arms, each with its own baseline:

    local only          n=26    LOO            the current published number
    external only       n=218   LOO            does it work at all, given n
    external -> local   fit 218, test 26       the honest transfer test
    local -> external   fit 26,  test 218      the reverse, for symmetry
    pooled              n=244   LOO            what a deployable model would use

Run: python experiments/14_pooled_regression/pooled.py

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
Hgb    haemoglobin, the spreadsheet's column name for it
WHO    World Health Organization, whose age/sex thresholds define anaemia here
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

from head_to_head import WORK, apply_model, leave_one_out, ridge, score  # noqa: E402

BUNDLE = ROOT / "runs/segmenter/unet.pt"
EXTERNAL = ROOT / "datasets/eyes_defy/dataset anemia"
MIN_MASK_PIXELS = 500
PERMUTATIONS = 5000
SEED = 0


def chroma(rgb, mask):
    """Mean r and g chromaticity — the representation the three-way selected."""

    px = rgb[mask > 0].astype(np.float64)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    return np.array([(R / total).mean(), (G / total).mean()])


def measure(path, extractor):
    rgb = gray_world_white_balance(resize(read_rgb(path), WORK))
    mask, _ = extractor(rgb)
    if np.count_nonzero(mask) < MIN_MASK_PIXELS:
        return None
    return chroma(rgb, mask)


def european_float(value):
    """Parse a number that may use a comma as its decimal separator.

    The Italy spreadsheet stores haemoglobin as `15,1` — European convention for
    15.1 g/dL. Left alone this either crashes or, worse, parses as 151, which no
    range check downstream would catch if the constant were plausible. Values
    outside a physiological range are rejected rather than trusted.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_external(extractor):
    """218 subjects: one photograph each, Hgb/Age/Gender from the spreadsheets."""

    rows = []
    rejected = {"unparseable hb": 0, "implausible hb": 0, "no mask": 0, "no photo": 0}
    for country in ("India", "Italy"):
        table = EXTERNAL / country / f"{country}.xlsx"
        if not table.exists():
            print(f"  missing {table}", file=sys.stderr)
            continue
        df = pd.read_excel(table, engine="openpyxl")
        labels = {int(r["Number"]): r for _, r in df.iterrows()
                  if pd.notna(r.get("Number")) and pd.notna(r.get("Hgb"))}

        for folder in sorted((EXTERNAL / country).iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            row = labels.get(int(folder.name))
            if row is None:
                continue
            photos = sorted(folder.glob("*.jpg"))
            if not photos:
                rejected["no photo"] += 1
                continue

            hb = european_float(row["Hgb"])
            if hb is None:
                rejected["unparseable hb"] += 1
                continue
            # A mis-parsed decimal separator turns 15,1 into 151. Anything
            # outside a survivable range is a parsing failure, not a patient.
            if not 3.0 <= hb <= 22.0:
                rejected["implausible hb"] += 1
                continue

            features = measure(photos[0], extractor)
            if features is None:
                rejected["no mask"] += 1
                continue

            sex = str(row.get("Gender", "")).strip().upper()[:1] or None
            age = european_float(row.get("Age"))
            rows.append({"id": f"{country}:{folder.name}", "x": features,
                         "hb": hb, "age": age, "sex": sex})

    dropped = ", ".join(f"{v} {k}" for k, v in rejected.items() if v)
    if dropped:
        print(f"  dropped: {dropped}")
    return rows


def load_local(extractor):
    per_patient: dict[str, dict] = {}
    for sample in load_local_cohort(ROOT):
        features = measure(sample.image_path, extractor)
        if features is None:
            continue
        entry = per_patient.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex, "x": []})
        entry["x"].append(features)
    return [{"id": p, "x": np.mean(e["x"], axis=0), "hb": e["hb"],
             "age": e["age"], "sex": e["sex"]}
            for p, e in sorted(per_patient.items(), key=lambda kv: int(kv[0].split(":")[1]))]


def arrays(rows):
    X = np.array([r["x"] for r in rows])
    y = np.array([r["hb"] for r in rows])
    meta = [(r["age"], r["sex"]) for r in rows]
    return X, y, meta


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def permutation_p(X, y, observed, held_out, n_perm=PERMUTATIONS):
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        pred = (leave_one_out(X, shuffled) if held_out
                else apply_model(X, *ridge(X, shuffled)))
        hits += r2_of(pred, shuffled) >= observed
    return hits / n_perm


def report(name, pred, y, meta, note=""):
    m = score(pred, y, meta)
    n = len(y)
    print(f"{name:<24}{n:>6}{m['mae']:>9.3f}{m['r2']:>9.3f}{m['f1']:>8.3f}"
          f"{m['accuracy']:>10.3f}  {note}")
    return m


def main() -> None:
    extractor = get_extractor("unet", model_dir=BUNDLE)

    print("measuring external cohort (Eyes-Defy-Anemia) with our own extractor...")
    external = load_external(extractor)
    print(f"  {len(external)} subjects with a usable mask and an Hgb value")
    print("measuring local cohort...")
    local = load_local(extractor)
    print(f"  {len(local)} patients\n")

    Xe, ye, me = arrays(external)
    Xl, yl, ml = arrays(local)

    print(f"Hb range — external {ye.min():.1f}-{ye.max():.1f} (mean {ye.mean():.2f}), "
          f"local {yl.min():.1f}-{yl.max():.1f} (mean {yl.mean():.2f})")
    # If the feature distributions differ, transfer will fail for reasons that
    # have nothing to do with the model.
    print(f"feature means — external {np.round(Xe.mean(axis=0), 4)}, "
          f"local {np.round(Xl.mean(axis=0), 4)}")
    print(f"feature  sds  — external {np.round(Xe.std(axis=0), 4)}, "
          f"local {np.round(Xl.std(axis=0), 4)}\n")

    print(f"{'arm':<24}{'n':>6}{'MAE':>9}{'R²':>9}{'F1':>8}{'Acc':>10}")
    print("-" * 74)

    results = {}

    # 1. local only — the current published number
    results["local only (LOO)"] = report("local only (LOO)", leave_one_out(Xl, yl), yl, ml)
    base_l = report("  baseline", np.array([np.delete(yl, i).mean() for i in range(len(yl))]),
                    yl, ml)

    # 2. external only
    results["external only (LOO)"] = report("external only (LOO)",
                                            leave_one_out(Xe, ye), ye, me)
    base_e = report("  baseline", np.array([np.delete(ye, i).mean() for i in range(len(ye))]),
                    ye, me)

    # 3. external -> local: the honest transfer test
    model = ridge(Xe, ye)
    results["external -> local"] = report("external -> local", apply_model(Xl, *model),
                                          yl, ml, "fit 218, test 26")

    # 4. local -> external
    model_l = ridge(Xl, yl)
    results["local -> external"] = report("local -> external", apply_model(Xe, *model_l),
                                          ye, me, "fit 26, test 218")

    # 5. pooled
    Xp = np.vstack([Xe, Xl])
    yp = np.concatenate([ye, yl])
    mp = me + ml
    results["pooled (LOO)"] = report("pooled (LOO)", leave_one_out(Xp, yp), yp, mp)
    base_p = report("  baseline", np.array([np.delete(yp, i).mean() for i in range(len(yp))]),
                    yp, mp)

    print("-" * 74)

    # Noise floors where n makes them informative.
    print(f"\npermutation tests ({PERMUTATIONS:,} shuffles):")
    for label, X, y in (("external only", Xe, ye), ("pooled", Xp, yp)):
        obs = results[f"{label} (LOO)"]["r2"]
        p = permutation_p(X, y, obs, held_out=True)
        print(f"  {label:<16} held-out R² {obs:+.3f}   p = {p:.4f}")
        results[f"{label} (LOO)"]["permutation_p"] = p

    # The fitted equation on the pooled data — what would actually ship.
    b0, b, mu, sd = ridge(Xp, yp)
    print(f"\npooled fitted model (standardised):")
    print(f"  Hb = {b0:.4f} + {b[0]:+.4f}*z(r_chroma) {b[1]:+.4f}*z(g_chroma)")
    print(f"  centres {np.round(mu, 6)}  scales {np.round(sd, 6)}")

    out = Path(__file__).resolve().parent / "pooled.json"
    out.write_text(json.dumps({
        "n_external": len(external), "n_local": len(local),
        "baselines": {"local": base_l, "external": base_e, "pooled": base_p},
        "results": results,
        "pooled_model": {"intercept": float(b0), "coefficients": b.tolist(),
                         "centres": mu.tolist(), "scales": sd.tolist()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
