"""Compare extractors on every labelled patient available, not just the local 26.

`compare_bundles.py` scores the downstream haemoglobin model on the 26 local
patients. That was the only labelled cohort when it was written; it no longer
is. Eyes-Defy-Anemia supplies **217 further subjects** with haemoglobin, age and
sex, which is nearly ten times the data and the difference between a comparison
that can resolve an effect and one that cannot.

The two cohorts are reported **separately, never pooled**. Experiment 14 measured
their features as more than three local standard deviations apart, with transfer
between them at R² −1.9; pooling would mostly teach a between-cohort offset and
the resulting number would describe neither site. An extractor that is genuinely
better should be better on *both* cohorts independently, and that is a much
stronger claim than winning on one.

Features come from running each extractor on the original photographs, so both
cohorts are measured by the same instrument.

    python experiments/13_learned_extractor/compare_extractors_full.py \\
        --bundles runs/segmenter/unet_v3.pt runs/segmenter/unet_v5.pt

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
px     pixels
"""

from __future__ import annotations

import argparse
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
from anemia.segment import get_extractor  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

from head_to_head import WORK, leave_one_out, score  # noqa: E402
from pooled import EXTERNAL, european_float  # noqa: E402

MIN_PIXELS = 500
PERMUTATIONS = 2000
ON_TISSUE = 0.0


def chroma(rgb, mask):
    px = rgb[mask > 0].astype(np.float64)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    total = np.maximum(R + G + B, 1e-6)
    return np.array([(R / total).mean(), (G / total).mean()])


def redness_of(rgb, mask):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    red = (lab[:, :, 1] - 128.0) - 0.5 * (lab[:, :, 2] - 128.0)
    return float(red[mask > 0].mean())


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def permutation_p(X, y, observed):
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(PERMUTATIONS):
        s = rng.permutation(y)
        hits += r2_of(leave_one_out(X, s), s) >= observed
    return hits / PERMUTATIONS


def measure_all(models):
    """One pass over every image, every extractor. Images are the expensive part."""

    cohorts = {"local": {}, "external": {}}
    placement = {n: {"local": [], "external": []} for n in models}
    empties = {n: {"local": 0, "external": 0} for n in models}

    def handle(cohort, key, path, hb, age, sex, aggregate):
        rgb = gray_world_white_balance(resize(read_rgb(path), WORK))
        entry = cohorts[cohort].setdefault(
            key, {"hb": hb, "age": age, "sex": sex, "f": {n: [] for n in models}})
        for name, seg in models.items():
            mask, _ = seg(rgb)
            if np.count_nonzero(mask) < MIN_PIXELS:
                empties[name][cohort] += 1
                continue
            entry["f"][name].append(chroma(rgb, mask))
            placement[name][cohort].append(redness_of(rgb, mask))

    print("measuring local cohort ...")
    for s in load_local_cohort(ROOT):
        handle("local", s.patient_id, s.image_path, s.hb, s.age_years, s.sex, True)

    print("measuring external cohort ...")
    import pandas as pd
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
            sex = str(row.get("Gender", "")).strip().upper()[:1] or None
            handle("external", f"{country}:{folder.name}", photos[0], hb,
                   european_float(row.get("Age")), sex, False)

    return cohorts, placement, empties


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundles", nargs="+", type=Path,
                    default=[ROOT / "runs/segmenter/unet_v3.pt"])
    ap.add_argument("--with-classical", action="store_true", default=True)
    args = ap.parse_args()

    models = {p.stem: LearnedSegmenter(p) for p in args.bundles}
    if args.with_classical:
        models["classical"] = get_extractor("refined")

    cohorts, placement, empties = measure_all(models)

    results = {}
    for cohort in ("local", "external"):
        data = cohorts[cohort]
        usable = [k for k, v in data.items() if all(v["f"][n] for n in models)]
        y = np.array([data[k]["hb"] for k in usable])
        meta = [(data[k]["age"], data[k]["sex"]) for k in usable]
        base = score(np.array([np.delete(y, i).mean() for i in range(len(y))]), y, meta)

        print(f"\n{'='*72}\n{cohort.upper()} COHORT — {len(usable)} patients usable by all "
              f"extractors\n{'='*72}")
        print(f"{'extractor':<14}{'med redness':>13}{'empty':>7}{'MAE':>9}{'R²':>9}"
              f"{'F1':>8}{'p':>8}")
        print("-" * 68)
        for name in models:
            X = np.array([np.mean(data[k]["f"][name], axis=0) for k in usable])
            m = score(leave_one_out(X, y), y, meta)
            p = permutation_p(X, y, m["r2"])
            red = np.array(placement[name][cohort])
            results.setdefault(name, {})[cohort] = {
                **m, "permutation_p": p, "n": len(usable),
                "median_redness": float(np.median(red)) if len(red) else float("nan"),
                "empty": empties[name][cohort]}
            print(f"{name:<14}{np.median(red):>13.2f}{empties[name][cohort]:>7}"
                  f"{m['mae']:>9.3f}{m['r2']:>9.3f}{m['f1']:>8.3f}{p:>8.3f}")
        print("-" * 68)
        print(f"{'baseline':<14}{'':>13}{'':>7}{base['mae']:>9.3f}{base['r2']:>9.3f}"
              f"{base['f1']:>8.3f}")
        results.setdefault("_baseline", {})[cohort] = base

    print(f"\n{'='*72}\nVERDICT — an extractor must win on BOTH cohorts to be better\n{'='*72}")
    names = [n for n in models]
    print(f"{'extractor':<14}{'local R²':>11}{'local p':>10}{'ext R²':>10}{'ext p':>9}")
    for name in names:
        l, e = results[name]["local"], results[name]["external"]
        print(f"{name:<14}{l['r2']:>11.3f}{l['permutation_p']:>10.3f}"
              f"{e['r2']:>10.3f}{e['permutation_p']:>9.3f}")

    out = Path(__file__).resolve().parent / "compare_extractors_full.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
