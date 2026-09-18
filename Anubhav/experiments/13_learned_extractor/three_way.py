"""Three-way head-to-head: classical, CIELAB, and the learned extractor.

Experiment 11 decided between two pipelines. This repeats that comparison
verbatim with a third, so the learned extractor is judged by the same protocol
that produced the existing deployment recommendation rather than by a friendlier
one written after the fact.

**The method is imported, not reimplemented.** `ridge`, `leave_one_out`,
`score`, `representations` and the patient aggregation all come from
`experiments/11_head_to_head/head_to_head.py`, so any difference in the numbers
below is a difference in the pipelines and not in how they were measured.

The three arms:

    A  gray-world white balance  + seeded-grabCut extractor   (classical)
    B  CIELAB normalisation      + CIELAB extractor
    C  gray-world white balance  + fine-tuned U-Net           (learned)

A and C share their colour normalisation and differ **only** in which pixels
the mask selects, which is what isolates the extractor's contribution. B
differs in both normalisation and extractor, exactly as in experiment 11.

Run: python experiments/13_learned_extractor/three_way.py

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
k      number of free feature parameters
CV     cross-validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import get_extractor, heuristic_segmenter  # noqa: E402

import cielab_pipeline as cp  # noqa: E402

# The measurement method, taken wholesale from the two-way experiment.
from head_to_head import (  # noqa: E402
    ALPHA,
    REPRESENTATIONS,
    WORK,
    apply_model,
    leave_one_out,
    representations,
    ridge,
    score,
)

PIPELINES = ("A", "B", "C")
LABELS = {"A": "gray-world + grabCut", "B": "CIELAB + CIELAB", "C": "gray-world + U-Net"}
PERMUTATIONS = 20000
BUNDLE = ROOT / "runs/segmenter/unet.pt"


def collect():
    """Per patient, per pipeline, every representation — one pass over the cohort."""

    learned = get_extractor("unet", model_dir=BUNDLE)
    data: dict[str, dict] = {}
    skipped = {"A": 0, "B": 0, "C": 0}

    for sample in load_local_cohort(ROOT):
        rgb = resize(read_rgb(sample.image_path), WORK)

        balanced = gray_world_white_balance(rgb)
        mask_a, _ = heuristic_segmenter(balanced)

        norm_rgb, _ = cp.cielab_normalization(rgb)
        mask_b, _ = cp.segment_conjunctiva(norm_rgb)

        # The learned extractor measures on the same gray-world image as A, so
        # A and C differ only in the mask.
        mask_c, _ = learned(balanced)

        masks = {"A": mask_a, "B": mask_b, "C": mask_c}
        for name, mask in masks.items():
            if not (mask > 0).any():
                skipped[name] += 1

        # A capture must be usable by ALL THREE, or the arms would be scored on
        # different patients — the confound that made the naive `fit-linear`
        # comparison meaningless (see RESULTS.md).
        if not all((m > 0).any() for m in masks.values()):
            continue

        entry = data.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex,
            "A": [], "B": [], "C": []})
        entry["A"].append(representations(balanced, mask_a))
        entry["B"].append(representations(norm_rgb, mask_b))
        entry["C"].append(representations(balanced, mask_c))

    return data, skipped


def stack(data, patients, pipeline, key):
    return np.array([np.mean([f[key] for f in data[p][pipeline]], axis=0)
                     for p in patients])


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def noise_floor(X, y, observed, held_out):
    """R² this many parameters would reach on shuffled labels (20,000 refits)."""

    rng = np.random.default_rng(0)
    null = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        shuffled = rng.permutation(y)
        pred = (leave_one_out(X, shuffled) if held_out
                else apply_model(X, *ridge(X, shuffled)))
        null[i] = r2_of(pred, shuffled)
    return float(null.mean()), float((null >= observed).mean())


def main() -> None:
    data, skipped = collect()
    patients = sorted(data, key=lambda p: int(p.split(":")[1]))
    y = np.array([data[p]["hb"] for p in patients])
    meta = [(data[p]["age"], data[p]["sex"]) for p in patients]
    n = len(y)

    print(f"{n} patients usable by all three pipelines, leave-one-patient-out")
    print(f"captures with an empty mask: "
          + ", ".join(f"{k} {v}" for k, v in skipped.items()) + "\n")

    baseline_loo = np.array([np.delete(y, i).mean() for i in range(n)])
    base = score(baseline_loo, y, meta)

    # ---------- 1. exploratory ----------
    print("=" * 86)
    print("1. EXPLORATORY — every representation, all three pipelines (held out)")
    print("   Picking the winner from this table is selection on the test set.")
    print("=" * 86)
    header = f"\n{'representation':<14}{'k':>3}"
    for p in PIPELINES:
        header += f"{p + ': MAE':>10}{p + ': R²':>9}"
    print(header)
    print("-" * 86)

    results: dict[str, dict] = {}
    for key in REPRESENTATIONS:
        row = {}
        for p in PIPELINES:
            X = stack(data, patients, p, key)
            row[p] = score(leave_one_out(X, y), y, meta)
            row[f"{p}_k"] = X.shape[1]
        results[key] = row
        line = f"{key:<14}{row['A_k']:>3}"
        for p in PIPELINES:
            line += f"{row[p]['mae']:>10.3f}{row[p]['r2']:>9.3f}"
        print(line)
    print("-" * 86)
    line = f"{'baseline':<14}{'-':>3}"
    for _ in PIPELINES:
        line += f"{base['mae']:>10.3f}{base['r2']:>9.3f}"
    print(line)

    best = {p: min(REPRESENTATIONS, key=lambda k: results[k][p]["mae"]) for p in PIPELINES}
    print("\nbest per pipeline: " + ",  ".join(f"{p} = {best[p]}" for p in PIPELINES))

    # ---------- 2. full metrics, held out ----------
    print("\n" + "=" * 86)
    print("2. FULL METRIC COMPARISON — best representation each, HELD OUT (LOO)")
    print("=" * 86)
    chosen = {p: results[best[p]][p] for p in PIPELINES}
    print(f"\n{'metric':<18}" + "".join(f"{p:>16}" for p in PIPELINES) + f"{'baseline':>14}")
    print(f"{'':18}" + "".join(f"{best[p]:>16}" for p in PIPELINES))
    print("-" * 80)
    rows = [("mse", "MSE (g/dL)²"), ("rmse", "RMSE (g/dL)"), ("mae", "MAE (g/dL)"),
            ("r2", "R²"), ("bias", "bias (g/dL)"), ("pearson_r", "Pearson r"),
            ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
            ("specificity", "Specificity"), ("f1", "F1")]
    for key, label in rows:
        print(f"{label:<18}"
              + "".join(f"{chosen[p][key]:>16.3f}" for p in PIPELINES)
              + f"{base.get(key, float('nan')):>14.3f}")
    print("-" * 80)
    print(f"{'confusion':<18}" + "".join(
        f"{'TP%d FP%d FN%d TN%d' % (chosen[p]['tp'], chosen[p]['fp'], chosen[p]['fn'], chosen[p]['tn']):>16}"
        for p in PIPELINES))

    # ---------- 3. fitted on all patients ----------
    print("\n" + "=" * 86)
    print(f"3. FITTED ON ALL {n} PATIENTS — scored in-sample (describes the fit)")
    print("=" * 86)
    print("\nEvery patient contributes to the fit, so this describes the MODEL,")
    print("not what to expect from a new patient. The in-sample baseline is the")
    print("full-sample mean, whose R² is 0.000 by definition — a fitted model")
    print("with an intercept essentially cannot lose to it, so each R² is also")
    print(f"reported against a {PERMUTATIONS:,}-permutation noise floor.\n")

    fitted_base = score(np.full(n, y.mean()), y, meta)
    fitted: dict[str, dict] = {}
    floors: dict[str, dict] = {}
    for p in PIPELINES:
        X = stack(data, patients, p, best[p])
        pred = apply_model(X, *ridge(X, y))
        fitted[p] = score(pred, y, meta)
        null_mean, p_value = noise_floor(X, y, fitted[p]["r2"], held_out=False)
        loo_null_mean, loo_p = noise_floor(X, y, chosen[p]["r2"], held_out=True)
        floors[p] = {"in_sample_null_mean": null_mean, "in_sample_p": p_value,
                     "loo_null_mean": loo_null_mean, "loo_p": loo_p,
                     "k": X.shape[1]}

    print(f"{'metric':<18}" + "".join(f"{p:>16}" for p in PIPELINES) + f"{'baseline':>14}")
    print("-" * 80)
    for key, label in rows:
        print(f"{label:<18}"
              + "".join(f"{fitted[p][key]:>16.3f}" for p in PIPELINES)
              + f"{fitted_base.get(key, float('nan')):>14.3f}")
    print("-" * 80)

    # ---------- 4. noise floors ----------
    print("\n" + "=" * 86)
    print("4. AGAINST THE NOISE FLOOR — shuffle Hb, refit, rescore")
    print("=" * 86)
    print(f"\n{'pipeline':<22}{'k':>3}{'in-sample R²':>14}{'null':>8}{'p':>8}"
          f"{'held-out R²':>14}{'null':>8}{'p':>8}")
    print("-" * 86)
    for p in PIPELINES:
        f = floors[p]
        print(f"{p + ' ' + LABELS[p]:<22}{f['k']:>3}{fitted[p]['r2']:>14.3f}"
              f"{f['in_sample_null_mean']:>8.3f}{f['in_sample_p']:>8.3f}"
              f"{chosen[p]['r2']:>14.3f}{f['loo_null_mean']:>8.3f}{f['loo_p']:>8.3f}")
    print("-" * 86)
    print(f"{'baseline':<25}{0.0:>14.3f}{'':>8}{'':>8}"
          f"{1 - (n / (n - 1)) ** 2:>14.3f}")

    # ---------- 5. paired comparison, C vs A ----------
    print("\n" + "=" * 86)
    print("5. PAIRED COMPARISON — C against A, per patient (held out)")
    print("=" * 86)
    err_a = np.abs(leave_one_out(stack(data, patients, "A", best["A"]), y) - y)
    err_c = np.abs(leave_one_out(stack(data, patients, "C", best["C"]), y) - y)
    diff = err_a - err_c  # positive means C is closer on that patient

    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, n))
    null = (signs * diff).mean(axis=1)
    p_paired = float((np.abs(null) >= abs(diff.mean())).mean())
    boot = rng.choice(diff, size=(10000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\nC is closer on {int((diff > 0).sum())} of {n} patients")
    print(f"mean advantage to C: {diff.mean():+.3f} g/dL  (95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"paired permutation p = {p_paired:.3f}")

    out = Path(__file__).resolve().parent / "three_way.json"
    out.write_text(json.dumps({
        "n_patients": n, "empty_masks": skipped, "labels": LABELS,
        "best_representation": best,
        "exploratory": {k: {p: results[k][p] for p in PIPELINES} for k in REPRESENTATIONS},
        "held_out": {**{p: chosen[p] for p in PIPELINES}, "baseline": base},
        "fitted_on_all": {**{p: fitted[p] for p in PIPELINES}, "baseline": fitted_base},
        "noise_floors": floors,
        "paired_C_vs_A": {"mean_advantage": float(diff.mean()),
                          "ci95": [float(lo), float(hi)],
                          "wins_C": int((diff > 0).sum()), "p_value": p_paired},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
