"""Score the person-safe models with a precision-weighted metric, and pick the mask threshold for it.

Why weight precision
--------------------
The extractor's mask decides which pixels the haemoglobin model reads. A pixel
wrongly included (eyelid skin, sclera, a lash, glare) pulls the measured colour
toward the wrong tissue; a pixel wrongly left out only removes some conjunctiva,
and the rest still reads roughly the same colour. Dice, which experiments 20-24
report, is F1 on pixels and counts both errors equally.

The metric, fixed before any model was rescored
-----------------------------------------------
F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP)

the F-beta score with beta = 0.5: precision weighs twice as much as recall.
Equivalently the Tversky index with weight 0.8 on false positives and 0.2 on
false negatives (Dice is 0.5 and 0.5). Scored per image and averaged over
images, the same convention as Dice in experiments 20-24. Dice stays the main
number for comparison with published work; F0.5 is the task-specific one.

The threshold
-------------
Each model gives every pixel a probability; experiments 20-24 keep pixels at
0.5 or above. Raising the threshold trades recall for precision without
retraining. Every architecture in experiment 24 has recall above precision at
0.5, so all of them lean toward over-segmenting there.

For each of the 30 models, the threshold with the highest mean F0.5 on that
split's *validation* images is chosen from 0.05, 0.10, ..., 0.95, 0.96, 0.97,
0.98, 0.99, 0.995 and 0.999, and the test images are scored at it. The test set never influences the choice. Validation
images are the ones that already chose each model's best epoch; using them for
one more setting is the usual practice.

Questions
---------
1. Does the ranking change under F0.5 at the default threshold of 0.5?
2. Does it change when every model gets its own validation-chosen threshold?
   This is the fairer comparison: where a model's 0.5 happens to fall is its
   calibration, not its ability to find the conjunctiva.
3. What does the chosen threshold buy and cost? Test F0.5 at the chosen
   threshold against 0.5, per architecture; and Dice, precision, recall, mask
   size and empty masks at each.
4. Control: is the gain specific to weighting precision? The same procedure
   with Dice in place of F0.5 (threshold chosen for validation Dice, test Dice
   at it against 0.5). If Dice also gained, 0.5 would simply be a poor default;
   if not, the higher threshold is the price of the precision weighting.

Statistics are those of experiments 21 and 24: mean over the five splits with a
95% interval, rank within each split, the corrected resampled t-test
(Nadeau-Bengio) on per-split scores, Holm within each family. Families: the two
from experiment 20, plus "chosen threshold minus 0.5", one test per
architecture.

Nothing is trained. The masks are the raw network output with no clean-up, as
in experiments 20-24. As a check, Dice, precision and recall at 0.5 must match
experiment 24's per_split.csv. Pixel counts are saved in pixel_counts.npz and
reused on later runs if the threshold grid is unchanged (pass --rescore to run
the models again).

Run
---
    python experiments/25_precision_weighted/run.py
    python experiments/25_precision_weighted/figure.py

Abbreviations
-------------
TP, FP, FN   true-positive, false-positive and false-negative pixels
Dice         2 TP / (2 TP + FP + FN), equal to F1
SD           standard deviation
CI           confidence interval
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP24 = ROOT / "experiments/24_grouped_splits"
sys.path.insert(0, str(ROOT / "src"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp24 = _load(EXP24 / "run.py", "exp24_run")
exp21, compare = exp24.exp21, exp24.compare
LABEL, FAMILIES, holm = compare.LABEL, compare.FAMILIES, compare.holm
SEEDS, ARCHS = exp24.SEEDS, exp24.ARCHS

BETA = 0.5
# Steps of 0.05, then finer near 1. The first version stopped at 0.95; the first
# model scored chose 0.95 on its validation images, an edge of the grid, so the
# grid was extended before any further model was scored.
THRESHOLDS = np.concatenate([np.round(np.arange(0.05, 0.951, 0.05), 2),
                             [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]])
DEFAULT = 0.5
I_DEFAULT = int(np.flatnonzero(np.isclose(THRESHOLDS, DEFAULT))[0])
EPS = 1e-6  # the same smoothing as experiment 20's score_pair


# ------------------------------------------------------------------ pixel counts

@torch.no_grad()
def counts(bundle: dict, pairs, device) -> np.ndarray:
    """TP, FP, FN per image and threshold: shape (images, thresholds, 3).

    Preprocessing copies experiment 20's evaluate_bundle line for line, so the
    0.5 column reproduces the masks experiments 20-24 scored.
    """

    config = bundle["config"]
    model = compare.build(config.get("arch", "unet"), config["encoder"], pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    size = 320

    out = np.zeros((len(pairs), len(THRESHOLDS), 3), dtype=np.int64)
    for n, (image_path, mask_path) in enumerate(pairs):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        truth = cv2.resize(truth, (size, size), interpolation=cv2.INTER_NEAREST) > 127

        x = ((image.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        predicted = prob[None] >= THRESHOLDS[:, None, None]
        out[n, :, 0] = (predicted & truth).sum(axis=(1, 2))
        out[n, :, 1] = (predicted & ~truth).sum(axis=(1, 2))
        out[n, :, 2] = (~predicted & truth).sum(axis=(1, 2))
    return out


def per_image(c: np.ndarray) -> dict:
    """Per-image metrics from counts of any leading shape (..., 3)."""

    tp, fp, fn = (c[..., k].astype(float) for k in range(3))
    b2 = BETA ** 2
    return {
        "f05": ((1 + b2) * tp + EPS) / ((1 + b2) * tp + b2 * fn + fp + EPS),
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "area_ratio": (tp + fp) / np.maximum(tp + fn, 1),  # predicted area over true area
        "empty": (tp + fp) == 0,
    }


# ------------------------------------------------------------------ scoring

def cached_counts() -> dict | None:
    path = HERE / "pixel_counts.npz"
    if not path.exists():
        return None
    saved = np.load(path)
    if not np.array_equal(saved["thresholds"], THRESHOLDS):
        return None
    return {k: saved[k] for k in saved.files if k != "thresholds"}


def score_all(rescore: bool = False) -> tuple[list, dict]:
    trainer = compare._trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pairs = exp24.labelled_pairs(trainer)
    cache = None if rescore else cached_counts()
    print("pixel counts: " + ("reused from pixel_counts.npz" if cache else "scoring every model"),
          flush=True)

    rows, arrays = [], {}
    for seed in SEEDS:
        for arch in ARCHS:
            path = exp24.bundle_path(seed, arch)
            assert exp24.trained_as_expected(path, seed, arch), f"missing bundle {path}"
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            test_ids = set(bundle["metrics"]["test_subjects"])
            val_ids = set(bundle["provenance"]["val_subjects"])
            test = [(i, m) for i, m, _ in pairs if trainer.subject_of(i) in test_ids]
            val = [(i, m) for i, m, _ in pairs if trainer.subject_of(i) in val_ids]
            assert len(test) == len(test_ids) and len(val) == len(val_ids)

            key = f"seed{seed}_{arch}"
            if cache:
                c_test, c_val = cache[f"{key}_test"], cache[f"{key}_val"]
                assert len(c_test) == len(test) and len(c_val) == len(val)
            else:
                c_test, c_val = counts(bundle, test, device), counts(bundle, val, device)
            arrays[f"{key}_test"] = c_test
            arrays[f"{key}_val"] = c_val
            m_test, m_val = per_image(c_test), per_image(c_val)

            val_curve = m_val["f05"].mean(axis=0)
            test_curve = m_test["f05"].mean(axis=0)
            k = int(np.argmax(val_curve))
            # Control: the same choice made for Dice instead of F0.5.
            k_dice = int(np.argmax(m_val["dice"].mean(axis=0)))
            row = {"seed": seed, "arch": arch, "n_test": len(test), "n_val": len(val),
                   "train_images": bundle["provenance"]["train_images"],
                   "threshold": float(THRESHOLDS[k]),
                   "val_f05_at_default": float(val_curve[I_DEFAULT]),
                   "val_f05_at_chosen": float(val_curve[k]),
                   "test_best_threshold": float(THRESHOLDS[int(np.argmax(test_curve))]),
                   "test_best_f05": float(test_curve.max()),
                   "max_probability_reached": float(THRESHOLDS[np.flatnonzero(
                       (c_test[:, :, 0] + c_test[:, :, 1]).sum(axis=0) > 0).max()]),
                   "dice_threshold": float(THRESHOLDS[k_dice]),
                   "dice_at_dice_threshold": float(m_test["dice"][:, k_dice].mean()),
                   "val_curve": val_curve.tolist(), "test_curve": test_curve.tolist(),
                   "per_image_f05_default": m_test["f05"][:, I_DEFAULT].tolist(),
                   "per_image_f05_chosen": m_test["f05"][:, k].tolist()}
            for tag, idx in (("default", I_DEFAULT), ("chosen", k)):
                for name in ("f05", "dice", "precision", "recall"):
                    row[f"{name}_{tag}"] = float(m_test[name][:, idx].mean())
                row[f"area_ratio_{tag}"] = float(np.median(m_test["area_ratio"][:, idx]))
                row[f"empty_{tag}"] = int(m_test["empty"][:, idx].sum())
            rows.append(row)
            print(f"  seed {seed:>2} {LABEL[arch]:<16} threshold {row['threshold']:.2f}  "
                  f"test F0.5 {row['f05_default']:.4f} -> {row['f05_chosen']:.4f}  "
                  f"Dice {row['dice_default']:.4f} -> {row['dice_chosen']:.4f}", flush=True)
    return rows, arrays


def check_against_exp24(rows: list) -> float:
    """Largest gap between our 0.5 column and experiment 24's stored scores."""

    stored = {(int(r["seed"]), r["arch"]): r for r in csv.DictReader(open(EXP24 / "per_split.csv"))}
    worst = 0.0
    for r in rows:
        s = stored[r["seed"], r["arch"]]
        for name in ("dice", "precision", "recall"):
            worst = max(worst, abs(r[f"{name}_default"] - float(s[name])))
    return worst


# ------------------------------------------------------------------ statistics

def split_ranks(rows: list, key: str) -> dict:
    by = {(r["seed"], r["arch"]): r for r in rows}
    ranks = {a: [] for a in ARCHS}
    for s in SEEDS:
        order = sorted(ARCHS, key=lambda a: by[s, a][key], reverse=True)
        for a in ARCHS:
            ranks[a].append(order.index(a) + 1)
    return {a: {"mean_rank": float(np.mean(v)), "times_first": int(sum(x == 1 for x in v))}
            for a, v in ranks.items()}


def summarise(rows: list) -> dict:
    by = {(r["seed"], r["arch"]): r for r in rows}
    n_train = int(round(np.mean([r["train_images"] for r in rows])))
    n_test = int(np.mean([r["n_test"] for r in rows]))
    per_split = lambda a, key: np.array([by[s, a][key] for s in SEEDS])

    keys = [f"{m}_{t}" for t in ("default", "chosen")
            for m in ("f05", "dice", "precision", "recall")]
    summary = {a: {k: exp21.describe(per_split(a, k)) for k in keys + ["threshold"]} for a in ARCHS}
    for a in ARCHS:
        summary[a]["thresholds"] = per_split(a, "threshold").tolist()
        summary[a]["dice_thresholds"] = per_split(a, "dice_threshold").tolist()
        summary[a]["dice_at_dice_threshold"] = exp21.describe(per_split(a, "dice_at_dice_threshold"))
        summary[a]["max_probability_reached"] = per_split(a, "max_probability_reached").tolist()
        summary[a]["test_best_thresholds"] = per_split(a, "test_best_threshold").tolist()
        summary[a]["empty_default"] = int(per_split(a, "empty_default").sum())
        summary[a]["empty_chosen"] = int(per_split(a, "empty_chosen").sum())
        summary[a]["area_ratio_default"] = float(np.median(per_split(a, "area_ratio_default")))
        summary[a]["area_ratio_chosen"] = float(np.median(per_split(a, "area_ratio_chosen")))
        summary[a]["test_curve"] = np.mean([by[s, a]["test_curve"] for s in SEEDS], axis=0).tolist()
        summary[a]["val_curve"] = np.mean([by[s, a]["val_curve"] for s in SEEDS], axis=0).tolist()

    ranks = {tag: split_ranks(rows, f"f05_{tag}") for tag in ("default", "chosen")}
    ranks["dice_default"] = split_ranks(rows, "dice_default")

    def family(pairs_, key_a, key_b=None):
        out = []
        for a, b in pairs_:
            res = exp21.corrected_ttest(per_split(a, key_a), per_split(b, key_b or key_a),
                                        n_train, n_test)
            res.update({"a": a, "b": b})
            out.append(res)
        return holm(out)

    families = {}
    for tag in ("default", "chosen"):
        families[tag] = {title: family(p, f"f05_{tag}") for title, p in FAMILIES.items()}
    tuning = family([(a, a) for a in ARCHS], "f05_chosen", "f05_default")
    tuning_dice = family([(a, a) for a in ARCHS], "dice_chosen", "dice_default")
    control = family([(a, a) for a in ARCHS], "dice_at_dice_threshold", "dice_default")

    return {"beta": BETA, "thresholds": THRESHOLDS.tolist(), "default": DEFAULT,
            "seeds": list(SEEDS), "n_train": n_train, "n_test": n_test,
            "summary": summary, "ranks": ranks, "families": families,
            "tuning_f05": tuning, "tuning_dice": tuning_dice, "control_dice_tuned": control}


# ------------------------------------------------------------------ report

def report(res: dict) -> None:
    s = res["summary"]
    order = sorted(ARCHS, key=lambda a: s[a]["f05_chosen"]["mean"], reverse=True)

    print("\nTEST SCORES, mean over 5 person-safe splits [95% CI]\n")
    print(f"   {'architecture':<18}{'Dice @0.5':>10}{'F0.5 @0.5':>22}{'threshold':>16}"
          f"{'F0.5 @chosen':>22}{'Dice @chosen':>14}")
    for a in order:
        d = s[a]
        ci = lambda k: f"{d[k]['mean']:.4f} [{d[k]['ci_low']:.3f}, {d[k]['ci_high']:.3f}]"
        th = ", ".join(f"{t:.2f}" for t in d["thresholds"])
        print(f"   {LABEL[a]:<18}{d['dice_default']['mean']:>10.4f}{ci('f05_default'):>22}"
              f"{d['threshold']['mean']:>8.2f} (sd {d['threshold']['sd']:.2f})"
              f"{ci('f05_chosen'):>22}{d['dice_chosen']['mean']:>14.4f}")
    at_edge = sum(t >= THRESHOLDS[-1] for a in ARCHS for t in s[a]["thresholds"])
    print(f"\n   models whose chosen threshold is the grid's top value ({THRESHOLDS[-1]}): {at_edge} of 30")
    print("\n   thresholds chosen per split (seeds 0, 1, 2, 3, 42) and the test-set optimum, for reference:")
    for a in order:
        print(f"   {LABEL[a]:<18} chosen {s[a]['thresholds']}   test optimum {s[a]['test_best_thresholds']}")

    print("\nPRECISION AND RECALL, test, at 0.5 -> at the chosen threshold\n")
    print(f"   {'architecture':<18}{'precision':>20}{'recall':>20}{'mask/true area':>18}{'empty':>10}")
    for a in order:
        d = s[a]
        pr = f"{d['precision_default']['mean']:.3f} -> {d['precision_chosen']['mean']:.3f}"
        rc = f"{d['recall_default']['mean']:.3f} -> {d['recall_chosen']['mean']:.3f}"
        ar = f"{d['area_ratio_default']:.2f} -> {d['area_ratio_chosen']:.2f}"
        em = f"{d['empty_default']} -> {d['empty_chosen']}"
        print(f"   {LABEL[a]:<18}{pr:>20}{rc:>20}{ar:>18}{em:>10}")

    print("\nRANK within each split")
    print(f"   {'architecture':<18}{'Dice @0.5':>14}{'F0.5 @0.5':>14}{'F0.5 @chosen':>16}")
    for a in order:
        cell = lambda t: f"{res['ranks'][t][a]['mean_rank']:.1f} ({res['ranks'][t][a]['times_first']}x 1st)"
        print(f"   {LABEL[a]:<18}{cell('dice_default'):>14}{cell('default'):>14}{cell('chosen'):>16}")

    def table(rows_, title):
        print(f"\n   {title}")
        print(f"   {'pair':<34}{'mean diff':>11}{'95% CI':>22}{'p':>8}{'p Holm':>9}{'splits':>9}")
        for r in rows_:
            ci = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
            name = (f"{LABEL[r['a']]}: chosen − 0.5" if r["a"] == r["b"]
                    else f"{LABEL[r['a']]} − {LABEL[r['b']]}")
            verdict = "  resolved" if r["p_holm"] < 0.05 else ""
            print(f"   {name:<34}{r['mean_diff']:>+11.4f}{ci:>22}{r['p']:>8.4f}{r['p_holm']:>9.4f}"
                  f"{str(r['splits_a_better']) + '/' + str(r['n_splits']):>9}{verdict}")

    for tag, label in (("default", "at threshold 0.5"), ("chosen", "at each model's chosen threshold")):
        print(f"\nCOMPARISONS on test F0.5 {label} — corrected resampled t-test "
              f"(n_train {res['n_train']}, n_test {res['n_test']}), Holm within each family")
        for title, rows_ in res["families"][tag].items():
            table(rows_, title)
    print("\nWHAT THE CHOSEN THRESHOLD BUYS — same test, Holm across the six architectures")
    table(res["tuning_f05"], "test F0.5, chosen threshold minus 0.5")
    table(res["tuning_dice"], "test Dice, chosen threshold minus 0.5 (the cost, if negative)")

    print("\nCONTROL — the threshold chosen for validation Dice instead of F0.5")
    for a in order:
        print(f"   {LABEL[a]:<18} chosen {s[a]['dice_thresholds']}")
    table(res["control_dice_tuned"], "test Dice, Dice-chosen threshold minus 0.5")

    print("\nHIGHEST THRESHOLD at which each model still marks any test pixel (grid value)")
    for a in order:
        print(f"   {LABEL[a]:<18} {s[a]['max_probability_reached']}")


def write(rows: list, res: dict, arrays: dict) -> None:
    fields = ["seed", "arch", "n_test", "n_val", "train_images", "threshold",
              "val_f05_at_default", "val_f05_at_chosen",
              "f05_default", "f05_chosen", "dice_default", "dice_chosen",
              "precision_default", "precision_chosen", "recall_default", "recall_chosen",
              "area_ratio_default", "area_ratio_chosen", "empty_default", "empty_chosen",
              "test_best_threshold", "test_best_f05", "max_probability_reached",
              "dice_threshold", "dice_at_dice_threshold"]
    with (HERE / "per_split.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(HERE / "pixel_counts.npz", thresholds=THRESHOLDS, **arrays)
    np.savez_compressed(HERE / "per_image_f05.npz",
                        **{f"seed{r['seed']}_{r['arch']}_default": np.array(r["per_image_f05_default"])
                           for r in rows},
                        **{f"seed{r['seed']}_{r['arch']}_chosen": np.array(r["per_image_f05_chosen"])
                           for r in rows})
    print(f"\nWrote per_split.csv, results.json, pixel_counts.npz, per_image_f05.npz in {HERE}")


def main() -> int:
    rescore = "--rescore" in sys.argv[1:]
    rows, arrays = score_all(rescore)
    gap = check_against_exp24(rows)
    print(f"\ncheck: largest gap to experiment 24's Dice/precision/recall at 0.5 = {gap:.2e}")
    if gap > 1e-3:
        print("ERROR: the 0.5 column does not reproduce experiment 24", file=sys.stderr)
        return 1
    res = summarise(rows)
    res["check_gap_vs_exp24"] = gap
    report(res)
    write(rows, res, arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
