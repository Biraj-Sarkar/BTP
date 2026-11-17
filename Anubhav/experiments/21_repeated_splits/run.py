"""Repeat the architecture comparison over several random splits.

Experiment 20 compared architectures on one split (seed 0). A single split
answers "which did better on these 129 test subjects", not "which is better":
a different draw of training and test subjects, a different weight
initialisation and a different augmentation order all move the score, and with
one split there is no way to see by how much. So every architecture is trained
again under several random states, and each metric is reported as a mean with
its spread across splits.

What a "random state" changes
-----------------------------
The trainer's `--seed` sets the subject shuffle that decides the
train / validation / test split (390 / 129 / 129 subjects) **and** the weight
initialisation and augmentation order. All architectures trained with one seed
face the same split, so within a seed they are compared on identical images.
Across seeds everything varies, which is the variance of the whole procedure —
the number that matters when someone else retrains on the same data.

Seeds 0, 1, 2, 3 and 42. Seed 0 is experiment 20's split, and its bundles are
reused after checking they were trained exactly as this script would train them.

How the results are summarised
------------------------------
Per architecture and metric: mean, standard deviation, variance, minimum,
maximum, and a 95% confidence interval for the mean across splits.

Comparing two architectures across splits needs care. The splits overlap — any
two seeds share about 60% of their training subjects — so the per-split scores
are correlated, and an ordinary paired t-test over seeds treats them as
independent and is overconfident. The **corrected resampled t-test** (Nadeau and
Bengio, 2003) inflates the variance by (1/J + n_test/n_train) instead of 1/J,
with J splits, which is the standard correction for repeated random splits.
Comparisons are the same two pre-specified families as experiment 20, each
Holm-corrected.

Pixel accuracy is reported because it is often asked for. It does not separate
models here: conjunctiva covers a few percent of the frame, so predicting
"background everywhere" already scores about 0.97, and that baseline is printed
beside it.

Run
---
    python experiments/21_repeated_splits/run.py               # train what is missing, then evaluate
    python experiments/21_repeated_splits/run.py --evaluate-only
    python experiments/21_repeated_splits/run.py --archs segformer   # add an architecture later

Training is resumable: a bundle already on disk is reused only if it was trained
with the same seed, epochs, augmentation and raw-pixel input.

Abbreviations
-------------
Dice   overlap score, 2|A∩B| / (|A|+|B|)
IoU    intersection over union
HD95   95th-percentile boundary distance, pixels
SD     standard deviation
CI     confidence interval
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP20 = ROOT / "experiments/20_architectures"
sys.path.insert(0, str(ROOT / "src"))

SEEDS = (0, 1, 2, 3, 42)
ARCHS = ("unet", "unetplusplus", "attention", "linknet", "fpn", "pspnet")
EPOCHS, AUGMENT, VAL_FRACTION, TEST_FRACTION = 40, "rotation", 0.2, 0.2
DATA = [Path.home() / "btp-local/datasets/eyes_defy_prepared",
        Path.home() / "btp-local/datasets/conjunctiva_seg_prepared"]
METRICS = ("dice", "iou", "accuracy", "precision", "recall", "hd95")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _load(EXP20 / "compare.py", "exp20_compare")
LABEL, FAMILIES, holm = compare.LABEL, compare.FAMILIES, compare.holm


def bundle_path(seed: int, arch: str) -> Path:
    if seed == 0:
        reused = EXP20 / "bundles" / f"{arch}.pt"
        if reused.exists():
            return reused
    return HERE / "bundles" / f"seed{seed}" / f"{arch}.pt"


def trained_as_expected(path: Path, seed: int, arch: str) -> bool:
    """True if a bundle was trained exactly as this script would train it."""

    if not path.exists():
        return False
    try:
        b = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    cfg, prov, met = b.get("config", {}), b.get("provenance", {}), b.get("metrics", {})
    return (cfg.get("arch", "unet") == arch and prov.get("seed") == seed
            and prov.get("epochs_requested") == EPOCHS and prov.get("augment") == AUGMENT
            and not cfg.get("gray_world", False) and met.get("test_subjects"))


def train(seed: int, arch: str) -> bool:
    out = HERE / "bundles" / f"seed{seed}" / f"{arch}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = HERE / "logs" / f"seed{seed}_{arch}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/train_segmenter.py"),
           "--data", *map(str, DATA), "--arch", arch, "--out", str(out),
           "--epochs", str(EPOCHS), "--seed", str(seed), "--augment", AUGMENT,
           "--val-fraction", str(VAL_FRACTION), "--test-fraction", str(TEST_FRACTION)]
    print(f"  training seed {seed} {LABEL[arch]} -> {log.relative_to(ROOT)}", flush=True)
    with log.open("w") as handle:
        rc = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        print(f"  FAILED (exit {rc}); see {log}", flush=True)
    return rc == 0


# ------------------------------------------------------------------ evaluation

def evaluate(seeds, archs) -> list:
    trainer = compare._trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pairs = [p for root in DATA for p in trainer.find_pairs(root)]

    rows = []
    for seed in seeds:
        present = {a: bundle_path(seed, a) for a in archs
                   if trained_as_expected(bundle_path(seed, a), seed, a)}
        if not present:
            continue
        splits = {a: frozenset(torch.load(p, map_location="cpu", weights_only=False)
                               ["metrics"]["test_subjects"]) for a, p in present.items()}
        reference = next(iter(splits.values()))
        assert all(s == reference for s in splits.values()), \
            f"seed {seed}: architectures were tested on different subjects"
        test_pairs = [p for p in pairs if trainer.subject_of(p[0]) in reference]
        for arch, path in present.items():
            per_image = compare.evaluate_bundle(path, test_pairs, 320, 0.5, device)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            row = {"seed": seed, "arch": arch, "n_test": len(per_image),
                   "best_epoch": bundle["metrics"].get("best_epoch"),
                   "params_m": round(sum(v.numel() for v in bundle["state_dict"].values()) / 1e6, 3),
                   "empty": int(sum(r["empty"] for r in per_image)),
                   "trivial_accuracy": float(np.mean([r["trivial_accuracy"] for r in per_image]))}
            for m in METRICS:
                values = np.array([r[m] for r in per_image], dtype=float)
                row[m] = float(np.nanmedian(values)) if m == "hd95" else float(values.mean())
            row["per_image_dice"] = [r["dice"] for r in per_image]
            rows.append(row)
            print(f"  scored seed {seed:>2} {LABEL[arch]:<16} Dice {row['dice']:.4f}  "
                  f"IoU {row['iou']:.4f}  acc {row['accuracy']:.4f}", flush=True)
    return rows


def describe(values: np.ndarray) -> dict:
    j = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if j > 1 else float("nan")
    half = float(stats.t.ppf(0.975, j - 1) * sd / np.sqrt(j)) if j > 1 else float("nan")
    return {"n_splits": j, "mean": mean, "sd": sd, "variance": sd ** 2 if j > 1 else float("nan"),
            "min": float(values.min()), "max": float(values.max()),
            "ci_low": mean - half, "ci_high": mean + half}


def corrected_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau-Bengio corrected resampled t-test on per-split scores of a minus b."""

    d = a - b
    j = len(d)
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = np.sqrt((1.0 / j + n_test / n_train) * var)
    t = mean / se if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), j - 1))
    half = float(stats.t.ppf(0.975, j - 1) * se)
    return {"mean_diff": mean, "ci": [mean - half, mean + half], "t": float(t), "p": p,
            "splits_a_better": int((d > 0).sum()), "n_splits": j,
            "naive_p": float(stats.ttest_rel(a, b).pvalue) if j > 1 and np.any(d != d[0]) else 1.0}


def summarise(rows: list, archs, n_train: int = 390) -> dict:
    by = {(r["seed"], r["arch"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    complete = [s for s in seeds if all((s, a) in by for a in archs)]

    summary = {a: {m: describe(np.array([by[s, a][m] for s in seeds if (s, a) in by]))
                   for m in METRICS + ("trivial_accuracy",)}
               for a in archs if any((s, a) in by for s in seeds)}
    for a in summary:
        ranks = []
        for s in complete:
            order = sorted(archs, key=lambda x: by[s, x]["dice"], reverse=True)
            ranks.append(order.index(a) + 1)
        summary[a]["mean_rank"] = float(np.mean(ranks)) if ranks else float("nan")
        summary[a]["times_first"] = int(sum(1 for r in ranks if r == 1))

    n_test = int(np.mean([r["n_test"] for r in rows]))
    families = {}
    for title, pairs_ in FAMILIES.items():
        out = []
        for a, b in pairs_:
            shared = [s for s in seeds if (s, a) in by and (s, b) in by]
            if len(shared) < 2:
                continue
            res = corrected_ttest(np.array([by[s, a]["dice"] for s in shared]),
                                  np.array([by[s, b]["dice"] for s in shared]), n_train, n_test)
            res.update({"a": a, "b": b, "seeds": shared})
            out.append(res)
        families[title] = holm(out) if out else []
    return {"seeds": seeds, "complete_seeds": complete, "n_train": n_train, "n_test": n_test,
            "summary": summary, "families": families}


def report(result: dict, archs) -> None:
    s = result["summary"]
    order = sorted(s, key=lambda a: s[a]["dice"]["mean"], reverse=True)
    print(f"\nSplits evaluated: seeds {result['seeds']}  "
          f"(complete for every architecture: {result['complete_seeds']})")
    for metric, label in (("dice", "DICE"), ("iou", "IoU"), ("accuracy", "PIXEL ACCURACY")):
        print(f"\n{label} across splits\n")
        print(f"   {'architecture':<18}{'n':>3}{'mean':>9}{'SD':>9}{'variance':>12}"
              f"{'min':>9}{'max':>9}{'95% CI of mean':>22}")
        for a in order:
            d = s[a][metric]
            ci = f"[{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
            print(f"   {LABEL[a]:<18}{d['n_splits']:>3}{d['mean']:>9.4f}{d['sd']:>9.4f}"
                  f"{d['variance']:>12.2e}{d['min']:>9.4f}{d['max']:>9.4f}{ci:>22}")
        if metric == "accuracy":
            t = np.mean([s[a]["trivial_accuracy"]["mean"] for a in s])
            print(f"\n   predicting background everywhere scores {t:.4f} — accuracy cannot separate models")

    print("\nRANK by Dice within each complete split\n")
    for a in order:
        print(f"   {LABEL[a]:<18} mean rank {s[a]['mean_rank']:.1f}   first in "
              f"{s[a]['times_first']} of {len(result['complete_seeds'])} splits")

    print(f"\nCOMPARISONS — corrected resampled t-test on per-split Dice "
          f"(n_train {result['n_train']}, n_test {result['n_test']}), Holm within each family\n")
    for title, rows in result["families"].items():
        if not rows:
            continue
        print(f"   {title}")
        print(f"   {'pair':<34}{'mean diff':>11}{'95% CI':>22}{'p':>8}{'p Holm':>9}{'splits':>9}")
        for r in rows:
            ci = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
            name = f"{LABEL[r['a']]} − {LABEL[r['b']]}"
            verdict = "  resolved" if r["p_holm"] < 0.05 else ""
            print(f"   {name:<34}{r['mean_diff']:>+11.4f}{ci:>22}{r['p']:>8.4f}{r['p_holm']:>9.4f}"
                  f"{str(r['splits_a_better']) + '/' + str(r['n_splits']):>9}{verdict}")
        print()
    print("   'splits' = splits where the first architecture had the higher Dice.")


def write(rows: list, result: dict) -> None:
    fields = ["seed", "arch", "n_test", "dice", "iou", "accuracy", "trivial_accuracy",
              "precision", "recall", "hd95", "empty", "best_epoch", "params_m"]
    with (HERE / "per_split.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["seed"], r["arch"])))
    with (HERE / "summary.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["arch", "metric", "n_splits", "mean", "sd", "variance", "min", "max",
                    "ci_low", "ci_high"])
        for a, by_metric in result["summary"].items():
            for m in METRICS + ("trivial_accuracy",):
                d = by_metric[m]
                w.writerow([a, m, d["n_splits"], *(round(d[k], 6) for k in
                            ("mean", "sd", "variance", "min", "max", "ci_low", "ci_high"))])
    (HERE / "results.json").write_text(json.dumps(result, indent=2, default=float))
    np.savez_compressed(HERE / "per_image_dice.npz",
                        **{f"seed{r['seed']}_{r['arch']}": np.array(r["per_image_dice"]) for r in rows})
    print(f"\nWrote per_split.csv, summary.csv, results.json, per_image_dice.npz in {HERE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    unknown = [a for a in args.archs if a not in LABEL]
    if unknown:
        print(f"Unknown architectures {unknown}; add them to compare.LABEL first", file=sys.stderr)
        return 1

    if not args.evaluate_only:
        plan = [(s, a) for s in args.seeds for a in args.archs
                if not trained_as_expected(bundle_path(s, a), s, a)]
        print(f"{len(plan)} training runs needed; "
              f"{len(args.seeds) * len(args.archs) - len(plan)} bundles reused", flush=True)
        for seed, arch in plan:
            train(seed, arch)
        if args.train_only:
            return 0

    rows = evaluate(args.seeds, args.archs)
    if not rows:
        print("Nothing to evaluate yet.", file=sys.stderr)
        return 1
    result = summarise(rows, args.archs)
    report(result, args.archs)
    write(rows, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
