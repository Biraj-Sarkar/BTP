"""Repeat the architecture comparison on splits that keep each person together.

Experiment 21 compared six architectures over five random splits. Experiment 23
then found that the Mendeley set's 547 images are repeat shots: at least 463 of
them fall into 115 runs of shots of one eye, and in every experiment-21 split
64-72% of the test images had a relative in training. This experiment repeats
experiment 21 exactly (same six architectures, seeds 0, 1, 2, 3 and 42, 40
epochs, rotation augmentation, EfficientNet-B0 encoder, 20% validation and 20%
test) with one change: the split.

The split
---------
Three trainer options, all in `scripts/train_segmenter.py`:

``--groups experiments/23_duplicates/groups.csv``
    Each run of repeat shots goes to one side of the split as a whole.

``--block 24``
    Grouping alone is not enough. Looking at where one group ends and the next
    begins, the two neighbouring runs are often the same person (the other eye,
    or the same eye re-framed): at least 14 of 24 sampled boundaries, against 4
    plainly different people. No matcher links them, so instead the Mendeley
    files are dealt out in consecutive blocks of about 24 files, never cutting
    through a group. A person can then straddle training and test only at a
    block edge. This is "blocked" cross-validation, the usual answer for data
    saved in time order.

``--buffer 4``
    Closes those edges: every group with a file within 4 numbers of a test
    file is left out of training and validation.

Each source is dealt out on its own, so 20 of the 101 Eyes-defy images and
about a fifth of the Mendeley images are tested in every split. The cost is
28-57 images per split that sit out; training sets are 339-366 images against
experiment 21's 390.

Two questions
-------------
1. **Does the ranking survive person-safe splits?** Same statistics as
   experiment 21 (mean and spread over splits, rank per split, corrected
   resampled t-test, Holm within the same two families).
2. **Did the leak inflate experiment 21's scores?** Mendeley test Dice in
   experiment 21 against here, per architecture. Eyes-defy test Dice is the
   control: Eyes-defy images had no relative in training in either experiment,
   so their change measures everything except the leak (different splits,
   fewer training images). The leak estimate is how much higher Mendeley
   scored relative to Eyes-defy in experiment 21 than it does here.

Run
---
    python experiments/24_grouped_splits/run.py --plan            # show the splits, train nothing
    python experiments/24_grouped_splits/run.py                   # train what is missing, then evaluate
    python experiments/24_grouped_splits/run.py --evaluate-only

Training is resumable: a bundle is reused only if it was trained with the same
seed, epochs, augmentation, raw-pixel input, groups file, block and buffer.

Abbreviations
-------------
Dice   overlap score, 2|A∩B| / (|A|+|B|)
IoU    intersection over union
SD     standard deviation
CI     confidence interval
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
EXP21 = ROOT / "experiments/21_repeated_splits"
GROUPS = ROOT / "experiments/23_duplicates/groups.csv"
sys.path.insert(0, str(ROOT / "src"))

SEEDS = (0, 1, 2, 3, 42)
ARCHS = ("unet", "unetplusplus", "attention", "linknet", "fpn", "pspnet")
EPOCHS, AUGMENT, VAL_FRACTION, TEST_FRACTION = 40, "rotation", 0.2, 0.2
BLOCK, BUFFER = 24, 4
DATA = {"eyes_defy": Path.home() / "btp-local/datasets/eyes_defy_prepared",
        "mendeley": Path.home() / "btp-local/datasets/conjunctiva_seg_prepared"}
METRICS = ("dice", "iou", "accuracy", "precision", "recall", "hd95")
SOURCES = ("mendeley", "eyes_defy")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp21 = _load(EXP21 / "run.py", "exp21_run")
compare = exp21.compare
LABEL = compare.LABEL


def groups_sha() -> str:
    return hashlib.sha256(GROUPS.read_bytes()).hexdigest()


def bundle_path(seed: int, arch: str) -> Path:
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
            and not cfg.get("gray_world", False) and met.get("test_subjects")
            and prov.get("groups_sha256") == groups_sha()
            and prov.get("block") == BLOCK and prov.get("buffer") == BUFFER)


def train(seed: int, arch: str) -> bool:
    out = bundle_path(seed, arch)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = HERE / "logs" / f"seed{seed}_{arch}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/train_segmenter.py"),
           "--data", *map(str, DATA.values()), "--arch", arch, "--out", str(out),
           "--epochs", str(EPOCHS), "--seed", str(seed), "--augment", AUGMENT,
           "--val-fraction", str(VAL_FRACTION), "--test-fraction", str(TEST_FRACTION),
           "--groups", str(GROUPS), "--block", str(BLOCK), "--buffer", str(BUFFER)]
    print(f"  training seed {seed} {LABEL[arch]} -> {log.relative_to(ROOT)}", flush=True)
    with log.open("w") as handle:
        rc = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        print(f"  FAILED (exit {rc}); see {log}", flush=True)
    return rc == 0


def labelled_pairs(trainer) -> list:
    """(image, mask, source) for every annotated image, in the trainer's order."""

    return [(image, mask, source) for source, root in DATA.items()
            for image, mask in trainer.find_pairs(root)]


def plan() -> None:
    """Print each seed's split without training, exactly as the trainer will make it."""

    trainer = compare._trainer()
    pairs = labelled_pairs(trainer)
    subject_roots = {trainer.subject_of(i): str(DATA[src]) for i, _, src in pairs}
    groups = trainer.read_groups(GROUPS)
    print(f"blocks of {BLOCK} files, buffer {BUFFER}, groups from {GROUPS.relative_to(ROOT)}\n")
    for seed in SEEDS:
        s = trainer.split_subjects(subject_roots, seed, VAL_FRACTION, TEST_FRACTION,
                                   groups=groups, buffer=BUFFER, block=BLOCK)
        count = lambda k, src: sum(1 for x in s[k] if subject_roots[x] == str(DATA[src]))
        print(f"  seed {seed:>2}: train {len(s['train'])}  val {len(s['val'])}  "
              f"test {len(s['test'])} ({count('test', 'mendeley')} Mendeley, "
              f"{count('test', 'eyes_defy')} Eyes-defy)  left out {len(s['excluded'])}")


# ------------------------------------------------------------------ evaluation

def evaluate(seeds, archs) -> list:
    trainer = compare._trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pairs = labelled_pairs(trainer)

    rows = []
    for seed in seeds:
        present = {a: bundle_path(seed, a) for a in archs
                   if trained_as_expected(bundle_path(seed, a), seed, a)}
        if not present:
            continue
        bundles = {a: torch.load(p, map_location="cpu", weights_only=False) for a, p in present.items()}
        splits = {a: frozenset(b["metrics"]["test_subjects"]) for a, b in bundles.items()}
        reference = next(iter(splits.values()))
        assert all(s == reference for s in splits.values()), \
            f"seed {seed}: architectures were tested on different subjects"
        test = [(i, m, src) for i, m, src in pairs if trainer.subject_of(i) in reference]
        source = np.array([src for _, _, src in test])
        for arch, path in present.items():
            per_image = compare.evaluate_bundle(path, [(i, m) for i, m, _ in test], 320, 0.5, device)
            bundle = bundles[arch]
            row = {"seed": seed, "arch": arch, "n_test": len(per_image),
                   "n_test_mendeley": int((source == "mendeley").sum()),
                   "train_images": bundle["provenance"]["train_images"],
                   "val_images": bundle["provenance"]["val_images"],
                   "left_out": len(bundle["provenance"].get("excluded_subjects", [])),
                   "best_epoch": bundle["metrics"].get("best_epoch"),
                   "val_dice": bundle["metrics"].get("val_dice"),
                   "empty": int(sum(r["empty"] for r in per_image)),
                   "trivial_accuracy": float(np.mean([r["trivial_accuracy"] for r in per_image]))}
            for m in METRICS:
                values = np.array([r[m] for r in per_image], dtype=float)
                row[m] = float(np.nanmedian(values)) if m == "hd95" else float(values.mean())
            dice = np.array([r["dice"] for r in per_image])
            for src in SOURCES:
                row[f"dice_{src}"] = float(dice[source == src].mean())
            row["per_image_dice"] = dice.tolist()
            row["per_image_source"] = source.tolist()
            row["per_image_name"] = [r["image"] for r in per_image]
            rows.append(row)
            print(f"  scored seed {seed:>2} {LABEL[arch]:<16} Dice {row['dice']:.4f}  "
                  f"(Mendeley {row['dice_mendeley']:.4f}, Eyes-defy {row['dice_eyes_defy']:.4f})",
                  flush=True)
    return rows


def experiment21_by_source() -> dict:
    """Experiment 21's per-split test Dice, overall and by source."""

    trainer = compare._trainer()
    pairs = labelled_pairs(trainer)
    dice = np.load(EXP21 / "per_image_dice.npz")
    out = {}
    for seed in SEEDS:
        b = torch.load(exp21.bundle_path(seed, "attention"), map_location="cpu", weights_only=False)
        test = set(b["metrics"]["test_subjects"])
        source = np.array([src for i, _, src in pairs if trainer.subject_of(i) in test])
        for arch in ARCHS:
            d = dice[f"seed{seed}_{arch}"]
            assert len(d) == len(source), "experiment-21 test order does not match"
            out[seed, arch] = {"dice": float(d.mean()),
                               **{f"dice_{s}": float(d[source == s].mean()) for s in SOURCES}}
    return out


def leak_comparison(rows: list, archs) -> dict:
    """How much higher Mendeley scored, relative to Eyes-defy, in experiment 21 than here."""

    old = experiment21_by_source()
    new = {(r["seed"], r["arch"]): r for r in rows}
    out = {}
    for arch in archs:
        seeds_new = [s for s in SEEDS if (s, arch) in new]
        if len(seeds_new) < 2:
            continue
        entry = {}
        for key in ("dice", "dice_mendeley", "dice_eyes_defy"):
            a = np.array([old[s, arch][key] for s in SEEDS])
            b = np.array([new[s, arch][key] for s in seeds_new])
            entry[key] = {"exp21": float(a.mean()), "exp24": float(b.mean()),
                          "change": float(b.mean() - a.mean()),
                          "p_welch": float(stats.ttest_ind(b, a, equal_var=False).pvalue)}
        # Per split, Mendeley minus Eyes-defy; then experiment 21 minus here.
        gap_old = np.array([old[s, arch]["dice_mendeley"] - old[s, arch]["dice_eyes_defy"] for s in SEEDS])
        gap_new = np.array([new[s, arch]["dice_mendeley"] - new[s, arch]["dice_eyes_defy"]
                            for s in seeds_new])
        estimate = float(gap_old.mean() - gap_new.mean())
        se = float(np.sqrt(gap_old.var(ddof=1) / len(gap_old) + gap_new.var(ddof=1) / len(gap_new)))
        dof = len(gap_old) + len(gap_new) - 2
        half = float(stats.t.ppf(0.975, dof) * se)
        entry["leak_estimate"] = {"value": estimate, "ci": [estimate - half, estimate + half],
                                  "gap_exp21": gap_old.tolist(), "gap_exp24": gap_new.tolist()}
        out[arch] = entry
    return out


def report_leak(leak: dict) -> None:
    print("\nDID THE LEAK INFLATE EXPERIMENT 21?  Mean test Dice over splits\n")
    print(f"   {'architecture':<18}{'all: 21 -> 24':>20}{'Mendeley: 21 -> 24':>24}"
          f"{'Eyes-defy: 21 -> 24':>25}{'leak estimate [95% CI]':>30}")
    for arch, e in leak.items():
        cell = lambda k: f"{e[k]['exp21']:.4f} -> {e[k]['exp24']:.4f}"
        le = e["leak_estimate"]
        ci = f"{le['value']:+.4f} [{le['ci'][0]:+.4f}, {le['ci'][1]:+.4f}]"
        print(f"   {LABEL[arch]:<18}{cell('dice'):>20}{cell('dice_mendeley'):>24}"
              f"{cell('dice_eyes_defy'):>25}{ci:>30}")
    print("\n   leak estimate = (Mendeley - Eyes-defy) in experiment 21 minus the same gap here;"
          "\n   positive means the leak raised Mendeley scores. Eyes-defy is the control: it had"
          "\n   no relatives in training in either experiment.")


def write(rows: list, result: dict) -> None:
    fields = ["seed", "arch", "n_test", "n_test_mendeley", "train_images", "val_images", "left_out",
              "dice", "dice_mendeley", "dice_eyes_defy", "iou", "accuracy", "trivial_accuracy",
              "precision", "recall", "hd95", "empty", "best_epoch", "val_dice"]
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
    np.savez_compressed(
        HERE / "per_image_dice.npz",
        **{f"seed{r['seed']}_{r['arch']}": np.array(r["per_image_dice"]) for r in rows},
        **{f"seed{r['seed']}_source": np.array(r["per_image_source"]) for r in rows},
        **{f"seed{r['seed']}_image": np.array(r["per_image_name"]) for r in rows})
    print(f"\nWrote per_split.csv, summary.csv, results.json, per_image_dice.npz in {HERE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--plan", action="store_true", help="print the splits and stop")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()

    if args.plan:
        plan()
        return 0

    if not args.evaluate_only:
        todo = [(s, a) for s in args.seeds for a in args.archs
                if not trained_as_expected(bundle_path(s, a), s, a)]
        print(f"{len(todo)} training runs needed; "
              f"{len(args.seeds) * len(args.archs) - len(todo)} bundles reused", flush=True)
        for seed, arch in todo:
            train(seed, arch)
        if args.train_only:
            return 0

    rows = evaluate(args.seeds, args.archs)
    if not rows:
        print("Nothing to evaluate yet.", file=sys.stderr)
        return 1
    n_train = int(round(np.mean([r["train_images"] for r in rows])))
    result = exp21.summarise(rows, args.archs, n_train=n_train)
    result["split"] = {"groups_file": str(GROUPS.relative_to(ROOT)), "groups_sha256": groups_sha(),
                       "block": BLOCK, "buffer": BUFFER}
    result["leak"] = leak_comparison(rows, args.archs)
    exp21.report(result, args.archs)
    report_leak(result["leak"])
    write(rows, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
