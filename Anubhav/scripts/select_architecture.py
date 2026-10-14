"""Choose a segmentation architecture on a split, then train the winner on everything.

Two stages, which is the standard way to use a train/test split:

**Stage 1 — selection.** Every candidate architecture is trained on the same
training subjects, its epoch chosen on the same validation subjects, and scored
on the same test subjects, which neither of the other two touched. The scores go
into a leaderboard file.

**Stage 2 — shipping.** The winner is retrained on *every* subject, including the
ones held out in stage 1, for the epoch count that selection found. More data
should give a slightly better model than the stage-1 bundle; the cost is that
nothing is held back any more, so this model has no score of its own.

    The number to quote for the shipped model is the stage-1 test score of its
    architecture. That is an estimate, not a measurement of these weights: it
    says what this architecture achieved when trained on 60% of the subjects.
    Retraining on all of them is expected to help, but that expectation is not
    measured anywhere, and it cannot be without holding data back again.

Adding an architecture later
----------------------------
The point of the leaderboard is that a fourth architecture does not require
retraining the first three:

    python scripts/select_architecture.py --archs segformer

trains only that one and appends it. This is sound **only if it faces the same
split**, so the leaderboard stores a fingerprint of the data roots, the seed and
the resulting test subjects. If a new run's fingerprint differs — a dataset grew,
the seed changed — the script refuses to append and says what to re-run. A
leaderboard that silently mixed two splits would rank architectures by which
subjects they happened to be tested on.

Examples
--------
    # Stage 1 on the three current architectures
    python scripts/select_architecture.py --archs unet unetplusplus attention

    # Add one later, same split, no retraining of the others
    python scripts/select_architecture.py --archs unet-scse

    # Stage 2: retrain the leader on everything, ready to deploy
    python scripts/select_architecture.py --ship

Abbreviations
-------------
Dice   overlap score, 2|A∩B| / (|A|+|B|); 1.0 is a perfect mask
IoU    intersection over union
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anemia.architectures import ARCHITECTURES  # noqa: E402

DEFAULT_DATA = [Path.home() / "btp-local/datasets/eyes_defy_prepared",
                Path.home() / "btp-local/datasets/conjunctiva_seg_prepared"]
LEADERBOARD = ROOT / "experiments/20_architectures/leaderboard.json"
BUNDLE_DIR = ROOT / "experiments/20_architectures/bundles"
SHIP_DIR = ROOT / "runs/segmenter"


def fingerprint(data: list[Path], seed: int, val_fraction: float, test_fraction: float,
                epochs: int, augment: str) -> dict:
    """Identify the comparison setting, so two runs can be checked as comparable.

    The test subjects themselves are derived rather than trusted, by repeating
    the trainer's split arithmetic. If a dataset gained images, the subject list
    changes and so does the hash.
    """

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ts", ROOT / "scripts/train_segmenter.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["ts"] = trainer
    spec.loader.exec_module(trainer)

    pairs = []
    for root in data:
        pairs.extend(trainer.find_pairs(root))
    subjects = sorted({trainer.subject_of(image) for image, _ in pairs})
    random.Random(seed).shuffle(subjects)
    n_test = int(len(subjects) * test_fraction)
    n_val = max(1, int(len(subjects) * val_fraction))
    test_subjects = sorted(subjects[:n_test])

    payload = json.dumps({
        "data": sorted(str(d) for d in data), "seed": seed,
        "val_fraction": val_fraction, "test_fraction": test_fraction,
        "epochs": epochs, "augment": augment,
        "n_subjects": len(subjects), "test_subjects": test_subjects,
    }, sort_keys=True)
    return {
        "hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "test_subjects": test_subjects,
        "n_subjects": len(subjects), "n_images": len(pairs),
        "n_test_subjects": len(test_subjects), "n_val_subjects": n_val,
        "seed": seed, "epochs": epochs, "augment": augment,
        "data": sorted(str(d) for d in data),
    }


def adopt(path: Path, current: dict) -> bool:
    """True when an existing bundle was trained against the split we are using.

    Checked against the subjects the bundle actually held out, not against its
    filename or its recorded arguments — a bundle whose test set differs is not
    comparable with the rest of the leaderboard no matter what it is called.
    """

    import torch
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    metrics = bundle.get("metrics", {})
    provenance = bundle.get("provenance", {})
    return (sorted(metrics.get("test_subjects", [])) == sorted(current["test_subjects"])
            and provenance.get("seed") == current["seed"]
            and provenance.get("augment") == current["augment"]
            and provenance.get("epochs_requested") == current["epochs"]
            and metrics.get("test_dice") is not None)


def load_board() -> dict:
    if LEADERBOARD.exists():
        return json.loads(LEADERBOARD.read_text())
    return {"fingerprint": None, "results": {}}


def train_one(arch: str, data: list[Path], out: Path, args) -> int:
    cmd = [sys.executable, str(ROOT / "scripts/train_segmenter.py"),
           "--data", *[str(d) for d in data], "--arch", arch, "--out", str(out),
           "--epochs", str(args.epochs), "--seed", str(args.seed),
           "--augment", args.augment,
           "--val-fraction", str(args.val_fraction),
           "--test-fraction", str(args.test_fraction)]
    print(f"\n{'=' * 66}\n  stage 1 · {arch}\n{'=' * 66}")
    return subprocess.call(cmd)


def read_metrics(path: Path) -> dict:
    import torch
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    m = bundle["metrics"]
    return {"test_dice": m.get("test_dice"), "test_iou": m.get("test_iou"),
            "val_dice": m.get("val_dice"), "best_epoch": m.get("best_epoch"),
            "test_images": m.get("test_images"), "empty_masks": m.get("test_empty_masks"),
            "params_m": sum(v.numel() for v in bundle["state_dict"].values()) / 1e6}


def show(board: dict) -> None:
    results = board["results"]
    if not results:
        print("leaderboard is empty")
        return
    order = sorted(results, key=lambda a: results[a]["test_dice"] or -1, reverse=True)
    print(f"\n{'rank':<6}{'architecture':<18}{'test Dice':>11}{'test IoU':>10}"
          f"{'val Dice':>10}{'epoch':>7}{'params':>9}{'empty':>7}")
    for i, arch in enumerate(order, 1):
        r = results[arch]
        print(f"{i:<6}{arch:<18}{r['test_dice']:>11.4f}{r['test_iou']:>10.4f}"
              f"{r['val_dice']:>10.4f}{r['best_epoch']:>7d}{r['params_m']:>8.2f}M"
              f"{r['empty_masks']:>7d}")
    f = board["fingerprint"]
    print(f"\nall scored on the same {f['n_test_subjects']} held-out subjects "
          f"(split {f['hash']}, seed {f['seed']}, {f['epochs']} epochs, {f['augment']} augmentation)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--archs", nargs="*", default=None,
                   help=f"architectures to train in stage 1; choose from {list(ARCHITECTURES)}")
    p.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment", default="rotation", choices=["standard", "rotation", "strong"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--ship", action="store_true",
                   help="stage 2: retrain the leaderboard's winner on every subject")
    p.add_argument("--show", action="store_true", help="print the leaderboard and exit")
    p.add_argument("--force", action="store_true",
                   help="retrain architectures already on the leaderboard")
    args = p.parse_args()

    board = load_board()
    if args.show:
        show(board)
        return 0

    current = fingerprint(args.data, args.seed, args.val_fraction, args.test_fraction,
                          args.epochs, args.augment)

    # ------------------------------------------------------------ stage 2
    if args.ship:
        if not board["results"]:
            print("Nothing on the leaderboard. Run stage 1 first.", file=sys.stderr)
            return 1
        winner = max(board["results"], key=lambda a: board["results"][a]["test_dice"] or -1)
        record = board["results"][winner]
        epochs = record["best_epoch"]
        out = SHIP_DIR / f"{winner}_full.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"stage 2 · shipping run\n  winner      {winner} "
              f"(test Dice {record['test_dice']:.4f})\n"
              f"  epochs      {epochs}, the epoch selection chose\n"
              f"  trained on  every subject, nothing held back\n"
              f"  output      {out}\n")
        rc = subprocess.call([
            sys.executable, str(ROOT / "scripts/train_segmenter.py"),
            "--data", *[str(d) for d in args.data], "--arch", winner, "--out", str(out),
            "--epochs", str(epochs), "--seed", str(args.seed),
            "--augment", args.augment, "--no-select"])
        if rc == 0:
            board["shipped"] = {"arch": winner, "epochs": epochs, "bundle": str(out),
                                "estimate_from_selection": record["test_dice"],
                                "estimate_note": "stage-1 test Dice for this architecture; "
                                                 "the shipped weights saw all subjects and have "
                                                 "no held-out score of their own"}
            LEADERBOARD.write_text(json.dumps(board, indent=2))
            print(f"\nleaderboard updated. Deploy with:  ANEMIA_SEGMENTER={out}")
            print("The deployed bundle at runs/segmenter/unet.pt is NOT replaced; "
                  "swap it deliberately once you have compared the two.")
        return rc

    # ------------------------------------------------------------ stage 1
    if board["fingerprint"] and board["fingerprint"]["hash"] != current["hash"]:
        print("The leaderboard was built on a different split:", file=sys.stderr)
        print(f"  stored  {board['fingerprint']['hash']}  "
              f"{board['fingerprint']['n_subjects']} subjects, seed {board['fingerprint']['seed']}, "
              f"{board['fingerprint']['epochs']} epochs", file=sys.stderr)
        print(f"  now     {current['hash']}  {current['n_subjects']} subjects, "
              f"seed {current['seed']}, {current['epochs']} epochs", file=sys.stderr)
        print("\nScores from two different splits are not comparable. Either match the "
              "stored settings, or delete the leaderboard and retrain every architecture.",
              file=sys.stderr)
        return 1

    archs = args.archs or list(ARCHITECTURES)
    unknown = [a for a in archs if a not in ARCHITECTURES]
    if unknown:
        print(f"Unknown architectures {unknown}; choose from {list(ARCHITECTURES)}", file=sys.stderr)
        return 1

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    board["fingerprint"] = current
    for arch in archs:
        if arch in board["results"] and not args.force:
            print(f"skipping {arch}: already on the leaderboard (use --force to retrain)")
            continue
        out = BUNDLE_DIR / f"{arch}.pt"
        if out.exists() and not args.force and adopt(out, current):
            print(f"adopting existing {out.name}: it was trained on this exact split")
            board["results"][arch] = read_metrics(out)
            LEADERBOARD.write_text(json.dumps(board, indent=2))
            continue
        if train_one(arch, args.data, out, args) != 0:
            print(f"training failed for {arch}", file=sys.stderr)
            return 1
        board["results"][arch] = read_metrics(out)
        LEADERBOARD.write_text(json.dumps(board, indent=2))   # written as we go

    show(board)
    winner = max(board["results"], key=lambda a: board["results"][a]["test_dice"] or -1)
    print(f"\nwinner: {winner}. Ship it with:  python scripts/select_architecture.py --ship")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
