"""Command line entry points.

    python -m anemia inspect  --data <root>
    python -m anemia train-hb --data <root> [--data <root2>] --out runs/exp1
    python -m anemia predict  --checkpoint runs/exp1/best.pt --image photo.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import List

from .config import Config, REPO_ROOT
from .data import Sample, load_dataset, summarise


def _collect(data_roots: List[Path]) -> List[Sample]:
    samples: List[Sample] = []
    for root in data_roots:
        loaded = load_dataset(root)
        print(f"  {root}: {len(loaded)} images")
        samples.extend(loaded)
    if not samples:
        raise SystemExit("No samples loaded — check --data paths.")
    return samples


def cmd_inspect(args: argparse.Namespace) -> None:
    """Describe the cohort before training anything.

    Worth running the moment a dataset lands: it is the cheapest way to catch a
    loader that silently matched 40 of 710 images.
    """

    samples = _collect(args.data)
    stats = summarise(samples)
    print("\nCohort summary")
    print(json.dumps(stats, indent=2))

    if stats["images"] < 100:
        print("\nWARNING: fewer than 100 images parsed. Verify the loader matched the layout.")
    if stats.get("with_masks", 0) == 0:
        print("\nNote: no ground-truth masks found — segmenter training needs Eyes-defy-anemia.")


def cmd_train_hb(args: argparse.Namespace) -> None:
    import torch

    from .segment import heuristic_segmenter, load_segmenter
    from .train import apply_quality_gate, build_crops, cross_validate, pick_device, set_seed

    config = Config(
        data_roots=tuple(args.data),
        output_root=args.out,
        cache_root=args.cache,
    )
    if args.epochs:
        config = replace(config, regression=replace(config.regression, epochs=args.epochs))
    if args.folds:
        config = replace(config, split=replace(config.split, folds=args.folds))

    set_seed(config.seed)
    device = pick_device()
    print(f"Device: {device}")

    samples = _collect(list(args.data))
    print(json.dumps(summarise(samples), indent=2))

    segmenter = load_segmenter(args.segmenter) if args.segmenter else heuristic_segmenter
    print(f"\nPreprocessing {len(samples)} images (cached at {config.cache_root})")
    prepared = build_crops(samples, segmenter, config)

    samples = apply_quality_gate(samples, prepared, config.quality.enforce)
    print(f"  {len(samples)} images passed QC")
    crops = {path: prep.crop for path, prep in prepared.items()}

    print(f"\nCross-validating ({config.split.folds} folds, patient-grouped)")
    results = cross_validate(samples, crops, config, device)

    args.out.mkdir(parents=True, exist_ok=True)
    config.save(args.out / "config.json")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {"regression": results["regression"], "screening": results["screening"],
             "per_fold": results["per_fold"]},
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = results["out_of_fold_predictions"]
    if rows:
        with (args.out / "out_of_fold_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    best = results["best_checkpoint"]
    if best and best.checkpoint:
        best.checkpoint.save(args.out / "best.pt")
        print(f"\nSaved checkpoint to {args.out / 'best.pt'}")

    reg = results["regression"]
    print("\nCross-validated results")
    print(f"  MAE  {reg.get('mae_mean', float('nan')):.3f} +/- {reg.get('mae_std', 0):.3f} g/dL")
    print(f"  RMSE {reg.get('rmse_mean', float('nan')):.3f} g/dL")
    print(f"  R^2  {reg.get('r2_mean', float('nan')):.3f}")
    gain = reg.get("mae_vs_baseline_mean")
    if gain is not None:
        verdict = "better than" if gain > 0 else "NO BETTER THAN"
        print(f"  {abs(gain):.3f} g/dL {verdict} predicting the training mean")
    scr = results["screening"]
    print(f"  sensitivity {scr.get('sensitivity_mean', float('nan')):.3f}, "
          f"specificity {scr.get('specificity_mean', float('nan')):.3f}")


def cmd_train_segmenter(args: argparse.Namespace) -> None:
    from .train import pick_device, set_seed
    from .train_segmenter import train_segmenter

    config = Config(data_roots=tuple(args.data), output_root=args.out)
    if args.epochs:
        config = replace(config, segmentation=replace(config.segmentation, epochs=args.epochs))

    set_seed(config.seed)
    samples = _collect(list(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    train_segmenter(samples, config, args.out, pick_device())


def cmd_predict(args: argparse.Namespace) -> None:
    from .predict import AnemiaPredictor

    predictor = AnemiaPredictor(args.checkpoint, segmenter_dir=args.segmenter)
    result = predictor.predict_path(args.image, age_years=args.age, sex=args.sex)
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anemia", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a dataset without training")
    inspect.add_argument("--data", type=Path, action="append", required=True)
    inspect.set_defaults(func=cmd_inspect)

    train = sub.add_parser("train-hb", help="cross-validate the Hb regressor")
    train.add_argument("--data", type=Path, action="append", required=True)
    train.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "hb")
    train.add_argument("--cache", type=Path, default=REPO_ROOT / ".cache" / "crops")
    train.add_argument("--segmenter", type=Path, default=None,
                       help="trained Mask2Former dir; omit to use the colour heuristic")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--folds", type=int, default=None)
    train.set_defaults(func=cmd_train_hb)

    segment = sub.add_parser("train-segmenter", help="fine-tune Mask2Former on ground-truth masks")
    segment.add_argument("--data", type=Path, action="append", required=True,
                         help="Eyes-defy-anemia root (the only dataset with masks)")
    segment.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "segmenter")
    segment.add_argument("--epochs", type=int, default=None)
    segment.set_defaults(func=cmd_train_segmenter)

    predict = sub.add_parser("predict", help="score a single photo")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--segmenter", type=Path, default=None)
    predict.add_argument("--age", type=float, default=None, help="patient age in years")
    predict.add_argument("--sex", type=str, default=None, choices=["M", "F"])
    predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
