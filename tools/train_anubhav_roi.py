"""Train Anubhav's Hb regressor directly from extracted ROI crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANUBHAV = ROOT / "algorithms" / "Anubhav"
sys.path.insert(0, str(ANUBHAV / "src"))

from anemia.config import Config  # noqa: E402
from anemia.data import Sample, _normalise_sex  # noqa: E402
from anemia.imaging import read_rgb  # noqa: E402
from anemia.train import cross_validate, pick_device, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "ROI" / "roi_manifest.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    frame = frame[frame["usable"].astype(str).str.lower().eq("true")].copy()
    samples = []
    crops = {}
    for _, row in frame.iterrows():
        image_path = args.manifest.parent / str(row["roi_224x224"])
        sample = Sample(
            patient_id=str(row["patient_id"]),
            source="extracted-roi",
            image_path=image_path,
            hb=float(row["hb"]),
            age_years=float(row["age"]) if str(row["age"]).strip() else None,
            sex=_normalise_sex(row["sex"]),
        )
        samples.append(sample)
        crops[image_path] = read_rgb(image_path)

    if len(samples) < args.folds * 3:
        raise SystemExit(f"Only {len(samples)} usable ROIs; too few for {args.folds} folds")
    config = Config(data_roots=(args.manifest,), output_root=args.out)
    config = config.__class__(
        **{**config.__dict__,
           "regression": config.regression.__class__(**{**config.regression.__dict__, "epochs": args.epochs, "pretrained": False}),
           "split": config.split.__class__(**{**config.split.__dict__, "folds": args.folds})}
    )
    set_seed(config.seed)
    device = pick_device()
    print(f"Using {len(samples)} quality-passed ROIs on {device}")
    results = cross_validate(samples, crops, config, device)
    args.out.mkdir(parents=True, exist_ok=True)
    config.save(args.out / "config.json")
    (args.out / "metrics.json").write_text(json.dumps({
        "regression": results["regression"],
        "screening": results["screening"],
        "per_fold": results["per_fold"],
    }, indent=2), encoding="utf-8")
    rows = results["out_of_fold_predictions"]
    if rows:
        with (args.out / "out_of_fold_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    best = results.get("best_checkpoint")
    if best is not None and best.checkpoint is not None:
        best.checkpoint.save(args.out / "best.pt")
    print(json.dumps(results["regression"], indent=2))
    print(f"Saved ROI training run to {args.out}")


if __name__ == "__main__":
    main()
