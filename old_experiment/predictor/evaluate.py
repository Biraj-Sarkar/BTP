"""Evaluation entry point for trained Hemoglobin Predictor checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader

from .data import (
    load_and_validate_manifest,
    create_patient_splits,
    get_val_transforms,
    HbDataset,
)
from .training import load_checkpoint, compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Hemoglobin predictor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest CSV")
    parser.add_argument("--split", choices=["test", "val", "all"], default="test", help="Partition to evaluate")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    model, bundle = load_checkpoint(args.checkpoint)
    print(f"Loaded {bundle.get('architecture', 'model')} checkpoint from {args.checkpoint}")

    cfg = bundle.get("config", {})
    manifest_path = args.manifest or Path(cfg.get("manifest_path", "ROI/roi_manifest.csv"))
    if not manifest_path.is_absolute():
        manifest_path = root_dir / manifest_path

    records, _ = load_and_validate_manifest(manifest_path, filter_usable=cfg.get("filter_usable", True))
    train_recs, val_recs, test_recs, _ = create_patient_splits(
        records=records,
        val_fraction=float(cfg.get("val_fraction", 0.15)),
        test_fraction=float(cfg.get("test_fraction", 0.15)),
        stratify_bins=int(cfg.get("stratify_bins", 4)),
        seed=int(cfg.get("seed", 42)),
    )

    eval_recs = test_recs if args.split == "test" else (val_recs if args.split == "val" else records)
    print(f"Evaluating on {len(eval_recs)} {args.split} samples...")

    tf = get_val_transforms()
    ds = HbDataset(eval_recs, transform=tf, target_mean=float(model.target_mean.item()), target_std=float(model.target_std.item()))
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []
    for batch in loader:
        preds = model.predict_hb(batch["image"]).cpu().numpy().reshape(-1)
        y_pred.extend(preds.tolist())
        y_true.extend(batch["hb_raw"])

    metrics = compute_metrics(y_true, y_pred)
    print("\n--- Evaluation Metrics ---")
    print(f"Samples: {metrics['count']}")
    print(f"MAE: {metrics['mae_g_dl']} g/dL")
    print(f"RMSE: {metrics['rmse_g_dl']} g/dL")
    print(f"R²: {metrics['r2']}")
    print(f"Pearson r: {metrics['pearson_r']} (p={metrics['pearson_p']:.4f})")
    print(f"Spearman ρ: {metrics['spearman_rho']}")
    print(f"Mean Bias: {metrics['bias_g_dl']} g/dL")
    print(f"95% LOA: [{metrics['loa_lower_g_dl']}, {metrics['loa_upper_g_dl']}] g/dL")
    print(f"Clinical Sensitivity: {metrics['screening']['sensitivity']}")
    print(f"Clinical Specificity: {metrics['screening']['specificity']}")
    print(f"Clinical Accuracy: {metrics['screening']['accuracy']}")


if __name__ == "__main__":
    main()

