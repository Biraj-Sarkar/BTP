"""Training entry point for ResNet, VGG, and MobileNet Hemoglobin Predictors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .models import build_model
from .data import (
    load_and_validate_manifest,
    create_patient_splits,
    get_train_transforms,
    get_val_transforms,
    HbDataset,
)
from .training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hemoglobin prediction model.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML training configuration file")
    parser.add_argument("--architecture", choices=["resnet", "vgg", "mobilenet"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_config(config_path: Path | None, cli_args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if config_path and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides
    if cli_args.architecture is not None:
        config["architecture"] = cli_args.architecture
    if cli_args.epochs is not None:
        config["epochs"] = cli_args.epochs
    if cli_args.batch_size is not None:
        config["batch_size"] = cli_args.batch_size
    if cli_args.lr is not None:
        config["learning_rate"] = cli_args.lr
    if cli_args.manifest is not None:
        config["manifest_path"] = str(cli_args.manifest)
    if cli_args.output is not None:
        config["output_dir"] = str(cli_args.output)

    # Defaults
    config.setdefault("architecture", "resnet")
    config.setdefault("variant", "resnet18" if config["architecture"] == "resnet" else ("vgg16_bn" if config["architecture"] == "vgg" else "mobilenet_v3_large"))
    config.setdefault("pretrained", True)
    config.setdefault("dropout", 0.3)
    config.setdefault("epochs", 25)
    config.setdefault("batch_size", 16)
    config.setdefault("learning_rate", 3e-4)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("manifest_path", "ROI/roi_manifest.csv")
    config.setdefault("output_dir", f"predictor/runs/{config['architecture']}")
    config.setdefault("val_fraction", 0.15)
    config.setdefault("test_fraction", 0.15)
    config.setdefault("seed", 42)
    config.setdefault("filter_usable", True)
    return config


def main() -> None:
    cli_args = parse_args()
    config = load_config(cli_args.config, cli_args)

    # Set random seed
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    root_dir = Path(__file__).resolve().parents[1]
    manifest_path = Path(config["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = root_dir / manifest_path

    print(f"Loading manifest from {manifest_path}...")
    records, audit = load_and_validate_manifest(
        manifest_path=manifest_path,
        filter_usable=config.get("filter_usable", True),
    )
    print(f"Loaded {len(records)} valid records ({audit['unique_patients']} unique patients).")

    # Patient-level split
    train_recs, val_recs, test_recs, split_info = create_patient_splits(
        records=records,
        val_fraction=float(config.get("val_fraction", 0.15)),
        test_fraction=float(config.get("test_fraction", 0.15)),
        stratify_bins=int(config.get("stratify_bins", 4)),
        seed=seed,
    )
    print(f"Partitions: Train={len(train_recs)} samples ({split_info['train_patients']} patients), "
          f"Val={len(val_recs)} samples ({split_info['val_patients']} patients), "
          f"Test={len(test_recs)} samples ({split_info['test_patients']} patients).")

    # Compute target normalization strictly on training patients
    train_hbs = [r.hb for r in train_recs]
    target_mean = float(np.mean(train_hbs))
    target_std = float(np.std(train_hbs))
    print(f"Training target stats: Mean={target_mean:.2f} g/dL, Std={target_std:.2f} g/dL")

    # Transforms & Datasets
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    train_ds = HbDataset(train_recs, transform=train_tf, target_mean=target_mean, target_std=target_std)
    val_ds = HbDataset(val_recs, transform=val_tf, target_mean=target_mean, target_std=target_std)
    test_ds = HbDataset(test_recs, transform=val_tf, target_mean=target_mean, target_std=target_std)

    batch_size = int(config.get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    arch = config["architecture"]
    variant = config.get("variant")
    pretrained = bool(config.get("pretrained", True))
    dropout = float(config.get("dropout", 0.3))

    model = build_model(
        architecture=arch,
        variant=variant,
        pretrained=pretrained,
        dropout=dropout,
        target_mean=target_mean,
        target_std=target_std,
    )
    param_info = model.count_parameters()
    print(f"Built {arch.upper()} model ({variant}): {param_info['trainable']:,} trainable parameters.")

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
        audit_summary={**audit, **split_info},
    )

    trainer.run()


if __name__ == "__main__":
    main()

