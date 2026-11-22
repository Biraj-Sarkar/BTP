from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models


ROOT = Path(__file__).resolve().parents[3]
V1_SCRIPT = ROOT.parent / "archive" / "prediction_scripts" / "hb_benchmark.py"
MANIFEST = ROOT / "experiments" / "prediction" / "data_audit" / "hb_dataset_manifest.csv"
OUT = ROOT / "experiments" / "prediction" / "benchmarks" / "v2"
CHECKPOINT_ROOT = ROOT.parent / "archive" / "prediction_models" / "hb_regression_v2"
PROTECTED = [
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0.metadata.json",
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0_best.pt",
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0_final.pt",
]
ARCHITECTURES = ("resnet18", "vgg16", "densenet121", "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny")
SEED = 20261116
BATCH_SIZE = 8
MAX_EPOCHS = 40
PATIENCE = 7
FREEZE_EPOCHS = 10
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4


def load_v1():
    spec = importlib.util.spec_from_file_location("hb_benchmark_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load V1 helpers from {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_v1()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def seed_everything(seed: int) -> None:
    V1.seed_everything(seed)


def read_train_validation_manifest() -> tuple[pd.DataFrame, dict[str, int]]:
    metadata_columns = [
        "image_id",
        "registration_no",
        "hb_inclusion_status",
        "extraction_split",
    ]
    metadata = pd.read_csv(MANIFEST, usecols=metadata_columns)
    if len(metadata) != 288:
        raise RuntimeError(f"Expected 288 manifest rows, found {len(metadata)}")
    eligible_metadata = metadata.loc[
        metadata["hb_inclusion_status"] == "eligible_for_supervised_hb_regression"
    ].copy()
    split_counts = eligible_metadata.groupby("extraction_split").size().to_dict()
    expected = {"train": 195, "validation": 43, "test": 41}
    if split_counts != expected:
        raise RuntimeError(f"Unexpected eligible split counts: {split_counts}")
    selected_rows = eligible_metadata.index[
        eligible_metadata["extraction_split"].isin(["train", "validation"])
    ].tolist()
    selected_row_set = set(selected_rows)
    skip_rows = [
        row_number + 1
        for row_number in metadata.index
        if row_number not in selected_row_set
    ]
    target_columns = [
        "image_id",
        "registration_no",
        "original_hb_unit",
        "hb_g_dl",
        "roi_crop_path",
        "extraction_split",
    ]
    frame = pd.read_csv(MANIFEST, usecols=target_columns, skiprows=skip_rows)
    frame = frame.loc[frame["extraction_split"].isin(["train", "validation"])].copy()
    for _, row in frame.iterrows():
        roi = ROOT / str(row["roi_crop_path"])
        if not roi.is_file():
            raise RuntimeError(f"Missing train/validation ROI: {roi}")
        with Image.open(roi) as image:
            image.verify()
        with Image.open(roi) as image:
            image.load()
            if image.mode != "RGB":
                raise RuntimeError(f"ROI is not RGB: {roi} mode={image.mode}")
        target = float(row["hb_g_dl"])
        if not math.isfinite(target):
            raise RuntimeError(f"Non-finite train/validation Hb target: {row['image_id']}={target}")
        if str(row["original_hb_unit"]).strip().lower() not in {"gm/dl", "g/dl"}:
            raise RuntimeError(f"Unexpected Hb unit: {row['image_id']}={row['original_hb_unit']}")
    registrations = {
        split: set(frame.loc[frame["extraction_split"] == split, "registration_no"].astype(str))
        for split in ("train", "validation")
    }
    if registrations["train"] & registrations["validation"]:
        raise RuntimeError("Train/validation registration overlap detected")
    if frame["registration_no"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate train/validation registration numbers")
    return frame, split_counts


def write_loader_audit(frame: pd.DataFrame, split_counts: dict[str, int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_targets = frame.loc[frame["extraction_split"] == "train", "hb_g_dl"].astype(float)
    correction = {
        "manifest": str(MANIFEST),
        "metadata_stage": {
            "columns_loaded": [
                "image_id",
                "registration_no",
                "hb_inclusion_status",
                "extraction_split",
            ],
            "rows_loaded": 288,
            "purpose": "row, eligibility, and split-count validation",
            "target_columns_loaded": False,
            "roi_paths_loaded": False,
        },
        "train_validation_stage": {
            "columns_loaded": [
                "image_id",
                "registration_no",
                "original_hb_unit",
                "hb_g_dl",
                "roi_crop_path",
                "extraction_split",
            ],
            "rows_loaded": 238,
            "rows_by_split": {"train": 195, "validation": 43},
            "purpose": "target validation and ROI readability validation",
        },
        "test_stage": {
            "rows_counted_from_metadata_only": split_counts["test"],
            "target_columns_loaded": False,
            "roi_paths_loaded": False,
            "images_loaded": False,
        },
        "audit_fields": {
            "test_rows_counted_from_metadata_only": True,
            "test_targets_loaded": False,
            "test_images_loaded": False,
            "test_predictions_generated": False,
            "test_used_for_selection": False,
        },
    }
    (OUT / "test_isolation_correction_audit.json").write_text(
        json.dumps(correction, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "manifest": str(MANIFEST),
        "manifest_rows": 288,
        "eligible_rows": sum(split_counts.values()),
        "split_counts": split_counts,
        "train_validation_rows_returned": len(frame),
        "registration_overlap": 0,
        "all_train_validation_roi_paths_readable": True,
        "target_mean_g_dl": float(train_targets.mean()),
        "target_std_g_dl": float(train_targets.std(ddof=0)),
        "test_rows_counted_from_metadata_only": True,
        "test_targets_loaded": False,
        "test_images_loaded": False,
        "test_predictions_generated": False,
        "test_used_for_selection": False,
    }
    (OUT / "dataset_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )


def bn_modules(model: nn.Module) -> list[tuple[str, nn.modules.batchnorm._BatchNorm]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]


def freeze_batchnorm_stats(model: nn.Module) -> None:
    for _, module in bn_modules(model):
        module.eval()


def frozen_bn_buffers(model: nn.Module) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    result = {}
    for name, module in bn_modules(model):
        result[name] = (
            module.running_mean.detach().clone() if module.running_mean is not None else None,
            module.running_var.detach().clone() if module.running_var is not None else None,
            module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
        )
    return result


def compare_bn_buffers(
    before: dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
    model: nn.Module,
) -> list[str]:
    changed = []
    current = frozen_bn_buffers(model)
    for name, values in before.items():
        for label, old, new in zip(("running_mean", "running_var", "num_batches_tracked"), values, current[name]):
            if old is not None and not torch.equal(old, new):
                changed.append(f"{name}.{label}")
    return changed


def head_module(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet18":
        return model.fc
    if architecture == "densenet121":
        return model.classifier
    return model.classifier[-1]


def tail_modules(model: nn.Module, architecture: str) -> list[tuple[str, nn.Module]]:
    if architecture == "resnet18":
        return [("layer4", model.layer4)]
    if architecture == "vgg16":
        return [(f"features[-{index}]", module) for index, module in enumerate(list(model.features[-6:])[::-1], start=1)]
    if architecture == "densenet121":
        return [("features.denseblock4", model.features.denseblock4), ("features.norm5", model.features.norm5)]
    if architecture == "efficientnet_b0":
        return [(f"features[-{index}]", module) for index, module in enumerate(list(model.features[-3:])[::-1], start=1)]
    if architecture == "mobilenet_v3_large":
        return [(f"features[-{index}]", module) for index, module in enumerate(list(model.features[-3:])[::-1], start=1)]
    if architecture == "convnext_tiny":
        return [(f"features[-{index}]", module) for index, module in enumerate(list(model.features[-2:])[::-1], start=1)]
    raise RuntimeError(f"Unknown architecture: {architecture}")


def create_model(architecture: str) -> nn.Module:
    constructors = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "vgg16": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "densenet121": lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
        "convnext_tiny": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
    }
    if architecture not in constructors:
        raise RuntimeError(f"Unknown architecture: {architecture}")
    model = constructors[architecture]()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif architecture == "vgg16":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif architecture == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif architecture == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif architecture == "mobilenet_v3_large":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif architecture == "convnext_tiny":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    for parameter in head_module(model, architecture).parameters():
        parameter.requires_grad = True
    freeze_batchnorm_stats(model)
    return model


def set_train_mode_with_frozen_bn(model: nn.Module) -> None:
    model.train()
    freeze_batchnorm_stats(model)


def unfreeze_tail(model: nn.Module, architecture: str) -> list[str]:
    names = []
    for name, module in tail_modules(model, architecture):
        names.append(name)
        for parameter in module.parameters():
            parameter.requires_grad = True
    # BatchNorm running statistics remain frozen for every architecture.
    freeze_batchnorm_stats(model)
    return names


def parameter_audit(model: nn.Module, architecture: str) -> dict:
    head_ids = {id(parameter) for parameter in head_module(model, architecture).parameters()}
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    head_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and id(parameter) in head_ids]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {
        "trainable_parameter_names": trainable,
        "head_trainable_parameter_names": head_trainable,
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_count": len(trainable),
        "frozen_batchnorm_modules_eval": all(not module.training for _, module in bn_modules(model)),
    }


def smoke_test(architecture: str) -> dict:
    seed_everything(SEED)
    model = create_model(architecture)
    model.train()
    freeze_batchnorm_stats(model)
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    if tuple(output.shape) != (2, 1):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    initial_buffers = frozen_bn_buffers(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HEAD_LR)
    optimizer.zero_grad(set_to_none=True)
    loss = output.square().mean()
    loss.backward()
    frozen_grads = [name for name, parameter in model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
    if frozen_grads:
        raise RuntimeError(f"Frozen parameters received gradients: {frozen_grads[:5]}")
    optimizer.step()
    changed_before_tail = compare_bn_buffers(initial_buffers, model)
    if changed_before_tail:
        raise RuntimeError(f"Frozen BatchNorm buffers changed: {changed_before_tail}")
    tail_names = unfreeze_tail(model, architecture)
    model.train()
    freeze_batchnorm_stats(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=BACKBONE_LR)
    optimizer.zero_grad(set_to_none=True)
    output = model(input_tensor)
    output.square().mean().backward()
    tail_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in {id(p) for p in head_module(model, architecture).parameters()}
        and parameter.grad is not None
    ]
    if not tail_gradients:
        raise RuntimeError("Unfrozen tail received no gradients")
    after_tail = compare_bn_buffers(initial_buffers, model)
    if after_tail:
        raise RuntimeError(f"BatchNorm buffers changed after tail smoke step: {after_tail}")
    checkpoint = OUT / "smoke_tests" / f"{architecture}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, checkpoint)
    restored = create_model(architecture)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    checkpoint.unlink()
    return {
        "architecture": architecture,
        "pretrained_weights": "ImageNet",
        "output_shape": [2, 1],
        "frozen_parameters_received_gradients": False,
        "frozen_batchnorm_eval": True,
        "frozen_batchnorm_buffers_unchanged": True,
        "unfrozen_tail": tail_names,
        "unfrozen_tail_received_gradients": True,
        "checkpoint_save_load": True,
        "parameter_audit_before_unfreeze": parameter_audit(model, architecture),
    }


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return V1.metrics(actual, predicted)


def predict(model, loader, target_mean, target_std, device):
    return V1.predict(model, loader, target_mean, target_std, device)


def train_one(architecture: str, frame: pd.DataFrame) -> dict:
    seed_everything(SEED)
    run_dir = OUT / "runs" / architecture / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train_frame = frame.loc[frame["extraction_split"] == "train"].copy()
    val_frame = frame.loc[frame["extraction_split"] == "validation"].copy()
    target_mean = float(train_frame["hb_g_dl"].mean())
    target_std = float(train_frame["hb_g_dl"].std(ddof=0))
    train_dataset = V1.HbRoiDataset(train_frame, True, target_mean, target_std)
    val_dataset = V1.HbRoiDataset(val_frame, False, target_mean, target_std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = create_model(architecture)
    device = torch.device("cpu")
    model.to(device)
    head_parameters = list(head_module(model, architecture).parameters())
    optimizer = torch.optim.AdamW([{"params": head_parameters, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_mae = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []
    tail_names: list[str] = []
    bn_audit = []
    checkpoint_path = CHECKPOINT_ROOT / architecture / f"{architecture}_seed_{SEED}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            tail_names = unfreeze_tail(model, architecture)
            tail_parameters = [
                parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad and id(parameter) not in {id(p) for p in head_parameters}
            ]
            optimizer = torch.optim.AdamW([
                {"params": head_parameters, "lr": HEAD_LR},
                {"params": tail_parameters, "lr": BACKBONE_LR},
            ], weight_decay=WEIGHT_DECAY)
        frozen_before = frozen_bn_buffers(model)
        set_train_mode_with_frozen_bn(model)
        model.train()
        freeze_batchnorm_stats(model)
        train_loss = 0.0
        batches = 0
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images).reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            frozen_grads = [
                name for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if frozen_grads:
                raise RuntimeError(f"Frozen gradients at epoch {epoch}: {frozen_grads[:5]}")
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        changed_buffers = compare_bn_buffers(frozen_before, model)
        if changed_buffers:
            raise RuntimeError(f"Frozen BatchNorm buffers changed at epoch {epoch}: {changed_buffers}")
        model.eval()
        val_predictions = predict(model, val_loader, target_mean, target_std, device)
        val_metric = metrics(val_predictions["actual_g_dl"].to_numpy(), val_predictions["predicted_g_dl"].to_numpy())
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_mae": val_metric["mae"],
            "val_rmse": val_metric["rmse"],
            "val_r2": val_metric["r2"],
            "tail_unfrozen": epoch > FREEZE_EPOCHS,
        })
        bn_audit.append({
            "epoch": epoch,
            "frozen_batchnorm_eval": all(not module.training for _, module in bn_modules(model)),
            "frozen_batchnorm_buffers_unchanged": not changed_buffers,
            "changed_buffers": changed_buffers,
        })
        if val_metric["mae"] < best_mae:
            best_mae = val_metric["mae"]
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "architecture": architecture,
                "seed": SEED,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_mean_g_dl": target_mean,
                "target_std_g_dl": target_std,
                "best_validation_mae": best_mae,
                "tail_unfrozen": tail_names,
                "batchnorm_policy": "all BatchNorm running statistics kept frozen and modules forced to eval",
            }, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "training_history.csv", index=False)
    history_frame.to_json(run_dir / "training_history.json", orient="records", indent=2)
    (run_dir / "batchnorm_integrity.json").write_text(json.dumps(bn_audit, indent=2) + "\n", encoding="utf-8")
    (run_dir / "frozen_layer_audit.json").write_text(json.dumps({
        "architecture": architecture,
        "head": [name for name, _ in head_module(model, architecture).named_parameters()],
        "tail_unfrozen": tail_names,
        "batchnorm_policy": "all BatchNorm running statistics frozen; eval mode restored after train()",
        "final_parameter_audit": parameter_audit(model, architecture),
    }, indent=2) + "\n", encoding="utf-8")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    val_predictions = predict(model, val_loader, target_mean, target_std, device)
    val_predictions.insert(0, "architecture", architecture)
    val_predictions.insert(1, "seed", SEED)
    val_predictions.insert(2, "split", "validation")
    val_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    result = {
        "architecture": architecture,
        "seed": SEED,
        "status": "completed",
        "best_epoch": best_epoch,
        "validation_metrics": metrics(val_predictions["actual_g_dl"].to_numpy(), val_predictions["predicted_g_dl"].to_numpy()),
        "runtime_seconds": time.time() - start,
        "checkpoint": str(checkpoint_path),
        "tail_unfrozen": tail_names,
        "batchnorm_policy": "all BatchNorm running statistics frozen; eval mode restored after train()",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    protected_before = {str(path): sha256(path) for path in PROTECTED}
    frame, split_counts = read_train_validation_manifest()
    smoke_results = []
    smoke_failures = {}
    for architecture in ARCHITECTURES:
        try:
            smoke_results.append(smoke_test(architecture))
        except Exception as exc:
            smoke_failures[architecture] = {"error": str(exc), "traceback": traceback.format_exc()}
    (OUT / "smoke_test_report.json").write_text(json.dumps({
        "generated_at": now(),
        "completed": smoke_results,
        "failures": smoke_failures,
    }, indent=2) + "\n", encoding="utf-8")
    if smoke_failures:
        raise RuntimeError(f"Smoke tests failed: {sorted(smoke_failures)}")

    train_targets = frame.loc[frame["extraction_split"] == "train", "hb_g_dl"].astype(float)
    (OUT / "benchmark_config.json").write_text(json.dumps({
        "version": "V2",
        "seed": SEED,
        "architectures": ARCHITECTURES,
        "input_size": [224, 224],
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "freeze_epochs": FREEZE_EPOCHS,
        "optimizer": "AdamW",
        "loss": "SmoothL1Loss",
        "head_learning_rate": HEAD_LR,
        "backbone_learning_rate": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
        "augmentation": "resize, mild rotation 5 degrees, horizontal flip, brightness/contrast 0.05",
        "target_standardization": "training targets only",
        "batchnorm_policy": "all BatchNorm running statistics frozen; eval mode restored after every train()",
        "test_access": False,
        "v1_reference": str(ROOT / "experiments" / "prediction" / "benchmarks" / "v1"),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "dataset_validation.json").write_text(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_rows": 288,
        "eligible_rows": sum(split_counts.values()),
        "split_counts": split_counts,
        "registration_overlap": 0,
        "all_roi_paths_readable": True,
        "target_mean_g_dl": float(train_targets.mean()),
        "target_std_g_dl": float(train_targets.std(ddof=0)),
        "test_targets_loaded": False,
    }, indent=2) + "\n", encoding="utf-8")

    train_mean = float(train_targets.mean())
    val = frame.loc[frame["extraction_split"] == "validation", "hb_g_dl"].astype(float).to_numpy()
    baseline = {
        "baseline": "training_set_mean",
        "training_mean_g_dl": train_mean,
        "validation_metrics": metrics(val, np.full(len(val), train_mean)),
        "test_used": False,
    }
    (OUT / "baseline_results.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    results = []
    for architecture in ARCHITECTURES:
        try:
            results.append(train_one(architecture, frame))
        except Exception as exc:
            result = {
                "architecture": architecture,
                "seed": SEED,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(result)
            (OUT / "logs").mkdir(parents=True, exist_ok=True)
            (OUT / "logs" / f"{architecture}_seed_{SEED}.log").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    prediction_frames = []
    for result in results:
        row = {
            "architecture": result["architecture"],
            "seed": SEED,
            "status": result["status"],
            "best_epoch": result.get("best_epoch", ""),
            "validation_mae": result.get("validation_metrics", {}).get("mae", ""),
            "validation_rmse": result.get("validation_metrics", {}).get("rmse", ""),
            "validation_r2": result.get("validation_metrics", {}).get("r2", ""),
            "runtime_seconds": result.get("runtime_seconds", ""),
            "checkpoint": result.get("checkpoint", ""),
        }
        rows.append(row)
        prediction_path = OUT / "runs" / result["architecture"] / f"seed_{SEED}" / "validation_predictions.csv"
        if prediction_path.is_file():
            prediction_frames.append(pd.read_csv(prediction_path))
    summary = pd.DataFrame(rows)
    summary["_mae"] = pd.to_numeric(summary["validation_mae"], errors="coerce")
    summary.sort_values("_mae", na_position="last").drop(columns="_mae").to_csv(OUT / "initial_benchmark_summary.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "validation_predictions.csv", index=False)
    pd.DataFrame([{
        "architecture": result["architecture"],
        "seed": SEED,
        "status": result["status"],
        "validation_mae": result.get("validation_metrics", {}).get("mae", ""),
        "validation_rmse": result.get("validation_metrics", {}).get("rmse", ""),
        "validation_r2": result.get("validation_metrics", {}).get("r2", ""),
    } for result in results]).to_csv(OUT / "seed_stability_summary.csv", index=False)

    (OUT / "batchnorm_integrity_report.json").write_text(json.dumps({
        "policy": "all BatchNorm running statistics frozen and eval mode restored after model.train()",
        "smoke_tests": smoke_results,
        "training_runs": [
            {"architecture": result["architecture"], "status": result["status"], "run_audit": str(OUT / "runs" / result["architecture"] / f"seed_{SEED}" / "batchnorm_integrity.json")}
            for result in results
        ],
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "frozen_layer_audit.json").write_text(json.dumps({
        "smoke_tests": smoke_results,
        "training_run_audits": [
            str(OUT / "runs" / result["architecture"] / f"seed_{SEED}" / "frozen_layer_audit.json")
            for result in results if result["status"] == "completed"
        ],
    }, indent=2) + "\n", encoding="utf-8")
    protected_after = {str(path): sha256(path) for path in PROTECTED}
    (OUT / "artifact_integrity.json").write_text(json.dumps({
        "protected_artifacts": {
            path: {
                "before_sha256": protected_before[path],
                "after_sha256": protected_after[path],
                "unchanged": protected_before[path] == protected_after[path],
            } for path in protected_before
        },
        "v1_outputs_modified": False,
        "test_evaluated": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
