from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARCHIVE_ROOT = Path(__file__).resolve().parents[3]
ROOT = ARCHIVE_ROOT.parent / "BTP-Anemia-Final"
INFERENCE_SCRIPT = ROOT / "experiments" / "prediction" / "scripts" / "inference.py"
INFERENCE_SPEC = importlib.util.spec_from_file_location("selected_hb_inference", INFERENCE_SCRIPT)
if INFERENCE_SPEC is None or INFERENCE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load maintained inference module from {INFERENCE_SCRIPT}")
INFERENCE = importlib.util.module_from_spec(INFERENCE_SPEC)
INFERENCE_SPEC.loader.exec_module(INFERENCE)
inverse_target = INFERENCE.inverse_target
load_model = INFERENCE.load_model
preprocess_image = INFERENCE.preprocess_image
MANIFEST = ROOT / "experiments" / "prediction" / "data_audit" / "hb_dataset_manifest.csv"
V2_DATASET_VALIDATION = ROOT / "experiments" / "prediction" / "benchmarks" / "v2" / "dataset_validation.json"
OUT = ROOT / "experiments" / "prediction" / "final_evaluation"
CHECKPOINT = ROOT / "models" / "predictors" / "densenet121_seed_20261118_best.pt"
ARCHITECTURE = "densenet121"
SEED = 20261118
PROTECTED = [
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0.metadata.json",
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0_best.pt",
    ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0_final.pt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_test_manifest() -> tuple[pd.DataFrame, dict[str, int]]:
    metadata_columns = ["image_id", "registration_no", "hb_inclusion_status", "extraction_split"]
    metadata = pd.read_csv(MANIFEST, usecols=metadata_columns)
    if len(metadata) != 288:
        raise RuntimeError(f"Expected 288 manifest rows, found {len(metadata)}")
    eligible = metadata.loc[
        metadata["hb_inclusion_status"] == "eligible_for_supervised_hb_regression"
    ].copy()
    counts = eligible.groupby("extraction_split").size().to_dict()
    if counts != {"train": 195, "validation": 43, "test": 41}:
        raise RuntimeError(f"Unexpected eligible split counts: {counts}")
    test_rows = eligible.index[eligible["extraction_split"] == "test"].tolist()
    test_row_set = set(test_rows)
    skip_rows = [row_number + 1 for row_number in metadata.index if row_number not in test_row_set]
    target_columns = [
        "image_id",
        "registration_no",
        "original_hb_unit",
        "hb_g_dl",
        "roi_crop_path",
        "extraction_split",
    ]
    frame = pd.read_csv(MANIFEST, usecols=target_columns, skiprows=skip_rows)
    if len(frame) != 41 or set(frame["extraction_split"]) != {"test"}:
        raise RuntimeError(f"Expected 41 test records, found {len(frame)}")
    if frame["registration_no"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate test registration numbers")
    train_registrations = set(
        metadata.loc[
            (metadata["hb_inclusion_status"] == "eligible_for_supervised_hb_regression")
            & (metadata["extraction_split"] == "train"),
            "registration_no",
        ].astype(str)
    )
    validation_registrations = set(
        metadata.loc[
            (metadata["hb_inclusion_status"] == "eligible_for_supervised_hb_regression")
            & (metadata["extraction_split"] == "validation"),
            "registration_no",
        ].astype(str)
    )
    test_registrations = set(frame["registration_no"].astype(str))
    if test_registrations & train_registrations or test_registrations & validation_registrations:
        raise RuntimeError("Test patient overlap detected")
    for _, row in frame.iterrows():
        roi = ROOT / str(row["roi_crop_path"])
        if not roi.is_file():
            raise RuntimeError(f"Missing test ROI: {roi}")
        with Image.open(roi) as image:
            image.verify()
        target = float(row["hb_g_dl"])
        if not math.isfinite(target):
            raise RuntimeError(f"Invalid test target: {row['image_id']}={target}")
        if str(row["original_hb_unit"]).strip().lower() not in {"gm/dl", "g/dl"}:
            raise RuntimeError(f"Unexpected Hb unit: {row['original_hb_unit']}")
    return frame, counts


def state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def state_changes(before: dict[str, torch.Tensor], model: torch.nn.Module) -> list[str]:
    after = model.state_dict()
    return [name for name, value in before.items() if not torch.equal(value, after[name])]


def frozen_bn_buffers(model: torch.nn.Module) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            result[name] = (
                module.running_mean.detach().clone() if module.running_mean is not None else None,
                module.running_var.detach().clone() if module.running_var is not None else None,
                module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
            )
    return result


def compare_bn_buffers(
    before: dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
    model: torch.nn.Module,
) -> list[str]:
    after = frozen_bn_buffers(model)
    changed = []
    for name, values in before.items():
        for label, old, new in zip(("running_mean", "running_var", "num_batches_tracked"), values, after[name]):
            if old is not None and not torch.equal(old, new):
                changed.append(f"{name}.{label}")
    return changed


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    absolute = np.abs(errors)
    mean_diff = float(np.mean(errors))
    sd_diff = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias": mean_diff,
        "median_absolute_error": float(np.median(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "bland_altman_mean_difference": mean_diff,
        "bland_altman_loa_low": mean_diff - 1.96 * sd_diff,
        "bland_altman_loa_high": mean_diff + 1.96 * sd_diff,
    }


def save_plots(predictions: pd.DataFrame) -> None:
    actual = predictions["actual_g_dl"].to_numpy()
    predicted = predictions["predicted_g_dl"].to_numpy()
    residual = predicted - actual
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.75)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "k--")
    ax.set(xlabel="Actual Hb (g/dL)", ylabel="Predicted Hb (g/dL)", title="DenseNet-121 test predictions")
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted, residual, alpha=0.75)
    ax.axhline(0, color="k", linestyle="--")
    ax.set(xlabel="Predicted Hb (g/dL)", ylabel="Residual (g/dL)", title="DenseNet-121 test residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_plot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.abs(residual), bins=10)
    ax.set(xlabel="Absolute error (g/dL)", ylabel="Count", title="Test absolute-error distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "absolute_error_distribution.png", dpi=150)
    plt.close(fig)

    means = (actual + predicted) / 2
    difference = float(np.mean(residual))
    sd = float(np.std(residual, ddof=1))
    low_loa = difference - 1.96 * sd
    high_loa = difference + 1.96 * sd
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(means, residual, alpha=0.75)
    ax.axhline(difference, color="r", label="mean difference")
    ax.axhline(low_loa, color="gray", linestyle="--", label="95% LoA")
    ax.axhline(high_loa, color="gray", linestyle="--")
    ax.set(xlabel="Mean actual/predicted Hb (g/dL)", ylabel="Predicted - actual (g/dL)", title="Bland-Altman plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "bland_altman.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)
    protected_before = {str(path): sha256(path) for path in PROTECTED}
    checkpoint_before = sha256(CHECKPOINT)
    selection = pd.read_csv(
        ROOT / "experiments" / "prediction" / "benchmarks" / "v2_stability" / "stability_results.csv"
    )
    candidate = selection.loc[
        (selection["architecture"] == ARCHITECTURE) & (selection["status"] == "completed")
    ].sort_values("validation_mae").iloc[0]
    if int(candidate["seed"]) != SEED or Path(candidate["checkpoint"]) != CHECKPOINT:
        raise RuntimeError("Selected checkpoint does not match the lowest DenseNet-121 validation MAE")
    model, checkpoint_metadata = load_model(CHECKPOINT)
    model_before = state_snapshot(model)
    bn_before = frozen_bn_buffers(model)
    test_frame, split_counts = load_test_manifest()
    dataset_validation = json.loads(V2_DATASET_VALIDATION.read_text(encoding="utf-8"))
    target_mean = checkpoint_metadata["target_mean_g_dl"]
    target_std = checkpoint_metadata["target_std_g_dl"]
    training_mean = float(dataset_validation["target_mean_g_dl"])

    smoke_image = preprocess_image(ROOT / str(test_frame.iloc[0]["roi_crop_path"]))
    with torch.inference_mode():
        smoke_output = model(smoke_image.unsqueeze(0))
    if tuple(smoke_output.shape) != (1, 1):
        raise RuntimeError(f"Single-record inference smoke test failed: {tuple(smoke_output.shape)}")

    rows = []
    with torch.inference_mode():
        for _, row in test_frame.iterrows():
            image = preprocess_image(ROOT / str(row["roi_crop_path"]))
            standardized = model(image.unsqueeze(0)).reshape(-1)
            predicted = inverse_target(standardized, checkpoint_metadata)
            rows.append({
                "registration_no": str(row["registration_no"]),
                "actual_g_dl": float(row["hb_g_dl"]),
                "predicted_g_dl": float(predicted.item()),
                "split": "test",
            })
    predictions = pd.DataFrame(rows)
    if len(predictions) != 41 or predictions["registration_no"].duplicated().any():
        raise RuntimeError("Test prediction count or uniqueness check failed")
    model_changes = state_changes(model_before, model)
    bn_changes = compare_bn_buffers(bn_before, model)
    checkpoint_after = sha256(CHECKPOINT)
    selected_metrics = metrics(predictions["actual_g_dl"].to_numpy(), predictions["predicted_g_dl"].to_numpy())
    baseline_predictions = np.full(len(predictions), training_mean)
    baseline_metrics = metrics(predictions["actual_g_dl"].to_numpy(), baseline_predictions)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    save_plots(predictions)

    final_selection = {
        "architecture": ARCHITECTURE,
        "selected_seed": SEED,
        "selection_metric": "validation_mae",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_before,
        "best_epoch": int(candidate["best_epoch"]),
        "validation_metrics": {
            "mae": float(candidate["validation_mae"]),
            "rmse": float(candidate["validation_rmse"]),
            "r2": float(candidate["validation_r2"]),
        },
        "configuration": {
            "input_size": [224, 224],
            "rgb": True,
            "preprocessing": "resize 224x224, ImageNet normalization; no color correction",
            "target_standardization": "training-target mean/std from selected checkpoint",
            "batch_size": 1,
            "batchnorm_policy": "frozen running statistics; model eval during inference",
        },
        "test_selection_used": False,
    }
    (OUT / "final_model_selection.json").write_text(json.dumps(final_selection, indent=2) + "\n", encoding="utf-8")
    (OUT / "test_metrics.json").write_text(json.dumps({
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "test_records": len(predictions),
        "model": selected_metrics,
        "training_mean_baseline": baseline_metrics,
        "training_mean_g_dl": training_mean,
        "test_used_for_selection": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "test_evaluation_audit.json").write_text(json.dumps({
        "manifest": str(MANIFEST),
        "metadata_columns_loaded": ["image_id", "registration_no", "hb_inclusion_status", "extraction_split"],
        "test_target_columns_loaded": [
            "image_id", "registration_no", "original_hb_unit", "hb_g_dl", "roi_crop_path", "extraction_split"
        ],
        "test_rows_loaded": 41,
        "test_predictions_generated": 41,
        "duplicate_test_records": False,
        "patient_overlap_train": 0,
        "patient_overlap_validation": 0,
        "training_operations_called": False,
        "optimizer_created": False,
        "backward_called": False,
        "weights_changed": model_changes,
        "batchnorm_buffers_changed": bn_changes,
        "single_record_smoke_test": True,
        "test_used_for_selection": False,
    }, indent=2) + "\n", encoding="utf-8")
    protected_after = {str(path): sha256(path) for path in PROTECTED}
    integrity = {
        path: {
            "before_sha256": protected_before[path],
            "after_sha256": protected_after[path],
            "unchanged": protected_before[path] == protected_after[path],
        }
        for path in protected_before
    }
    (OUT / "artifact_integrity.json").write_text(json.dumps({
        "selected_checkpoint": {
            "path": str(CHECKPOINT),
            "before_sha256": checkpoint_before,
            "after_sha256": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
            "model_state_changed": model_changes,
            "batchnorm_buffers_changed": bn_changes,
        },
        "protected_extraction_artifacts": integrity,
        "v1_outputs_modified": False,
        "v2_outputs_modified": False,
        "stability_outputs_modified": False,
    }, indent=2) + "\n", encoding="utf-8")

    (OUT / "final_model_selection_report.md").write_text(
        f"""# Final Hb Model Selection

DenseNet-121 seed {SEED} was selected using validation MAE only from the
three-seed Phase 7.2 stability results. Its validation MAE was
{float(candidate['validation_mae']):.4f} g/dL, RMSE
{float(candidate['validation_rmse']):.4f} g/dL, and R2
{float(candidate['validation_r2']):.4f}, at epoch {int(candidate['best_epoch'])}.

Checkpoint: `{CHECKPOINT}`

The test set was not used for this selection. No checkpoint averaging,
retraining, or test-driven tuning was performed.
""",
        encoding="utf-8",
    )
    (OUT / "test_evaluation_report.md").write_text(
        f"""# Held-Out Test Evaluation

The selected DenseNet-121 checkpoint was evaluated once on the 41-record
patient-wise test split using inference mode only.

## Model

- MAE: {selected_metrics['mae']:.4f} g/dL
- RMSE: {selected_metrics['rmse']:.4f} g/dL
- R2: {selected_metrics['r2']:.4f}
- Bias: {selected_metrics['bias']:.4f} g/dL
- Median absolute error: {selected_metrics['median_absolute_error']:.4f} g/dL
- Maximum absolute error: {selected_metrics['maximum_absolute_error']:.4f} g/dL
- Bland-Altman 95% LoA: {selected_metrics['bland_altman_loa_low']:.4f} to {selected_metrics['bland_altman_loa_high']:.4f} g/dL

## Training-mean baseline

- Training mean: {training_mean:.4f} g/dL
- MAE: {baseline_metrics['mae']:.4f} g/dL
- RMSE: {baseline_metrics['rmse']:.4f} g/dL
- R2: {baseline_metrics['r2']:.4f}

The V1 Phase 7 VGG16 test result exists as historical evidence. This V2
evaluation is not independent external validation, clinical validation, or
evidence that the model can diagnose anemia. It is an internal held-out
research evaluation on a small single-source cohort.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
