"""Conjunctiva preprocessing pipeline for anemia detection (CIELAB Version).

This script is designed for the current dataset layout in:
    /Users/yatikajena/Desktop/AnemiaDetection/dataset anemia/India

Modifications for CIELAB:
1. Removed Gray World White Balance (preserves redness).
2. Applies CLAHE to the L* channel only to normalize lighting.
3. Modifies ResNet18 to accept only the a* and b* channels (2-channel input),
   forcing the network to learn hemoglobin levels purely from color.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

WORKSPACE_CACHE = Path(__file__).resolve().parent / ".hf_cache"
os.environ.setdefault("HF_HOME", str(WORKSPACE_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(WORKSPACE_CACHE / "hub"))

import cv2
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = object
    Dataset = object
    DataLoader = object
    AdamW = object

try:
    from torchvision.models import ResNet18_Weights, resnet18
except Exception:  # pragma: no cover
    ResNet18_Weights = None
    resnet18 = None

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
except Exception:  # pragma: no cover
    Mask2FormerForUniversalSegmentation = None
    Mask2FormerImageProcessor = None


@dataclass(frozen=True)
class PipelineConfig:
    dataset_root: Path
    output_root: Path
    train_limit: int = 50
    demo_limit: int = 5
    segmentation_checkpoint: str = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    target_size: Tuple[int, int] = (320, 320)
    training_size: Tuple[int, int] = (512, 512)
    blur_threshold: float = 35.0
    min_mask_ratio: float = 0.015
    max_mask_ratio: float = 0.85
    train_epochs: int = 3
    train_batch_size: int = 2
    learning_rate: float = 5e-5
    regression_image_size: Tuple[int, int] = (224, 224)
    hb_train_epochs: int = 25
    hb_batch_size: int = 8
    hb_learning_rate: float = 1e-4


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_raw_images(india_root: Path) -> List[Path]:
    images = sorted(
        [
            path
            for path in india_root.glob("*/*.jpg")
            if path.is_file() and "_" not in path.stem.split("/")[-1]
        ]
    )
    if not images:
        images = sorted(india_root.glob("*/*.jpg"))
    return images


def discover_mask_candidates(image_path: Path) -> List[Path]:
    stem = image_path.stem
    candidates = [
        image_path.with_name(f"{stem}_forniceal_palpebral.png"),
        image_path.with_name(f"{stem}_forniceal.png"),
        image_path.with_name(f"{stem}_palpebral.png"),
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def find_first_jpg(image_dir: Path) -> Optional[Path]:
    jpgs = sorted(image_dir.glob("*.jpg"))
    return jpgs[0] if jpgs else None


def load_india_hb_records(india_root: Path, limit: int) -> List[dict]:
    workbook_path = india_root / "India.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(f"Missing Hb label workbook: {workbook_path}")

    table = pd.read_excel(workbook_path)
    required_columns = {"Number", "Hgb"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"India.xlsx is missing columns: {sorted(missing_columns)}")

    records: List[dict] = []
    for _, row in table.sort_values("Number").head(limit).iterrows():
        number = int(row["Number"])
        sample_dir = india_root / str(number)
        image_path = find_first_jpg(sample_dir)
        if image_path is None:
            continue
        records.append(
            {
                "number": number,
                "image_path": image_path,
                "hgb": float(row["Hgb"]),
            }
        )

    if not records:
        raise RuntimeError(f"No labeled Indian samples found under {india_root}")

    return records


def read_binary_mask(mask_path: Path) -> Optional[np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = mask.max(axis=2)
    return (mask > 0).astype(np.uint8)


def load_training_mask(image_path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    mask_candidates = discover_mask_candidates(image_path)
    if not mask_candidates:
        return None

    combined = None
    for mask_path in mask_candidates:
        candidate = read_binary_mask(mask_path)
        if candidate is None:
            continue
        if combined is None:
            combined = candidate.copy()
        else:
            combined = np.maximum(combined, candidate)

    if combined is None:
        return None

    resized = cv2.resize(combined, target_size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def read_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(image_rgb: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def write_gray(image_gray: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), image_gray)


def resize_image(image_rgb: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)


def cielab_normalization(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts to LAB space and normalizes only the Lightness channel."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L-channel to normalize lighting variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    normalized_lab = cv2.merge((cl, a_channel, b_channel))
    # Return both the normalized RGB (for visualization) and the LAB image (for processing)
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB), normalized_lab


def focus_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_score(image_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    return float(l_channel.std())


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8 * 255

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return largest_connected_component(closed)


def heuristic_conjunctiva_mask(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    valid = np.ones_like(l_channel, dtype=bool)
    l_threshold = np.percentile(l_channel[valid], 64)
    a_center = np.median(a_channel[valid])
    b_center = np.median(b_channel[valid])
    chroma = np.abs(a_channel.astype(np.int16) - int(a_center)) + np.abs(
        b_channel.astype(np.int16) - int(b_center)
    )
    chroma_threshold = np.percentile(chroma[valid], 55)

    candidate = np.logical_and(l_channel >= l_threshold, chroma <= chroma_threshold)
    candidate_mask = (candidate.astype(np.uint8)) * 255
    candidate_mask = smooth_mask(candidate_mask)
    return candidate_mask


def grabcut_mask(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    rect = (
        int(width * 0.08),
        int(height * 0.08),
        int(width * 0.84),
        int(height * 0.84),
    )
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grabcut = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        grabcut = np.zeros((height, width), dtype=np.uint8)

    return smooth_mask(grabcut)


def segment_conjunctiva(
    image_rgb: np.ndarray,
    segmenter: Optional[callable] = None,
) -> Tuple[np.ndarray, str]:
    if segmenter is not None:
        return segmenter(image_rgb)

    heuristic_mask = heuristic_conjunctiva_mask(image_rgb)
    ratio = float(np.count_nonzero(heuristic_mask)) / float(heuristic_mask.size)

    if ratio < 0.01 or ratio > 0.90:
        fallback_mask = grabcut_mask(image_rgb)
        fallback_ratio = float(np.count_nonzero(fallback_mask)) / float(fallback_mask.size)
        if fallback_ratio > 0:
            return fallback_mask, "grabcut"
        return heuristic_mask, "heuristic"

    return heuristic_mask, "heuristic"


def tight_bbox(mask: np.ndarray, pad_ratio: float = 0.08) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    pad_x = max(int(width * pad_ratio), 4)
    pad_y = max(int(height * pad_ratio), 4)

    h, w = mask.shape[:2]
    left = max(0, x_min - pad_x)
    top = max(0, y_min - pad_y)
    right = min(w, x_max + pad_x + 1)
    bottom = min(h, y_max + pad_y + 1)
    return left, top, right, bottom


def apply_mask(image_numpy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_numpy, image_numpy, mask=mask)


def build_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    red = np.zeros_like(image_rgb)
    red[:, :, 0] = 255
    alpha = 0.35
    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * red[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def pad_to_square(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    
    padded_img = cv2.copyMakeBorder(image, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_mask = cv2.copyMakeBorder(mask, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=0)
    
    return padded_img, padded_mask


def crop_to_mask(image_numpy: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = tight_bbox(mask)
    cropped_img = image_numpy[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    if cropped_img.size > 0:
        cropped_img, cropped_mask = pad_to_square(cropped_img, cropped_mask)
    return cropped_img, cropped_mask


class Mask2FormerConjunctivaDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], config: PipelineConfig):
        self.image_paths = list(image_paths)
        self.config = config

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = resize_image(read_rgb(image_path), self.config.training_size)
        
        # We pass the CLAHE normalized RGB image to Mask2Former for segmentation training
        normalized_rgb, _ = cielab_normalization(image)
        
        mask = load_training_mask(image_path, self.config.training_size)
        return {
            "image": normalized_rgb,
            "mask": mask,
            "image_path": image_path,
        }


def build_mask2former_components(config: PipelineConfig, device: Any):
    if Mask2FormerImageProcessor is None or Mask2FormerForUniversalSegmentation is None:
        raise RuntimeError("transformers Mask2Former components are not available in this environment")

    legacy_root = Path(__file__).resolve().parent / "conjunctiva_pipeline_outputs"
    candidate_model_dirs = [
        config.output_root / "mask2former_model",
        legacy_root / "mask2former_model",
    ]
    candidate_processor_dirs = [
        config.output_root / "mask2former_processor",
        legacy_root / "mask2former_processor",
    ]

    local_model_dir = next((path for path in candidate_model_dirs if path.exists()), None)
    local_processor_dir = next((path for path in candidate_processor_dirs if path.exists()), None)
    pretrained_source: Path | str
    processor_source: Path | str

    if local_model_dir is not None and local_processor_dir is not None:
        pretrained_source = local_model_dir
        processor_source = local_processor_dir
    else:
        pretrained_source = config.segmentation_checkpoint
        processor_source = config.segmentation_checkpoint

    try:
        processor = Mask2FormerImageProcessor.from_pretrained(processor_source, local_files_only=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_source,
            num_labels=2,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        if hasattr(processor, "do_reduce_labels"):
            processor.do_reduce_labels = False
        model.to(device)
        return model, processor
    except Exception as exc:
        print(f"Mask2Former unavailable offline, falling back to heuristic segmentation: {exc}")
        return None, None


def train_mask2former(config: PipelineConfig, train_paths: Sequence[Path], device: Any):
    dataset = Mask2FormerConjunctivaDataset(train_paths, config)
    usable_items = [item for item in (dataset[i] for i in range(len(dataset))) if item["mask"] is not None]
    if not usable_items:
        raise RuntimeError("No usable image/mask pairs were found for Mask2Former training")

    model, processor = build_mask2former_components(config, device)
    if model is None or processor is None:
        return None, None
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    model.train()

    print("\n--- Training Mask2Former segmentation model ---")
    print(f"Checkpoint: {config.segmentation_checkpoint}")
    print(f"Training pairs: {len(usable_items)}")

    for epoch in range(config.train_epochs):
        epoch_loss = 0.0
        random.shuffle(usable_items)
        steps = 0
        for start in range(0, len(usable_items), config.train_batch_size):
            batch = usable_items[start : start + config.train_batch_size]
            images = [item["image"] for item in batch]
            masks = [item["mask"] for item in batch]
            encoding = processor(images=images, segmentation_maps=masks, return_tensors="pt")
            encoding = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoding.items()
            }

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**encoding)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            steps += 1

        mean_loss = epoch_loss / max(steps, 1)
        print(f"Epoch {epoch + 1}/{config.train_epochs} | loss={mean_loss:.4f}")

    model_dir = config.output_root / "mask2former_model"
    processor_dir = config.output_root / "mask2former_processor"
    ensure_dir(model_dir)
    ensure_dir(processor_dir)
    model.save_pretrained(model_dir)
    processor.save_pretrained(processor_dir)
    print(f"Saved trained model to {model_dir}")
    return model, processor


def make_mask2former_segmenter(model, processor, device: Any):
    def _segment(image_rgb: np.ndarray) -> Tuple[np.ndarray, str]:
        model.eval()
        with torch.no_grad():
            encoding = processor(images=image_rgb, return_tensors="pt")
            encoding = {key: value.to(device) for key, value in encoding.items()}
            outputs = model(**encoding)
            target_sizes = [(image_rgb.shape[0], image_rgb.shape[1])]
            semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
            mask = (semantic_map.detach().cpu().numpy() == 1).astype(np.uint8) * 255
            return smooth_mask(mask), "mask2former"

    return _segment


def evaluate_mask2former(
    model,
    processor,
    image_paths: Sequence[Path],
    config: PipelineConfig,
    device: Any,
) -> dict:
    if not image_paths:
        return {"dice": 0.0, "iou": 0.0, "count": 0}

    model.eval()
    dice_values: List[float] = []
    iou_values: List[float] = []

    with torch.no_grad():
        for image_path in image_paths:
            image = resize_image(read_rgb(image_path), config.training_size)
            normalized_rgb, _ = cielab_normalization(image)
            target_mask = load_training_mask(image_path, config.training_size)
            if target_mask is None:
                continue

            encoding = processor(images=normalized_rgb, return_tensors="pt")
            encoding = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoding.items()
            }
            outputs = model(**encoding)
            target_sizes = [(image.shape[0], image.shape[1])]
            semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
            pred_mask = (semantic_map.detach().cpu().numpy() == 1).astype(np.uint8) * 255

            dice_values.append(dice_score(pred_mask, target_mask))
            iou_values.append(iou_score(pred_mask, target_mask))

    if not dice_values:
        return {"dice": 0.0, "iou": 0.0, "count": 0}

    return {
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "count": len(dice_values),
    }


def train_hb_regressor(
    config: PipelineConfig,
    train_records: Sequence[dict],
    eval_records: Sequence[dict],
    segmenter: Any,
    device: Any,
) -> dict:
    if not train_records:
        raise RuntimeError("No Hb training records available")

    train_targets = [record["hgb"] for record in train_records]
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets)) if float(np.std(train_targets)) > 1e-6 else 1.0
    train_weights = regression_sample_weights(train_targets)
    for record, sample_weight in zip(train_records, train_weights):
        record["sample_weight"] = float(sample_weight)

    for record in eval_records:
        record["sample_weight"] = 1.0

    train_dataset = HbRegressionDataset(train_records, config, segmenter, target_mean, target_std, augment=True)
    eval_dataset = HbRegressionDataset(eval_records, config, segmenter, target_mean, target_std, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.hb_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.hb_batch_size, shuffle=False)

    model = HbRegressionNet(use_pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=config.hb_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print("\n--- Training Hb regression network ---")
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    for epoch in range(config.hb_train_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, hb_targets, sample_weights, _, _ in train_loader:
            images = images.to(device)
            hb_targets = hb_targets.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = torch.mean(sample_weights * torch.nn.functional.smooth_l1_loss(predictions, hb_targets, reduction="none"))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            batch_count += 1

        mean_loss = running_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{config.hb_train_epochs} | weighted MSE={mean_loss:.4f}")
        scheduler.step(mean_loss)

    model.eval()
    eval_rows = []
    predictions_all: List[float] = []
    targets_all: List[float] = []
    with torch.no_grad():
        for images, hb_targets, _, numbers, filenames in eval_loader:
            images = images.to(device)
            outputs = model(images).detach().cpu().numpy().reshape(-1)
            targets = hb_targets.numpy().reshape(-1)
            outputs_denorm = outputs * target_std + target_mean
            targets_denorm = targets * target_std + target_mean
            predictions_all.extend(outputs_denorm.tolist())
            targets_all.extend(targets_denorm.tolist())
            for number, filename, target, pred in zip(numbers, filenames, targets_denorm, outputs_denorm):
                eval_rows.append(
                    {
                        "number": int(number),
                        "filename": filename,
                        "actual_hgb": float(target),
                        "predicted_hgb": float(pred),
                    }
                )

    metrics = {
        "count": len(targets_all),
        "mae": float(np.mean(np.abs(np.asarray(predictions_all) - np.asarray(targets_all)))) if targets_all else 0.0,
        "rmse": root_mean_squared_error(np.asarray(predictions_all), np.asarray(targets_all)) if targets_all else 0.0,
    }

    all_records = list(train_records) + list(eval_records)
    all_dataset = HbRegressionDataset(all_records, config, segmenter, target_mean, target_std, augment=False)
    all_loader = DataLoader(all_dataset, batch_size=config.hb_batch_size, shuffle=False)
    all_predictions: List[float] = []
    with torch.no_grad():
        for images, _, _, _, _ in all_loader:
            images = images.to(device)
            outputs = model(images).detach().cpu().numpy().reshape(-1)
            all_predictions.extend((outputs * target_std + target_mean).tolist())

    package_root = config.output_root / "final_prediction_package"

    hb_model_path = config.output_root / "hb_regression_model.pt"
    hb_metrics_path = config.output_root / "hb_regression_metrics.json"
    hb_predictions_path = config.output_root / "hb_regression_predictions.csv"

    ensure_dir(config.output_root)
    torch.save(model.state_dict(), hb_model_path)
    save_eval_metrics(metrics, hb_metrics_path)
    save_predictions_csv(eval_rows, hb_predictions_path)
    save_prediction_package(all_records, all_predictions, config, segmenter, package_root)

    print(
        f"Held-out Hb evaluation on {metrics['count']} images: "
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}"
    )
    print(f"Saved Hb regression model to {hb_model_path}")
    print(f"Saved full prediction package to {package_root}")
    return metrics


def make_contact_sheet(records: Sequence[dict], output_path: Path) -> None:
    panels: List[np.ndarray] = []
    for record in records:
        original = cv2.cvtColor(cv2.imread(record["raw_path"]), cv2.COLOR_BGR2RGB)
        overlay = cv2.cvtColor(cv2.imread(record["overlay_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        normalized = cv2.cvtColor(cv2.imread(record["normalized_path"]), cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        def fit(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)

        top_row = np.concatenate([fit(original), fit(overlay)], axis=1)
        bottom_row = np.concatenate([fit(mask_rgb), fit(normalized)], axis=1)
        panel = np.concatenate([top_row, bottom_row], axis=0)
        panels.append(panel)

    if not panels:
        return

    sheet = np.concatenate(panels, axis=0)
    write_rgb(sheet, output_path)


def save_manifest(records: Sequence[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)


def save_config(config: PipelineConfig) -> None:
    ensure_dir(config.output_root)
    payload = {
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "train_limit": config.train_limit,
        "demo_limit": config.demo_limit,
        "segmentation_checkpoint": config.segmentation_checkpoint,
        "target_size": list(config.target_size),
        "training_size": list(config.training_size),
        "blur_threshold": config.blur_threshold,
        "min_mask_ratio": config.min_mask_ratio,
        "max_mask_ratio": config.max_mask_ratio,
        "train_epochs": config.train_epochs,
        "train_batch_size": config.train_batch_size,
        "learning_rate": config.learning_rate,
        "regression_image_size": list(config.regression_image_size),
        "hb_train_epochs": config.hb_train_epochs,
        "hb_batch_size": config.hb_batch_size,
        "hb_learning_rate": config.hb_learning_rate,
    }
    with (config.output_root / "pipeline_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_eval_metrics(metrics: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_predictions_csv(rows: Sequence[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def preprocess_hb_image(image_path: Path, config: PipelineConfig, segmenter: Any) -> dict:
    raw_rgb = read_rgb(image_path)
    resized_rgb = resize_image(raw_rgb, config.target_size)
    
    # We apply the CIELAB normalization to correct lighting
    balanced_rgb, balanced_lab = cielab_normalization(resized_rgb)
    
    # Segment based on the lighting-normalized RGB
    mask, _ = segment_conjunctiva(balanced_rgb, segmenter=segmenter)
    
    # Apply the mask directly to the LAB image, since that is what ResNet needs
    masked_lab = apply_mask(balanced_lab, mask)
    
    # Crop the LAB image to the mask
    cropped_lab, cropped_mask = crop_to_mask(masked_lab, mask)
    cropped_mask = (cropped_mask > 0).astype(np.uint8) * 255
    
    normalized_lab = cropped_lab if cropped_lab.size else masked_lab
    
    return {
        "raw_rgb": raw_rgb,
        "resized_rgb": resized_rgb,
        "balanced_rgb": balanced_rgb,
        "mask": mask,
        "cropped_mask": cropped_mask,
        "normalized_lab": normalized_lab,
    }


def augment_hb_image(image_lab: np.ndarray) -> np.ndarray:
    augmented = image_lab.copy()

    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 1)

    if random.random() < 0.35:
        angle = random.uniform(-8.0, 8.0)
        height, width = augmented.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        augmented = cv2.warpAffine(
            augmented,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if random.random() < 0.2:
        kernel = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (kernel, kernel), 0)

    return augmented


def save_prediction_package(
    records: Sequence[dict],
    predictions: Sequence[float],
    config: PipelineConfig,
    segmenter: Any,
    package_root: Path,
) -> None:
    original_dir = package_root / "original_images"
    mask_dir = package_root / "masks"
    overlay_dir = package_root / "overlays"
    normalized_dir = package_root / "normalized_a_channel"
    ensure_dir(package_root)

    rows: List[dict] = []
    for record, prediction in zip(records, predictions):
        prep = preprocess_hb_image(record["image_path"], config, segmenter)
        sample_name = f"{record['number']}_{record['image_path'].name}"

        ensure_dir(original_dir)
        shutil.copy2(record["image_path"], original_dir / sample_name)
        write_gray(prep["mask"], mask_dir / f"{sample_name}_mask.png")
        write_rgb(build_overlay(prep["balanced_rgb"], prep["mask"]), overlay_dir / f"{sample_name}_overlay.png")
        
        # Save just the a* channel visually as a grayscale image for debugging
        _, a_channel, _ = cv2.split(prep["normalized_lab"])
        write_gray(a_channel, normalized_dir / f"{sample_name}_normalized_a.png")

        rows.append(
            {
                "number": int(record["number"]),
                "filename": record["image_path"].name,
                "actual_hgb": float(record["hgb"]),
                "predicted_hgb": float(prediction),
                "final_prediction": "Anemic" if float(prediction) < 11.0 else "Normal",
                "actual_anemia": "Anemic" if float(record["hgb"]) < 11.0 else "Normal",
            }
        )

    save_predictions_csv(rows, package_root / "all_predictions.csv")


def regression_sample_weights(targets: Sequence[float], bins: int = 10) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)

    if values.min() == values.max():
        return np.ones_like(values, dtype=np.float32)

    hist, edges = np.histogram(values, bins=bins)
    bin_ids = np.clip(np.digitize(values, edges[1:-1], right=True), 0, bins - 1)
    weights = np.asarray([1.0 / max(hist[bin_id], 1) for bin_id in bin_ids], dtype=np.float32)
    weights *= float(weights.size) / float(weights.sum())
    return weights


def weighted_mse(predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if weights.shape != targets.shape:
        raise ValueError("weights and targets must have the same shape")
    return float(np.mean(weights * np.square(predictions - targets)))


def root_mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(predictions - targets))))


def dice_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    pred = (np.asarray(pred_mask) > 0).astype(np.float32)
    target = (np.asarray(target_mask) > 0).astype(np.float32)
    intersection = float((pred * target).sum())
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1e-6) -> float:
    pred = (np.asarray(pred_mask) > 0).astype(np.float32)
    target = (np.asarray(target_mask) > 0).astype(np.float32)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum() - intersection)
    return float((intersection + smooth) / (union + smooth))


class ConjunctivaDemoDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], config: PipelineConfig):
        self.image_paths = list(image_paths)
        self.config = config

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = resize_image(read_rgb(image_path), self.config.target_size)
        balanced_rgb, _ = cielab_normalization(image)
        mask, _ = segment_conjunctiva(balanced_rgb)
        normalized = image.copy()
        if torch is None:
            return image, mask, normalized, image_path.name
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        normalized_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float() / 255.0
        return image_tensor, mask_tensor, normalized_tensor, image_path.name


class HbRegressionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        config: PipelineConfig,
        segmenter: Any,
        target_mean: float,
        target_std: float,
        augment: bool = False,
    ):
        self.records = list(records)
        self.config = config
        self.segmenter = segmenter
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) > 1e-6 else 1.0
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _preprocess(self, image_path: Path) -> np.ndarray:
        prep = preprocess_hb_image(image_path, self.config, self.segmenter)
        # Returns the LAB image
        return cv2.resize(prep["normalized_lab"], self.config.regression_image_size, interpolation=cv2.INTER_AREA)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_lab = self._preprocess(record["image_path"])
        
        if self.augment:
            image_lab = augment_hb_image(image_lab)
            
        # IMPORTANT: Extract ONLY the a* and b* channels (indices 1 and 2)
        # We discard the L* channel (index 0) so lighting cannot affect the network
        ab_channels = image_lab[:, :, 1:3]
        
        # Convert to tensor and normalize (LAB channels typically range 0-255 in OpenCV uint8 representation)
        image_tensor = torch.from_numpy(ab_channels.transpose(2, 0, 1)).float() / 255.0
        
        normalized_hgb = (float(record["hgb"]) - self.target_mean) / self.target_std
        hb_tensor = torch.tensor([normalized_hgb], dtype=torch.float32)
        weight_tensor = torch.tensor([record.get("sample_weight", 1.0)], dtype=torch.float32)
        
        return image_tensor, hb_tensor, weight_tensor, record["number"], record["image_path"].name


class HbRegressionNet(nn.Module):
    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        self.use_pretrained = False
        if resnet18 is not None:
            try:
                weights = ResNet18_Weights.DEFAULT if use_pretrained and ResNet18_Weights is not None else None
                backbone = resnet18(weights=weights)
                self.use_pretrained = weights is not None
                
                # IMPORTANT: Modify the first convolutional layer to accept 2 channels (a* and b*) instead of 3
                original_conv1 = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    2, # Changed from 3 to 2
                    original_conv1.out_channels, 
                    kernel_size=original_conv1.kernel_size, 
                    stride=original_conv1.stride, 
                    padding=original_conv1.padding, 
                    bias=False
                )
                
                # If using pretrained weights, we average the weights of the original 3 channels 
                # and duplicate them across our 2 new channels so we don't destroy the pretraining
                if self.use_pretrained:
                    with torch.no_grad():
                        backbone.conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)

            except Exception:
                backbone = resnet18(weights=None)
        else:
            backbone = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1), # Changed from 3 to 2
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        if hasattr(backbone, "fc"):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 1),
            )

            if self.use_pretrained:
                for parameter in backbone.parameters():
                    parameter.requires_grad = False
                for parameter in backbone.layer4.parameters():
                    parameter.requires_grad = True
                for parameter in backbone.fc.parameters():
                    parameter.requires_grad = True
                # Ensure the new conv1 layer is trainable
                for parameter in backbone.conv1.parameters():
                    parameter.requires_grad = True

        self.backbone = backbone

    def forward(self, image):
        return self.backbone(image)


def run_demo(config: PipelineConfig) -> None:
    ensure_dir(config.output_root)
    save_config(config)

    labeled_records = load_india_hb_records(config.dataset_root, config.train_limit)
    selected_images = [record["image_path"] for record in labeled_records]
    demo_images = selected_images[: config.demo_limit]

    if torch is None:
        raise RuntimeError("PyTorch is required for Mask2Former training")

    segmentation_device = torch.device("cpu")
    regression_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_split = max(1, int(round(len(selected_images) * 0.8)))
    train_paths = selected_images[:train_split]
    eval_paths = selected_images[train_split:]

    trained_model, processor = train_mask2former(config, train_paths, segmentation_device)
    if trained_model is not None and processor is not None:
        eval_metrics = evaluate_mask2former(trained_model, processor, eval_paths, config, segmentation_device)
        save_eval_metrics(eval_metrics, config.output_root / "evaluation_metrics.json")
        print(
            f"\nHeld-out evaluation on {eval_metrics['count']} images: "
            f"Dice={eval_metrics['dice']:.4f}, IoU={eval_metrics['iou']:.4f}"
        )
        segmenter = make_mask2former_segmenter(trained_model, processor, segmentation_device)
    else:
        eval_metrics = {"dice": 0.0, "iou": 0.0, "count": 0}
        save_eval_metrics(eval_metrics, config.output_root / "evaluation_metrics.json")
        print("\nHeld-out evaluation skipped because Mask2Former is unavailable offline.")
        segmenter = lambda image_rgb: (heuristic_conjunctiva_mask(image_rgb), "heuristic")

    hb_train_records = labeled_records[:train_split]
    hb_eval_records = labeled_records[train_split:]
    hb_metrics = train_hb_regressor(config, hb_train_records, hb_eval_records, segmenter, regression_device)

    records = []
    for image_path in demo_images:
        sample_name = image_path.parent.name + "_" + image_path.stem
        raw_rgb = resize_image(read_rgb(image_path), config.target_size)
        
        balanced_rgb, balanced_lab = cielab_normalization(raw_rgb)
        
        mask, backend = segment_conjunctiva(balanced_rgb, segmenter=segmenter)
        masked_rgb = apply_mask(balanced_rgb, mask)
        cropped_rgb, cropped_mask = crop_to_mask(masked_rgb, mask)
        cropped_mask = (cropped_mask > 0).astype(np.uint8) * 255
        normalized_rgb = cropped_rgb if cropped_rgb.size else masked_rgb

        focus = focus_score(raw_rgb)
        exposure = exposure_score(raw_rgb)
        mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        accepted = focus >= config.blur_threshold and config.min_mask_ratio <= mask_ratio <= config.max_mask_ratio

        outputs_dir = config.output_root
        raw_dir = outputs_dir / "raw"
        mask_dir = outputs_dir / "masks"
        overlay_dir = outputs_dir / "overlays"
        crop_dir = outputs_dir / "crops"
        normalized_dir = outputs_dir / "normalized"

        raw_path = raw_dir / f"{sample_name}_raw.png"
        mask_path = mask_dir / f"{sample_name}_mask.png"
        overlay_path = overlay_dir / f"{sample_name}_overlay.png"
        crop_path = crop_dir / f"{sample_name}_crop.png"
        normalized_path = normalized_dir / f"{sample_name}_lab.png"

        write_rgb(balanced_rgb, raw_path)
        write_gray(mask, mask_path)
        write_rgb(build_overlay(balanced_rgb, mask), overlay_path)
        write_rgb(cropped_rgb if cropped_rgb.size else masked_rgb, crop_path)
        write_rgb(normalized_rgb, normalized_path)

        record = {
            "sample_name": sample_name,
            "source_path": str(image_path),
            "raw_path": str(raw_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "crop_path": str(crop_path),
            "normalized_path": str(normalized_path),
            "segmentation_backend": backend,
            "focus_score": round(focus, 4),
            "exposure_score": round(exposure, 4),
            "mask_ratio": round(mask_ratio, 6),
            "accepted_for_training": bool(accepted),
        }
        records.append(record)
        print(
            f"{sample_name}: backend={record['segmentation_backend']}, "
            f"focus={record['focus_score']:.2f}, exposure={record['exposure_score']:.2f}, "
            f"mask_ratio={record['mask_ratio']:.4f}, accepted={record['accepted_for_training']}"
        )

    manifest_path = config.output_root / "manifest.csv"
    save_manifest(records, manifest_path)

    contact_sheet_path = config.output_root / "contact_sheet.png"
    make_contact_sheet(records, contact_sheet_path)

    print("\nSaved outputs to:")
    print(f"  {config.output_root}")
    print(f"  manifest: {manifest_path}")
    print(f"  contact sheet: {contact_sheet_path}")
    print("\nTraining subset policy:")
    print(f"  raw Indian images reserved for segmentation training/eval: {len(selected_images)}")
    print(f"  train split: {len(train_paths)} | held-out split: {len(eval_paths)}")
    print(f"  demo images processed now: {len(demo_images)}")
    print(f"  Hb regression held-out count: {hb_metrics['count']}")
    print("  imbalance handling utilities are included for the later Hb regression stage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conjunctiva preprocessing pipeline for anemia detection")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/yatikajena/Desktop/AnemiaDetection/dataset anemia/India"),
        help="Path to the Indian eye-image subset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/yatikajena/Desktop/AnemiaDetection/cielab_pipeline_outputs_v1"),
        help="Where to store masks, crops, overlays, and manifests",
    )
    parser.add_argument("--train-limit", type=int, default=50, help="Number of raw Indian images reserved for training")
    parser.add_argument("--demo-limit", type=int, default=5, help="Number of images to process in the demo run")
    parser.add_argument(
        "--segmentation-checkpoint",
        type=str,
        default="facebook/mask2former-swin-tiny-cityscapes-semantic",
        help="Mask2Former checkpoint used to initialize the segmentation model",
    )
    parser.add_argument("--train-epochs", type=int, default=3, help="Fine-tuning epochs for Mask2Former")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Batch size for Mask2Former training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for Mask2Former training")
    parser.add_argument("--blur-threshold", type=float, default=35.0, help="Minimum Laplacian variance to accept an image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        train_limit=args.train_limit,
        demo_limit=args.demo_limit,
        segmentation_checkpoint=args.segmentation_checkpoint,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        blur_threshold=args.blur_threshold,
    )
    run_demo(config)


if __name__ == "__main__":
    main()