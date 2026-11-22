from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "datasets" / "raw" / "hospital" / "images"
CAPTURED_DATA = ROOT / "datasets" / "raw" / "hospital" / "captured_data.csv"
MANIFEST = ROOT / "experiments" / "annotation" / "masks" / "reports" / "annotation_manifest.csv"
MASK_ROOT = ROOT / "experiments" / "annotation"
REPORT_DIR = ROOT / "experiments" / "extraction" / "reports"
MANIFEST_DIR = ROOT / "experiments" / "extraction" / "manifests"
CONFIG_DIR = ROOT / "experiments" / "extraction" / "configs"
LOG_DIR = ROOT / "experiments" / "extraction" / "logs"
MODEL_DIR = ROOT / "models" / "extraction"

IMAGE_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEED = 20261116


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def capture_registration_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with CAPTURED_DATA.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        images = json.loads(row["Images"])
        if len(images) != 1:
            raise RuntimeError(f"Expected one captured image for {row['RegistrationNo']}")
        exif = images[0].get("exif", {})
        timestamp = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if not timestamp or timestamp in mapping:
            raise RuntimeError(f"Missing or duplicate capture timestamp for {row['RegistrationNo']}")
        mapping[timestamp] = row["RegistrationNo"]
    return mapping


def image_capture_timestamp(path: Path) -> str:
    with Image.open(path) as image:
        exif = image.getexif()
        timestamp = exif.get(36867) or exif.get(306)
    if not timestamp:
        raise RuntimeError(f"Missing EXIF capture timestamp: {path.name}")
    return str(timestamp)


def oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unreadable mask: {path}")
    values = np.unique(mask)
    if not np.all(np.isin(values, [0, 255])):
        raise RuntimeError(f"Non-binary mask: {path}")
    return mask


def validate_dataset() -> list[dict[str, str]]:
    rows = read_manifest()
    if len(rows) != 288:
        raise RuntimeError(f"Expected 288 manifest rows, found {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate image IDs in active manifest")
    statuses = {row["status"] for row in rows}
    if statuses != {"verified"}:
        raise RuntimeError(f"Unexpected manifest statuses: {sorted(statuses)}")
    registration_by_timestamp = capture_registration_map()
    registrations: set[str] = set()
    source_names = {path.name for path in IMAGE_DIR.iterdir() if path.is_file()}
    records: list[dict[str, str]] = []
    for row in rows:
        image_path = IMAGE_DIR / row["image_file"]
        mask_path = MASK_ROOT / Path(row["mask_file"])
        if image_path.name not in source_names or not image_path.is_file():
            raise RuntimeError(f"Missing source image: {row['image_file']}")
        if not mask_path.is_file():
            raise RuntimeError(f"Missing mask: {row['mask_file']}")
        image = oriented_rgb(image_path)
        mask = load_mask(mask_path)
        if image.size != (mask.shape[1], mask.shape[0]):
            raise RuntimeError(
                f"Oriented dimension mismatch for {row['image_id']}: "
                f"{image.size} versus {(mask.shape[1], mask.shape[0])}"
            )
        if not np.any(mask):
            raise RuntimeError(f"Empty mask: {row['image_id']}")
        timestamp = image_capture_timestamp(image_path)
        registration = registration_by_timestamp.get(timestamp)
        if registration is None:
            raise RuntimeError(f"No captured_data match for {row['image_file']}")
        if registration in registrations:
            raise RuntimeError(f"Duplicate RegistrationNo assignment: {registration}")
        registrations.add(registration)
        records.append(
            {
                "image_id": row["image_id"],
                "image_file": row["image_file"],
                "mask_file": row["mask_file"],
                "registration_no": registration,
                "capture_timestamp": timestamp,
            }
        )
    if len(registrations) != 288:
        raise RuntimeError(f"Expected 288 unique registrations, found {len(registrations)}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dataset_validation.json").write_text(
        json.dumps(
            {
                "source_images": 288,
                "manifest_rows": 288,
                "verified_rows": 288,
                "draft_rows": 0,
                "pending_rows": 0,
                "readable_images": 288,
                "readable_masks": 288,
                "binary_masks": 288,
                "non_empty_masks": 288,
                "missing_masks": 0,
                "orphan_masks": 0,
                "duplicate_image_ids": 0,
                "duplicate_mask_stems": 0,
                "oriented_dimension_mismatches": 0,
                "unique_registration_numbers": len(registrations),
                "mapping_method": "exact embedded EXIF capture timestamp to captured_data.csv",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def make_splits(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.70)
    validation_end = train_end + round(len(shuffled) * 0.15)
    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    registrations = [set(item["registration_no"] for item in values) for values in splits.values()]
    if set.intersection(*registrations):
        raise RuntimeError("Patient leakage detected between splits")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        pd.DataFrame(values).to_csv(MANIFEST_DIR / f"{name}_manifest.csv", index=False)
    pd.DataFrame(
        [{"split": name, "registration_no": item["registration_no"], "image_id": item["image_id"]}
         for name, values in splits.items() for item in values]
    ).to_csv(MANIFEST_DIR / "patient_grouping.csv", index=False)
    (REPORT_DIR / "split_summary.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "patient_grouping_key": "RegistrationNo",
                "counts": {name: len(values) for name, values in splits.items()},
                "overlap_count": len(set.intersection(*registrations)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return splits


class SegmentationDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(self, records: list[dict[str, str]], augment: bool = False):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        record = self.records[index]
        image = np.asarray(oriented_rgb(IMAGE_DIR / record["image_file"]), dtype=np.float32) / 255.0
        mask = load_mask(MASK_ROOT / Path(record["mask_file"]))
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image, mask = self._augment(image, mask)
        image = (image - MEAN) / STD
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)[None, ...])
        return image_tensor, mask_tensor, record["image_id"]

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        angle = random.uniform(-8.0, 8.0)
        matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            image = np.clip(image * factor, 0.0, 1.0)
        return image, mask


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int):
        super().__init__()
        hidden = max(8, min(gate_channels, skip_channels) // 2)
        self.gate = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        attention = self.psi(self.gate(gate) + self.skip(skip))
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.attention = AttentionGate(input_channels, skip_channels)
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(value, skip)
        return self.convolution(torch.cat([value, skip], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            "timm-efficientnet-b0", in_channels=3, depth=5, weights="imagenet"
        )
        channels = self.encoder.out_channels
        self.decoder4 = DecoderBlock(channels[5], channels[4], 256)
        self.decoder3 = DecoderBlock(256, channels[3], 128)
        self.decoder2 = DecoderBlock(128, channels[2], 64)
        self.decoder1 = DecoderBlock(64, channels[1], 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        features = self.encoder(image)
        value = self.decoder4(features[5], features[4])
        value = self.decoder3(value, features[3])
        value = self.decoder2(value, features[2])
        value = self.decoder1(value, features[1])
        return self.final(value)


def dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()


def batch_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    tp = (prediction & truth).sum().item()
    tn = (~prediction & ~truth).sum().item()
    fp = (prediction & ~truth).sum().item()
    fn = (~prediction & truth).sum().item()
    return {
        "dice": (2 * tp) / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
    }


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "pixel_accuracy": 0.0}
    batches = 0
    for images, masks, _ in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        metrics = batch_metrics(logits.detach(), masks)
        totals["loss"] += float(loss.detach())
        for key, value in metrics.items():
            totals[key] += value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def smoke_test() -> dict[str, Any]:
    seed_everything(SEED)
    records = validate_dataset()
    splits = make_splits(records)
    dataset = SegmentationDataset(splits["train"], augment=False)
    images, masks, _ = dataset[0]
    model = AttentionUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits = model(images.unsqueeze(0))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, masks.unsqueeze(0)) + dice_loss(logits, masks.unsqueeze(0))
    loss.backward()
    optimizer.step()
    path = MODEL_DIR / "smoke_test.pt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    reloaded = AttentionUNet()
    reloaded.load_state_dict(torch.load(path, map_location="cpu")["model"])
    path.unlink()
    result = {"dataset_sample_shape": list(images.shape), "mask_shape": list(masks.shape), "output_shape": list(logits.shape), "loss": float(loss.detach()), "checkpoint_reload": True}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cpu_smoke_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train(epochs: int = 30, batch_size: int = 2, patience: int = 7) -> dict[str, Any]:
    seed_everything(SEED)
    records = validate_dataset()
    splits = make_splits(records)
    model = AttentionUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loaders = {
        "train": DataLoader(SegmentationDataset(splits["train"], True), batch_size=batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(SegmentationDataset(splits["validation"], False), batch_size=batch_size, shuffle=False, num_workers=0),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / "attention_unet_efficientnet_b0_best.pt"
    final_path = MODEL_DIR / "attention_unet_efficientnet_b0_final.pt"
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, loaders["train"], optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, loaders["validation"], None)
        row = {"epoch": epoch, "seconds": time.time() - started, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in validation_metrics.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": {"seed": SEED, "image_size": IMAGE_SIZE}}, best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"], "config": {"seed": SEED, "image_size": IMAGE_SIZE}}, final_path)
    pd.DataFrame(history).to_csv(LOG_DIR / "training_history.csv", index=False)
    import matplotlib.pyplot as plt
    history_frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    axes[0].plot(history_frame["epoch"], history_frame["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_frame["epoch"], history_frame["train_dice"], label="train")
    axes[1].plot(history_frame["epoch"], history_frame["validation_dice"], label="validation")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)
    summary = {"epochs_completed": len(history), "best_epoch": max(history, key=lambda row: row["validation_dice"])["epoch"], "best_validation_dice": best_dice, "best_checkpoint": str(best_path), "final_checkpoint": str(final_path), "seed": SEED, "batch_size": batch_size, "train_count": len(splits["train"]), "validation_count": len(splits["validation"]), "test_count": len(splits["test"])}
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (MODEL_DIR / "attention_unet_efficientnet_b0.metadata.json").write_text(
        json.dumps(
            {
                "architecture": "Attention U-Net",
                "encoder": "EfficientNet-B0",
                "encoder_weights": "ImageNet",
                "checkpoint": str(best_path),
                "input_size": [IMAGE_SIZE, IMAGE_SIZE],
                "orientation": "Pillow ImageOps.exif_transpose",
                "normalization_mean": MEAN.tolist(),
                "normalization_std": STD.tolist(),
                "seed": SEED,
                "best_epoch": summary["best_epoch"],
                "best_validation_dice": best_dice,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_and_infer(checkpoint: Path) -> dict[str, Any]:
    seed_everything(SEED)
    records = validate_dataset()
    splits = make_splits(records)
    model = AttentionUNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.eval()
    test_loader = DataLoader(SegmentationDataset(splits["test"], False), batch_size=1, shuffle=False, num_workers=0)
    per_image: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, masks, ids in test_loader:
            logits = model(images)
            metrics = batch_metrics(logits, masks)
            per_image.append({"image_id": ids[0], **metrics})
    pd.DataFrame(per_image).to_csv(REPORT_DIR / "test_metrics_per_image.csv", index=False)
    metrics = pd.DataFrame(per_image)
    test_summary = {"count": len(per_image), **{key: {"mean": float(metrics[key].mean()), "std": float(metrics[key].std(ddof=0))} for key in metrics.columns if key != "image_id"}}
    (REPORT_DIR / "test_evaluation.json").write_text(json.dumps(test_summary, indent=2) + "\n", encoding="utf-8")
    outputs = {
        "masks": ROOT / "experiments" / "extraction" / "predicted_masks",
        "overlays": ROOT / "experiments" / "extraction" / "overlays",
        "roi": ROOT / "experiments" / "extraction" / "roi_crops",
    }
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    empty = 0
    for record in records:
        try:
            image = np.asarray(oriented_rgb(IMAGE_DIR / record["image_file"]))
            resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(((resized.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()).unsqueeze(0)
            with torch.no_grad():
                probability = torch.sigmoid(model(tensor))[0, 0].numpy()
            predicted = cv2.resize((probability >= 0.5).astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
            cv2.imwrite(str(outputs["masks"] / f"{record['image_id']}.png"), predicted)
            if not np.any(predicted):
                empty += 1
                continue
            overlay = image.copy()
            foreground = predicted > 0
            overlay[foreground] = (0.55 * overlay[foreground] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(overlay).save(outputs["overlays"] / f"{record['image_id']}.jpg", quality=90)
            ys, xs = np.where(foreground)
            pad = 20
            x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad + 1)
            y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad + 1)
            Image.fromarray(image[y0:y1, x0:x1]).save(outputs["roi"] / f"{record['image_id']}.jpg", quality=90)
        except Exception as exc:
            failures.append({"image_id": record["image_id"], "error": str(exc)})
    inference_summary = {"requested": len(records), "successful_masks": len(records) - len(failures), "failed": failures, "empty_predicted_masks": empty, "generated_masks": len(records) - len(failures), "generated_overlays": len(records) - len(failures) - empty, "generated_roi_crops": len(records) - len(failures) - empty, "checkpoint": str(checkpoint)}
    (REPORT_DIR / "full_dataset_inference_summary.json").write_text(json.dumps(inference_summary, indent=2) + "\n", encoding="utf-8")
    return {"test": test_summary, "inference": inference_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "smoke", "train", "evaluate"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "attention_unet_efficientnet_b0_best.pt")
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps({"records": len(validate_dataset())}, indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke_test(), indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.epochs, args.batch_size), indent=2))
    else:
        print(json.dumps(evaluate_and_infer(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
