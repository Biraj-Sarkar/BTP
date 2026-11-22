from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

# Ensure repository root is on sys.path for robust absolute imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.extraction.src.extraction_pipeline import AttentionUNet

DEFAULT_CHECKPOINT = ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0_best.pt"
DEFAULT_METADATA = ROOT / "models" / "extraction" / "attention_unet_efficientnet_b0.metadata.json"

IMAGE_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD = 20


@dataclass
class ExtractionResult:
    """Encapsulates the complete output of Stage 1 conjunctiva extraction."""
    success: bool
    original_image: Image.Image | None
    roi_image: Image.Image | None
    mask: np.ndarray | None
    overlay_image: Image.Image | None
    bbox: tuple[int, int, int, int] | None
    mask_area_pixels: int
    error_message: str | None = None


def load_oriented_rgb(image_input: Union[Path, str, Image.Image]) -> Image.Image:
    """Load image, apply EXIF orientation correction, and convert to RGB."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    elif isinstance(image_input, Image.Image):
        return ImageOps.exif_transpose(image_input).convert("RGB")
    else:
        raise TypeError(f"Expected file path or PIL.Image.Image, got {type(image_input).__name__}")


def load_extraction_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> tuple[torch.nn.Module, dict]:
    """Load Attention U-Net model in eval mode on CPU with checkpoint weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Extraction checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint format in {checkpoint_path}; expected dict with 'model' key")

    model = AttentionUNet()
    model.load_state_dict(checkpoint["model"])
    model.to("cpu")
    model.eval()

    metadata = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config", {}),
    }
    if DEFAULT_METADATA.is_file():
        try:
            metadata["metadata_file"] = json.loads(DEFAULT_METADATA.read_text(encoding="utf-8"))
        except Exception:
            pass

    return model, metadata


class ExtractionPredictor:
    """Inference-only conjunctiva segmentation and ROI extraction engine."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.metadata = load_extraction_model(self.checkpoint_path)

    def extract(
        self,
        image_input: Union[Path, str, Image.Image],
        generate_overlay: bool = True,
    ) -> ExtractionResult:
        """
        Run inference on a single full image:
        1. EXIF orientation correction and RGB conversion.
        2. Bilinear resize to 320x320 and ImageNet normalization.
        3. Attention U-Net forward pass with sigmoid activation.
        4. Threshold at 0.5 and nearest-neighbor resize to original dimensions.
        5. Bounding box computation with 20px padding (clipped to image bounds).
        6. ROI crop generation and optional red overlay rendering.
        """
        try:
            oriented_image = load_oriented_rgb(image_input)
        except Exception as exc:
            return ExtractionResult(
                success=False,
                original_image=None,
                roi_image=None,
                mask=None,
                overlay_image=None,
                bbox=None,
                mask_area_pixels=0,
                error_message=f"Failed to load image: {exc}",
            )

        image_np = np.asarray(oriented_image)
        h, w = image_np.shape[:2]

        resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(
            ((resized.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        binary_mask_small = (probability >= 0.5).astype(np.uint8)
        predicted_mask = cv2.resize(
            binary_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
        ) * 255

        foreground = predicted_mask > 0
        mask_area = int(foreground.sum())

        if mask_area == 0:
            return ExtractionResult(
                success=False,
                original_image=oriented_image,
                roi_image=None,
                mask=predicted_mask,
                overlay_image=None,
                bbox=None,
                mask_area_pixels=0,
                error_message="No conjunctiva detected: predicted segmentation mask is empty.",
            )

        ys, xs = np.where(foreground)
        x0, x1 = max(0, int(xs.min()) - PAD), min(w, int(xs.max()) + PAD + 1)
        y0, y1 = max(0, int(ys.min()) - PAD), min(h, int(ys.max()) + PAD + 1)
        bbox = (x0, y0, x1, y1)

        roi_crop_np = image_np[y0:y1, x0:x1]
        roi_image = Image.fromarray(roi_crop_np)

        overlay_image = None
        if generate_overlay:
            overlay_np = image_np.copy()
            overlay_np[foreground] = (
                0.55 * overlay_np[foreground] + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)
            overlay_image = Image.fromarray(overlay_np)

        return ExtractionResult(
            success=True,
            original_image=oriented_image,
            roi_image=roi_image,
            mask=predicted_mask,
            overlay_image=overlay_image,
            bbox=bbox,
            mask_area_pixels=mask_area,
            error_message=None,
        )


def extract_roi(
    image_input: Union[Path, str, Image.Image],
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    generate_overlay: bool = True,
) -> ExtractionResult:
    """Convenience functional wrapper for one-off extraction."""
    predictor = ExtractionPredictor(checkpoint_path=checkpoint_path)
    return predictor.extract(image_input=image_input, generate_overlay=generate_overlay)
