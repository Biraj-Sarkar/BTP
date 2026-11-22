from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.extraction.src.extraction_inference import (
    DEFAULT_CHECKPOINT as DEFAULT_EXTRACTION_CHECKPOINT,
    ExtractionPredictor,
    ExtractionResult,
)
from experiments.prediction.scripts.inference import (
    DEFAULT_CHECKPOINT as DEFAULT_DENSENET_CHECKPOINT,
    inverse_target,
    load_model as load_densenet_model,
    preprocess_image,
)


@dataclass
class EndToEndResult:
    """Encapsulates the complete result of the end-to-end Hb estimation pipeline."""
    success: bool
    predicted_hb_g_dl: float | None
    bbox: tuple[int, int, int, int] | None
    mask_area_pixels: int
    extraction_success: bool
    stage: str | None = None
    error_message: str | None = None
    roi_image: Image.Image | None = None
    mask: np.ndarray | None = None
    overlay_image: Image.Image | None = None
    metadata: dict[str, Any] | None = None


class EndToEndPipeline:
    """
    Two-stage end-to-end inference pipeline:
    Stage 1: Eye Image -> Attention U-Net -> Conjunctiva Mask & ROI Crop.
    Stage 2: ROI Crop -> JPEG quality=90 encoding -> DenseNet-121 -> Hemoglobin estimate in g/dL.
    """

    def __init__(
        self,
        extraction_checkpoint: Path = DEFAULT_EXTRACTION_CHECKPOINT,
        predictor_checkpoint: Path = DEFAULT_DENSENET_CHECKPOINT,
    ):
        self.extraction_checkpoint = Path(extraction_checkpoint)
        self.predictor_checkpoint = Path(predictor_checkpoint)

        # Stage 1: Load Attention U-Net
        self.extraction_predictor = ExtractionPredictor(checkpoint_path=self.extraction_checkpoint)

        # Stage 2: Load DenseNet-121
        self.densenet_model, self.densenet_metadata = load_densenet_model(self.predictor_checkpoint)

    def predict(
        self,
        image_input: Union[Path, str, Image.Image],
        return_artifacts: bool = False,
    ) -> EndToEndResult:
        """
        Run full-image end-to-end inference:
        1. Extract conjunctival ROI via Attention U-Net.
        2. Handle extraction failures or empty masks gracefully.
        3. Match DenseNet training provenance: encode ROI with JPEG quality=90.
        4. Preprocess ROI using inference.py:preprocess_image().
        5. Forward pass through DenseNet-121 and invert standardization.
        6. Return structured EndToEndResult.
        """
        # --- Stage 1: Extraction ---
        try:
            extraction_result: ExtractionResult = self.extraction_predictor.extract(
                image_input=image_input,
                generate_overlay=return_artifacts,
            )
        except Exception as exc:
            return EndToEndResult(
                success=False,
                predicted_hb_g_dl=None,
                bbox=None,
                mask_area_pixels=0,
                extraction_success=False,
                stage="extraction",
                error_message=f"Extraction stage encountered an unexpected exception: {exc}",
            )

        if not extraction_result.success or extraction_result.roi_image is None:
            return EndToEndResult(
                success=False,
                predicted_hb_g_dl=None,
                bbox=extraction_result.bbox,
                mask_area_pixels=extraction_result.mask_area_pixels,
                extraction_success=False,
                stage="extraction",
                error_message=extraction_result.error_message or "Extraction failed to produce an ROI.",
                mask=extraction_result.mask if return_artifacts else None,
                overlay_image=extraction_result.overlay_image if return_artifacts else None,
            )

        # --- Stage 2: Hb Regression ---
        try:
            # Reproduce the exact training-time ROI encoding behavior from extraction_pipeline.py:
            # Image.fromarray(image[y0:y1, x0:x1]).save(outputs["roi"] / f"{record['image_id']}.jpg", quality=90)
            # DenseNet was trained on these lossy JPEG Q=90 crops. Reusing inference.py:preprocess_image()
            # on this encoded representation guarantees exact parity with standalone predictions.
            temp_fd, temp_name = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_fd)
            temp_roi_path = Path(temp_name)
            try:
                extraction_result.roi_image.save(temp_roi_path, format="JPEG", quality=90)
                roi_tensor = preprocess_image(temp_roi_path)
            finally:
                if temp_roi_path.is_file():
                    temp_roi_path.unlink()

            with torch.inference_mode():
                standardized = self.densenet_model(roi_tensor.unsqueeze(0)).reshape(-1)[0]
                prediction = inverse_target(standardized, self.densenet_metadata)

            if not torch.isfinite(prediction):
                return EndToEndResult(
                    success=False,
                    predicted_hb_g_dl=None,
                    bbox=extraction_result.bbox,
                    mask_area_pixels=extraction_result.mask_area_pixels,
                    extraction_success=True,
                    stage="predictor",
                    error_message="Model returned a non-finite Hb prediction.",
                    roi_image=extraction_result.roi_image if return_artifacts else None,
                    mask=extraction_result.mask if return_artifacts else None,
                    overlay_image=extraction_result.overlay_image if return_artifacts else None,
                )

            predicted_hb = float(prediction.item())

            metadata = {
                "extraction_checkpoint": str(self.extraction_checkpoint),
                "predictor_checkpoint": str(self.predictor_checkpoint),
                "target_mean_g_dl": self.densenet_metadata["target_mean_g_dl"],
                "target_std_g_dl": self.densenet_metadata["target_std_g_dl"],
                "standardized_prediction": float(standardized.item()),
                "roi_encoding": "JPEG quality=90 (matching extraction_pipeline.py)",
            }

            return EndToEndResult(
                success=True,
                predicted_hb_g_dl=predicted_hb,
                bbox=extraction_result.bbox,
                mask_area_pixels=extraction_result.mask_area_pixels,
                extraction_success=True,
                stage=None,
                error_message=None,
                roi_image=extraction_result.roi_image if return_artifacts else None,
                mask=extraction_result.mask if return_artifacts else None,
                overlay_image=extraction_result.overlay_image if return_artifacts else None,
                metadata=metadata,
            )

        except Exception as exc:
            return EndToEndResult(
                success=False,
                predicted_hb_g_dl=None,
                bbox=extraction_result.bbox,
                mask_area_pixels=extraction_result.mask_area_pixels,
                extraction_success=True,
                stage="predictor",
                error_message=f"Predictor stage encountered an exception: {exc}",
                roi_image=extraction_result.roi_image if return_artifacts else None,
                mask=extraction_result.mask if return_artifacts else None,
                overlay_image=extraction_result.overlay_image if return_artifacts else None,
            )


def predict_end_to_end(
    image_input: Union[Path, str, Image.Image],
    extraction_checkpoint: Path = DEFAULT_EXTRACTION_CHECKPOINT,
    predictor_checkpoint: Path = DEFAULT_DENSENET_CHECKPOINT,
    return_artifacts: bool = False,
) -> EndToEndResult:
    """Convenience function for running one-off end-to-end inference."""
    pipeline = EndToEndPipeline(
        extraction_checkpoint=extraction_checkpoint,
        predictor_checkpoint=predictor_checkpoint,
    )
    return pipeline.predict(image_input=image_input, return_artifacts=return_artifacts)
