"""Inference orchestration package for FastAPI server."""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

try:
    from model_loader import resolve_unet_path, resolve_predictor_path
    from inference.roi_pipeline import BackendROIPipeline
    from inference.hb_predictor import BackendHbPredictor
except ImportError:
    from ..model_loader import resolve_unet_path, resolve_predictor_path
    from .roi_pipeline import BackendROIPipeline
    from .hb_predictor import BackendHbPredictor


class BackendModels:
    """Orchestrates ROI extraction and Hemoglobin prediction for the FastAPI server."""

    def __init__(self, extractor: BackendROIPipeline, predictor: BackendHbPredictor):
        self.extractor = extractor
        self.predictor = predictor

    @classmethod
    def load(cls) -> "BackendModels":
        unet_path = resolve_unet_path()
        predictor_path = resolve_predictor_path()

        extractor = BackendROIPipeline(unet_path=unet_path)
        predictor = BackendHbPredictor(checkpoint_path=predictor_path)
        return cls(extractor=extractor, predictor=predictor)

    def predict(self, image_rgb: np.ndarray, age_years: float, sex: str) -> Dict[str, Any]:
        """Full inference pipeline: image -> U-Net ROI extraction -> Hb prediction -> WHO screening."""
        roi_result = self.extractor.process(image_rgb)
        pred_result = self.predictor.predict(roi_result.crop, age_years=age_years, sex=sex)

        return {
            "roi_backend": roi_result.backend,
            "bbox": list(roi_result.bbox),
            "usable": True,
            "hb_g_dl": pred_result["hb_g_dl"],
            "model": pred_result["model"],
            "anemic": pred_result["anemic"],
            "threshold_g_dl": pred_result["threshold_g_dl"],
            "quality": roi_result.quality.to_dict(),
        }

    def extract_roi(self, image_rgb: np.ndarray) -> ExtractionResult:
        """Run only the approved production ROI extractor."""
        return self.extractor.process(image_rgb)


__all__ = ["BackendModels", "BackendROIPipeline", "BackendHbPredictor"]
