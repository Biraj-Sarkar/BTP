"""ROI extraction pipeline for server-side inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

SUITE_ROOT = Path(__file__).resolve().parents[2]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from extractor import ROIExtractor, load_unet, ExtractionResult


class BackendROIPipeline:
    """Server-side conjunctival ROI extraction manager."""

    def __init__(self, unet_path: Optional[Path | str] = None):
        self.extractor = ROIExtractor(checkpoint_path=unet_path)

    def process(self, image_rgb: np.ndarray) -> ExtractionResult:
        """Runs gray-world balance, U-Net inference, cleanup, and 224x224 crop."""
        result = self.extractor.extract(image_rgb)
        if not result.mask.any():
            raise ValueError("U-Net could not identify a valid conjunctival ROI in the submitted image.")
        return result

