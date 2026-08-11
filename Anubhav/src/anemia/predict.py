"""Serving path: photo in, haemoglobin estimate and screening verdict out.

Imports nothing from `train`. The API process holds one `AnemiaPredictor` for
its lifetime and does a single forward pass per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import PreprocessConfig, QualityConfig
from .data import anemia_threshold
from .imaging import read_rgb
from .model import Checkpoint, to_tensor
from .preprocess import prepare
from .segment import heuristic_segmenter, load_segmenter


@dataclass
class Prediction:
    hb: Optional[float]
    """Estimated haemoglobin in g/dL, or None when the capture failed QC."""

    anemic: Optional[bool]
    threshold: float
    usable: bool
    quality: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "hb_g_dl": round(self.hb, 2) if self.hb is not None else None,
            "anemic": self.anemic,
            "threshold_g_dl": self.threshold,
            "usable": self.usable,
            "quality": self.quality,
            "message": self.message,
        }


class AnemiaPredictor:
    """Stateless-per-request predictor around a loaded checkpoint.

    Preprocessing settings come from the checkpoint rather than from ambient
    config, so a server running an older model still conditions images the way
    that model was trained.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        segmenter_dir: Optional[Path] = None,
        device: Optional[str] = None,
        quality: Optional[QualityConfig] = None,
    ):
        self.checkpoint = Checkpoint.load(Path(checkpoint_path))
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = self.checkpoint.build(self.device)

        self.preprocess_config = PreprocessConfig(
            work_size=tuple(self.checkpoint.work_size),
            model_size=tuple(self.checkpoint.model_size),
            gray_world=self.checkpoint.gray_world,
        )
        self.quality_config = quality or QualityConfig()

        try:
            self.segmenter = load_segmenter(segmenter_dir)
        except (FileNotFoundError, ImportError) as exc:
            if segmenter_dir is not None:
                raise
            self.segmenter = heuristic_segmenter
            self._segmenter_note = str(exc)

    def predict_array(
        self,
        image_rgb: np.ndarray,
        age_years: Optional[float] = None,
        sex: Optional[str] = None,
        pregnant: bool = False,
    ) -> Prediction:
        prepared = prepare(
            image_rgb, self.segmenter, self.preprocess_config, self.quality_config
        )
        threshold = anemia_threshold(age_years, sex, pregnant=pregnant)

        if not prepared.quality.passed:
            return Prediction(
                hb=None,
                anemic=None,
                threshold=threshold,
                usable=False,
                quality=prepared.quality.to_dict(),
                message="Could not read this photo clearly: "
                + "; ".join(prepared.quality.reasons)
                + ". Please retake with the lower eyelid pulled down and steady lighting.",
            )

        with torch.inference_mode():
            tensor = to_tensor(prepared.crop).unsqueeze(0).to(self.device)
            standardised = self.model(tensor).cpu().numpy().reshape(-1)

        hb = float(self.checkpoint.denormalise(standardised)[0])
        anemic = hb < threshold

        return Prediction(
            hb=hb,
            anemic=anemic,
            threshold=threshold,
            usable=True,
            quality=prepared.quality.to_dict(),
            message=(
                f"Estimated haemoglobin {hb:.1f} g/dL. "
                f"{'Below' if anemic else 'At or above'} the {threshold:.1f} g/dL "
                "screening threshold for this patient. "
                "This is a screening estimate, not a diagnosis — confirm with a lab test."
            ),
        )

    def predict_path(self, image_path: Path, **kwargs) -> Prediction:
        return self.predict_array(read_rgb(Path(image_path)), **kwargs)
