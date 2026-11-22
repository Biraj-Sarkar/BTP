"""FastAPI server for the Anubhav ROI haemoglobin model.

Each request follows one deterministic path: decode the uploaded eye image,
extract and quality-check the conjunctival ROI, then run the ROI-trained
checkpoint for haemoglobin and anaemia screening.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.data import anemia_threshold  # noqa: E402
from anemia.imaging import build_overlay, tight_bbox  # noqa: E402
from anemia.model import Checkpoint, to_tensor  # noqa: E402
from anemia.predict import Prediction  # noqa: E402
from anemia.preprocess import prepare  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402


class RoiAnubhavPredictor:
    """Loads the ROI checkpoint and separates extraction from inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        extractor_name: str = "refined",
        segmenter_path: Optional[Path] = None,
    ):
        self.checkpoint = Checkpoint.load(Path(checkpoint_path))
        self.device = torch.device("cpu")
        self.model = self.checkpoint.build(self.device)
        self.preprocess_config = PreprocessConfig(
            work_size=tuple(self.checkpoint.work_size),
            model_size=tuple(self.checkpoint.model_size),
            gray_world=self.checkpoint.gray_world,
        )
        self.quality_config = QualityConfig()
        self.extractor_name = extractor_name
        self.segmenter = get_extractor(extractor_name, segmenter_path)

    def extract(self, image_rgb: np.ndarray):
        return prepare(
            image_rgb, self.segmenter, self.preprocess_config, self.quality_config
        )

    def predict_prepared(
        self,
        prepared,
        age_years: Optional[float] = None,
        sex: Optional[str] = None,
        pregnant: bool = False,
    ) -> Prediction:
        threshold = anemia_threshold(age_years, sex, pregnant=pregnant)
        if not prepared.quality.passed:
            return Prediction(
                hb=None,
                anemic=None,
                threshold=threshold,
                usable=False,
                quality=prepared.quality.to_dict(),
                message=(
                    "Could not read this photo clearly: "
                    + "; ".join(prepared.quality.reasons)
                    + ". Please retake with the lower eyelid pulled down and steady lighting."
                ),
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
                "screening threshold for this patient. This is a screening estimate, "
                "not a diagnosis — confirm with a lab test."
            ),
        )


_predictor: Optional[RoiAnubhavPredictor] = None


def get_predictor() -> RoiAnubhavPredictor:
    global _predictor
    if _predictor is None:
        checkpoint = os.environ.get(
            "ANEMIA_CHECKPOINT",
            str(Path(__file__).resolve().parent / "runs" / "roi_training" / "best.pt"),
        )
        extractor = os.environ.get("ANEMIA_EXTRACTOR", "refined")
        segmenter = os.environ.get("ANEMIA_SEGMENTER")
        _predictor = RoiAnubhavPredictor(
            Path(checkpoint), extractor, Path(segmenter) if segmenter else None
        )
    return _predictor


app = FastAPI(title="Anubhav Anaemia Screening API", version="2.0")


async def _decode_upload(image: UploadFile) -> np.ndarray:
    raw = await image.read()
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _png_base64(rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode visualisation")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.get("/health")
def health() -> dict:
    predictor = get_predictor()
    return {
        "status": "ok",
        "pipeline": "extract -> ROI prediction",
        "extractor": predictor.extractor_name,
        "backbone": predictor.checkpoint.backbone,
        "metrics": predictor.checkpoint.metrics,
    }


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    extract_target: str = Form("roi"),
    include_visuals: bool = Form(False),
):
    del extract_target  # kept for compatibility with the existing app
    predictor = get_predictor()
    image_rgb = await _decode_upload(image)
    prepared = predictor.extract(image_rgb)
    left, top, right, bottom = tight_bbox(
        prepared.mask, predictor.preprocess_config.bbox_pad_ratio
    )
    payload = {
        "backend": prepared.backend,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
        "quality": prepared.quality.to_dict(),
    }
    if include_visuals:
        payload["roi_png_base64"] = _png_base64(prepared.crop)
        payload["overlay_png_base64"] = _png_base64(
            build_overlay(image_rgb, prepared.mask)
        )
    return JSONResponse(payload)


@app.post("/anemia-detection")
async def anemia_detection(
    image: UploadFile = File(...),
    age_years: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    pregnant: bool = Form(False),
):
    predictor = get_predictor()
    image_rgb = await _decode_upload(image)
    prepared = predictor.extract(image_rgb)
    return predictor.predict_prepared(prepared, age_years, sex, pregnant).to_dict()


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age_years: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    pregnant: bool = Form(False),
):
    return await anemia_detection(image, age_years, sex, pregnant)
