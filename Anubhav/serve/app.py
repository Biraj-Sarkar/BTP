"""FastAPI service backing the mobile app.

The phone uploads a photo plus the patient's age and sex; the server returns an
Hb estimate and a screening verdict. One model is loaded at startup and reused,
so a request costs one segmentation pass plus one ResNet forward pass.

    ANEMIA_CHECKPOINT=runs/hb/best.pt \
    ANEMIA_SEGMENTER=runs/segmenter \
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anemia.predict import AnemiaPredictor  # noqa: E402

MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(
    title="Anemia screening",
    description="Non-invasive haemoglobin estimation from a conjunctiva photograph.",
    version="0.2.0",
)

_predictor: Optional[AnemiaPredictor] = None


def get_predictor() -> AnemiaPredictor:
    global _predictor
    if _predictor is None:
        checkpoint = os.environ.get("ANEMIA_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("Set ANEMIA_CHECKPOINT to a trained .pt bundle")
        segmenter = os.environ.get("ANEMIA_SEGMENTER")
        _predictor = AnemiaPredictor(
            Path(checkpoint),
            segmenter_dir=Path(segmenter) if segmenter else None,
        )
    return _predictor


@app.on_event("startup")
def warm_up() -> None:
    """Load the model at boot so the first real request is not the slow one."""

    try:
        get_predictor()
    except Exception as exc:  # noqa: BLE001 - surfaced via /health
        print(f"Model not loaded at startup: {exc}")


@app.get("/health")
def health() -> dict:
    try:
        predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "unhealthy", "detail": str(exc)}, status_code=503)
    return {
        "status": "ok",
        "backbone": predictor.checkpoint.backbone,
        "metrics": predictor.checkpoint.metrics,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age_years: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    pregnant: bool = Form(False),
) -> dict:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = get_predictor().predict_array(
        cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
        age_years=age_years,
        sex=sex,
        pregnant=pregnant,
    )

    # A failed-QC capture is a 200 with usable=false, not an error: the app
    # should show a retake prompt, which is a normal outcome rather than a bug.
    return result.to_dict()
