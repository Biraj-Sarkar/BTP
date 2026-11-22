"""Standalone API: upload photo and demographics, receive ROI/Hb screening."""
from __future__ import annotations
import logging
import base64
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from inference import BackendModels
app=FastAPI(title="BTP Anaemia Backend",version="1.0"); models=None; model_error=None
logger = logging.getLogger("btp_backend")
def get_models():
    global models, model_error
    if models is None:
        try:
            models=BackendModels.load(); model_error=None
        except Exception as exc:
            model_error=str(exc)
            raise
    return models
async def decode(image):
    raw=await image.read()
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image is empty or exceeds the 10 MB limit")
    decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if decoded is None: raise HTTPException(400,"Invalid image file")
    return cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)


def encode_image(image: np.ndarray, extension: str = ".jpg") -> str:
    ok, encoded = cv2.imencode(extension, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode extracted image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
@app.get("/health")
def health():
    try:
        loaded = get_models()
        return {"status":"ok","extractor":"unet","model": f"{loaded.predictor.arch_name}_{loaded.predictor.variant_name}".strip("_"), "models_loaded":True}
    except Exception:
        return {"status":"error","extractor":"unet","models_loaded":False,"error":model_error}
@app.get("/")
def root(): return {"status":"ok","service":"btp-anemia-backend","docs":"/docs"}
@app.post("/predict")
@app.post("/anemia-detection")
async def predict(image: UploadFile=File(...), age_years: float=Form(...), sex: str=Form(...)):
    try: return get_models().predict(await decode(image),age_years,sex)
    except (ValueError,RuntimeError) as exc: raise HTTPException(422,str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(503, "Model service is unavailable; check the server logs") from exc


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    extract_target: str = Form("conjunctiva"),
    include_visuals: bool = Form(False),
):
    if extract_target != "conjunctiva":
        raise HTTPException(422, "Only conjunctiva extraction is supported")
    try:
        raw_image = await image.read()
        if not raw_image or len(raw_image) > 10 * 1024 * 1024:
            raise HTTPException(400, "Image is empty or exceeds the 10 MB limit")
        decoded = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(400, "Invalid image file")
        original = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        loaded = get_models()
        result = loaded.extract_roi(original)
        response = {
            "extract_target": extract_target,
            "backend": result.backend,
            "usable": result.passed,
            "bbox": list(result.bbox),
            "quality": result.quality.to_dict(),
            "message": "" if result.passed else "; ".join(result.quality.reasons),
        }
        if include_visuals:
            working = cv2.resize(original, (result.mask.shape[1], result.mask.shape[0]))
            masked = np.where(result.mask[:, :, None] > 0, working, 0).astype(np.uint8)
            response["visuals"] = {
                "original": encode_image(original),
                "mask": encode_image(cv2.cvtColor(result.mask, cv2.COLOR_GRAY2RGB), ".png"),
                "masked": encode_image(masked),
                "roi": encode_image(result.crop),
            }
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("ROI extraction request failed")
        raise HTTPException(503, "Model service is unavailable; check the server logs") from exc
