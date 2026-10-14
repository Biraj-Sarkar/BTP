"""Standalone API: upload photo and demographics, receive ROI/Hb screening."""
from __future__ import annotations
import logging
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
    raw=await image.read(); decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if decoded is None: raise HTTPException(400,"Invalid image file")
    return cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
@app.get("/health")
def health():
    try:
        get_models()
        return {"status":"ok","extractor":"unet","model": "anubhav", "models_loaded":True}
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
