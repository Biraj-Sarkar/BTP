"""Standalone API: upload photo and demographics, receive ROI/Hb screening."""
from __future__ import annotations
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from inference import BackendModels
app=FastAPI(title="BTP Anaemia Backend",version="1.0"); models=None
def get_models():
    global models
    if models is None: models=BackendModels.load()
    return models
async def decode(image):
    raw=await image.read(); decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if decoded is None: raise HTTPException(400,"Invalid image file")
    return cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
@app.get("/health")
def health(): return {"status":"ok","extractor":"unet","models_loaded":models is not None}
@app.post("/predict")
@app.post("/anemia-detection")
async def predict(image: UploadFile=File(...), age_years: float=Form(...), sex: str=Form(...)):
    try: return get_models().predict(await decode(image),age_years,sex)
    except (ValueError,RuntimeError) as exc: raise HTTPException(422,str(exc)) from exc
