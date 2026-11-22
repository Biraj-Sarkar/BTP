FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Backend/requirements.txt ./Backend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2+cpu torchvision==0.17.2+cpu \
    && python -m pip install -r Backend/requirements.txt

COPY Backend/server.py Backend/model_loader.py ./Backend/
COPY Backend/inference ./Backend/inference
COPY extractor/*.py ./extractor/
COPY predictor/*.py ./predictor/
COPY predictor/data ./predictor/data
COPY predictor/models ./predictor/models
COPY predictor/training ./predictor/training
COPY Models/UNet_extractor/unet.pt ./Models/UNet_extractor/unet.pt
COPY predictor/runs/resnet/best.pt ./predictor/runs/resnet/best.pt

EXPOSE 7860
WORKDIR /app/Backend
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
