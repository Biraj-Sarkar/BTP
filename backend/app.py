from __future__ import annotations

import base64
import io
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.prediction.scripts.end_to_end_inference import (
    EndToEndPipeline,
    EndToEndResult,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

DISCLAIMER_TEXT = (
    "This is an academic research prototype for hemoglobin estimation from conjunctiva images. "
    "It is not a clinically validated diagnostic device and must not be used as a substitute for "
    "laboratory testing or professional medical advice."
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Configure structured server-side logging
logger = logging.getLogger("btp_anemia_backend")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Global application state: pipeline singleton, initialization status, and concurrency lock
_pipeline: EndToEndPipeline | None = None
_pipeline_initialized: bool = False
_init_error: str | None = None
_inference_lock = threading.Lock()


def init_pipeline(app: FastAPI | None = None) -> EndToEndPipeline | None:
    """
    Initializes the application-wide EndToEndPipeline singleton once during startup.
    Catches and logs any startup failures, recording the error for diagnostics.
    """
    global _pipeline, _pipeline_initialized, _init_error
    if _pipeline is not None:
        return _pipeline

    try:
        logger.info("Initializing EndToEndPipeline models (Attention U-Net & DenseNet-121)...")
        _pipeline = EndToEndPipeline()
        _pipeline_initialized = True
        _init_error = None
        if app is not None:
            app.state.pipeline = _pipeline
            app.state.models_loaded = True
            app.state.init_error = None
        logger.info("EndToEndPipeline successfully initialized and ready for inference.")
        return _pipeline
    except Exception as exc:
        _pipeline = None
        _pipeline_initialized = False
        _init_error = str(exc)
        if app is not None:
            app.state.pipeline = None
            app.state.models_loaded = False
            app.state.init_error = str(exc)
        logger.error("Failed to initialize EndToEndPipeline: %s", exc, exc_info=True)
        return None


def get_pipeline(app: FastAPI | None = None) -> EndToEndPipeline | None:
    """
    Returns the singleton EndToEndPipeline instance if successfully loaded, or None.
    Does NOT lazily load models on inference requests; models must be initialized at startup.
    """
    global _pipeline
    if app is not None and hasattr(app.state, "pipeline") and app.state.pipeline is not None:
        return app.state.pipeline
    return _pipeline


def set_pipeline_state_for_testing(
    pipeline: EndToEndPipeline | None,
    models_loaded: bool,
    init_error: str | None = None,
    app: FastAPI | None = None,
) -> None:
    """
    Test helper allowing explicit injection or clearing of pipeline state for test isolation.
    """
    global _pipeline, _pipeline_initialized, _init_error
    _pipeline = pipeline
    _pipeline_initialized = models_loaded
    _init_error = init_error
    if app is not None:
        app.state.pipeline = pipeline
        app.state.models_loaded = models_loaded
        app.state.init_error = init_error


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    FastAPI lifespan context manager: handles application startup and shutdown.
    Initializes models once during startup and logs lifecycle transitions.
    """
    logger.info("Application starting up: executing lifespan startup hook...")
    init_pipeline(fastapi_app)
    yield
    logger.info("Application shutting down: executing lifespan shutdown hook...")


def pil_to_data_url(image: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 Data URL string."""
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        # Convert RGBA to RGB for JPEG compatibility
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG")
        mime = "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


app = FastAPI(
    title="Anemia Detection BTP Inference Service",
    description="Academic prototype for conjunctival hemoglobin estimation.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the single-page user interface."""
    if not INDEX_HTML.is_file():
        logger.error("Template index.html not found at %s", INDEX_HTML)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="UI template index.html not found.",
        )
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
def health_check():
    """
    Health status and model readiness endpoint.
    Returns HTTP 200 when models are loaded and ready, or HTTP 503 if unavailable.
    """
    pipeline = get_pipeline(app)
    is_ready = pipeline is not None
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    content = {
        "status": "healthy" if is_ready else "unhealthy",
        "models_loaded": is_ready,
        "disclaimer": DISCLAIMER_TEXT,
    }
    return JSONResponse(status_code=status_code, content=content)


@app.post("/api/predict")
def predict_single_image(file: UploadFile = File(...)):
    """
    User-facing endpoint for single-image hemoglobin estimation.
    Accepts one image (JPG, JPEG, PNG), runs EndToEndPipeline, and returns structured result.
    """
    # 0. Check pipeline availability
    pipeline = get_pipeline(app)
    if pipeline is None:
        logger.error("Inference requested but EndToEndPipeline is not initialized or unavailable.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "Inference service is temporarily unavailable. Models are not loaded.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 1. Validate file presence
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "No file was uploaded.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 2. Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": (
                    f"Unsupported file format '{file_ext}'. "
                    f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
                ),
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 3. Read uploaded content
    try:
        content = file.file.read()
    except Exception as exc:
        logger.error("Failed to read upload payload: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "Failed to read upload data. Please re-select the file and try again.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    if len(content) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "The uploaded file is empty (0 bytes).",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 4. Open image via PIL (validating file integrity)
    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()
    except Exception as exc:
        logger.warning("Corrupted or unreadable image uploaded: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "Corrupted or invalid image data: cannot decode image.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 5. Concurrency protection: Reject repeated concurrent requests with HTTP 429
    if not _inference_lock.acquire(blocking=False):
        logger.info("Inference lock busy: rejecting concurrent request with HTTP 429.")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "Another inference request is currently processing. Please wait.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )

    # 6. Execute inference under concurrency lock
    try:
        e2e_result: EndToEndResult = pipeline.predict(
            pil_image,
            return_artifacts=True,
        )
    except Exception as exc:
        logger.exception("Unexpected error during pipeline execution: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "error_message": "An unexpected error occurred during image processing. Please try again later.",
                "disclaimer": DISCLAIMER_TEXT,
            },
        )
    finally:
        _inference_lock.release()

    # 7. Format structured output
    roi_data_url = None
    if e2e_result.roi_image is not None:
        try:
            roi_data_url = pil_to_data_url(e2e_result.roi_image, format="JPEG")
        except Exception as exc:
            logger.warning("Failed to encode ROI image: %s", exc)

    overlay_data_url = None
    if e2e_result.overlay_image is not None:
        try:
            overlay_data_url = pil_to_data_url(e2e_result.overlay_image, format="JPEG")
        except Exception as exc:
            logger.warning("Failed to encode overlay image: %s", exc)

    response_payload = {
        "success": e2e_result.success,
        "extraction_success": e2e_result.extraction_success,
        "predicted_hb_g_dl": e2e_result.predicted_hb_g_dl,
        "bbox": list(e2e_result.bbox) if e2e_result.bbox else None,
        "mask_area_pixels": e2e_result.mask_area_pixels,
        "error_message": e2e_result.error_message,
        "stage": e2e_result.stage,
        "roi_base64": roi_data_url,
        "overlay_base64": overlay_data_url,
        "disclaimer": DISCLAIMER_TEXT,
    }

    status_code = status.HTTP_200_OK if e2e_result.success else 422
    return JSONResponse(status_code=status_code, content=response_payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
