from __future__ import annotations

# IMPORTANT: ZeroGPU patches CUDA initialization; this must precede torch/torchvision.
import spaces

import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTRACTOR_REPO_ID = os.getenv(
    "EXTRACTOR_REPO_ID", "Biraj0016/Anemia-Extractor"
)
PREDICTOR_REPO_ID = os.getenv(
    "PREDICTOR_REPO_ID", "Biraj0016/Anemia-Predictor"
)
EXTRACTOR_FILENAME = os.getenv(
    "EXTRACTOR_FILENAME", "attention_unet_efficientnet_b0_best.pt"
)
PREDICTOR_FILENAME = os.getenv(
    "PREDICTOR_FILENAME", "densenet121_seed_20261118_best.pt"
)
EXTRACTOR_METADATA_FILENAME = os.getenv(
    "EXTRACTOR_METADATA_FILENAME", "attention_unet_efficientnet_b0.metadata.json"
)

IMAGE_SIZE = 320
PREDICTOR_SIZE = 224
PAD = 20
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

DISCLAIMER_TEXT = (
    "This is an academic research prototype for hemoglobin estimation from "
    "conjunctiva images. It is not a clinically validated diagnostic device and "
    "must not be used as a substitute for laboratory testing or professional "
    "medical advice."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anemia_space")

# ---------------------------------------------------------------------------
# Exact production Attention U-Net architecture
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int):
        super().__init__()
        hidden = max(8, min(gate_channels, skip_channels) // 2)
        self.gate = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.psi(self.gate(gate) + self.skip(skip))
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.attention = AttentionGate(input_channels, skip_channels)
        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = nn.functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        skip = self.attention(value, skip)
        return self.convolution(torch.cat([value, skip], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # weights=None avoids an unnecessary ImageNet download. The complete
        # trained encoder weights are restored from the uploaded checkpoint.
        self.encoder = smp.encoders.get_encoder(
            "timm-efficientnet-b0", in_channels=3, depth=5, weights=None
        )
        channels = self.encoder.out_channels
        self.decoder4 = DecoderBlock(channels[5], channels[4], 256)
        self.decoder3 = DecoderBlock(256, channels[3], 128)
        self.decoder2 = DecoderBlock(128, channels[2], 64)
        self.decoder1 = DecoderBlock(64, channels[1], 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        value = self.decoder4(features[5], features[4])
        value = self.decoder3(value, features[3])
        value = self.decoder2(value, features[2])
        value = self.decoder1(value, features[1])
        return self.final(value)


# ---------------------------------------------------------------------------
# Exact production DenseNet-121 architecture
# ---------------------------------------------------------------------------

def build_predictor() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def _download_model_files() -> tuple[Path, Path]:
    token = os.getenv("HF_TOKEN") or None
    extractor_path = Path(hf_hub_download(
        repo_id=EXTRACTOR_REPO_ID,
        filename=EXTRACTOR_FILENAME,
        token=token,
    ))
    predictor_path = Path(hf_hub_download(
        repo_id=PREDICTOR_REPO_ID,
        filename=PREDICTOR_FILENAME,
        token=token,
    ))
    # Metadata is informative, but inference constants and model state are
    # taken from the actual production checkpoint contract.
    try:
        hf_hub_download(
            repo_id=EXTRACTOR_REPO_ID,
            filename=EXTRACTOR_METADATA_FILENAME,
            token=token,
        )
    except Exception as exc:
        logger.warning("Optional extractor metadata file was not downloaded: %s", exc)
    return extractor_path, predictor_path


def _load_models() -> tuple[nn.Module, nn.Module, dict[str, float]]:
    extractor_path, predictor_path = _download_model_files()

    extractor_checkpoint = torch.load(
        extractor_path, map_location="cpu", weights_only=False
    )
    if not isinstance(extractor_checkpoint, dict) or "model" not in extractor_checkpoint:
        raise RuntimeError(
            "Extractor checkpoint format mismatch: expected a dictionary containing 'model'. "
            "Check that the production Attention U-Net checkpoint was uploaded."
        )

    extractor = AttentionUNet()
    extractor.load_state_dict(extractor_checkpoint["model"], strict=True)
    extractor.eval()
    # Required ZeroGPU pattern: put weights on CUDA at module startup.
    extractor.to("cuda")

    predictor_checkpoint = torch.load(
        predictor_path, map_location="cpu", weights_only=False
    )
    required = {
        "architecture", "model_state_dict", "seed",
        "target_mean_g_dl", "target_std_g_dl",
    }
    if not isinstance(predictor_checkpoint, dict) or not required.issubset(predictor_checkpoint):
        raise RuntimeError(
            "Predictor checkpoint format mismatch: expected architecture, model_state_dict, "
            "seed, target_mean_g_dl and target_std_g_dl."
        )
    if predictor_checkpoint["architecture"] != "densenet121":
        raise RuntimeError(
            f"Wrong predictor architecture: {predictor_checkpoint['architecture']!r}; "
            "expected 'densenet121'."
        )

    target_std = float(predictor_checkpoint["target_std_g_dl"])
    target_mean = float(predictor_checkpoint["target_mean_g_dl"])
    if target_std <= 0:
        raise RuntimeError("Predictor checkpoint contains an invalid target standard deviation.")

    predictor = build_predictor()
    predictor.load_state_dict(predictor_checkpoint["model_state_dict"], strict=True)
    predictor.eval()
    predictor.to("cuda")

    return extractor, predictor, {
        "target_mean_g_dl": target_mean,
        "target_std_g_dl": target_std,
    }


# Load once when the Space process starts. A failure is reflected by /api/health.
EXTRACTOR_MODEL: nn.Module | None = None
PREDICTOR_MODEL: nn.Module | None = None
PREDICTOR_METADATA: dict[str, float] = {}
MODEL_LOAD_ERROR: str | None = None
try:
    EXTRACTOR_MODEL, PREDICTOR_MODEL, PREDICTOR_METADATA = _load_models()
    logger.info("Both production models loaded successfully from their model repositories.")
except Exception as exc:
    MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    logger.exception("Model initialization failed")


# ---------------------------------------------------------------------------
# Inference and response helpers
# ---------------------------------------------------------------------------

_predict_lock = threading.Lock()
predictor_transform = transforms.Compose([
    transforms.Resize((PREDICTOR_SIZE, PREDICTOR_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def _image_data_url(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _gradio_image_from_data_url(image_data: str | None) -> Image.Image | None:
    """Convert an API-style data URI into a PIL image for gr.Image outputs.

    FastAPI/Flutter keep using the Base64 data-URI strings. Gradio Image
    components, however, require a PIL image, NumPy array, or file path.
    """
    if not image_data:
        return None

    try:
        encoded = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            return decoded.convert("RGB").copy()
    except Exception:
        logger.exception("Could not decode an inference image for the Gradio UI")
        return None


def _predict_core(image_input: Image.Image) -> dict[str, Any]:
    if EXTRACTOR_MODEL is None or PREDICTOR_MODEL is None:
        return {
            "success": False,
            "extraction_success": False,
            "predicted_hb_g_dl": None,
            "bbox": None,
            "mask_area_pixels": 0,
            "error_message": "Inference service is unavailable because one or both models failed to load.",
            "stage": "initialization",
            "roi_base64": None,
            "overlay_base64": None,
            "disclaimer": DISCLAIMER_TEXT,
        }

    try:
        oriented = ImageOps.exif_transpose(image_input).convert("RGB")
        image_np = np.asarray(oriented)
        height, width = image_np.shape[:2]

        resized = cv2.resize(
            image_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )
        normalized = (
            resized.astype(np.float32) / 255.0 - np.asarray(MEAN, dtype=np.float32)
        ) / np.asarray(STD, dtype=np.float32)
        input_tensor = torch.from_numpy(
            np.ascontiguousarray(normalized.transpose(2, 0, 1))
        ).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            logits = EXTRACTOR_MODEL(input_tensor)
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

        small_mask = (probability >= 0.5).astype(np.uint8)
        full_mask = cv2.resize(
            small_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        foreground = full_mask > 0
        mask_area = int(foreground.sum())

        if mask_area == 0:
            return {
                "success": False,
                "extraction_success": False,
                "predicted_hb_g_dl": None,
                "bbox": None,
                "mask_area_pixels": 0,
                "error_message": "No conjunctiva detected: predicted segmentation mask is empty.",
                "stage": "extraction",
                "roi_base64": None,
                "overlay_base64": None,
                "disclaimer": DISCLAIMER_TEXT,
            }

        ys, xs = np.where(foreground)
        x0 = max(0, int(xs.min()) - PAD)
        x1 = min(width, int(xs.max()) + PAD + 1)
        y0 = max(0, int(ys.min()) - PAD)
        y1 = min(height, int(ys.max()) + PAD + 1)
        bbox = [x0, y0, x1, y1]

        # Preserve the production ROI JPEG provenance: encode Q=90 before
        # DenseNet preprocessing. Do not pass the raw crop directly.
        roi_crop = Image.fromarray(image_np[y0:y1, x0:x1])
        jpeg_buffer = io.BytesIO()
        roi_crop.save(jpeg_buffer, format="JPEG", quality=90)
        jpeg_buffer.seek(0)
        with Image.open(jpeg_buffer) as jpeg_roi:
            predictor_tensor = predictor_transform(jpeg_roi.convert("RGB")).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            standardized = PREDICTOR_MODEL(predictor_tensor).reshape(-1)[0]
            predicted_hb = (
                standardized * PREDICTOR_METADATA["target_std_g_dl"]
                + PREDICTOR_METADATA["target_mean_g_dl"]
            )

        if not torch.isfinite(predicted_hb):
            return {
                "success": False,
                "extraction_success": True,
                "predicted_hb_g_dl": None,
                "bbox": bbox,
                "mask_area_pixels": mask_area,
                "error_message": "Model returned a non-finite Hb prediction.",
                "stage": "predictor",
                "roi_base64": _image_data_url(roi_crop),
                "overlay_base64": None,
                "disclaimer": DISCLAIMER_TEXT,
            }

        overlay_np = image_np.copy()
        overlay_np[foreground] = (
            0.55 * overlay_np[foreground] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay = Image.fromarray(overlay_np)

        return {
            "success": True,
            "extraction_success": True,
            "predicted_hb_g_dl": round(float(predicted_hb.item()), 3),
            "bbox": bbox,
            "mask_area_pixels": mask_area,
            "error_message": None,
            "stage": "complete",
            "roi_base64": _image_data_url(roi_crop),
            "overlay_base64": _image_data_url(overlay),
            "disclaimer": DISCLAIMER_TEXT,
        }
    except Exception as exc:
        logger.exception("Inference failed")
        return {
            "success": False,
            "extraction_success": False,
            "predicted_hb_g_dl": None,
            "bbox": None,
            "mask_area_pixels": 0,
            "error_message": f"Inference failed: {type(exc).__name__}: {exc}",
            "stage": "inference",
            "roi_base64": None,
            "overlay_base64": None,
            "disclaimer": DISCLAIMER_TEXT,
        }


@spaces.GPU(duration=30)
def predict_api_image(image_input: Image.Image) -> dict[str, Any]:
    # The decorated function is called by the FastAPI endpoint.
    with _predict_lock:
        return _predict_core(image_input)


@spaces.GPU(duration=30)
def predict_gradio_image(image_input: Image.Image):
    # This decorated function is directly registered as the Gradio event handler.
    with _predict_lock:
        result = _predict_core(image_input)
    if result["success"]:
        hb_text = f"**Estimated Hb: {result['predicted_hb_g_dl']:.2f} g/dL**"
    else:
        hb_text = f"**Unable to estimate Hb.** {result['error_message']}"
    # Convert data URIs to PIL images only for Gradio's Image components.
    # The original Base64 strings remain untouched in the FastAPI response.
    roi_for_gradio = _gradio_image_from_data_url(result.get("roi_base64"))
    overlay_for_gradio = _gradio_image_from_data_url(result.get("overlay_base64"))

    return (
        hb_text,
        roi_for_gradio,
        overlay_for_gradio,
        result,
    )


# ---------------------------------------------------------------------------
# Gradio interface and FastAPI-compatible Flutter API
# ---------------------------------------------------------------------------

api_app = FastAPI(
    title="Anemia Detection BTP Inference Service",
    description="Academic prototype for conjunctival hemoglobin estimation.",
)


@api_app.get("/api/health")
def health_check():
    ready = EXTRACTOR_MODEL is not None and PREDICTOR_MODEL is not None
    payload = {
        "status": "healthy" if ready else "unhealthy",
        "models_loaded": ready,
        "disclaimer": DISCLAIMER_TEXT,
    }
    if not ready and MODEL_LOAD_ERROR:
        payload["model_load_error"] = MODEL_LOAD_ERROR
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload,
    )


@api_app.post("/api/predict")
async def predict_single_image(file: UploadFile = File(...)):
    if EXTRACTOR_MODEL is None or PREDICTOR_MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference models are not loaded. Check Space logs and /api/health.",
        )

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Allowed: JPG, JPEG, PNG.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    try:
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            image_input = source.copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    result = predict_api_image(image_input)
    http_code = status.HTTP_200_OK if result["success"] else (
        status.HTTP_422_UNPROCESSABLE_ENTITY
        if result.get("stage") == "extraction"
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(status_code=http_code, content=result)


with gr.Blocks(title="Anemia Detection — Research Prototype") as demo:
    gr.Markdown(
        "# Conjunctival Hemoglobin Estimation\n"
        "Upload an eye image to run the two-stage research pipeline. "
        "The uploaded image is processed for this request and is not intentionally stored by this app."
    )
    with gr.Row():
        input_image = gr.Image(type="pil", label="Eye image")
        with gr.Column():
            run_button = gr.Button("Estimate hemoglobin", variant="primary")
            result_text = gr.Markdown()
    with gr.Row():
        roi_image = gr.Image(label="Extracted conjunctiva ROI")
        overlay_image = gr.Image(label="Segmentation overlay")
    result_json = gr.JSON(label="API-style result")
    run_button.click(
        fn=predict_gradio_image,
        inputs=input_image,
        outputs=[result_text, roi_image, overlay_image, result_json],
    )
    gr.Markdown(f"**Disclaimer:** {DISCLAIMER_TEXT}")

# Mount Gradio at "/" while retaining the FastAPI routes above it.
# The `spaces` package patches mount_gradio_app to add ZeroGPUMiddleware, which
# runs ZeroGPU startup automatically on the ASGI lifespan event. Do NOT call
# spaces.zero.startup() manually.
# ssr_mode=False is required: Hugging Face sets GRADIO_SSR_MODE=true, which makes
# Gradio start a Node.js SSR proxy on port 7860 and breaks our own uvicorn bind.
app = gr.mount_gradio_app(api_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Anemia Detection API on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")