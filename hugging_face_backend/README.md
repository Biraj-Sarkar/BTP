---
title: Anemia Detection API
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# Anemia Detection — ZeroGPU Space

Academic research prototype for estimating hemoglobin from conjunctiva images.

## Hardware

In Space Settings, select **ZeroGPU**. This app uses Gradio as required by ZeroGPU and mounts FastAPI endpoints for the Flutter client.

## Model repositories

- Extractor: `Biraj0016/Anemia-Extractor`
- Predictor: `Biraj0016/Anemia-Predictor`

The app downloads these files at startup:

- `attention_unet_efficientnet_b0_best.pt`
- `densenet121_seed_20261118_best.pt`
- Optional metadata: `attention_unet_efficientnet_b0.metadata.json`

If your uploaded filenames differ, set the corresponding Space variables under **Settings → Variables and secrets**:

- `EXTRACTOR_FILENAME`
- `PREDICTOR_FILENAME`
- `EXTRACTOR_METADATA_FILENAME`

For private model repositories, add `HF_TOKEN` as a Space **Secret** with read access to both repositories. Do not put a token in source code or ordinary variables.

## API contract (Flutter compatible)

### Health

`GET /api/health`

Returns `status`, `models_loaded`, and `disclaimer`. Returns HTTP 503 until both models load successfully.

### Prediction

`POST /api/predict`

Send `multipart/form-data` with field name `file` containing a JPG, JPEG, or PNG image (maximum 15 MB).

Success JSON retains the Flutter contract:

- `success`
- `extraction_success`
- `predicted_hb_g_dl`
- `bbox`
- `mask_area_pixels`
- `error_message`
- `stage`
- `roi_base64`
- `overlay_base64`
- `disclaimer`

The ROI and overlay are JPEG data URLs.

## Important model fidelity details

- Extractor: Attention U-Net with `timm-efficientnet-b0`, input 320×320 RGB, ImageNet normalization, sigmoid threshold 0.5.
- The binary mask is resized to the original image dimensions using nearest-neighbor interpolation.
- Bounding box uses 20-pixel padding.
- The extracted ROI is encoded as JPEG quality 90 before DenseNet preprocessing. This is required to match the predictor's training-time crop provenance.
- Predictor: DenseNet-121, input 224×224 RGB, ImageNet normalization.
- Hb inversion uses `target_mean_g_dl` and `target_std_g_dl` stored in the predictor checkpoint.

The model weights are loaded at module startup and moved to `"cuda"` at module scope, following the current ZeroGPU model-loading pattern. The inference functions registered with Gradio are decorated with `@spaces.GPU`.

## Deploy

1. Open `https://huggingface.co/spaces/Biraj0016/anemia-detection-api`.
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the Space repository (or push them with Git).
3. Confirm the Space hardware is **ZeroGPU**.
4. Wait for the build and startup.
5. Open `/api/health` on the running Space. Continue only when it returns HTTP 200 and `"models_loaded": true`.
6. Test `/api/predict` with a non-patient, approved test image before changing Flutter's base URL.
7. Set Flutter `SERVER_BASE_URL` to the running Space host, e.g. `https://biraj0016-anemia-detection-api.hf.space`.

## Limitations

This is an academic research prototype, not a clinically validated diagnostic device. It must not replace laboratory hemoglobin testing or professional medical advice. Do not upload identifiable patient photographs or clinical data to a public demo.
