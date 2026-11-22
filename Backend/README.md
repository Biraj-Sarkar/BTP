# Single-Image Inference Web Interface

This directory contains the user-facing web interface and backend service for the two-stage **Anemia Detection** research pipeline (Attention U-Net conjunctiva segmentation $\to$ DenseNet-121 hemoglobin regression).

---

## 1. Important Academic Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE NOTICE**  
> This is an academic research prototype for hemoglobin estimation from conjunctiva images. It is not a clinically validated diagnostic device and must not be used as a substitute for laboratory testing or professional medical advice.

---

## 2. Architecture & Pipeline Overview

The interface connects the validated two-stage pipeline directly to a lightweight browser UI:

1. **User Upload**: Receives a single image (JPG, JPEG, or PNG) via drag-and-drop or file selector.
2. **Stage 1 (Attention U-Net)**: Performs EXIF orientation correction, segments the palpebral conjunctiva, and extracts the bounding-box ROI crop ($+20\text{px}$ padding).
3. **Stage 2 (DenseNet-121)**: Encodes the ROI at JPEG $Q=90$, applies standard ImageNet normalization, and estimates hemoglobin concentration in $\text{g/dL}$.
4. **Interactive Display**: Shows original preview, extracted conjunctival ROI, segmentation overlay, predicted Hb value, and extraction metadata.

---

## 3. Application Lifecycle & Startup

The backend service is managed via FastAPI's modern `lifespan` context manager:

- **Startup (`lifespan`)**:
  - `init_pipeline(app)` is executed once when the server initializes.
  - Both checkpoints (`Attention U-Net` and `DenseNet-121`) are loaded into memory and verified.
  - The resulting `EndToEndPipeline` instance is cached as an application-wide singleton (`app.state.pipeline` and `_pipeline`).
  - If startup succeeds, `/api/health` returns `HTTP 200 OK` with `models_loaded: true`.
  - If initialization fails (e.g. missing checkpoint), the full traceback is recorded in server logs, the service remains online, `/api/health` returns `HTTP 503 Service Unavailable` with `models_loaded: false`, and inference requests return `HTTP 503`.
- **Inference Lifecycle**:
  - All inference requests retrieve the pre-initialized singleton via `get_pipeline()`. Models are never reloaded on per-request basis.
- **Concurrency & Thread Safety**:
  - Requests are protected by a non-blocking execution lock (`threading.Lock`).
  - If an inference request arrives while another is executing, it is immediately rejected with `HTTP 429 Too Many Requests`.
  - Once active inference finishes, the lock is guaranteed to be released in a `finally` block, allowing subsequent requests to proceed.

---

## 4. API Endpoints & Status Codes

| Endpoint | Method | Expected Status Codes | Description |
|---|:---:|:---:|---|
| `/` | `GET` | `200`, `500` | Serves the single-page HTML/JS user interface. |
| `/api/health` | `GET` | `200`, `503` | Readiness probe. Returns `200` (`healthy`) when models are loaded; `503` (`unhealthy`) if not ready. |
| `/api/predict` | `POST` | `200`, `400`, `422`, `429`, `500`, `503` | Single-image hemoglobin prediction endpoint (accepts multipart/form-data `file`). |

### Status Code Semantics:
- **`HTTP 200 OK`**: Successful inference; returns predicted Hb in $\text{g/dL}$, bounding box, mask pixel count, and base64 preview artifacts.
- **`HTTP 400 Bad Request`**: Client validation failure (missing upload, unsupported file format, empty 0-byte file, or unreadable/corrupted image data).
- **`HTTP 422 Unprocessable Content`**: Model extraction failure caused by submitted image (e.g. eye closed, blurry, or no palpebral conjunctiva detected).
- **`HTTP 429 Too Many Requests`**: Concurrency busy protection; another inference operation is actively running.
- **`HTTP 503 Service Unavailable`**: Pipeline models are not loaded or startup initialization failed.
- **`HTTP 500 Internal Server Error`**: Unexpected server error. Internal exception details and system paths are logged server-side and never exposed to the client.

---

## 5. Prerequisites & Environment Setup

- **Operating System**: Windows 11 (CPU-only execution).
- **Python**: Python 3.14+ in virtual environment `D:\BTP\.venv`.
- **Dependencies**: `fastapi`, `uvicorn`, `python-multipart`, `torch`, `torchvision`, `pillow`, `opencv-python`, `segmentation-models-pytorch`.

---

## 6. How to Launch the Interface

From `D:\BTP\BTP-Anemia-Final` in PowerShell:

```powershell
D:\BTP\.venv\Scripts\python.exe run_interface.py
```

Or using Uvicorn directly:

```powershell
D:\BTP\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Once started, open your web browser at:

**`http://127.0.0.1:8000`**

---

## 7. How to Use the Interface

1. **Select an Image**: Click the upload area or drag a JPG or PNG eye photograph into the drop zone. An instant preview is displayed.
2. **Run Prediction**: Click the **"Estimate Hemoglobin"** button. The button shows a loading spinner and disables during processing to prevent duplicate requests.
3. **Review Results**:
   - **Estimated Hemoglobin**: Displayed prominently in $\text{g/dL}$.
   - **Extracted ROI**: High-resolution crop of the detected conjunctiva tissue.
   - **Segmentation Overlay**: Full-image overlay with the detected conjunctival region highlighted in red.
   - **Extraction Metadata**: Displays detected pixel area and bounding box coordinates $[x_0, y_0, x_1, y_1]$.
4. **Process Another Image**: Click **"Select Another Image"** to reset the input and upload a new photograph without restarting the server.

---

## 8. Expected Checkpoint Locations & Integrity

The server automatically loads the authorized model checkpoints on startup:

- **Extraction Model**:  
  `models/extraction/attention_unet_efficientnet_b0_best.pt`  
  *(SHA-256: `1F53832F8DC2A8FFBFCBEEBD426B8A53F8C9EC69B0A9229C21A4D7F66AD86AF3`)*
- **Hb Regression Model**:  
  `models/predictors/densenet121_seed_20261118_best.pt`  
  *(SHA-256: `57EA961A29524563059F5293147E962945AE9BDB1042E357570057F4C533288B`)*
- **Metadata**:  
  `models/extraction/attention_unet_efficientnet_b0.metadata.json`  
  *(SHA-256: `EE23E2FBB1BCD683056090F804B95282DF2EC026D7473D569EDBE7D881C0A2BB`)*

---

## 9. Common Errors & Troubleshooting

| Error | Cause | Resolution |
|---|---|---|
| **Invalid file type** | Uploaded file is not `.jpg`, `.jpeg`, or `.png` | Provide a valid JPG or PNG eye image. |
| **Empty file (0 bytes)** | Corrupted or zero-length file uploaded | Ensure the image file is valid before uploading. |
| **No conjunctiva detected** | Eye is closed, out of frame, blurry, or poor lighting | Provide a focused photograph showing the palpebral conjunctiva. |
| **Concurrent request busy (HTTP 429)** | Multiple simultaneous inference requests sent | Wait a moment for the active request to finish; requests are locked during processing. |
| **Service Unavailable (HTTP 503)** | Models failed to load at startup | Check console log for missing or corrupted checkpoint paths. |
| **Port 8000 already in use** | Another service is using port 8000 | Launch on an alternate port: `python run_interface.py --port 8080`. |
