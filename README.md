# BTP Anemia Detection

An academic research project for conjunctiva-image processing and hemoglobin (Hb) estimation. The repository contains the dataset organization, manual annotation workflow, extraction and prediction experiments, selected model artifacts, inference backend, Flutter application, and project reports.

The current application uses a two-stage image pipeline:

1. **Conjunctiva extraction:** an Attention U-Net with an EfficientNet-B0 encoder segments the conjunctiva from an input eye image and extracts a region of interest (ROI).
2. **Hemoglobin estimation:** a DenseNet-121 regression model estimates Hb in g/dL from the extracted ROI.

This is a research prototype. It is not a clinically validated diagnostic device and must not replace laboratory hemoglobin testing or professional medical advice.

---

## Repository structure

```text
BTP-Anemia-Final/
├── backend/                         # API and inference service source
├── datasets/
│   ├── manifests/                   # Dataset and provenance manifests
│   ├── processed/                   # Processed dataset artifacts
│   ├── raw/
│   │   └── hospital/
│   │       ├── images/              # Original hospital images (restricted)
│   │       └── captured_data.csv    # Captured image metadata and Hb records
│   └── splits/                      # Patient-wise/image split definitions
├── docs/                            # Project documentation
├── experiments/
│   ├── annotation/
│   │   ├── scripts/                 # Manual annotation and audit utilities
│   │   └── masks/
│   │       ├── drafts/              # In-progress manual masks
│   │       ├── verified/            # Verified ground-truth masks
│   │       └── reports/             # Annotation manifest and audit reports
│   ├── extraction/                  # Extraction training, evaluation and outputs
│   └── prediction/                  # Hb data audit, benchmarks and evaluation
├── flutter_app/                     # Flutter Android application
├── models/
│   ├── annotation/                  # Annotation-related assets, if applicable
│   ├── extraction/                  # Attention U-Net checkpoints and metadata
│   └── prediction/                  # Selected Hb predictor checkpoints
├── reports/                         # Consolidated project reports
├── requirements.txt                 # Root Python dependencies
└── README.md
```

Original hospital images and associated patient data are sensitive research data. Do not commit or publish identifiable images, captured metadata, or patient-level manifests to a public repository. Follow the institution's data-access and consent requirements.

---

## Current workflows

### 1. Manual annotation

Manual annotation is the active annotation workflow. Verified masks are the ground-truth labels used by the extraction work. Draft masks remain separate from verified masks; annotation reports and manifests document the review state.

- Verified ground-truth masks: `experiments/annotation/masks/verified/`
- Draft masks: `experiments/annotation/masks/drafts/`
- Active GUI manifest: `experiments/annotation/masks/reports/annotation_manifest.csv`
- Annotation scripts and validation utilities: `experiments/annotation/scripts/`

The Attention U-Net is part of the **extraction/segmentation phase**, not a model-assisted annotation workflow. It does not replace or modify the verified manual ground-truth masks.

See [`experiments/annotation/README.md`](experiments/annotation/README.md) for the active annotation GUI workflow and audit details.

### 2. Conjunctiva extraction

The extraction stage uses an EXIF-aware Attention U-Net with an ImageNet EfficientNet-B0 encoder.

- Architecture: Attention U-Net + `timm-efficientnet-b0`
- Input: RGB image resized to 320 × 320
- Normalization: ImageNet mean and standard deviation
- Mask threshold: sigmoid probability ≥ 0.5
- Original-image mask resizing: nearest-neighbor interpolation
- ROI bounding box: 20-pixel padding, clipped to image boundaries
- EXIF orientation: applied before image processing

Training, evaluation, prediction, masks, overlays, ROI crops, and extraction audits are organized under `experiments/extraction/`. The selected extraction checkpoint and metadata are stored under `models/extraction/`.

Reported held-out extraction test results:

| Metric | Result |
|---|---:|
| Dice | 0.8273 ± 0.0694 |
| IoU | 0.7112 ± 0.0962 |
| Precision | 0.8180 ± 0.1039 |
| Recall | 0.8555 ± 0.1123 |
| Specificity | 0.9987 ± 0.0009 |

The extraction test set contained 43 images. Full-dataset inference generated masks, overlays, and ROI crops for 288 images, with zero reported failures in the documented run.

See [`experiments/extraction/README.md`](experiments/extraction/README.md) for training, evaluation, and output details.

### 3. Hemoglobin prediction

The prediction stage estimates Hb from an extracted conjunctiva ROI. The selected model is DenseNet-121, trained using the patient-wise split and the documented ROI preprocessing.

- Selected architecture: DenseNet-121
- Selected seed: `20261118`
- Checkpoint: `models/predictors/densenet121_seed_20261118_best.pt`
- Input: RGB ROI resized to 224 × 224
- Normalization: ImageNet mean and standard deviation
- Output: Hb estimate in g/dL, inverse-transformed using the target mean and standard deviation stored in the checkpoint

Selected checkpoint validation metrics:

| Metric | Result |
|---|---:|
| MAE | 0.7736 g/dL |
| RMSE | 1.1696 g/dL |
| R² | 0.4510 |

The selected checkpoint's target standardization values are stored in its checkpoint metadata. Inference must use those checkpoint values rather than hard-coded or recomputed values.

The held-out test evaluation is documented separately from model selection. The test set must not be used for further tuning, architecture selection, threshold selection, or repeated model selection.

See [`experiments/prediction/README.md`](experiments/prediction/README.md) for the dataset audit, benchmarks, selected model, preprocessing, and final evaluation.

---

## End-to-end inference application

The deployed inference service connects the two stages:

```text
Input eye image
      │
      ▼
EXIF orientation correction
      │
      ▼
Attention U-Net conjunctiva segmentation
      │
      ├── Segmentation mask / overlay
      │
      ▼
Padded conjunctiva ROI extraction
      │
      ▼
JPEG encoding (quality 90)
      │
      ▼
DenseNet-121 Hb regression
      │
      ▼
Estimated Hb (g/dL) + ROI + overlay
```

The deployed backend is hosted on Hugging Face Spaces using ZeroGPU and Gradio, with FastAPI endpoints for the Flutter client.

- Space: [Biraj0016/anemia-detection-api](https://huggingface.co/spaces/Biraj0016/anemia-detection-api)
- API base URL: `https://biraj0016-anemia-detection-api.hf.space`
- Extractor model repository: [Biraj0016/Anemia-Extractor](https://huggingface.co/Biraj0016/Anemia-Extractor)
- Predictor model repository: [Biraj0016/Anemia-Predictor](https://huggingface.co/Biraj0016/Anemia-Predictor)

### API endpoints

#### Health check

```http
GET /api/health
```

Returns service status, whether both models are loaded, and the research disclaimer. The service is ready only when `models_loaded` is `true`.

#### Image prediction

```http
POST /api/predict
Content-Type: multipart/form-data
```

Upload the image using the multipart field name `file`. JPG, JPEG, and PNG are supported; the current backend upload limit is 15 MB.

The JSON response includes:

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

The ROI and overlay are returned as JPEG data URLs. The Gradio interface converts these data URLs to image objects for display; the API response contract remains Base64/data-URL based for Flutter compatibility.

---

## Flutter application

The Flutter client is in `flutter_app/`. It communicates with the deployed FastAPI service using the `SERVER_BASE_URL` compile-time setting.

From PowerShell:

```powershell
cd flutter_app
flutter pub get
flutter run --dart-define=SERVER_BASE_URL=https://biraj0016-anemia-detection-api.hf.space
```

Build a release APK:

```powershell
flutter build apk --release --dart-define=SERVER_BASE_URL=https://biraj0016-anemia-detection-api.hf.space
```

The APK is generated at:

```text
build/app/outputs/flutter-apk/app-release.apk
```

For local Android emulator development, the app configuration may use `http://10.0.2.2:8000` as the local backend address. For a physical device, configure a reachable backend address. Use the production URL above for the deployed service.

---

## Dataset and experiment safeguards

- Preserve original source images and captured metadata; do not overwrite source data during preprocessing.
- Keep draft masks separate from verified manual ground-truth masks.
- Keep patient-level grouping intact across train, validation, and test splits.
- Do not regenerate established splits without documenting and justifying the change.
- Do not use held-out test data for training, tuning, model selection, or threshold selection.
- Preserve the selected checkpoints and their preprocessing contracts; any replacement model must have its own documented evaluation and provenance.
- Keep archived or superseded experiments outside the active workflow. They should not be presented as current models, current annotation procedures, or current application dependencies.

---

## Reproducibility and documentation

Each active phase maintains its own README and supporting audit files:

- `experiments/annotation/README.md`
- `experiments/extraction/README.md`
- `experiments/prediction/README.md`
- `docs/`
- `reports/`

Use the phase-specific documentation as the source of truth for exact scripts, checkpoint paths, dataset manifests, experiment configurations, and detailed results.

## Limitations

This project is an academic research prototype for image-based Hb estimation. The models and application have not been clinically validated and have not been established as a substitute for laboratory measurement. Performance may vary with image quality, acquisition conditions, population, and clinical setting. Do not use the app to make diagnosis or treatment decisions. Use approved, de-identified test images when demonstrating the public API; do not upload identifiable patient images or clinical data to a public demo.
