# Hemoglobin Prediction Experiments & Inference Pipeline

This directory contains the machine learning experiments, model evaluations, verification reports, and inference modules for hemoglobin (Hb) regression from conjunctival images.

---

## 1. Important Academic Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE NOTICE**
>
> This software is an academic research prototype for hemoglobin estimation from conjunctiva photographs. It is not a clinically validated diagnostic device and must not be used as a substitute for certified laboratory testing, standard complete blood counts (CBC), or professional medical advice.

---

## 2. End-to-End Two-Stage Pipeline Architecture

The complete system connects conjunctiva segmentation and hemoglobin regression in an integrated two-stage pipeline.

```text
[Full Eye Image (JPG/PNG)]
          |
          v
+--------------------------------------------------------+
| Stage 1: Conjunctiva Extraction (Attention U-Net)       |
|  - EXIF orientation correction                         |
|  - Resize to 320x320, ImageNet normalization            |
|  - EfficientNet-B0 encoder, sigmoid output (thr = 0.5) |
|  - Nearest-neighbor mask resize to original dimensions |
|  - Bounding box calculation with +20px padding         |
+--------------------------------------------------------+
          |
          v
     [Conjunctival ROI Crop]
          |
          v
+--------------------------------------------------------+
| Inter-Stage Coupling: JPEG Q=90 In-Memory Encoding     |
|  - Reproduces the JPEG-encoded ROI input distribution  |
|    used during DenseNet training                       |
+--------------------------------------------------------+
          |
          v
+--------------------------------------------------------+
| Stage 2: Hemoglobin Regression (DenseNet-121)          |
|  - ImageNet normalization: mean (0.485, 0.456, 0.406) |
|    std (0.229, 0.224, 0.225)                           |
|  - Forward pass on 224x224 normalized crop             |
|  - Inverse target standardization using training split |
|    statistics (mean: 10.3629 g/dL, std: 2.1462 g/dL)   |
+--------------------------------------------------------+
          |
          v
[Predicted Hemoglobin in g/dL + ROI Artifacts]
```

---

## 3. Active Prediction Scripts (`scripts/`)

The following four active modules are maintained in `scripts/`:

| Script | Purpose |
|---|---|
| `end_to_end_inference.py` | Production two-stage inference module (`EndToEndPipeline`, `EndToEndResult`). Integrates Attention U-Net extraction, JPEG Q=90 ROI encoding, DenseNet-121 regression, and structured artifact generation. Reused by the FastAPI backend. |
| `inference.py` | Standalone ROI inference module. Constructs the DenseNet-121 regression architecture, loads checkpoint weights, provides deterministic `preprocess_image()`, and executes ROI-to-Hb prediction. |
| `hb_benchmark_v2.py` | Corrected V2 multi-architecture training and evaluation benchmark script, preserved for experimental reproducibility. |
| `hb_final_test_evaluation.py` | Held-out test evaluation script for the selected DenseNet-121 model, preserved for evaluation reproducibility. |

### Archived Historical Scripts

Historical Phase 6 data preparation scripts, V1 benchmark scripts, and Phase 8 test scripts have been archived under `D:\BTP\archive\`:

- `D:\BTP\archive\prediction_scripts\hb_benchmark.py` — V1 benchmark.
- `D:\BTP\archive\prediction_scripts\hb_benchmark_v2_stability.py` — Multi-seed stability runs.
- `D:\BTP\archive\prediction_scripts\phase8_verification\test_phase83_robustness.py` — Phase 8.3 test harness.
- `D:\BTP\archive\prediction_scripts\phase8_verification\phase84\test_phase84_interface.py` — Phase 8.4 interface test harness.
- `D:\BTP\archive\prediction_scripts\phase8_verification\phase841\test_phase841_hardening.py` — Phase 8.4.1 interface hardening test harness.

**Note:** `extraction_inference.py` remains active and unchanged under `experiments/extraction/src/`.

---

## 4. Benchmark Evolution & Model Selection

### V1 vs V2 Benchmark Distinction

- **V1 Benchmark (`benchmarks/v1/`):** Initial multi-architecture baseline benchmark evaluating six backbone architectures: ResNet-18, VGG-16, DenseNet-121, EfficientNet-B0, MobileNetV3-Large, and ConvNeXt-Tiny.
- **V2 Benchmark (`benchmarks/v2/` and `benchmarks/v2_stability/`):** Standardized and audited benchmark pipeline with corrected BatchNorm handling, training-target standardization, and multi-seed stability evaluation using seeds `20261116`, `20261117`, and `20261118`.

### Final Selected Model: DenseNet-121

- **Architecture:** DenseNet-121.
- **Selected seed:** `20261118`.
- **Best epoch:** `9`.
- **Checkpoint:** `models/predictors/densenet121_seed_20261118_best.pt`.
- **Checkpoint SHA-256:** `57EA961A29524563059F5293147E962945AE9BDB1042E357570057F4C533288B`.
- **Selection basis:** Lowest validation MAE among the three evaluated seeds. Test-set results were not used for model selection.

### Validation Performance Used for Model Selection

| Metric | Validation result |
|---|---:|
| MAE | 0.7736 g/dL |
| RMSE | 1.1696 g/dL |
| R² | 0.4510 |

### Held-Out Test Set Performance

The selected checkpoint was evaluated once on the patient-wise held-out test set containing **41 records**.

| Metric | Final test result |
|---|---:|
| MAE | **1.3452 g/dL** |
| RMSE | **1.7005 g/dL** |
| R² | **0.2097** |
| Bias | +0.4078 g/dL |
| Median absolute error | 1.0662 g/dL |
| Maximum absolute error | 4.5003 g/dL |

### Training-Mean Baseline on the Test Set

| Metric | Baseline result |
|---|---:|
| MAE | 1.3713 g/dL |
| RMSE | 1.9149 g/dL |
| R² | -0.0021 |

The selected DenseNet-121 model achieved a modest improvement over the training-mean baseline on the held-out test set. These results represent internal held-out evaluation, not independent external validation or clinical validation.

**Metric provenance:** The validation figures are recorded in `final_evaluation/final_model_selection.json`. The held-out test figures are recorded in `final_evaluation/test_metrics.json`.

---

## 5. Verification Evidence & Historical Reports

Formal Phase 8 verification reports and artifact integrity audits are organized under `reports/inference_verification/`.

### Phase 8.3 — End-to-End Inference Robustness

Location: `reports/inference_verification/phase83/`

- Evaluates pipeline robustness across 15 training and validation samples.
- Audits EXIF orientation transformations, distinguishing verified orientations from inconclusive empty-mask scenarios.
- Verifies that the DenseNet-121 forward pass is bypassed when conjunctiva extraction produces an empty mask.
- Confirms deterministic predictions across repeated inference runs.
- Records checkpoint integrity and robustness results.

### Phase 8.4.1 — Interface Hardening & Verification

Location: `reports/inference_verification/phase841/`

- Verifies FastAPI lifespan lifecycle initialization and shutdown.
- Tests actual startup constructor failure through the lifespan context and verifies HTTP 503 service-unavailable behavior.
- Verifies handler-level overlapping-request rejection with HTTP 429 using synchronized worker threads and `threading.Event`.
- Tests user-facing error sanitization to ensure internal paths and stack traces are not exposed.
- Confirms numerical parity between direct pipeline calls and HTTP uploads for JPG and PNG inputs, within the tested tolerance of `1e-8 g/dL`.
- Verifies response schema compatibility, academic disclaimer presence, and checkpoint integrity.

The concurrency test is a deterministic handler-level overlapping-request test; it is not a network-level load or stress test.

---

## 6. How to Run Inference

Run the following commands from the project root:

`D:\BTP\BTP-Anemia-Final`

### 1. Standalone ROI Inference (Pre-extracted Crop)

```powershell
& "D:\BTP\.venv\Scripts\python.exe" experiments\prediction\scripts\inference.py --image experiments\extraction\roi_crops\0115H-569.jpg
```

### 2. End-to-End Full-Image Inference (Python API)

```python
from experiments.prediction.scripts.end_to_end_inference import EndToEndPipeline

pipeline = EndToEndPipeline()
result = pipeline.predict(
    "path/to/full_eye_image.jpg",
    return_artifacts=True
)

if result.success:
    print(f"Predicted Hb: {result.predicted_hb_g_dl:.2f} g/dL")
    print(f"Conjunctiva Bounding Box: {result.bbox}")
    print(f"Mask Area: {result.mask_area_pixels} px")
else:
    print(f"Extraction / Prediction Failed: {result.error_message}")
```

### 3. Interactive Web Interface

```powershell
& "D:\BTP\.venv\Scripts\python.exe" run_interface.py
```

Open the application in your browser:

`http://127.0.0.1:8000`

---

## 7. Reproducibility and Scientific Safeguards

- **Patient-wise splitting:** Training, validation, and test partitions are maintained separately.
- **Model selection isolation:** Validation MAE was used to select the final DenseNet-121 checkpoint; test results were not used for selection.
- **Test-set isolation:** The final test evaluation was performed separately after model selection. Phase 8 inference verification and cleanup smoke tests used training and validation samples only.
- **Model integrity:** The selected DenseNet-121 and Attention U-Net checkpoints are preserved and verified using SHA-256 hashes.
- **Preprocessing consistency:** EXIF correction, extraction resizing, ROI padding, JPEG Q=90 encoding, ImageNet normalization, and inverse target standardization follow the documented training/inference pipeline.
- **No clinical claim:** The reported metrics demonstrate experimental model performance only. Further independent validation, larger datasets, and clinical studies are required before any clinical use.

