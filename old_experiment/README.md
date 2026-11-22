# BTP Anemia Model Suite

This directory is a combined, reviewable copy of the anemia and haemoglobin
prediction work. The individual algorithm directories are kept separate so
their source files and training instructions remain easy to inspect.

## Contents

- `App/` – Flutter mobile client for anemia detection.
- `Backend/` – Server-side FastAPI inference service and model loading.
- `extractor/` – Conjunctival ROI extraction using pretrained U-Net and quality control.
- `predictor/` – Production hemoglobin prediction pipeline supporting ResNet, VGG, and MobileNet.
- `algorithms/Anubhav/` – Retained experimental reference pipeline.
- `Models/` – U-Net segmentation checkpoint (`Models/UNet_extractor/unet.pt`) and trained predictor weights.
- `ROI/` – Pre-extracted 224x224 conjunctiva ROIs and manifest.
- `datasets/` – Clinical captures, metadata CSV, and reference images.

## Architecture & Workflows

1. **Extraction (`extractor/`)**: Eye image → Color balancing → U-Net inference → Mask morphological refinement → Quality checks → 224x224 ROI crop.
2. **Prediction (`predictor/`)**: ROI → Preprocessing & Augmentation → ResNet / VGG / MobileNet Regressor → Standardized score denormalization → Hb prediction (g/dL).
3. **Backend (`Backend/`)**: FastAPI server orchestrating extraction, prediction, and WHO threshold-based anemia classification.

