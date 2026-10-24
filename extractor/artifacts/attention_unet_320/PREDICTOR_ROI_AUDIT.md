# Hemoglobin predictor ROI audit — training not started

Audit date: 2026-10-22

## Findings

- Authoritative source labels exist in `datasets/captured_data.csv`: 183 registration rows with hemoglobin values.
- The saved segmentation split exists at `extractor/experiments/final_pretrained320/split_manifest.csv`: 129 train, 27 validation, 27 test. All 183 manifest image stems matched a captured-data hemoglobin label; no labels were inferred from row order or filenames.
- The expected production ROI directory `ROI/` is present but empty. `ROI/roi_manifest.csv` and `ROI/extraction_manifest.csv` are absent. Therefore production U-Net ROI image files and the authoritative image→ROI→label mapping needed for predictor training are unavailable.
- `ROI_attention/` contains 183 ROI files and `extraction_manifest.csv`, but that manifest records every non-empty row as `backend=unet` and one row as `unet-empty` (`1951.jpg`). It has no candidate-architecture provenance. Because the task explicitly forbids assuming that an uploaded production manifest identifies candidate ROIs, these files cannot be treated as verified Attention-U-Net outputs.
- The candidate artifact and prior `preproduction_assessment/rows.json` provide inference/quality metadata, but not a complete candidate ROI image set with an authoritative manifest suitable for training.
- The existing ResNet18 checkpoint is `predictor/runs/resnet/best.pt`; its configuration references `ROI/roi_manifest.csv`, confirming that the established predictor was trained from the missing production ROI manifest. Its preprocessing is 224×224 RGB tensor conversion with ImageNet normalization inside the model.

## Decision

Training the production-ROI and Attention-ROI predictors is blocked. No training, ROI regeneration, split modification, checkpoint writing, or four-way evaluation was performed. Labels, splits, extractors, and existing checkpoints remain unchanged.

Required before training can begin: restore or provide the authoritative production ROI images plus `ROI/roi_manifest.csv` (or an equivalent manifest containing image ID, ROI path, label, quality/backend, and split membership), and provide a separately authoritative Attention-U-Net ROI manifest for the same 183 source images. The single empty extraction must be handled explicitly rather than silently dropped or replaced.

Reproducible audit commands:

```powershell
Get-ChildItem ROI -Force
Import-Csv ROI_attention/extraction_manifest.csv | Group-Object backend
Import-Csv extractor/experiments/final_pretrained320/split_manifest.csv | Group-Object split
```
