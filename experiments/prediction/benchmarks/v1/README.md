# Hb regression benchmark

This directory contains the Phase 7 image-only hemoglobin regression benchmark
for the 279 eligible records in `hb_dataset_manifest.csv`.

## Dataset

- Eligible records: 279
- Train: 195
- Validation: 43
- Test: 41
- Split grouping: `registration_no`
- Split overlap: zero
- Target: `hb_g_dl`
- Input: existing RGB ROI JPEG crops

## Protocol

- Seed: 20261116
- Resolution: 224 x 224
- Device: CPU
- Batch size: 8
- Optimizer: AdamW
- Loss: Smooth L1
- Primary metric: validation MAE
- Early stopping: validation MAE, patience 7
- Maximum epochs: 40
- Target normalization: training-set mean and standard deviation only
- Augmentation: mild 5-degree rotation, horizontal flip, and brightness/contrast
  variation of 5 percent
- No gray-world white balance or other colour correction
- No metadata inputs
- Test set not used during Stage A or model selection

All architectures used ImageNet-pretrained weights. No model fell back to random
weights.

## Outputs

- `benchmark_config.json`
- `dataset_validation.json`
- `baseline_results.json`
- `initial_benchmark_summary.csv`
- `validation_predictions.csv`
- `seed_stability_summary.csv`
- `final_model_selection.json`
- `test_predictions.csv`
- `test_metrics.json`
- `training_curves/`
- `plots/`
- `logs/`

Checkpoints are under `D:/BTP/archive/prediction_models/hb_regression/`.

See `benchmark_report.md` for the results and limitations.
