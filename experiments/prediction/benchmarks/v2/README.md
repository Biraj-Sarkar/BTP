# Hb Regression Benchmark V2

This directory contains the corrected Phase 7.1 validation-only benchmark. It
uses the finalized Phase 6.1 manifest, ROI images only, ImageNet-pretrained
backbones, and the existing patient-wise train/validation/test assignments.

V2 corrects the V1 frozen-backbone behavior. After every `model.train()` call,
all BatchNorm modules are explicitly returned to evaluation mode, including
BatchNorm modules in the partially unfrozen tail. Their running means,
variances, and batch counters are audited after every epoch. Frozen parameter
gradients and tail gradients are also checked.

## Dataset and protocol

- 288 manifest rows; 279 eligible records
- Train/validation/test: 195/43/41
- 224 x 224 RGB ROI inputs with ImageNet normalization
- Target: `hb_g_dl`, standardized using training targets only
- AdamW, Smooth L1 loss, batch size 8
- Maximum 40 epochs; validation-MAE early stopping with patience 7
- Seed: `20261116`
- No metadata features and no test-set access

The corrected loader counts test rows from metadata columns only. It loads
targets, ROI paths, and images only for the 195 train and 43 validation rows.
The audit fields `test_targets_loaded`, `test_images_loaded`,
`test_predictions_generated`, and `test_used_for_selection` are all false.
Additional seed stability runs were not performed; `seed_stability_summary.csv`
records the initial seed only.

## Outputs

- `initial_benchmark_summary.csv`: validation ranking
- `validation_predictions.csv`: per-record validation predictions
- `smoke_test_report.json`: pre-training correctness checks
- `batchnorm_integrity_report.json`: BatchNorm policy and audit references
- `frozen_layer_audit.json`: frozen/head/tail audit references
- `test_isolation_correction_audit.json`: columns and rows loaded by stage
- `training_curves/` and `plots/`: generated visual diagnostics
- `runs/<architecture>/seed_20261116/`: histories, predictions, and audits

Checkpoints are stored separately under
`D:/BTP/archive/prediction_models/hb_regression_v2/`. V1 outputs remain under
`experiments/prediction/benchmarks/v1/` and
`D:/BTP/archive/prediction_models/hb_regression/`.

This is an internal research benchmark on a small single-source dataset. It
does not establish clinical diagnostic validity or external generalization.
