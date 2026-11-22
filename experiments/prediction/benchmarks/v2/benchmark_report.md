# Phase 7.1 Corrected Hb Regression Benchmark

## Scope

V2 repeats the six-architecture ROI-only benchmark while correcting the V1
frozen-backbone and BatchNorm behavior. V1 was not modified. The corrected
loader counts test rows from metadata columns only, then reads target-bearing
columns and ROI paths only for train and validation rows. No test predictions,
losses, gradients, or model-selection decisions were produced.

## Correction implemented

V1 set `requires_grad=False` for the frozen backbone but called `model.train()`
without restoring frozen BatchNorm modules to evaluation mode. Consequently,
BatchNorm running statistics could change even without parameter gradients.
V2 calls a BatchNorm evaluation-mode helper after every training-mode change,
keeps all BatchNorm running statistics frozen in both phases, and audits all
running means, variances, and batch counters after every epoch. Smoke tests
also verify frozen gradients, tail gradients, output shape, pretrained weight
loading, and checkpoint round trips.

The documented unfrozen tails were: ResNet-18 `layer4`, VGG16 final six
feature modules, DenseNet-121 `features.denseblock4` and `features.norm5`,
EfficientNet-B0 final three feature modules, MobileNetV3-Large final three
feature modules, and ConvNeXt-Tiny final two feature modules.

## Dataset and environment

- Manifest rows: 288
- Eligible records: 279
- Train/validation/test: 195/43/41
- Registration overlap: zero
- ROI paths: all readable
- Target: `hb_g_dl` in gm/dL, numerically equivalent to g/dL
- Input: 224 x 224 RGB, ImageNet normalization
- Python 3.14.6; PyTorch 2.13.0+cpu; torchvision 0.28.0+cpu
- CPU-only execution

## V2 validation results

| Architecture | Best epoch | MAE | RMSE | R2 | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 11 | 0.9392 | 1.3372 | 0.2825 | 569.3 |
| MobileNetV3-Large | 5 | 0.9531 | 1.3233 | 0.2974 | 99.6 |
| EfficientNet-B0 | 10 | 0.9612 | 1.3414 | 0.2779 | 251.5 |
| VGG16 | 5 | 0.9922 | 1.2705 | 0.3523 | 777.6 |
| ResNet-18 | 20 | 1.0277 | 1.3174 | 0.3036 | 398.1 |
| ConvNeXt-Tiny | 4 | 1.0646 | 1.3988 | 0.2149 | 340.5 |

The training-set mean baseline remains in `baseline_results.json`. V2 model
selection is based on validation results only. DenseNet-121 has the lowest
validation MAE; VGG16 has the lowest validation RMSE and highest validation
R2, so the ranking is not uniform across metrics. No final test model was
selected or evaluated in V2.

## Comparison with V1

| Architecture | V1 MAE | V2 MAE | V2 - V1 |
|---|---:|---:|---:|
| ResNet-18 | 1.1138 | 1.0277 | -0.0861 |
| VGG16 | 1.0095 | 0.9922 | -0.0173 |
| DenseNet-121 | 1.0434 | 0.9392 | -0.1041 |
| EfficientNet-B0 | 1.1144 | 0.9612 | -0.1532 |
| MobileNetV3-Large | 1.0392 | 0.9531 | -0.0861 |
| ConvNeXt-Tiny | 1.0325 | 1.0646 | +0.0321 |

These are validation comparisons from separate single-seed runs, not an
independent clinical confirmation. V1's historical held-out test result
remains preserved, but it must not be interpreted as V2 confirmation. The
completed benchmark metrics remain unchanged; this correction affects only the
loader and audit documentation.

## Strict test-target isolation audit

The metadata pass loads only `image_id`, `registration_no`,
`hb_inclusion_status`, and `extraction_split` for all 288 manifest rows. The
second pass loads `image_id`, `registration_no`, `original_hb_unit`, `hb_g_dl`,
`roi_crop_path`, and `extraction_split` only for the 238 train/validation rows.
Test rows are counted from metadata only. Test target values, ROI paths, images,
and predictions are not loaded or generated.

## Integrity and limitations

All six smoke tests passed. Every training run completed, and every epoch
reported unchanged frozen BatchNorm buffers. Protected Attention U-Net
metadata, best, and final checkpoint SHA-256 hashes were unchanged before and
after V2. V1 outputs, source data, masks, extraction outputs, and manifests
were not modified.

Only one seed was run because the environment is CPU-only. The sample is
small and comes from one hospital source. Results are internal research
measurements and do not establish clinical diagnostic validity. A genuinely
independent cohort is recommended for any future external evaluation.
