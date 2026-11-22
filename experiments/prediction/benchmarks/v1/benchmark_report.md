# Phase 7 Hb regression benchmark report

## Scope

This benchmark compares six ImageNet-pretrained ROI-only hemoglobin regressors
against a training-mean baseline. It uses the finalized Phase 6.1 manifest and
the existing patient-wise 202/43/43 extraction split. The test set was kept
untouched until final VGG16 evaluation.

## Dataset and leakage control

- Manifest rows: 288
- Eligible rows: 279
- Train eligible: 195
- Validation eligible: 43
- Test eligible: 41
- Registration overlap: zero
- Duplicate eligible registrations: none
- Target units: gm/dL, numerically equivalent to g/dL
- All 279 targets and 288 ROI paths passed dataset validation
- Training target mean: 11.3764102564 g/dL
- Training target standard deviation: 1.8684581429 g/dL

## Environment

- Python: 3.14.6
- PyTorch: 2.13.0+cpu
- torchvision: 0.28.0+cpu
- CUDA: unavailable
- Logical CPUs: 12
- Platform: Windows 11

## Training protocol

All models used 224 x 224 RGB ROI images, batch size 8, AdamW, weight decay
0.0001, Smooth L1 loss, target standardization fitted only on training targets,
validation-MAE early stopping with patience 7, and maximum 40 epochs.

The classifier/regression head was trained initially. The requested limited
backbone tail was unfrozen after epoch 10. VGG16 unfrozen its final feature
layers, DenseNet121 unfrozen its fourth dense block and final normalization,
EfficientNet-B0 unfrozen its final three feature stages, MobileNetV3-Large
unfrozen its final three feature stages, ConvNeXt-Tiny unfrozen its final two
feature stages, and ResNet-18 unfrozen `layer4`.

## Initial Stage A validation comparison

| Architecture | Seed | Status | Best epoch | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---|---:|---:|---:|---:|
| VGG16 | 20261116 | completed | 11 | 1.0095 | 1.2928 | 0.3294 |
| ConvNeXt-Tiny | 20261116 | completed | 12 | 1.0325 | 1.3673 | 0.2498 |
| MobileNetV3-Large | 20261116 | completed | 5 | 1.0392 | 1.3844 | 0.2309 |
| DenseNet-121 | 20261116 | completed | 4 | 1.0434 | 1.4054 | 0.2075 |
| ResNet-18 | 20261116 | completed | 15 | 1.1138 | 1.5330 | 0.0569 |
| EfficientNet-B0 | 20261116 | completed | 2 | 1.1144 | 1.4783 | 0.1231 |

No model failed in the final Stage A run. DenseNet121 had one initial
implementation failure in the code used for tail-unfreezing; the module path
was corrected and only DenseNet121 was rerun. The completed rerun is the result
reported above.

## Mean-prediction baseline

The training-set mean was 11.3764102564 g/dL. On the 43-record validation set:

- MAE: 1.2223 g/dL
- RMSE: 1.6193 g/dL
- R²: -0.0522

Five of the six models outperformed the mean baseline by validation MAE.
EfficientNet-B0 performed slightly worse.

## Seed stability

Additional fixed-seed stability experiments were not run. The CPU-only VGG16
seed-20261116 run took approximately 1381 seconds, and further top-candidate
multi-seed runs were not practical within this benchmark session. The
`seed_stability_summary.csv` therefore contains only the initial seed.

No test result was used to select or retrain a model.

## Final selection and held-out test

VGG16 was selected because it had the lowest Stage A validation MAE
1.0095109274 g/dL, the lowest validation RMSE 1.2927791583 g/dL, and the
highest validation R² 0.3293706207 among the completed seed-20261116 runs.

The saved VGG16 checkpoint was evaluated once on the untouched 41-record test
set. It was not retrained or modified after test evaluation.

Test metrics:

- MAE: 1.2252477902 g/dL
- RMSE: 1.6283113484 g/dL
- R²: 0.2754000559
- Mean prediction error / bias: 0.3585535957 g/dL
- Median absolute error: 0.8770790100 g/dL
- Maximum absolute error: 5.0703582764 g/dL
- Bland-Altman mean difference: 0.3585535957 g/dL
- Bland-Altman 95% limits: -2.7932751655 to 3.5103823568 g/dL

## Limitations

This is a small, single-source, internal dataset and a screening-research
experiment. It does not establish clinical diagnostic validity or external
generalization. All measurements and splits originate from the same hospital
cohort. Results may be sensitive to capture devices, lighting, patient
demographics, and annotation/extraction processing.

Only one seed was run for every architecture. The lack of seed stability
experiments limits confidence in ranking differences. The test set was
evaluated once for the selected VGG16 checkpoint.

## Preservation

Before and after the benchmark, the three protected extraction artifacts had
matching SHA-256 hashes:

- Metadata: `EE23E2FBB1BCD683056090F804B95282DF2EC026D7473D569EDBE7D881C0A2BB`
- Best checkpoint: `1F53832F8DC2A8FFBFCBEEBD426B8A53F8C9EC69B0A9229C21A4D7F66AD86AF3`
- Final checkpoint: `959CCF4EF94E93EFB5CBC83926CB537A764D55E21C071597541B62881A70AD9E`

The original CSV, images, masks, extraction outputs, Phase 6/6.1 reports,
existing split manifests, and legacy repositories were not modified.
