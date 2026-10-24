# Matched ROI Predictor Experiment

Date: 2026-10-22

## 1. Label audit

The authoritative source is `datasets/captured_data.csv`, joined by
`RegistrationNo`/image stem. All 183 source IDs matched uniquely. The seven
previously flagged IDs (`19-`, `2268`, `2428`, `308-3407`, `3142-2`, `43071-2`,
and `4380-2`) are present in the CSV, but their `Haemoglobin` fields are
genuinely blank. No labels were inferred, imputed, or mapped by row order.

## 2. Matched dataset

The joined manifest is [matched_manifest.csv](./matched_manifest.csv). Excluded
rows and reasons are in [exclusions.csv](./exclusions.csv), with the summary in
[matched_manifest.audit.json](./matched_manifest.audit.json).

| Split | Usable matched records |
|---|---:|
| Train | 122 |
| Validation | 27 |
| Test | 25 |
| **Total** | **174** |

The nine excluded records comprise the seven missing-label rows and the
non-overlapping failed extraction rows: production extraction failure
`112-3643.jpg` and Attention extraction failure `1951.jpg`. Successful ROI
files were required to exist and be `224x224x3` `uint8` for both extractors.
Original split membership was preserved.

## 3. Training configuration and commands

Both models are ResNet18 regressors initialized with ImageNet-pretrained
weights and trained with the same data IDs, labels, splits, preprocessing,
augmentation, loss, optimizer, schedule, seed, and stopping policy:

- 8-epoch budget; batch size 16; seed 42
- AdamW, learning rate `0.0003`, weight decay `0.0001`
- Smooth L1 loss with severe-Hb weight `1.2`
- ReduceLROnPlateau scheduler and validation-MAE early stopping, patience 3
- 224x224 RGB tensors with the existing ImageNet normalization in the model

Commands run:

```powershell
python -m predictor.build_matched_experiment_manifest `
  --production predictor\experiments\roi_provenance\production_manifest.csv `
  --attention predictor\experiments\roi_provenance\attention_manifest.csv `
  --output predictor\experiments\matched_roi_predictors\matched_manifest.csv

python -m predictor.train_matched `
  --manifest predictor\experiments\roi_provenance\production_manifest.csv `
  --paired-manifest predictor\experiments\roi_provenance\attention_manifest.csv `
  --output predictor\experiments\matched_roi_predictors\production_trained `
  --seed 42 --epochs 8 --batch-size 16 --lr 0.0003 `
  --weight-decay 0.0001 --patience 3

python -m predictor.train_matched `
  --manifest predictor\experiments\roi_provenance\attention_manifest.csv `
  --paired-manifest predictor\experiments\roi_provenance\production_manifest.csv `
  --output predictor\experiments\matched_roi_predictors\attention_trained `
  --seed 42 --epochs 8 --batch-size 16 --lr 0.0003 `
  --weight-decay 0.0001 --patience 3
```

Experimental checkpoints:

- [production_trained/best.pt](./production_trained/best.pt), best epoch 2
- [attention_trained/best.pt](./attention_trained/best.pt), best epoch 7

## 4. Held-out four-way evaluation

Evaluation used the same 25 matched test images for every combination. Bias is
`prediction - reference Hb`; positive values indicate overprediction.

| Predictor trained on | ROI evaluated | N | MAE (g/dL) | RMSE (g/dL) | Bias (g/dL) | Failures |
|---|---|---:|---:|---:|---:|---:|
| Production U-Net | Production U-Net | 25 | 1.401 | 1.904 | +0.171 | 0 |
| Production U-Net | Attention U-Net | 25 | 1.385 | 1.799 | +0.439 | 0 |
| Attention U-Net | Production U-Net | 25 | 1.363 | 1.710 | +0.719 | 0 |
| Attention U-Net | Attention U-Net | 25 | 1.146 | 1.456 | +0.327 | 0 |

Full per-image predictions and absolute errors are in
[per_image.csv](./evaluation/per_image.csv); aggregate results are in
[summary.json](./evaluation/summary.json).

For the intended production-versus-candidate comparison
(Attention-trained/Attention-ROI minus Production-trained/Production-ROI),
the mean paired absolute-error difference was **−0.255 g/dL**, with a
deterministic 20,000-resample paired bootstrap 95% interval of
**[−0.683, +0.142] g/dL**.

## 5. Limitations and recommendation

The current experiment used validation data for checkpoint selection and
early stopping; the test split was evaluated only afterward. The test set is
small (25 images), and patient identifiers are unavailable, so
patient-independent generalization has not been demonstrated. Earlier project
work repeatedly used a separate 27-image validation split for exploratory
downstream comparison; that historical test-set limitation is disclosed here,
although this matched run retained the fixed test split for final evaluation.

The Attention-trained predictor substantially improves the point estimate on
Attention ROIs, but the paired confidence interval includes no improvement.
The candidate pipeline therefore merits further controlled evaluation, not
promotion based on this experiment alone. Keep the production U-Net as the
default.

Production checkpoints, extractor defaults and threshold, segmentation split,
and production predictor checkpoint were not modified or replaced.
