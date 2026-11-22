# Legacy hemoglobin model review

## Scope and evidence standard

This review covers the hemoglobin-prediction code, configurations, saved
checkpoints, experiment reports, and logs inspected in
`D:\BTP\BTP-Anemia-Model-Suite`. The legacy repository was treated as
research context only; no legacy code, data, or weights were copied into the
current project.

Observed facts are stated separately from recommendations. A model is marked
**trained** only when a saved run, checkpoint, and/or result report was found.
An implementation without a corresponding run is marked **implemented only**.

## 1. Legacy model inventory

### 1.1 Main `predictor/` framework

The framework defines a common image-regression interface in
`predictor/models/common.py`. All models:

- accept a 3-channel RGB tensor;
- use 224 x 224 input transforms;
- apply ImageNet mean and standard deviation normalization;
- predict one continuous standardized Hb value through a scalar regression head;
- store training target mean and standard deviation for inverse conversion to
  g/dL.

| Model | Architecture and head | Backbone initialization | Status |
|---|---|---|---|
| ResNet18 | ResNet-18 with dropout 0.3 and `Linear(features, 1)` | ImageNet pretrained | **Trained**: `predictor/runs/resnet/` |
| ResNet34 | ResNet-34 with dropout 0.3 and `Linear(features, 1)` | ImageNet pretrained option | **Implemented only**; no ResNet34 run found |
| VGG16 | VGG-16 features, adaptive average pool, `Linear(512,128)`, ReLU, dropout 0.3, scalar output | ImageNet pretrained option | **Implemented only**; no VGG16 run found |
| VGG16-BN | VGG-16 with batch normalization, same compact regression head | ImageNet pretrained | **Trained**: `predictor/runs/vgg/` |
| MobileNetV3-Large | MobileNetV3-Large features, `Linear -> Hardswish -> Dropout -> Linear(1)` | ImageNet pretrained | **Trained**: `predictor/runs/mobilenet/` |
| MobileNetV3-Small | MobileNetV3-Small with the same MobileNet head | ImageNet pretrained option | **Implemented only**; no MobileNetV3-Small run found |

The model definitions are in
`predictor/models/resnet.py`, `predictor/models/vgg.py`, and
`predictor/models/mobilenet.py`. The framework is image-only for the
regression head: age and sex are parsed into records but are not concatenated
with image features.

### 1.2 Matched ROI predictor experiment

`predictor/experiments/matched_roi_predictors/` is not a new architecture
family. It trains two ResNet18 instances, one on production-U-Net ROIs and one
on Attention-U-Net ROIs, using the same fixed image IDs and configuration.
Both are ImageNet-pretrained ResNet18 regressors with a 0.3 dropout head.

The matched manifest contains 183 source IDs and 174 matched training-eligible
records: 122 train, 27 validation, and 25 test. The seven missing-Hb rows and
two extraction-failure rows were excluded. The report states that patient IDs
were unavailable; `train_matched.py` consequently constructs each
`SampleRecord` with `patient_id=image_id`. This is an image-level split, not
proof of patient-independent generalization.

### 1.3 `algorithms/Anubhav/` models

The separate Anubhav pipeline contains three predictor approaches:

1. **RGB ResNet18 Hb regressor**: `src/anemia/model.py` defines an ImageNet
   ResNet18 with a dropout-0.3 scalar head. The default configuration in
   `src/anemia/config.py` uses `pretrained=true`, freezes through `layer3`,
   and fine-tunes the remaining layers.
2. **Two-channel CIELAB ResNet18**: `cielab_pipeline.py` changes the first
   convolution to two channels, keeps only OpenCV LAB `a*` and `b*`, and
   replaces the classifier with dropout 0.3 plus one scalar output. If
   pretrained weights are used, the original three-channel `conv1` weights
   are averaged and duplicated into two channels; `layer4`, `fc`, and the new
   first convolution remain trainable.
3. **Ridge colour-statistics model**: `src/anemia/linear_model.py` implements
   ridge regression over masked colour representations: raw means, chromaticity,
   erythema ratios, LAB `a*/b*`, or LAB `a*` alone. It is implemented as a
   self-describing JSON model, but no saved trained result was found in the
   inspected run artifacts.

The actual saved Anubhav configurations under `runs/` use the RGB ResNet18
with `pretrained=false`, 224 x 224 input, AdamW, learning rate `1e-4`,
weight decay `1e-4`, batch size 16, 30 epochs, weighted Smooth L1 loss,
inverse-frequency Hb-bin weights, cosine annealing, and patient-grouped
three-fold cross-validation with a patient-grouped 20% inner holdout for early
stopping.

## 2. Architecture and training comparison

### 2.1 Main predictor runs

These are actual saved test results from the three completed runs. They use
the generic `predictor/train.py` pipeline and its fixed configuration files.

| Run | Best epoch | Batch | Learning rate | Epoch budget | Test N | MAE | RMSE | R² | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 7 | 16 | 0.0003 | 8 | 27 | 1.097 | 1.397 | 0.4234 | 0.789 | 0.500 |
| VGG16-BN | 2 | 16 | 0.0001 | 2 | 27 | 1.389 | 1.814 | 0.0285 | 1.000 | 0.000 |
| MobileNetV3-Large | 5 | 16 | 0.0003 | 8 | 27 | 1.427 | 1.849 | -0.0098 | 1.000 | 0.000 |

The full metric records are in
`predictor/runs/resnet/metrics.json`,
`predictor/runs/vgg/metrics.json`, and
`predictor/runs/mobilenet/metrics.json`. The runs are not directly comparable
to the current 288-image extraction test report: they use the legacy ROI
manifest and 27-image test split.

### 2.2 Matched ROI results

The matched experiment evaluated the same 25 test images in all four
train/evaluate combinations:

| Predictor trained on | ROI evaluated on | MAE | RMSE | Bias |
|---|---|---:|---:|---:|
| Production U-Net | Production U-Net | 1.401 | 1.904 | +0.171 |
| Production U-Net | Attention U-Net | 1.385 | 1.799 | +0.439 |
| Attention U-Net | Production U-Net | 1.363 | 1.710 | +0.719 |
| Attention U-Net | Attention U-Net | 1.146 | 1.456 | +0.327 |

The intended paired difference, Attention-trained/Attention-ROI minus
Production-trained/Production-ROI, was -0.255 g/dL with a deterministic
20,000-resample interval of [-0.683, +0.142] g/dL. The interval included zero.
The report explicitly recommends further controlled evaluation rather than
promoting the candidate.

### 2.3 Anubhav saved cross-validation results

The saved `captured_data_full` run used 57 captured images, all 57 passing its
quality gate, and three patient-grouped folds. Its mean +/- fold standard
deviation was:

- MAE: 1.653 +/- 0.149 g/dL
- RMSE: 2.065 +/- 0.174 g/dL
- R²: -0.032 +/- 0.095
- Sensitivity: 0.823 +/- 0.157
- Specificity: 0.089 +/- 0.078

The saved `roi_training` run used 246 quality-passed ROIs and three
patient-grouped folds:

- MAE: 1.825 +/- 0.214 g/dL
- RMSE: 2.263 +/- 0.279 g/dL
- R²: 0.013 +/- 0.240
- Sensitivity: 0.570 +/- 0.268
- Specificity: 0.568 +/- 0.182

These are cross-validation summaries, not held-out test results. The
`captured_data_warmup` run is also a three-fold cross-validation run and
records MAE 1.721 +/- 0.066 g/dL, RMSE 2.078 +/- 0.070 g/dL, and R²
-0.046 +/- 0.050. No CIELAB-specific saved Hb metric report was identified in
the inspected artifacts, so its performance is unavailable.

## 3. Data preparation and leakage review

### Observed useful practices

- `predictor/data/manifest.py` joins labels through explicit columns and image
  paths, validates Hb values in the range 3 to 25 g/dL, rejects missing Hb,
  checks ROI files, and normalizes sex values.
- The generic splitter groups by the record's `patient_id`, stratifies by
  patient-level mean Hb, verifies zero patient overlap, and computes target
  normalization from training records only.
- The Anubhav loader maps the local cohort by `RegistrationNo`, retains both
  eyes where present, and documents that both eyes must share a patient group.
- Anubhav training creates an inner training-only validation holdout for
  early stopping and evaluates each outer test fold only after training.
- Both pipelines save target-normalization constants and preprocessing metadata
  with checkpoints.
- The Anubhav pipeline emits out-of-fold predictions and reports performance
  against a training-mean baseline.

### Weaknesses and leakage risks

- The generic predictor's patient grouping is only as good as the supplied
  `patient_id`. The matched ROI experiment explicitly lacks patient IDs and
  sets `patient_id=image_id`, so its 25-image test result cannot establish
  patient-independent performance.
- The generic predictor uses a single 15% validation and 15% test split and
  one seed. The model comparison script selects a winner by test MAE if several
  run directories are present; this is unsuitable for unbiased architecture
  selection.
- Generic training uses `ColorJitter` for brightness, contrast, and saturation
  in `predictor/data/transforms.py`. This is inappropriate for a controlled
  white-balance comparison because it can reduce sensitivity to the colour
  intervention.
- Generic clinical metrics use a fixed 12.0 g/dL cutoff. They do not apply the
  age- and sex-specific WHO thresholds implemented by
  `algorithms/Anubhav/src/anemia/data.py`.
- The backend wrapper applies only adult male/female thresholds and accepts an
  `age_years` argument without using it to select pediatric cutoffs.
- The legacy cohort is small and imbalanced. Several reported runs have
  specificity near zero, negative R², or performance close to a mean-prediction
  baseline.
- Some source pipelines have one photograph per `RegistrationNo`, while other
  loaders support multiple captures or both eyes. The grouping key must be
  verified rather than inferred from filenames.
- The generic matched experiment preserves an existing split rather than
  proving patient-level independence. Its results are therefore useful for
  pipeline comparison but not for a clinical generalization claim.

## 4. White-balance experiment review

### Proposed controlled design

`WhiteBalanceTest_1_Steps.txt` and `WhiteBalanceTest_2_Prompt.txt` propose two
arms using the same photographs: raw crops versus gray-world-balanced crops.
The proposed controls are:

- intersect the extractor acceptance sets so both arms use identical images;
- split by patient with five folds;
- use identical folds, seeds, architecture, pretrained initialization,
  optimizer, learning rate, batch size, loss, normalization, and stopping rule;
- disable all colour augmentation, including brightness, contrast, saturation,
  hue, gamma, random grayscale, extra white balance, and per-image colour
  normalization;
- preserve the existing 224 x 224 crops without colour-changing resizing;
- produce out-of-fold predictions;
- average predictions over seeds and photographs at patient level;
- compare per-patient absolute errors with
  `d = |error_gw| - |error_raw|`;
- use a 2,000-resample patient bootstrap interval and a Wilcoxon signed-rank
  test;
- decide only from the confidence interval, not from mean MAE alone.

These are useful design principles for the new project. In particular, the
same-photo intersection, patient grouping, fixed seeds, colour-augmentation
ban, out-of-fold predictions, paired patient-level analysis, and explicit
baseline/error-noise reporting should be retained.

### Required adaptation

The documents assume two extractors and two extractor checkpoints. The current
project has one active EXIF-aware Attention U-Net extraction pipeline and its
288 saved predicted masks, overlays, and ROI crops. The old two-extractor
experiment must therefore not be copied literally and its old extractor
checkpoints must not be imported.

A current-project experiment should instead:

1. define raw and gray-world crop generation from the same current source
   photographs and the same current segmentation result, or explicitly
   document a new segmentation comparison;
2. derive a verified patient identifier from the current laboratory metadata
   before splitting;
3. use five patient-grouped outer folds with an inner validation procedure or
   fixed training schedule;
4. turn off the legacy `ColorJitter` policy and any other photometric
   augmentation;
5. preserve current EXIF orientation and 224 x 224 crop semantics;
6. run multiple identical seeds and emit out-of-fold predictions for every
   arm;
7. apply the age/sex-specific anemia threshold function, including the
   documented unknown-demographic fallback;
8. report paired patient-level results and the predeclared bootstrap and
   Wilcoxon analyses.

## 5. Recommendations for the new hemoglobin predictor

These are recommendations, not completed current-project results:

1. Use the current 288-image annotation and extraction outputs as the starting
   image source; do not use the archived annotation-assistance model.
2. Build an auditable Hb/image manifest by explicit laboratory identifier and
   verify one-to-many photograph relationships before creating splits.
3. Make patient grouping a hard validation invariant. Refuse to run if a
   patient occurs in more than one outer fold.
4. Start with a clearly specified ResNet18 regression baseline because it is
   the only legacy architecture with repeated, inspectable implementations and
   saved runs. Treat this as a baseline choice, not evidence that it is the
   final architecture.
5. Compare alternatives only with identical patient-grouped folds, seeds,
   preprocessing, and stopping rules. Do not select architectures on the
   final test fold.
6. Keep target standardization statistics, image orientation, crop settings,
   augmentation policy, label mapping, and checkpoint provenance in one
   self-describing run record.
7. Report MAE, RMSE, R², bias, Bland-Altman limits, a mean-Hb baseline, and
   age/sex-specific anemia sensitivity and specificity. Include uncertainty
   across seeds or folds.
8. Treat the white-balance question as a separate paired experiment. Do not
   claim that the old two-extractor result transfers to the current single
   extraction pipeline.

## 6. Supporting legacy paths

Key source and evidence files inspected:

- `predictor/models/common.py`
- `predictor/models/resnet.py`
- `predictor/models/vgg.py`
- `predictor/models/mobilenet.py`
- `predictor/configs/resnet.yaml`
- `predictor/configs/vgg.yaml`
- `predictor/configs/mobilenet.yaml`
- `predictor/data/manifest.py`
- `predictor/data/splits.py`
- `predictor/data/transforms.py`
- `predictor/train.py`
- `predictor/train_matched.py`
- `predictor/training/trainer.py`
- `predictor/training/losses.py`
- `predictor/training/metrics.py`
- `predictor/runs/resnet/metrics.json`
- `predictor/runs/vgg/metrics.json`
- `predictor/runs/mobilenet/metrics.json`
- `predictor/experiments/matched_roi_predictors/FINAL_REPORT.md`
- `predictor/experiments/matched_roi_predictors/evaluation/summary.json`
- `algorithms/Anubhav/src/anemia/model.py`
- `algorithms/Anubhav/src/anemia/data.py`
- `algorithms/Anubhav/src/anemia/splits.py`
- `algorithms/Anubhav/src/anemia/train.py`
- `algorithms/Anubhav/src/anemia/metrics.py`
- `algorithms/Anubhav/src/anemia/linear_model.py`
- `algorithms/Anubhav/src/anemia/config.py`
- `algorithms/Anubhav/cielab_pipeline.py`
- `algorithms/Anubhav/runs/captured_data_full/metrics.json`
- `algorithms/Anubhav/runs/captured_data_warmup/metrics.json`
- `algorithms/Anubhav/runs/roi_training/metrics.json`
- `algorithms/Anubhav/README.md`
- `algorithms/Anubhav/README_cielab_pipeline.md`
- `WhiteBalanceTest_1_Steps.txt`
- `WhiteBalanceTest_2_Prompt.txt`

The two white-balance files were supplied outside the legacy repository and
were read from `C:\Users\Biraj\Downloads\`.
