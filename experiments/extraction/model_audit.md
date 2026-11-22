# Extractor model audit

## Current implementation

The extraction workflow is implemented in
[`src/extraction_pipeline.py`](D:/BTP/BTP-Anemia-Final/experiments/extraction/src/extraction_pipeline.py).
It provides:

- Read-only validation of the 288 verified image-mask pairs.
- EXIF-aware RGB image loading with `Pillow.ImageOps.exif_transpose`.
- Exact capture-timestamp mapping to the 288 unique `RegistrationNo` values.
- Reproducible 202/43/43 patient-wise train/validation/test splits.
- Attention U-Net with an ImageNet-pretrained EfficientNet-B0 encoder.
- BCE-with-logits plus soft Dice loss, AdamW optimization, CPU smoke testing,
  early stopping, checkpointing, test evaluation, and full-dataset inference.

The stored manual masks are never rewritten. The old annotation-assistance
checkpoint is not used for initialization.

## Dataset validation

The current validation report is
[`dataset_validation.json`](D:/BTP/BTP-Anemia-Final/experiments/extraction/reports/dataset_validation.json).

- Source images: **288**
- Verified manifest rows: **288**
- Draft/pending rows: **0/0**
- Readable images and masks: **288/288**
- Binary, non-empty masks: **288/288**
- Oriented dimension mismatches: **0**
- Missing, orphaned, duplicate IDs, or duplicate stems: **0**

The 89 apparent dimension mismatches are resolved by applying EXIF Orientation 6
to the source images. The masks remain unchanged.

## Training and evaluation

The split is recorded in
[`split_summary.json`](D:/BTP/BTP-Anemia-Final/experiments/extraction/reports/split_summary.json).
The run used seed `20261116`, batch size `2`, and early stopping after 25
epochs. The best validation Dice was **0.8301** at epoch 18.

Held-out test results for 43 images are recorded in
[`test_evaluation.json`](D:/BTP/BTP-Anemia-Final/experiments/extraction/reports/test_evaluation.json):

- Dice: **0.8273 ± 0.0694**
- IoU: **0.7112 ± 0.0962**
- Precision: **0.8180 ± 0.1039**
- Recall: **0.8555 ± 0.1123**
- Specificity: **0.9987 ± 0.0009**
- Pixel accuracy: **0.9975 ± 0.0019**

These are held-out test metrics, not training metrics.

## Inference outputs

Using the best validation checkpoint, full-dataset inference generated 288
binary masks, 288 overlays, and 288 ROI crops with zero failures and zero
empty predictions. Outputs are separated under:

- `experiments/extraction/predicted_masks/`
- `experiments/extraction/overlays/`
- `experiments/extraction/roi_crops/`

No separate all-288 deployment-model training run was performed.
