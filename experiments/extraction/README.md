# Conjunctiva extraction

The completed extraction workflow uses an Attention U-Net with an ImageNet-
pretrained EfficientNet-B0 encoder for conjunctiva segmentation. It is separate
from the manual annotation workflow and does not write into the verified
annotation directory.

## Dataset and preprocessing

The dataset contains 288 hospital images paired with 288 verified manual masks
from the read-only active manifest:
`experiments/annotation/masks/reports/annotation_manifest.csv`.

Images are loaded with Pillow `ImageOps.exif_transpose`, converted to RGB, and
resized to 320x320 with bilinear interpolation. Masks are loaded as grayscale
PNG files without EXIF transformation and resized with nearest-neighbor
interpolation. This preserves alignment for the 89 portrait JPEGs whose EXIF
Orientation 6 changes their displayed dimensions.

The saved patient-wise manifests use `RegistrationNo` as the grouping key:

- Train: 202 images
- Validation: 43 images
- Test: 43 images

The split manifests are under `experiments/extraction/manifests/`, and the
counts and zero-overlap check are recorded in
`experiments/extraction/reports/split_summary.json`.

## Completed training and evaluation

The completed CPU run used seed `20261116`, batch size `2`, BCE-with-logits plus
soft Dice loss, AdamW, and early stopping. The best checkpoint was selected at
epoch 18 with validation Dice `0.8301`.

On the untouched 43-image test split, the saved report
`experiments/extraction/reports/test_evaluation.json` records:

- Dice: mean `0.8273228591`, standard deviation `0.0693862543`
- IoU: mean `0.7111709134`, standard deviation `0.0961989404`
- Precision: mean `0.8179871203`, standard deviation `0.1038808753`
- Recall: mean `0.8555273022`, standard deviation `0.1122925706`
- Specificity: mean `0.9987227615`, standard deviation `0.0009279541`
- Pixel accuracy: mean `0.9974902344`, standard deviation `0.0019417090`

## Inference outputs

Full-dataset inference generated 288 predicted binary masks, 288 overlays, and
288 ROI crops with zero failures and zero empty predictions. For each image,
the predicted probability map is thresholded at `0.5`, resized back to the
oriented image dimensions with nearest-neighbor interpolation, and used to
find the foreground bounding box. The ROI is the oriented RGB image cropped to
that box with 20 pixels of padding, clipped to the image boundaries.

The active pipeline and model outputs remain under `experiments/extraction/`,
with checkpoints under `models/extraction/`. No separate deployment model
trained on all 288 images was created.

## Archived quality assessment

The completed quality assessment and detailed visual comparisons are archived
at `D:\BTP\archive\BTP-Anemia-Final_extraction_quality_assessment`. The archive
retains test-set comparisons, worst-performing test cases, suspicious
predictions, contact sheets, per-image metrics, and quality summaries for
documentation and research reproducibility. The active extraction pipeline
does not depend on this archived material.
