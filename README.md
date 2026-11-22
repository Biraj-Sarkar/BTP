# BTP Anemia

This project contains the anemia image-processing application, datasets, models,
experiments, and reports. Original hospital images, captured metadata, manifests,
splits, extraction models, prediction models, backend files, and Flutter files are
preserved.

## Project layout

```text
backend/
datasets/
  manifests/
  processed/
  raw/hospital/images/
  raw/hospital/captured_data.csv
  splits/
docs/
experiments/
  annotation/
    scripts/
    masks/drafts/
    masks/verified/
    masks/reports/
  extraction/
  prediction/
flutter_app/
models/
  annotation/
  extraction/
  prediction/
reports/
requirements.txt
```

## Annotation

Manual annotation is the only active annotation workflow. The verified masks in
`experiments/annotation/masks/verified/` are the ground-truth labels, and the
active GUI manifest is
`experiments/annotation/masks/reports/annotation_manifest.csv`.

Old model-assisted predictions, model-selection reports, and pilot annotation
reports have been archived under `D:\BTP\archive\annotation\`. The Attention
U-Net belongs to the extraction/segmentation phase; it is not part of the active
manual annotation workflow.

See [`experiments/annotation/README.md`](experiments/annotation/README.md) for
the active GUI workflow and current audit.

The extraction phase uses a fresh EXIF-aware Attention U-Net with an ImageNet
EfficientNet-B0 encoder. Its training, evaluation, prediction, and ROI outputs
are kept under `experiments/extraction/`; it does not modify manual masks.
See [`experiments/extraction/README.md`](experiments/extraction/README.md).

The annotation cleanup did not start training; extraction training is documented
separately in `experiments/extraction/README.md`.
