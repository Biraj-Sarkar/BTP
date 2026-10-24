# Segmentation training

This package trains `unet`, `unet++`, and `attention_unet` using real image/mask pairs. A valid dataset must contain sibling `images/` and `masks/` directories with identical filename stems. Masks are binary (`>0` is foreground); predicted ROI crops are not valid masks. Splits are grouped by image parent/stem group and use seed 42 by default.

The current annotated dataset is `segmentation_dataset/` (183 images and 183 reviewed masks). The validation gate reports 183 eligible samples and training is permitted. The reproducible CPU benchmark was run with the same split and settings for all architectures:

```powershell
python -m extractor.training.validate_annotations --root segmentation_dataset
python -m extractor.training.train --data segmentation_dataset --model unet --output extractor/experiments --epochs 10 --patience 3 --size 224 --batch-size 8 --device cpu
python -m extractor.training.train --data segmentation_dataset --model unet++ --output extractor/experiments --epochs 10 --patience 3 --size 224 --batch-size 8 --device cpu
python -m extractor.training.train --data segmentation_dataset --model attention_unet --output extractor/experiments --epochs 10 --patience 3 --size 224 --batch-size 8 --device cpu
python -m extractor.training.compare --root extractor/experiments
```

The fixed split is 129 training, 27 validation, and 27 held-out test images. Validation Dice selected `unetplusplus` (0.0156; the other values were 0.0002 and 0.0000), so only that checkpoint was evaluated on the held-out test set. Its test metrics are recorded by `extractor.training.evaluate` and in `extractor/experiments/unetplusplus/test_predictions.csv`. These preliminary scores are poor and should not be treated as production quality; collect better masks and/or tune the training recipe before deployment.

Comparison artifacts are written to `extractor/experiments/comparison/summary.csv`, `summary.json`, and `report.md`. The selected checkpoint can be loaded by the existing inference interface, while the legacy `Models/UNet_extractor/unet.pt` remains unchanged.
