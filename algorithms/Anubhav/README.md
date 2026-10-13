# Anubhav conjunctival Hb pipeline

This directory contains the image preprocessing, conjunctiva extraction,
quality gate, and Hb regression pipeline. It now uses only the shared dataset
at `../../datasets/`.

## Canonical dataset

`datasets/captured_data.csv` has these columns:

- `RegistrationNo` – patient identifier and image stem
- `Gender` – demographic sex input
- `AgeInYears` – demographic age input
- `Haemoglobin` – supervised Hb target in g/dL
- `Images` – original export metadata; the loader matches the staged
  `{RegistrationNo}.jpg` reference image

Rows without Hb or without a matching image are excluded. The loader groups
captures by `RegistrationNo` so no patient can leak across validation folds.

## Inspect the shared data

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m anemia inspect --data ..\..\datasets\captured_data.csv
```

## Train the Anubhav Hb model

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m anemia train-hb `
  --data ..\..\datasets\captured_data.csv `
  --out runs\captured_data `
  --epochs 30 `
  --folds 3
```

The current reference captures do not include segmentation masks, so this run
uses the built-in colour-based conjunctiva extractor. Outputs are written only
under `runs/` and include configuration, cross-validated metrics, and
out-of-fold predictions.

## CIELAB pipeline

`cielab_pipeline.py` and `cielab_stages.py` remain available for the CIELAB
preprocessing/extraction path. See `README_cielab_pipeline.md` for its command
line options. It should use the same images under `../../datasets/`.

## Notes

This is research software and the small captured cohort is not a clinical
validation set. Predictions must not be treated as a diagnosis.
