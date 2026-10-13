# Conjunctival ROI haemoglobin predictor

This package trains a continuous Hb (g/dL) regressor using the shared capture
dataset, age, and sex. The loader accepts the canonical
`datasets/captured_data.csv` export directly.

## Dataset schema

The CSV columns are `RegistrationNo`, `Gender`, `AgeInYears`, `Haemoglobin`,
and `Images`. The loader converts them to the model fields, matches each
labelled registration to `{RegistrationNo}.jpg` in the shared datasets
directory, and resizes the available reference image to 224x224. Rows without
Hb or without a matching image are skipped. These images are provisional ROIs
until a dedicated conjunctival-ROI export is created.

## Train

```powershell
cd D:\BTP\BTP-Anemia-Model-Suite\algorithms\HBPredictor
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts\train.py --manifest ..\..\datasets\captured_data.csv --output-dir runs\captured_data
```

The split is patient-level (70/15/15). The run directory contains split CSVs,
training-only demographic statistics, the best checkpoint, predictions, and
regression metrics. Use `--no-pretrained` when ImageNet weights are unavailable.

This is a research prediction aid, not a validated diagnostic device.
