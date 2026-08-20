# BTP — Non-Invasive Anemia Detection

> **Documentation:** `PIPELINE.md` explains how the method works, stage by
> stage. `CODE_GUIDE.md` explains what every file does and the design rationale
> behind each stage. `RESEARCH_LOG.md` records every approach tried including
> those that failed, with reasons. `GLOSSARY.md` defines every abbreviation.
> `WORKLOG.md` is the week-by-week record. `CONTRIBUTING.md` explains how to
> add an extractor, a feature representation, a metric or a dataset.
> `UI_INTEGRATION.md` is the step-by-step guide for building the client
> against the API, and is the only document the app developer needs.

Estimate haemoglobin from a photograph of the palpebral conjunctiva, and screen
for anemia against the patient's WHO threshold.

The app is a thin client: the phone captures the eye and uploads it, a server
segments the conjunctiva and runs the regressor, and the response carries an Hb
estimate in g/dL plus an anemic / not-anemic verdict.

> **Status: no CNN has been trained on real data yet.** The linear model fitted
> on all 26 patients reaches **MAE 0.944 g/dL, R² 0.133, F1 0.727, accuracy
> 0.654** (pipeline A), against a predict-the-mean baseline of MAE 1.041,
> R² 0.000, F1 0.722. Held out on unseen patients it gives MAE 1.062,
> R² −0.094 — the gap is overfitting, and the two candidate pipelines are
> **not statistically separable** at this sample size (paired test p = 0.408).
> Full comparison: `experiments/11_head_to_head/RESULTS.md`. **No accuracy
> figure in this repo is a clinical result.**

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

On macOS, Python's SSL store may be missing, which breaks the pretrained-weight
download. If you see `CERTIFICATE_VERIFY_FAILED`:

```bash
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")
```

## Data

Neither dataset is redistributed here — download them yourself.

| Dataset | Images | Masks | Notes |
|---|---|---|---|
| [Eyes-defy-anemia](https://ieee-dataport.org/documents/eyes-defy-anemia) ([Kaggle mirror](https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia)) | 218 | yes | India + Italy, adults. The only source of pixel-level ground truth. |
| [CP-AnemiC](https://data.mendeley.com/datasets/m53vz6b7fx/1) | 710 | no | Ghana, children 6–59 months. |
| Local cohort (`left_eye/`, `right_eye/`, `DATASAMPLE.csv`) | 52 | no | 26 patients, both eyes. Hb + full blood count. Children 3–17y, 50% anemic. |

The segmenter can only be trained on Eyes-defy-anemia. It is then applied to
CP-AnemiC and to the hospital data to produce crops for the regressor.

Check any new dataset before training on it:

```bash
python -m anemia inspect --data data/eyes-defy-anemia
python -m anemia inspect --data .          # the local cohort
```

If the image count does not match the published figure, the loader did not match
the layout — fix that before anything else.

## Running the pipeline

```bash
python -m anemia train-segmenter --data data/eyes-defy-anemia --out runs/segmenter
```

```bash
python -m anemia train-hb --data data/eyes-defy-anemia --data data/cp-anemic --segmenter runs/segmenter --out runs/hb
```

```bash
python -m anemia predict --checkpoint runs/hb/best.pt --image photo.jpg --age 34 --sex F
```

Quantify the segmentation rework against real ground truth:

```bash
python scripts/benchmark_segmenters.py --data data/eyes-defy-anemia --segmenter runs/segmenter
```

## Checking segmentation on real photographs

No trained model or network access needed. Works on any folder of eye photos:

```bash
python -m anemia debug --dir left_eye --out debug/left
```

Writes `quality_report.csv` (per-image focus, clipping, mask ratio, pass/fail)
and `overlay_sheet.png`. **Look at the overlay sheet** — green must sit on the
red inner lid, not on lashes or cheek skin. Mask ratio alone will not tell you
this. This check exposed the plain colour heuristic bleeding onto lashes and
skin, which is why extraction now runs through the refined seeded-grabCut
extractor (`refined_conjunctiva_mask`); the current per-image results live in
`experiments/02_extraction_check/`.

For a single image:

```bash
python -m anemia debug --image photo.jpg --out debug
```

## Stage-by-stage walkthrough

To see exactly what the pipeline does to a photo, stage by stage:

```bash
python -m anemia stages --dir left_eye --out experiments/01_stage_traces/main_pipeline
```

Each capture produces **one contact sheet** (`<name>_stages.jpg`) with every
stage side by side, so each can be compared against the one before it without
opening separate files. Panels read left to right, top to bottom:

```
00 original capture        05 mask over photo
01 resized to 512x512      06 outside mask removed
02 gray-world white balance 07 tight crop
03 redness map (a*-0.5b*)  08 letterboxed to square
04 segmentation mask       09 model input 224x224
```

The header strip carries the filename, segmentation backend, mask ratio, focus
score and QC verdict. Panels are letterboxed rather than stretched, so the
crescent geometry the pipeline protects is not misrepresented in the figure.

Options: `--image` for a single photo, `--segmenter` for a trained model,
`--panel` / `--per-row` to change the layout, and `--separate` to additionally
write each stage as its own file. Pre-generated traces live in
`experiments/01_stage_traces/`; see `experiments/README.md` for the full index.

## Predicting on new patients (no training)

Fitting and predicting are separate. The model is fitted **once** and saved as a
plain JSON file carrying its coefficients, feature specification, extractor
name and standardisation constants. Prediction loads that file — nothing is
trained, so it runs in seconds and is reproducible.

**Step 1 — fit** (only when new *labelled* data arrives):

```bash
python -m anemia fit-linear --data . --out runs/linear_model.json
```

Options: `--representation` (means / chroma / erythema / lab / a_only),
`--extractor` (refined / redness / brightness / grabcut / cielab), `--alpha`,
or `--segmenter <dir>` for a trained segmentation model.

**Step 2 — predict** (any time, any number of patients, no labels needed):

```bash
python predict_folder.py --dir new_patients --out predictions.csv
```

Expected layout:

```
new_patients/
    left_eye/   1.jpg  2.jpg  ...
    right_eye/  1.jpg  2.jpg  ...
    patients.csv        (optional: ID, Date Of Birth or Age, Gender)
```

Filenames are the patient ID; both eyes are averaged when present, exactly as
during fitting. Output columns: `patient, hb_g_dl, anemic, threshold_g_dl,
usable, eyes_used, age_years, sex, note`.

The extractor is read back from the model file rather than defaulting, because
features measured inside a different mask are not comparable.

## Serving

```bash
ANEMIA_CHECKPOINT=runs/hb/best.pt ANEMIA_SEGMENTER=runs/segmenter uvicorn serve.app:app --port 8000
```

`POST /predict` takes an image plus optional `age_years`, `sex`, `pregnant`, and
returns `hb_g_dl`, `anemic`, `threshold_g_dl`, `usable`, and a `quality` block.
A capture that fails QC returns `usable: false` with a retake message — the app
should show that rather than a number.

Building the client: **`UI_INTEGRATION.md`** carries the full contract, both
response shapes, a runnable mock server so the app can be built before a model
file is available, and the screen-by-screen requirements.

## Developing without the datasets

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic --patients 40
python -m anemia train-hb --data data/synthetic --out runs/smoke --epochs 12 --folds 3
pytest tests/ -q
```

The phantoms are flat-coloured and trivially separable. They prove the plumbing
works; they prove nothing about clinical accuracy.

## Layout

```
src/anemia/
  config.py           tunables; serialised next to every checkpoint
  imaging.py          colour ops, QC scores, letterbox crop (no torch)
  segment.py          segmentation backends behind one callable contract
  preprocess.py       capture -> model-ready crop, with the QC gate
  data.py             dataset loaders + WHO thresholds
  splits.py           patient-grouped, Hb-stratified CV
  metrics.py          regression / screening / segmentation metrics
  model.py            regressor + self-describing checkpoint
  train.py            regressor training
  train_segmenter.py  Mask2Former fine-tuning
  predict.py          serving path (imports nothing from train)
serve/app.py          FastAPI service
scripts/              benchmarking and synthetic data
```

## Honest reporting

Small cohorts make it easy to report a number that means nothing, so the
pipeline reports these by construction:

- **Baseline comparison.** Always-predict-the-mean is reported next to model
  MAE. On a cohort clustered near normal Hb, that baseline is already decent.
- **Cross-validated mean ± std**, patient-grouped, not a single lucky split.
- **Out-of-fold predictions only** in the results CSV.
- **Sensitivity and specificity**, not just accuracy — a missed anemic patient
  is the costly error.
- **Bland–Altman limits of agreement**, the standard comparison against a lab
  reference in the clinical literature.

Full metric definitions, the mechanism that produces the Hb value, and current
numbers on both synthetic and real data:
`experiments/06_metrics_evaluation/METRICS.md`.
