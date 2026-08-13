# BTP — Non-Invasive Anemia Detection

> **Documentation:** `PIPELINE.md` explains how the method works, stage by
> stage. `CODE_GUIDE.md` explains what every file does and the design
> rationale behind each stage.

Estimate haemoglobin from a photograph of the palpebral conjunctiva, and screen
for anemia against the patient's WHO threshold.

The app is a thin client: the phone captures the eye and uploads it, a server
segments the conjunctiva and runs the regressor, and the response carries an Hb
estimate in g/dL plus an anemic / not-anemic verdict.

> **Status: the pipeline runs end to end on synthetic phantoms. No model has
> been trained on real data yet, and no accuracy figure in this repo should be
> quoted until it has.**

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
`figures/extraction_left_eye/` and `figures/extraction_right_eye/`.

For a single image:

```bash
python -m anemia debug --image photo.jpg --out debug
```

## Serving

```bash
ANEMIA_CHECKPOINT=runs/hb/best.pt ANEMIA_SEGMENTER=runs/segmenter uvicorn serve.app:app --port 8000
```

`POST /predict` takes an image plus optional `age_years`, `sex`, `pregnant`, and
returns `hb_g_dl`, `anemic`, `threshold_g_dl`, `usable`, and a `quality` block.
A capture that fails QC returns `usable: false` with a retake message — the app
should show that rather than a number.

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
