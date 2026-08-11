# BTP — Non-Invasive Anemia Detection

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

The segmenter can only be trained on Eyes-defy-anemia. It is then applied to
CP-AnemiC and to the hospital data to produce crops for the regressor.

Check any new dataset before training on it:

```bash
python -m anemia inspect --data data/eyes-defy-anemia
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

## What changed from the inherited pipeline

The original is preserved at `gemini_anemia_pipeline.py` on `main`. The rework
was driven by these defects:

| Defect | Consequence |
|---|---|
| Heuristic selected bright, low-chroma pixels | Segmented the **sclera**, not the red conjunctiva. Zero overlap with ground truth on phantoms. |
| `local_files_only=True` on the Mask2Former load | Always failed on a clean machine, silently fell back to the heuristic; "trained segmentation" runs had trained nothing. |
| `accepted_for_training` computed, never read | QC was decorative; blurred captures trained the model. |
| `target_mean` / `target_std` never saved | Checkpoints could not be served — the network emits a standardised value with no way back to g/dL. |
| `all_predictions.csv` mixed train and test rows | Headline results were ~80% training-set predictions. |
| Segmentation re-run inside `__getitem__` | Every image re-segmented once per epoch. |
| `/255` with no ImageNet normalisation | Input distribution mismatched the pretrained weights. |
| `requires_grad = False` alone to freeze layers | BatchNorm running stats kept updating, drifting the frozen features. |
| Hardcoded `Hb < 11.0` | Correct only for young children and pregnant women; misses anemic adult men (threshold 13.0). |
| Split on images, sorted by patient number | Leakage across captures of the same eye; no shuffling. |
| One image per patient, capped at 50 | Used ~5% of available data. |

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
