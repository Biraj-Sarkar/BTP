# Code Guide

Two things: the design decisions behind the pipeline, and what every file does.
Read `PIPELINE.md` first for how the method works — this document is about the
codebase.

---

# Part 1 — Design decisions and rationale

The pipeline's design is driven by two constraints: an eventually deployed
screening app, and small clinical cohorts. Both punish silent failure, so the
recurring theme below is *fail loudly, measure honestly*.

## 1.1 Extraction targets redness, not brightness

The haemoglobin signal lives in the red palpebral conjunctiva. A
brightness-based mask (bright, low-chroma pixels) selects the *sclera* instead
— measured on phantoms it has zero overlap with the target tissue. Extraction
is therefore built on a redness score in CIELAB (`a* − 0.5·b*`), and the
brightness approach is kept only as a benchmark baseline
(`legacy_bright_neutral_mask`) so the choice is justified by a Dice comparison
rather than an assertion.

On real photographs a plain redness threshold is still not enough — lashes and
lid skin are red enough to pass it — which is what the refined seeded-grabCut
extractor addresses (see `segment.py` below and `PIPELINE.md` §3.2).

## 1.2 Model loads fail loudly

If a trained-segmenter load fails and the pipeline silently degrades to the
colour heuristic, a run can report "trained segmentation" results that are
nothing of the sort. An explicitly requested model directory that fails to
load therefore raises instead of falling back.

## 1.3 Quality control is a gate, not a note

QC (blur, mask area, clipping) is *enforced*: failing captures are dropped from
training, and at serve time the API returns `usable: false` with a retake
prompt. A confident number derived from a blurred photo is the most dangerous
output a screening tool can produce.

## 1.4 Checkpoints are self-describing

The network is trained on standardised labels, so its raw output needs the
training mean and standard deviation to become g/dL. `Checkpoint` bundles the
weights, those constants, the preprocessing settings, and the metrics in one
file — weights saved alone can never be served, and the loader refuses an
incomplete bundle.

## 1.5 Reported results are out-of-fold only

The results CSV contains only predictions made on data the model did not train
on. Mixing train-set predictions into the same file would inflate headline
results with no visible sign.

## 1.6 Clinical thresholds are per-patient

WHO anemia cutoffs vary by age and sex (11.0 for under-fives, 11.5 for 5–11,
12.0 for 12–14 and adult women, 13.0 for adult men). A single fixed cutoff
misclassifies anyone in the wrong band — measured on the local cohort, a fixed
11.0 g/dL rule mislabels 4 of 26 real patients, all anemic children called
normal. The threshold is looked up per patient, defaulting to 12.0 when
demographics are unknown so an unknown patient is not assumed healthy.

## 1.7 Splits are patient-grouped and stratified

Multiple captures of one patient (or both eyes of one child) are highly
correlated; letting them straddle a train/test split lets the model score by
recognising the subject. Splits group on `patient_id` and stratify by Hb so no
fold ends up without anemic cases.

## 1.8 Training details that are easy to get wrong

| Decision | Why |
|---|---|
| ImageNet mean/std normalisation inside `forward()` | `[0,1]` scaling alone mismatches the pretrained weights; putting it in the module means serving cannot forget it |
| Frozen BatchNorm forced to `eval()` | `requires_grad=False` alone does not stop BN running statistics drifting the frozen features |
| Rotation augmentation pads black, not reflected | Reflection would mirror the letterbox back into frame, inventing tissue never photographed |
| Global seed set per run | Reproducibility |
| Loss is smooth-L1 with inverse-frequency sample weights | Robust to label noise; rare severe-anemia cases carry larger gradients |

## 1.9 Structure

**Preprocessing is cached** (`CropCache`): segmentation runs once per image,
not once per epoch, keyed on image content, settings, and the extractor
version (`cache_key`). **The serving path is isolated**: `predict.py` imports
nothing from `train.py`. **Datasets are pluggable**: every loader emits the
same `Sample`, so the hospital data needs one adapter, not training changes.

## 1.10 Known gap

Visual audit artifacts (`overlays/`, contact sheets, manifests) are produced
by the `debug` command but not yet by training runs; worth wiring in before
the write-up.

---

# Part 2 — What each file does

## Core pipeline — `src/anemia/`

### `config.py`
Every tunable, as frozen dataclasses grouped by stage (`PreprocessConfig`,
`QualityConfig`, `SegmentationConfig`, `RegressionConfig`, `SplitConfig`).
Serialised next to every checkpoint so any run can be reproduced from one blob.

### `imaging.py`
Colour and geometry primitives: white balance, redness index, lightness,
letterbox padding, tight bounding box, mask smoothing, overlay rendering, and
the QC scores (focus, exposure, clipping).

Deliberately imports no torch — the serving path uses these without pulling in
the training stack.

### `segment.py`
All segmentation backends behind one contract: `(rgb) → (mask, name)`.

- `refined_conjunctiva_mask` — the production classical extractor: seeded
  grabCut with texture, sclera, and darkness gates, tuned on the 52 real
  captures. `heuristic_segmenter` tries it first (backend name `refined`)
- `heuristic_conjunctiva_mask` — the plain redness prior, now the first
  fallback
- `legacy_bright_neutral_mask` — a brightness-based baseline, kept so the
  redness-based design can be justified with a measured Dice comparison
- `grabcut_mask` — deterministic fallback when the colour prior degenerates
- `Mask2FormerSegmenter` — the trained model, loaded once and reused

Because everything is one callable, backends swap without touching
preprocessing, training, or serving. `heuristic_segmenter.cache_key`
(`classical-v2`) namespaces the crop cache — bump it whenever a classical
algorithm changes, or training may silently reuse crops cut by the old one.

### `preprocess.py`
The `prepare()` function every image passes through, at train time and at serve
time: white balance → segment → tight crop → letterbox → resize.

Also holds `assess_quality()` (the enforced QC gate) and `CropCache` (the disk
cache described in §1.3).

### `data.py`
Dataset adapters and clinical labelling.

- `load_eyes_defy_anemia()` — India + Italy subsets, with mask discovery
- `load_cp_anemic()` — Ghana cohort (**unverified against a real download**)
- `load_local_cohort()` — the locally collected cohort: `DATASAMPLE.csv` keyed
  by image ID, plus `left_eye/` and `right_eye/` folders. Assigns **both eyes of
  a subject the same `patient_id`**, without which the grouped splitter cannot
  stop the two correlated captures straddling train and test. Computes age at
  the date of capture rather than today
- `load_unlabelled()` — a plain folder of photographs with no metadata sheet,
  such as the local `left_eye/` and `right_eye/` collection. Sets `hb` to NaN so
  that training on unlabelled images fails loudly instead of silently fitting a
  placeholder value
- `has_labels()` — guard used before anything that needs a target
- `Sample` — the common record every loader emits
- `anemia_threshold()` / `is_anemic()` — the WHO cutoff table
- `summarise()` — cohort description, and a sanity check that the loader
  actually matched the layout

### `splits.py`
Patient-grouped, Hb-stratified k-fold and holdout. Grouping stops the same eye
appearing in train and test; stratification stops a fold ending up with no
anemic patients.

### `metrics.py`
- Regression: MAE, RMSE, bias, R², Pearson r, Bland–Altman limits of agreement
- Screening: sensitivity, specificity, PPV, NPV, confusion counts
- Segmentation: Dice, IoU
- `inverse_frequency_weights()` for imbalanced Hb
- `aggregate_folds()` for mean ± std across CV

`regression_metrics()` always carries `mae_vs_baseline` — the comparison against
always-predicting-the-training-mean.

### `model.py`
`HbRegressor`: ResNet-18/34 with a scalar head. ImageNet normalisation happens
inside `forward()` so the serving path cannot forget it, and frozen BatchNorm
layers are forced to `eval()` in `train()`.

`Checkpoint`: the self-describing bundle — weights, label standardisation
constants, preprocessing settings, metrics, config.

### `train.py`
Regressor training. Crop building with cache reuse, QC filtering, augmentation,
the per-fold loop with early stopping, and `cross_validate()`.

### `train_segmenter.py`
Mask2Former fine-tuning on the ground-truth masks, with patient-grouped
validation driving model selection. Only runs on Eyes-defy-anemia, the sole
source of pixel-level ground truth.

### `predict.py`
The serving path. `AnemiaPredictor` loads a checkpoint, reads its preprocessing
settings from the bundle (so an older model is still conditioned the way it was
trained), and returns Hb, verdict, threshold, and quality — or a retake message
when QC fails.

### `cli.py` / `__main__.py`
Four commands:

| Command | Purpose |
|---|---|
| `inspect` | Summarise a dataset without training. Run this first on any new data. |
| `debug` | Dump masks and overlays for one photo (`--image`) or a whole folder (`--dir`). Needs no trained model. |
| `train-segmenter` | Fine-tune Mask2Former on ground-truth masks |
| `train-hb` | Cross-validate the Hb regressor |
| `predict` | Score a single photo |

`debug` is the tool for unlabelled photographs. In batch mode it writes a
`quality_report.csv` of per-image QC statistics plus an `overlay_sheet.png`.
Without ground-truth masks there is no Dice to compute, so the overlay sheet is
the instrument — segmentation quality on real tissue is judged by eye until
masks are available. This is what exposed the colour heuristic's failure on
real captures.

## Serving — `serve/app.py`
FastAPI service. Model loaded once at startup. `POST /predict` accepts an image
plus optional age, sex, and pregnancy, and returns the prediction as JSON. A
failed-QC capture is a 200 with `usable: false`, not an error — a retake is a
normal outcome, not a bug.

## Scripts

### `scripts/benchmark_segmenters.py`
Scores every backend against real ground-truth masks and prints a Dice/IoU
table. This produces the before/after comparison that justifies the segmentation
rework in the report.

### `scripts/make_synthetic_dataset.py`
Writes phantoms in the real Eyes-defy-anemia on-disk layout, including a
fraction of deliberately blurred captures so the QC gate has something to
reject.

## Tests

### `tests/synthetic.py`
Eye phantom generator: skin surround, bright sclera, dark iris, and a
conjunctiva strip whose colour interpolates with Hb.

### `tests/test_pipeline.py`
19 tests covering segmentation (including the explicit check that the legacy
heuristic selects sclera), preprocessing and the QC gate, the WHO threshold
table, split integrity, and the metrics.

Runs without any real data.

## Experiments — `experiments/`

Exploratory studies, each self-contained with its own findings. Indexed in
`experiments/README.md`; failed approaches are recorded separately in
`RESEARCH_LOG.md`.

| Folder | Script | Produces |
|---|---|---|
| `01_stage_traces/` | — | One contact sheet per capture showing every pipeline stage side by side, for both pipelines |
| `02_extraction_check/` | via `anemia debug --dir` | Overlay sheets and per-image QC CSVs for the cohort |
| `03_colour_normalisation/` | — | Gray-world vs CIELAB scored against laboratory Hb |
| `04_lighting_stress/` | — | One eye under six illuminants; masks, cast ratios, capture metadata |
| `05_representation_ablation/` | `run_ablation.py` | Four colour representations scored on Hb prediction |
| `06_metrics_evaluation/` | `evaluate.py`, `plot_results.py` | Full metric suite on synthetic and real data (`metrics.json`), plus three-panel diagnostic plots |
| `07_pipeline_comparison/` | `compare_pipelines.py` | Both pipelines end-to-end and shared-mask, linear models only (`comparison.json`) |
| `08_regularisation/` | `compare_regularisers.py` | OLS vs ridge vs lasso vs elastic net, with coefficient-stability measurements |

### `cielab_stages.py`
Stage tracer for the CIELAB pipeline. Imports `cielab_pipeline.py` unmodified
and emits one captioned contact sheet per capture, ending with the a\*/b\*
channels that pipeline's network would receive.

## Documentation map

| Document | Covers |
|---|---|
| `README.md` | Setup, commands, data sources |
| `PIPELINE.md` | How the method works, stage by stage, with per-stage advantages and limitations |
| `CODE_GUIDE.md` | This file — design rationale and what every file does |
| `RESEARCH_LOG.md` | Every approach tried and rejected, with measured reasons |
| `GLOSSARY.md` | All abbreviations, defined once |
| `WORKLOG.md` | Week-by-week record |
| `experiments/README.md` | Index of the eight studies |
| `Progress_Report.pdf` | The formal report |

## Reference

### `gemini_anemia_pipeline.py`
An earlier prototype script, kept unchanged for reference.

---

## Suggested reading order

`data.py` (what a sample is) → `preprocess.py` (what the model sees) →
`model.py` (what it does) → `train.py` (how it learns) → `predict.py` (what the
app calls).
