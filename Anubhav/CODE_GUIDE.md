# Code Guide

Two things: what changed from the inherited implementation, and what every file
does. Read `PIPELINE.md` first for how the method works — this document is about
the codebase.

---

# Part 1 — What changed from the original

The inherited implementation was a single 1,205-line script,
`gemini_anemia_pipeline.py`, kept unchanged in this folder for reference.

Its core design decisions were sound and are preserved: gray-world white
balance, aspect-preserving square padding, Mask2Former for segmentation,
ResNet-18 for regression, inverse-frequency weighting for imbalanced Hb, and
Dice/IoU for segmentation quality. What changed is the parts that did not work.

## 1.1 Defects that changed results

### Segmentation targeted the wrong tissue

The colour heuristic kept **bright, low-chroma** pixels. Bright and neutral is
the *sclera* — the white of the eye. The haemoglobin signal is in the red
palpebral conjunctiva. Measured against phantom ground truth, the original
heuristic scores Dice **0.000**; it has no overlap with the target tissue at
all.

Now: pixels are scored by redness in CIELAB (`a* − 0.5·b*`), with the threshold
chosen by Otsu and an absolute (not percentile) lightness gate. See §2.2 of
`PIPELINE.md` for why both of those details matter.

### The trained segmenter never actually ran

The Mask2Former load passed `local_files_only=True`, and no checkpoint was
bundled. On any clean machine that raises. The exception was caught, a warning
printed, and the pipeline silently fell back to the colour heuristic.

The consequence is worse than a crash: a run could report "trained
segmentation" results having trained nothing, and the only sign was one line of
console output.

Now: the flag is gone, and an explicitly requested model directory that fails to
load raises instead of degrading silently.

### Quality control was decorative

`accepted_for_training` was computed for every image, written to the manifest,
and **never read**. Blurred and badly framed captures trained the model exactly
like good ones.

Now: failing images are dropped from training, and at serve time the API returns
`usable: false` with a retake prompt instead of a confident number.

### Saved checkpoints could not be served

The network is trained on standardised labels, so its output is a
standardised value. Recovering g/dL requires the training mean and standard
deviation. The original saved `model.state_dict()` alone and discarded both.

That makes every checkpoint it ever wrote unusable for inference — which is a
blocker for building an app on top of it.

Now: `Checkpoint` bundles the weights, `target_mean`, `target_std`, the
preprocessing settings, and the metrics in one file, and refuses to load an
incomplete bundle.

### Reported results mixed training and test data

`all_predictions.csv` was built from `train_records + eval_records` with no
column distinguishing them. With an 80/20 split, roughly **80% of the headline
rows were training-set predictions**.

Now: the results CSV contains out-of-fold predictions only — every row is a
prediction on data that fold did not train on.

### The anemia threshold was wrong for most adults

`Hb < 11.0` was applied to everyone. That is the WHO cutoff for children 6–59
months and pregnant women. For an adult man the cutoff is 13.0.

So a man at 12.0 g/dL — genuinely anemic — was labelled **normal**. In a
screening tool that is the dangerous direction of error.

Now: the threshold is looked up per patient from age and sex, and defaults to
12.0 when demographics are unknown rather than assuming health.

### Splits leaked between train and test

Splitting was `selected_images[:train_split]` on a list sorted by patient
number: no shuffling, no grouping, no stratification.

Now: patient-grouped and Hb-stratified k-fold, so no patient's captures can
straddle the split and no fold can end up without anemic cases.

## 1.2 Training defects

| Defect | Effect |
|---|---|
| `/255` scaling with no ImageNet mean/std | Input distribution mismatched the pretrained weights being fine-tuned |
| `requires_grad = False` used alone to freeze layers | BatchNorm keeps updating running statistics in train mode, drifting the frozen features |
| Rotation augmentation used `BORDER_REFLECT_101` | Mirrored the black letterbox back into frame, inventing tissue that was never photographed |
| Loss printed as "weighted MSE" | It was smooth-L1; `weighted_mse()` was defined and never called |
| No seed anywhere | Runs were not reproducible |
| One image per patient, capped at 50 | Used roughly 5% of the available data |

## 1.3 Structural changes

**Preprocessing is cached.** Segmentation ran inside `Dataset.__getitem__`, so a
40-epoch run re-segmented every image 40 times. Crops are now computed once and
cached to disk, keyed on image content plus preprocessing settings. Augmentation
still runs per-epoch on the cached crop, so no diversity is lost.

**The serving path is isolated.** `predict.py` imports nothing from `train.py`.
The API loads one model at startup and does one segmentation pass plus one
forward pass per request.

**Datasets are pluggable.** Loaders emit a common `Sample`, so adding the
hospital data means writing one adapter, not editing training code.

**Paths are arguments.** The original hardcoded
`/Users/yatikajena/Desktop/AnemiaDetection/...` as CLI defaults.

**Dead code removed.** `process_image()`, `weighted_mse()`, and
`ConjunctivaDemoDataset` were all defined and never called. `run_demo()`
duplicated `process_image()`'s body inline.

**Added:** 18 tests, synthetic phantoms, a segmenter benchmark, and a FastAPI
service.

## 1.4 Things deliberately not carried over

The original wrote a set of visual audit artifacts — `overlays/`, `masks/`,
`crops/`, `contact_sheet.png`, and `manifest.csv`. These are genuinely useful
for eyeballing whether segmentation is behaving, and for figures in the report.

They are **not** currently produced. `build_overlay()` still exists in
`imaging.py` but nothing calls it. Worth re-adding as an `export` CLI command
before the write-up.

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

- `heuristic_conjunctiva_mask` — the fixed redness prior
- `legacy_bright_neutral_mask` — the inherited heuristic, kept so the rework can
  be justified with a measured Dice comparison rather than an assertion
- `grabcut_mask` — deterministic fallback when the colour prior degenerates
- `Mask2FormerSegmenter` — the trained model, loaded once and reused

Because everything is one callable, backends swap without touching
preprocessing, training, or serving.

### `preprocess.py`
The `prepare()` function every image passes through, at train time and at serve
time: white balance → segment → tight crop → letterbox → resize.

Also holds `assess_quality()` (the enforced QC gate) and `CropCache` (the disk
cache described in §1.3).

### `data.py`
Dataset adapters and clinical labelling.

- `load_eyes_defy_anemia()` — India + Italy subsets, with mask discovery
- `load_cp_anemic()` — Ghana cohort (**unverified against a real download**)
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
| `train-segmenter` | Fine-tune Mask2Former on ground-truth masks |
| `train-hb` | Cross-validate the Hb regressor |
| `predict` | Score a single photo |

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
18 tests covering segmentation (including the explicit check that the legacy
heuristic selects sclera), preprocessing and the QC gate, the WHO threshold
table, split integrity, and the metrics.

Runs without any real data.

## Reference

### `gemini_anemia_pipeline.py`
The original inherited implementation, unchanged. Kept for reference and for the
before/after comparison in the report.

---

## Suggested reading order

`data.py` (what a sample is) → `preprocess.py` (what the model sees) →
`model.py` (what it does) → `train.py` (how it learns) → `predict.py` (what the
app calls).
