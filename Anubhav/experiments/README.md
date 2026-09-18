# Experiments

Investigations kept separate from the pipeline itself. Everything here is
exploratory: outputs, one-off analyses, and findings. The pipeline that
produces them lives at the top level (`src/`, `serve/`, `scripts/`).

| Folder | What it is | Headline |
|---|---|---|
| `01_stage_traces/` | One contact sheet per capture showing every pipeline stage side by side, for all three extractors: `main_pipeline/` (classical), `cielab_pipeline/`, and `learned_pipeline/` (U-Net) | Learning and debugging aid; open a single sheet and compare each stage against the previous one. In `learned_pipeline/` panel 03 is the **conjunctiva probability map** (absolute 0-1, comparable across captures) rather than the per-image-stretched redness map |
| `02_extraction_check/` | Overlay sheets and per-image QC for the 52 cohort captures | Extraction lands on conjunctiva across the cohort; 49/52 pass QC |
| `03_colour_normalisation/` | Gray-world vs CIELAB measured against laboratory Hb | Gray-world correlates better with Hb (ρ +0.30 vs +0.02) and gives more consistent left/right eye readings. *Superseded by 18: on 217 external patients no correction is better, within and across sites* |
| `04_lighting_stress/` | One eye under six coloured illuminants | Both pipelines segment correctly only under neutral light; extreme casts destroy the measurement physically, not just numerically |
| `05_representation_ablation/` | Four colour representations scored on Hb prediction, including the gray-world + a\*/b\* hybrid | *Superseded in part.* No representation beats baseline **at channel-mean scale**; gray-world consistently beats its uncorrected counterpart. Illumination-invariant features later changed this — see `11` |
| `06_metrics_evaluation/` | How the Hb number is produced, the full metric suite on synthetic and real data, plus diagnostic plots | *Real-cohort figures superseded.* Synthetic R² 0.888, F1 0.986 (trivial by construction). The metric definitions and the mechanism write-up are current |
| `07_pipeline_comparison/` | Both pipelines end-to-end and shared-mask, scored with linear models (OLS + ridge) — the floor a CNN must clear | *Superseded in part.* Pipeline A wins consistently but narrowly; **24 features on 25 points blows up to R² −118.9**, the overfitting warning for the network stage |
| `08_regularisation/` | OLS vs ridge vs lasso vs elastic net, with coefficient-stability measurements | *Numbers superseded.* Features are near-collinear (max \|r\| = 0.993), which is why ridge is used; unregularised OLS wobbles ~300× more, and lasso at moderate penalty zeroes every coefficient |
| `09_train_test_protocol/` | In-sample vs out-of-sample performance, protocol agreement, and the instability of a single split at n=26 | *Numbers superseded; protocol findings stand.* A single 60/20/20 split reports test R² anywhere from **−2.43 to +0.78** on identical data — the strongest argument for getting the public datasets |
| `10_fitted_equation/` | The channel-mean equation in standardised and raw form, bootstrap confidence intervals, and a training-size comparison | *Superseded — not the deployed equation.* Its channel-mean coefficients are physiologically backwards; **the deployed erythema model is not**, with a dominant **+1.049 on log(R/G)**. Read §1–2 (protocol) as current, §3–4 as history |
| `11_head_to_head/` | **The deployment decision.** Both pipelines scored on identical patients: five feature representations, full metric suite, nested CV, a paired test, and the noise floor for each fitted R² | A leads B on every matched framing, but **p = 0.408 — the two are not separable at n=26**, and no fitted R² clears its own permutation null |
| `12_mask_placement/` | Where each pipeline's mask actually lands, measured by redness index inside the mask on all 52 captures | **A lands on tissue in 52 of 52; B is off-tissue in 46 of 52** and its placement is statistically indistinguishable from the sclera-selecting brightness baseline. The two masks overlap by a **median of 0.000**. *Caveat added by 13: this test detects a mask that is off the tissue, not one diluted by it* |
| `13_learned_extractor/` | **Three-way head-to-head**, repeating experiment 11's protocol with a U-Net extractor as a third arm (measurement functions imported from it, not reimplemented) | **C is the only arm to clear its permutation noise floor** — held-out R² **+0.111** (p = 0.015) vs A's −0.104 (p = 0.225) and B's −0.129 (p = 0.274), and the only positive held-out R² in the project. **But C vs A paired is p = 0.554** — not separable at n=26, and A keeps the better recall (0.923 vs 0.846) and F1. Also shows A's masks are **diluted with sclera** (redness +9.36 vs +15.89 on the same pixels) |
| `13_learned_extractor/` (continued) | The extractor comparison repeated on **243 patients**, cohorts kept separate; robustness to rotation, zoom, exposure and blur; augmentation ablation; the colour–haemoglobin direction check | **Settles the extractor question:** learned R² **0.354** vs classical **0.074** on 116 external photographs no segmenter trained on. Rotation augmentation helps, scale augmentation costs accuracy. **The local cohort lacks the colour signal** that the external cohort shows (investigated in 16) |
| `14_pooled_regression/` | Fits on the 217-subject external cohort and tests transfer to the local cohort and back | Within the external cohort R² **+0.350** (p < 0.0001). **Transfer fails both ways** (−1.944, −1.160): the cohorts' features sit 3+ SD apart |
| `15_sclera_relative/` | Measures the conjunctiva relative to the sclera in the same photograph, to cancel camera differences | The only features that transfer at all (a* difference: +0.032 / +0.081), at a cost in within-cohort accuracy. 51 penalised relative features do **not** transfer (−2.3): penalisation optimises accuracy, not portability |
| `16_local_inversion/` | Why the local cohort lacks the colour signal: capture metadata, sclera and whole-frame colour, framing, demographics, a processing simulation on external photos, and a blind rating of eyelid eversion | **Processing ruled out**: external photos shrunk to local crescent size and JPEG-compressed twice keep r = +0.673. Illumination, session, demographics and framing ruled out. **Eyelid eversion changes measured colour under both extractors** (redder, less blue with a harder pull; p 0.008–0.033). Whether eversion also tracked Hb is suggestive but not significant under fully blind rating (ρ −0.24) |
| `17_screening_categories/` | Replaces a number or yes/no verdict with three messages (likely / possibly / unlikely anaemic), with each patient's probability taken from the model's own error record and every cut chosen on other patients | On 217 patients the deployed extractor gives a clear answer to **78%** ("Likely" 69% truly anaemic, "Unlikely" 15%) and catches 88% of anaemic patients; the classical extractor puts 47% in "Possibly". **Misses fall on women** (17% vs 5%): predictions shrink toward 12.8 g/dL, above the women's cutoff. Adding sex as an input moves the misses onto men (4% vs 19%); **sex as an input plus a per-sex category cut fixes it** (women 4%, men 12%, R² 0.48, 12% in "Possibly"). One severe case (7.0 g/dL) is missed by every variant |
| `18_white_balance/` | Gray-world white balance versus no colour correction: within each site, across sites, and on the screening outcome, with the mask held fixed | **No correction wins every test.** Within site R² 0.467 vs 0.335 on 217 (MAE difference CI [+0.09, +0.23]); across sites gray-world is worse than guessing the average in both directions. Best app configuration improves to R² 0.544, 9% in "Possibly", and the Hb 7.0 patient is caught. Deployed pipeline not changed: one study's two sites, hospital camera untested |

## The deployed model is not in this folder

These are studies. The model that ships is fitted by `anemia fit-linear` and
lives in `runs/linear_model.json`.

**Status: no CNN has been trained on real data yet.** The deployed model is a
ridge regression on mean chromaticity inside the **learned (U-Net) extractor's**
mask, fitted on **190 patients** from Eyes-Defy-Anemia with the quality gate
applied first (`runs/linear_model.json`, extractor `runs/segmenter/unet.pt`):

```
Hb = −39.2123 + 70.042126·r + 75.186694·g     [g/dL]
r, g = mean R/(R+G+B) and G/(R+G+B) inside the mask
```

| | in-sample | held out (unseen patients) | baseline |
|---|---|---|---|
| MAE (g/dL) | 1.482 | **1.505** | 1.965 |
| R² | +0.336 | **+0.317** | −0.011 |

The learned extractor was chosen on 243 patients: on 116 photographs it never
saw in training it explains R² 0.354 against the classical extractor's 0.074.
Three limits matter more than those numbers:

- **It is calibrated to the cameras it was fitted on.** Models measured so far
  do not transfer between the two cohorts' cameras (R² −1.9); that has not yet
  been re-measured with this extractor. A new site needs recalibration.
- **The local 26-patient cohort is not used to fit.** It shows no
  physiological colour signal under either extractor. Image processing,
  illumination, capture session and demographics are ruled out; how far the
  eyelid was pulled down varies widely between its captures and measurably
  changes conjunctival colour (`experiments/16_local_inversion/`).
- **Predictions shrink toward the cohort mean**, so classification near the WHO
  thresholds is weak.

Full account: `experiments/13_learned_extractor/RESULTS.md` and §7 of
`Progress_Report.pdf`. **No accuracy figure in this repo is a clinical result.**

## Reproducing

Most folders contain their outputs directly. Two are regenerable:

```bash
python -m anemia stages --dir left_eye --out experiments/01_stage_traces/main_pipeline
python cielab_stages.py --dir left_eye --out experiments/01_stage_traces/cielab_pipeline
python -m anemia stages --dir left_eye --extractor unet --out experiments/01_stage_traces/learned_pipeline/left_eye
python experiments/05_representation_ablation/run_ablation.py
python experiments/06_metrics_evaluation/evaluate.py
python experiments/06_metrics_evaluation/plot_results.py
python experiments/07_pipeline_comparison/compare_pipelines.py
python experiments/08_regularisation/compare_regularisers.py
python experiments/09_train_test_protocol/protocol_comparison.py
python experiments/10_fitted_equation/fit_equation.py
python experiments/11_head_to_head/head_to_head.py
python experiments/12_mask_placement/measure_placement.py
python experiments/13_learned_extractor/learned_vs_classical.py
python experiments/13_learned_extractor/three_way.py
```

Experiment 13 needs a trained segmenter bundle first:

```bash
python scripts/prepare_eyes_defy.py --source datasets/eyes_defy --out datasets/eyes_defy_prepared
python scripts/train_segmenter.py --data datasets/eyes_defy_prepared --out runs/segmenter/unet.pt --epochs 60
```

## Conventions

- Results on synthetic phantoms are always labelled as such and are never
  clinical claims.
- Every predictive result is reported against a predict-the-mean baseline.
- Negative results are kept, not discarded — `05` and `07` are the clearest
  examples, and `../RESEARCH_LOG.md` records every failed approach with its
  reason.
- Models here are **linear only** (OLS / ridge). They exist to establish the
  floor a neural network has to clear, not to be the final model.
