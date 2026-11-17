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
| `19_coverage_and_vessels/` | Does a wider mask help, and do the conjunctiva's visible vessels carry signal? Threshold sweep 0.3–0.9 plus erode/dilate, and texture features, on 217 patients at 1024px | **Wider is worse, monotonically** (R² 0.544 eroded → 0.520 dilated 32px), reproducing experiment 13 on a cohort that can answer it. **Vessel texture looked strong pooled (edge strength r +0.40, R² 0.539 → 0.563) but is a site effect**: within site r +0.18 / +0.14 (n.s.), and controlling site and surrounding tissue r −0.02. Not adopted. Surrounding non-conjunctival tissue also predicts Hb within site (+0.31) — a warning for end-to-end networks |
| `20_architectures/` | Does a different encoder-decoder decoder beat the U-Net extractor? U-Net vs UNet++ vs Attention U-Net, identical encoder, seed, split, augmentation and epochs, scored by segmentation metrics alone | **Attention U-Net selected** (Dice 0.8476, IoU 0.7427, precision 0.8320 on 129 held-out subjects). On this split it is the only architecture whose advantage over U-Net resolves (+0.0105, 95% CI [+0.003, +0.020], p 0.007) — *not* reproduced over five splits (21); it is **not** separable from UNet++. Both new architectures lift the worst-case image from 0.36 to ~0.54. **Also: white balance was hurting the extractor and was a train/serve mismatch** — training read raw pixels while `prepare()` served balanced ones. Matched-raw beats matched-gray-world (+0.0069, p 0.0015) and the mismatch cost +0.0116, more than the architecture upgrade gained. **LinkNet, FPN, PSPNet added**: all significantly worse than Attention U-Net (Dice 0.801, 0.826, 0.759) — they over-segment (high recall, low precision); a published claim that LinkNet beats the others is not reproduced. Deployed extractor not swapped |
| `21_repeated_splits/` | Experiment 20's six-way comparison repeated under random states 0, 1, 2, 3 and 42 — each a fresh 390 / 129 / 129 subject split and fresh training randomness, 30 trained models — reporting Dice, IoU and pixel accuracy as mean, SD, variance, range and 95% CI, with a corrected resampled t-test (Nadeau–Bengio) and the same two Holm families | **Two groups.** U-Net, UNet++ and Attention U-Net are **tied** (mean Dice 0.8445 / 0.8462 / 0.8464, all p > 0.5; U-Net first in 3 of 5 splits) — the single-split advantage of Attention U-Net over U-Net was the split. FPN (0.835), LinkNet (0.819) and PSPNet (0.765) rank 4th, 5th and 6th in **every** split. LinkNet is worse than U-Net, UNet++ and FPN in 5/5 splits (Holm p 0.03–0.05): the published claim is not reproduced. *These splits let repeat shots of one subject straddle train and test (23); on person-safe splits the LinkNet gaps shrink and lose significance (24).* Pixel accuracy (0.994–0.996) cannot separate models against an all-background baseline of 0.986. Attention U-Net stays selected because nothing beats it |
| `22_resolution/` | What do resizing and squashing cost? On the annotators' own masks, no training: the best Dice possible on each grid against the 12-megapixel annotation, colour features before and after each resize, and fine detail kept by feature size — following the app's real path (512×512 frame, 320×320 segmenter grid, 224×224 neural input) | **Mask: not limited** — ideal Dice 0.992 at 320 (0.989 on the thinnest masks) against models at ~0.845. **Colour: survives** — 2–4% of the between-person spread at the app's 512 frame, order of people unchanged (r ≥ 0.999). **Structure: lost** — at 512, features of 2–4 / 4–8 px keep 33% / 66% of their detail. The neural input is cut from the 512 frame and enlarged; cutting it from the original keeps 1.5–2.4× the fine detail at no cost. Mendeley training labels are 1.6% fatter than the annotation in every image. No pipeline code changed |
| `23_duplicates/` | Are some Mendeley photographs of the same subject? The files carry no subject ids, so the trainer treats each as its own subject. Pixel hashes, perceptual hashes under all flips and rotations, and SIFT + RANSAC on every one of 204,578 pairs (Mendeley × Mendeley and Eyes-defy × Mendeley); cut-off set by file-number adjacency and by eye; leakage into experiment 21's splits measured | **Yes: 463 of 547 images in 115 groups** (mostly four shots each), so at most 199 subjects — fewer in fact, since SIFT misses some repeats. No exact copies; no overlap with Eyes-defy. **83–93 of 129 test images per split had a same-subject photograph in training.** Within Mendeley those images scored only +0.003 to +0.010 Dice higher (5 of 6 architectures; per-split differences both ways), so little sign of inflated scores and no reason to expect a changed ranking — but per-image statistics were overconfident. Neighbouring groups are often one person's two eyes (≥ 14 of 24 boundaries inspected), so people may number nearer half of 199. `groups.csv` is read by the trainer's `--groups` |
| `24_grouped_splits/` | Experiment 21 again (six architectures × seeds 0, 1, 2, 3, 42, same recipe) on person-safe splits: repeat-shot groups kept together (`--groups`), Mendeley dealt out in consecutive blocks of ~24 files so a person's two eyes can straddle the split only at a block edge (`--block 24`), and every group within 4 file numbers of a test file left out (`--buffer 4`). Does the ranking hold, and did the leak inflate experiment 21? Eyes-defy test Dice serves as the control | **The order holds; the margins soften.** UNet++ 0.833, Attention U-Net 0.831, U-Net 0.826 (tied; nothing beats Attention U-Net), FPN 0.821, LinkNet 0.810, PSPNet 0.764 (last in every split). LinkNet is never first, but its deficit shrinks (−0.017 vs U-Net, was −0.026) and **no LinkNet comparison survives Holm** (best: vs UNet++ −0.023, 0/5 splits, Holm p 0.18). Split-to-split SD doubled (0.012–0.029): a test set of a few capture sessions is a harder, more variable draw. **Leak estimate +0.013 to +0.025 Dice for all six, every 95% CI spans zero** — Mendeley fell 0.006–0.021, the Eyes-defy control did not. Best epochs came earlier (mean 32 → 27), consistent with the leak having flattered validation. More splits would not resolve 0.01–0.02 gaps (corrected test). Splits: 339–366 train, 20 Eyes-defy in every test set, zero test images with a group partner or a file within 4 numbers in training |
| `25_precision_weighted/` | Experiment 24's 30 models rescored, no training: **F0.5** (the F-beta score with beta 0.5, precision weighted twice recall, fixed in advance), because a wrongly included pixel pulls the measured colour toward the wrong tissue while a missed one only removes some conjunctiva; plus a threshold sweep (0.05–0.999), each model's threshold chosen on its own validation images; control: the same choice made for Dice | **Same order at 0.5** (Attention U-Net 0.825, U-Net 0.824, UNet++ 0.823, FPN 0.809, LinkNet 0.782, PSPNet 0.740); LinkNet's gap doubles (−0.043) but is not significant (Holm p 0.30). **Best F0.5 threshold ≈ 0.97** (0.85–0.995) in 29 of 30 models: test F0.5 rose in all 29 (+0.021 to +0.050 per architecture), precision +0.06 to +0.12, recall to 0.71–0.77, Dice −0.018 to −0.033, masks ≈ 5/6 of the true area, 6 of 3,750 masks empty. Per architecture no gain survives Holm (five-split power limit), but it is unanimous. **Control: a Dice-chosen threshold gains nothing** (−0.001 to +0.002), so the high threshold is the price of weighting precision. With tuned thresholds the middle closes up (UNet++ 0.853, Attention 0.846, U-Net 0.846, FPN 0.841, LinkNet 0.832, PSPNet 0.781; LinkNet −0.014 vs Attention, Holm 0.45): much of LinkNet's and FPN's shortfall was over-segmentation at 0.5. Nothing beats Attention U-Net. One Attention U-Net (seed 0) never exceeds probability 0.53, so thresholds must be chosen per trained model. Deployed threshold (0.5) not changed |
| `26_correction_prechecks/` | Checks before screening further colour corrections for the regressor's crop (brightest pixels as white, sclera as white) on the 217 Eyes-defy photographs, no haemoglobin model: a clipping census, sclera colour against age and Hb, and the device's white plastic as a reference for the light | **Plain max-RGB is no correction** (all channels hit 255 in 96%); half the brightest unclipped pixels are the device plastic, so brightest-pixel must be taken above the lower lid. Against the plastic, brightest-above-lid (0.8–1.5°) and sclera (1.0–2.3°) estimate the light far better than gray-world (~5°, worse than none). **The sclera is not white**: b* rises ~0.5–0.6 per decade with the light removed (all groups), and its a* follows Hb (ρ +0.28 to +0.51); dividing by it in India makes older patients read ~0.4 g/dL per decade lower. **The Italian set is two capture batches** (patients 1–31: older, Hb 11.5, less red light; 32–123: Hb 14.6), which tilts experiment 18's Italian results toward no correction |

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
