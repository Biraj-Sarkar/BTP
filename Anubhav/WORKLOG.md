# Work Log

Running record of work on the anemia-detection BTP, newest week first.
Every substantive change gets an entry: what was done, and where it lives.

---

## Week of 8–14 Oct 2026

**14 Oct — Attention U-Net beats the U-Net extractor; selection and shipping made repeatable**
The extractor was a U-Net because U-Net is the default encoder-decoder segmentation
architecture, not because anything had been measured against it. Two candidates were
compared, and the comparison was moved off the regression probe onto **segmentation
metrics alone**.

**Why the change of method.** Extractors had been ranked by fitting a ridge
regression on colour inside each mask and reporting R² against laboratory
haemoglobin. That scores the extractor and a stand-in model together, and the
stand-in is far weaker than the network that will read these crops. Segmentation
models have their own evaluation, which measures only the mask. The comparison now
reports Dice, IoU, precision, recall, HD95 and the failure distribution, and no
haemoglobin figure at all.

**Result on 129 held-out subjects** — none of the three produced a single empty mask:

| architecture | params | Dice | IoU | precision | recall | HD95 px |
|---|---|---|---|---|---|---|
| U-Net (deployed) | 6.30M | 0.8372 | 0.7288 | 0.8239 | 0.8637 | 2.8 |
| UNet++ | 6.61M | 0.8446 | 0.7379 | 0.8266 | 0.8746 | **2.2** |
| **Attention U-Net** | 6.33M | **0.8476** | **0.7427** | **0.8320** | 0.8737 | 2.5 |

**Only one comparison resolves.** Paired per-image Wilcoxon with a bootstrap
interval: Attention U-Net beats U-Net by **+0.0105 Dice, 95% CI [+0.003, +0.020],
p 0.007**, winning 75 of 129 images. UNet++ beats U-Net by a similar margin but its
interval touches zero (p 0.088), and the two new architectures are **not separable**
from each other (p 0.155). The honest claim is therefore narrow: Attention U-Net is
better than the extractor we run; it is not demonstrably better than UNet++.

Two details worth more than the headline. The gain is **mostly in precision**, which
is the error direction that costs us — a mask spilling onto lid skin contaminates the
colour read inside it — whereas UNet++ gains almost entirely in recall. And the
**worst case improves far more than the mean**: U-Net's worst image scores 0.363,
both new architectures reach ~0.54, with the spread narrowing. For a screening tool
the floor matters more than the average. Counterweight recorded too: UNet++ has the
best median, so the mean advantage comes from the bad cases, not typical ones. HD95
disagrees with Dice and favours UNet++ (2.2 against 2.5).

**Infrastructure, so this is not a one-off.**
- `src/anemia/architectures.py` is now the only place a model is built. Both the
  trainer and the read-only loader call it, and the bundle records `arch`, so a model
  cannot be served by a decoder it was not trained with. Existing bundles load
  unchanged — `arch` defaults to `unet`.
- Attention U-Net is **not in `segmentation_models_pytorch`**, so the additive gates
  and decoder are implemented here, with decoder widths matched to the library's
  U-Net so the comparison is about the gates rather than capacity.
- `--test-fraction` added to the trainer. The epoch was being chosen on validation
  Dice and that same number reported, which is the maximum of a noisy sequence and
  optimistic by construction. A third split, untouched by training and selection, is
  what gets reported now.
- `--no-select` trains on every subject for a fixed epoch count: the shipping run.
- `scripts/select_architecture.py` runs both stages and keeps a leaderboard with a
  **split fingerprint** (data roots, seed, fractions, epochs, augmentation, resulting
  test subjects). A fourth architecture later is one command; a run whose fingerprint
  differs is refused, because a leaderboard mixing two splits would rank architectures
  by which subjects they were tested on. Bundles already on disk are adopted only
  after checking the subjects they actually held out, not their filename.

**Shipped:** Attention U-Net retrained on all 648 images for 36 epochs →
`runs/segmenter/attention_full.pt`, verified to load through the normal serving path.
Its estimate is the stage-1 figure of 0.8476; those weights saw every subject and have
no held-out score of their own, and the leaderboard records that caveat.
**The deployed `runs/segmenter/unet.pt` is deliberately not replaced** — swapping it
invalidates every cached crop and every downstream feature, and the existing
robustness question (experiment 13) is still open.

**Limits recorded:** one seed per architecture, and the margin is about one point of
Dice, so it is not yet separable from run-to-run variance — a second seed would settle
this and the older stability question together. And Dice measures agreement with the
annotator, who is a proxy: 117 of 218 masks in one dataset were bounding rectangles
and were discarded.
→ `experiments/20_architectures/`, `src/anemia/architectures.py`,
`scripts/select_architecture.py`, `scripts/train_segmenter.py`, `CODE_GUIDE.md`

---

## Week of 24–30 Sep 2026

**30 Sep — How to use the extractor from other training code, documented**
The extractor had no documented library-level entry point. `UI_INTEGRATION.md`
covers the HTTP contract for the application, and `CODE_GUIDE.md` described
`prepare()` without saying how to call it from outside the package — so anyone
training their own regressor on extracted crops had to read the source.

`CODE_GUIDE.md` now carries a worked example under `preprocess.py`: resolve the
segmenter, build the two configs, call `prepare_path()`, check the QC gate, and
take `Prepared.crop` — a 224×224 RGB array with everything outside the mask
blacked out, which is already the input size a ResNet-18 expects. The snippet was
run verbatim from an unrelated working directory before being written down.

Four traps are called out because each produces a silently wrong result rather
than an error:

- `get_extractor("unet")` resolves `DEFAULT_BUNDLE` **relative to the working
  directory**, so it only works from the repository root; an absolute
  `model_dir` is required from anywhere else. Verified: `FileNotFoundError`.
- `PreprocessConfig.gray_world` still defaults to `True`, while experiment 18
  found no correction better on every test. Callers should pass `False`.
- Widening the mask for more context lowers accuracy monotonically
  (experiment 19), so the threshold should be left alone.
- Non-conjunctival tissue predicts haemoglobin within each site, so training on
  full frames rather than the masked crop can score well without reading the
  conjunctiva. `.crop` is the defence; full-frame training needs site-held-out
  splits and a masked-input comparison.

Also recorded: `out.backend` and `segmenter.cache_key` identify which extractor
and which bundle produced a crop, and both belong in any checkpoint that is
trained on them.
→ `CODE_GUIDE.md`

**29 Sep — Report gains the screening-categories and texture sections**
An audit of `Progress_Report.pdf` against the current findings showed the document
had fallen behind the experiments in two places. Both are now written up.

**New §7.8, "What the application should report".** The three-category output had
only ever reached the report as two rows inside the colour-correction table, with
the terms "Possibly" and "per-sex cut" used before being defined. It now has its
own section: the category rule, the probability taken from the model's own error
record rather than an assumed distribution, the nested protocol that tunes nothing
on the patients being scored, and the table showing what the app would say against
who is actually anaemic. The no-information column is kept beside it, because a
model that answers "possibly" to everyone catches 100% of anaemic patients and is
useless — sensitivity is never reported here without the undecided share.

The **fairness result was absent from the report entirely** and is the most
consequential thing added: anaemic women told "unlikely" 17% of the time against
5% for men, the structural reason (predictions shrink toward 12.8 g/dL, above the
women's cutoff and below the men's), the finding that adding sex as an input moves
the misses onto men rather than removing them, and the per-sex cut that brings both
groups to 4% and 12%. Stated as a general rule about any hedging estimator, since
it applies equally to the network.

**New §7.9, "Mask coverage and tissue texture".** Experiment 19 should have gone
into the report when it was run on 18 Sep and did not. Both negative results are
now recorded: mask widening hurting monotonically across eight variants, and the
vessel-texture gain (R² 0.539 → 0.563) dissolving under its controls to r −0.021,
p 0.76. The warning for the network work — that non-conjunctival tissue predicts
haemoglobin within each site, so an end-to-end model can score well without
reading the conjunctiva — is a callout rather than a buried sentence.

§8 updated accordingly: the category result under Verified, texture and mask
widening under Measured and found wanting, and two new entries under Not yet
established (the clinical category targets, and whether a network reads the
conjunctiva or the photograph around it).

**15 pages**, up from 14. Diplomacy grep clean on the extracted PDF text.
→ `scripts/make_progress_report.py`, `Progress_Report.pdf`

---

## Week of 10–16 Sep 2026

**19 Sep — Both original pipeline scripts made runnable on any machine**
`gemini_anemia_pipeline.py` and `cielab_pipeline.py` defaulted `--dataset-root`
and `--output-root` to absolute paths under a home directory that exists on no
current machine, so both failed on a bare `python cielab_pipeline.py` and had to
be driven with explicit flags every time.

Defaults are now relative to the script: `datasets/eyes_defy/dataset anemia/India`
for input and `runs/<pipeline>_outputs_*` for output, via a module-level `HERE`
matching the `WORKSPACE_CACHE` idiom already in both files. Header docstrings
updated to name the default and the override flag. **No logic touched** — the
pipelines themselves are unchanged and remain the comparison baselines; only the
two argparse defaults and the docstring paths moved.

Verified: both import cleanly and resolve `--dataset-root` to the real 96-entry
India subset, `--output-root` inside the gitignored `runs/`.
→ `gemini_anemia_pipeline.py`, `cielab_pipeline.py`

**18 Sep — Experiment 06 made reproducible off this machine**
`evaluate.py` read its synthetic-phantom predictions from an absolute path outside
the repository, so `evaluate_synthetic()` returned `None` for anyone else and
section A of the output silently vanished — the metric code's only end-to-end test
was passing on one laptop and skipping everywhere else.

The 74-row prediction file and the training configuration that produced it are now
checked in at `experiments/06_metrics_evaluation/artifacts/`, and the script reads
them from beside itself. `artifacts/`, not `data/`: `.gitignore` excludes every
directory named `data` at any depth, so the obvious name would have reproduced the
bug in a quieter form.

Numbers are unchanged and now reproduce anywhere — synthetic R² 0.888, MAE 0.666,
F1 0.986, all still labelled as trivial by construction.
→ `experiments/06_metrics_evaluation/evaluate.py`, `METRICS.md` §3.1

**18 Sep — Reading list assembled and ordered in `papers/`**
Five downloaded papers renamed by title, two more fetched, and everything filed in
`papers/conjunctiva-anemia/` numbered in reading order, with `papers/README.md`
saying what to take from each *for this project*:

01 Collings 2016 (erythema index with an in-frame calibration card) → 02 Suner 2021
(the pipeline closest to our linear model) → 03 Zhao 2024 (deployed app, LoA ±4.8
g/dL, transfusion-threshold framing) → 04 Mannino 2018 (**personalised calibration**,
published support for per-site recalibration) → 05 Kasiviswanathan 2020 (the
Eyes-Defy-Anemia paper, whose masks we train on) → 06 Dhalla 2023 (segmentation
architectures for this exact task) → 07 Kato 2024 (end-to-end, for the network work)
→ 08, 09 (existing 2025 papers).

Added 04 and 05, neither previously held: Mannino because our transfer failure points
exactly at per-patient/per-site calibration, and Kasiviswanathan because we train on
that dataset and rejected ~45% of its masks as bounding boxes. Zhao 2024 turned out
to be a duplicate of a paper already stored and was consolidated. The README's claim
that MDPI blocks scripted downloads is now qualified: their CDN serves the PDF.
Licence table added; the Wiley paper is flagged as unverified — if it is not open
access the PDF should be replaced by its DOI.
→ `papers/`

**18 Sep — Wider masks don't help; the "vessel signal" was the site, not the vessels**
Two questions on 217 patients at 1024px, no colour correction, sex included.

**Coverage.** Swept the extractor's probability threshold (0.9 → 0.3) and eroded
and dilated the 0.5 mask. **Tighter is better, monotonically:** R² 0.544 (erode
8px) → 0.539 (deployed) → 0.520 (dilate 32px), MAE moving with it. Lowering the
threshold to capture more tissue does not help; widening dilutes the measurement.
This reproduces experiment 13, which mattered because that sweep was run on the 26
local patients, since shown unable to answer it. **Deployed threshold unchanged**
— the erode gain is 0.005 R², picked from eight variants on the same patients, and
tighter masks come back empty more often on hard captures.

**Vessels — a negative result worth the space.** Texture features (variability,
edge strength, vessel-like darkness) looked convincing pooled: edge strength
correlates **+0.403** with Hb, R² 0.539 → **0.563**, and structure alone with no
colour reached 0.434. Three controls killed it:
- edge strength on tissue *surrounding* the conjunctiva also correlates +0.328, so
  it is not conjunctiva-specific (whole-frame blur does not, +0.094 n.s.);
- the two sites differ in both Hb (11.47 vs 13.83) and image detail (edges 14.3 vs
  20.1), so "more edges" partly means "Italian patient". **Within site: +0.180
  (p 0.081) and +0.143 (p 0.117); controlling site and surroundings: −0.021**;
- within-site prediction gains +0.005 to +0.007 R², and the full structure set is
  *worse* than colour alone.

**Not adopted.** Recorded because the pooled number would have looked like a real
improvement in any report that skipped the site control.

**Open observation:** surrounding, non-conjunctival tissue predicts Hb within each
site (+0.319 India, +0.309 Italy). It cannot be pallor. Warning for the network
work: an end-to-end model on whole photographs can score well on photo-level cues
without looking at the conjunctiva — split by site, and consider evaluating on
masked images.
→ `experiments/19_coverage_and_vessels/`

**17 Sep — Worklog audit: gaps filled, order fixed, personal notes de-referenced**
Three entries were missing and are now in place under their dates: experiment 14
(217-patient cohort, transfer failure), experiment 15 (sclera-relative features),
and the decision to keep datasets out of git with rebuild scripts and manifests
committed instead. Two 15 Sep entries that sat above the 17 Sep ones in this
newest-first log were moved to their day. Older entries and one experiment write-up
named personal, non-shared notes files by filename; those references are now
generic, since this log is shared.

**17 Sep — Gray-world vs no colour correction: no correction wins every test**
Mask held fixed (extractor runs on the balanced image, as deployed); only the pixels
colour is read from change. 10-fold CV, 5 splits, patient-level bootstrap.

- **Within site:** Italy R² 0.409 vs 0.301 (MAE diff CI [+0.03, +0.16]); India 0.251
  vs 0.192 (MAE CI [−0.02, +0.09], R² CI excludes 0); both 0.467 vs 0.335 (MAE CI
  [+0.09, +0.23]).
- **Across sites** — gray-world's reason to exist: India→Italy MAE 2.84 none vs
  **4.00** gray-world (baseline 2.82); Italy→India 2.39 vs 2.87 (baseline 2.63).
  **Gray-world is worse than guessing the average both ways.**
- **App configuration** (colour + sex, per-sex cut): R² 0.544 vs 0.482, "Possibly"
  9% vs 12%, "Likely" 74% vs 70% anaemic; **the Hb 7.0 patient is caught** without
  correction.

Why: gray-world rescales by whole-frame content, and framing (how much sclera, iris,
skin) varies shot to shot, so it adds noise unrelated to Hb. The experiment-03
preference for gray-world was measured on the 26-patient cohort, now known to lack
the signal; annotated as superseded in the experiments index, study notes and talking
points, and report §5 now points to a new §7.7. **Deployed pipeline not changed:**
both sites are one study; the hospital camera is untested (added as next step 8).
→ `experiments/18_white_balance/`

**17 Sep — The Hb 7.0 miss: gray-world white balance over-corrected her photo**
Looked directly at the one severe patient every variant misses (`eyesdefy-italy:105`,
Hb 7.0, estimate 13.6). The mask is correct. She is the most anaemic Italian subject
(none other ≤ 8.5). Her raw photo has the coolest cast of 40 Italians checked (sclera
red share 0th percentile, blue 100th) — lavender as photographed. Gray-world assumes
the frame averages grey; this close-up is dominated by bluish sclera and iris, so it
over-corrects (sclera visibly turns mint-cyan) and pushes the conjunctiva toward
orange-red. **Without white balance both her red and blue cues read anaemic; with
gray-world the blue cue — the model's main one — reads healthy.** Sclera-relative
colour fails too (her sclera is the bluest).

Population-level support: within Italy, colour tracks Hb more strongly without
gray-world (blue −0.60 vs −0.56, red +0.66 vs +0.59); across 217 external patients red
r is +0.67 raw vs +0.47 after gray-world. Gray-world was chosen on the local cohort,
now known to be anomalous. Needs a held-out gray-world-vs-none comparison, within and
across sites, before changing the default — uncorrected colour is camera-dependent.
→ `experiments/17_screening_categories/RESULTS.md`, `patient_italy105_hb7.jpg`

**17 Sep — Sex as a model input moves the misses; a per-sex category cut fixes them**
Tested the obvious fix for anaemic women being told "unlikely". **Adding sex as an
input** improves the estimate a lot (R² 0.335 → 0.482, MAE 1.53 → 1.33) and cuts
women's misses 17% → 4% — **but men's rise 4% → 19%**. The model now shrinks toward
each sex's own average: women 11.17 (below their 12.0 cutoff), men **13.86 (above
their 13.0 cutoff)**, so the same mechanism now hits anaemic men. General rule: a
hedging model under-flags any group whose average sits above its own cutoff, and
a better model can't remove that — the category rule has to.

**Choosing the lower category cut separately per sex, with sex as an input,
fixes it:** women 4%, men 12% missed (target 10%; ~34 anaemic men per fold makes
the men's cut noisy), 93% caught overall, smallest "Possibly" group (12%),
"Unlikely" right 93% of the time — best on every row. Sex alone (no photo) reaches
R² 0.295 but can't sort anyone (60% "Possibly", no "unlikely"), so the photo adds
substantially on top. One woman at Hb 7.0 reads as healthy (estimate 13.6) in
every variant: a measurement failure no cutoff can fix.
→ `experiments/17_screening_categories/sex_feature.py`

**17 Sep — Screening categories instead of a number: tested on 217 patients**
Design discussion: the app is a screening aid, so rather than a haemoglobin value
or a hard verdict it should say "likely anaemic — get tested", "possibly — consider
a test", or "unlikely from this photo". Each patient's P(anaemic) is taken from the
model's own error record (the share of past errors that would put them below
their WHO cutoff), so the method needs no bell curve and works for any model. The
lower category cut is set so the top two catch 90% of anaemic patients; nothing is
tuned on the patients being scored (outer and inner 10-fold, 5 repeats).

On 217 external patients (42% anaemic), the deployed extractor gives a clear
answer to **78%**: "Likely" is 69% truly anaemic, "Unlikely" 15%; top two catch 88%.
The classical extractor catches slightly more (93%) only by putting **47%** of
patients in "Possibly", with a "Likely" group barely above the base rate (51%). A
no-information model catches 100% by saying "possibly" to everyone — the reason the
share in the middle category has to be reported beside sensitivity.

**Misses concentrate in women: 9 of 53 anaemic women told "unlikely" (17%) vs 2 of
38 men (5%).** Predictions shrink toward the cohort average of 12.8 g/dL, which is
above the women's cutoff (12.0) and below the men's (13.0), so hedging under-flags
women. Structural, and will apply to any shrinking model including a network.
Options (untested): sex as a feature, sex-specific shrinkage, per-sex cuts. One
severe case (Hb 7.0, estimated 13.6) was told "unlikely".
→ `experiments/17_screening_categories/`

**17 Sep — Why the local cohort shows no colour signal; report now in two columns**
**The "inversion" was overstated.** Tested for significance, the local red
chromaticity correlation is r −0.26, **p = 0.20**, not distinguishable from zero.
What is firm: the local cohort *lacks* the signal every external subset shows
(difference from external p < 10⁻⁶), and under gray-world its blue channel
correlates positively with Hb (p = 0.006, one of many tests). Every document now
says "lacks the signal" rather than "inverted".

**What the local images are:** hand-cropped exports. No EXIF or colour profile,
52 different dimensions, 29–97 KB at JPEG ~75. Labels map correctly and the two
eyes of each child agree (r ≈ 0.6).

**Ruled out, each by measurement:**
- **Image processing, decisively.** External photos downscaled until their
  crescents matched local thickness in pixels (×0.146), then JPEG q75 once and
  twice and q40 twice: red vs Hb stays **+0.673** in every variant.
- Illumination / phone white balance: sclera and whole-frame colour vs Hb all
  |r| < 0.08.
- Capture session (nine-month span): r +0.01. Age, sex, SES: all p > 0.2.
- Crop framing: it correlates with Hb (sclera area r +0.53) but does not predict
  colour, and controlling for it leaves colour–Hb unchanged. **My own framing
  hypothesis, recorded as rejected.**

**Found: eyelid eversion changes colour.** Mask overlays suggested lids were
pulled down very differently and that the lowest-Hb conjunctivae looked red.
Eversion was rated **blind**: 52 captures shuffled and coded, fixed 0/1/2 rubric,
joined to Hb only afterwards. Harder eversion gives redder, less blue tissue
under **both** extractors (learned blue ρ −0.51, p 0.008; classical red ρ +0.42,
p 0.033), and the rating tracks an objective quantity (less sclera in frame,
ρ −0.42). Plausibly mechanical congestion, which masks pallor.

**Not established: that low-Hb children were everted harder.** Both eyes averaged
ρ −0.46 (p 0.018), but **fully blind right eyes ρ −0.24 (p 0.23)**. I had seen ten
left-eye images with their Hb while forming the hypothesis, and the left/right gap
is what rater bias would produce. Controlling for eversion moves every association
toward zero without eliminating it. Practical conclusion: capture protocol is
part of the measurement (fixed gentle eversion, original files, time since blood
draw recorded; a second blind rater before quoting eversion vs Hb).
→ `experiments/16_local_inversion/`

**A bug I introduced, found by this work.** The `sed` that wired `_to_float` into
the Eyes-Defy loader also matched an identical line in `load_local_cohort`,
leaving it referencing an undefined variable. No test loaded the local cohort, so
all 43 passed. Restored; the diff of `data.py` now contains only the intended
Eyes-Defy change; added a test that builds the cohort's layout in miniature.
**44 tests.** No reported number was affected (nothing loaded the local cohort
between the break and the fix).

**Progress report reformatted as a two-column paper** (13 pages): centred title,
author, full-width abstract and keywords, text in balanced columns, and tables of
four or more columns plus figures spanning the page. §7.4 rewritten with the
investigation; summary, §8–11 and the assessment revised. Two layout bugs fixed
while proofreading rendered pages: columns ~2 pt narrower than the tables placed
in them (frame padding), and the final column's default 10% overrun being drawn
over by the next full-width table.

**17 Sep — Deployed model replaced; a check before deploying caught an inverted cohort**
Three tasks: refit the haemoglobin model with the recommended extractor, make
the server answer the mobile app, and regenerate the progress report. The first
turned up the most important finding of the month.

**Quality gate, calibrated per extractor.** The first refit kept only 14 of 26
patients: the gate's 1.5%-of-frame mask floor, set for the classical extractor's
~10% masks, rejected 30 of 52 captures as "conjunctiva too small" under the
learned extractor (~1.9%). It was discarding captures *because* the mask was
precise. An extractor can now declare its own floor (`LearnedSegmenter.
min_mask_ratio`, the same 0.2% it already applies to its own output); every
other gate and extractor is unchanged. Serving goes through the same `prepare()`,
so without this the app would have told most patients to retake a good photo.

**The local refit worked — and was wrong.** With all 26 patients it reached
held-out R² **+0.260**, the best local number yet. Applying the report's own
physiological check before deploying: the dominant term was blue-over-green,
and in the raw data **redder conjunctiva went with *lower* haemoglobin**
(r −0.375). The external cohort shows the physiological direction strongly
(r +0.672). Gray-world amplifies the local inversion but does not cause it; the
classical extractor shows a weak correct sign locally. Cause not established.
**Not deployed.** (An earlier statement in this session that the local model's
coefficients were physiologically fine was wrong and has not been carried into
any document.)

**Leakage check on the extractor decision.** The learned extractor trained on
101 of the external photographs. Restricting to the 116 it never saw: R² **0.354
vs 0.074**, a *larger* advantage than on seen photos (0.286 vs 0.134). The
decision stands on clean evidence.

**Deployed model: learned extractor, chromaticity, 190 external patients**
(27 blur-rejected): `Hb = −39.2123 + 70.042126·r + 75.186694·g`, held-out MAE
**1.505** vs baseline 1.965, R² **+0.317** (in-sample 0.336). Physiological
check passes on the dominant term (bluer → lower Hb, −1.4 g/dL per SD) but the
red-over-green term is ≈ 0. Predictions shrink toward the mean, so
classification near WHO thresholds is weak. Previous model kept as
`runs/linear_model_refined_2026-08-27.json`.

**Eyes-Defy loader bug fixed.** `load_eyes_defy_anemia` crashed on the real
download: 19 of 123 Italy haemoglobin values are text with a decimal comma
(`15,1`). New `_to_float` plus a 3–22 g/dL plausibility gate; verified at 217
patients, all ages and sexes parsed. Test added.

**Server answers the app.** `POST /anemia-detection` added beside `/predict`
(one handler). New `tests/test_api.py` checks both paths return every field
`lib/results.dart` reads. Verified end to end with the deployed model. `/extract`
still does not exist — the app's extract mode will 404.

**Report regenerated** (17 pages): new §7 on learned extraction, the 243-patient
comparison, robustness, the inverted cohort, the deployed model and transfer;
§8–11 rewritten; §6 marked as the superseded 26-patient phase. One correction
made while proofreading: transfer figures were measured with the *first*
learned extractor and are now labelled so rather than attributed to the deployed
one.

**Tests: 32 → 43.** iCloud had evicted three model bundles, including the
deployed extractor; the BTP folders are now set to stay downloaded.
→ `src/anemia/preprocess.py`, `segment_learned.py`, `data.py`, `serve/app.py`,
`tests/test_api.py`, `experiments/13_learned_extractor/unseen_external.py`,
`Progress_Report.pdf`, README, `experiments/README.md`, `UI_INTEGRATION.md`

**16 Sep — v5 is the default. Rotation and scale augmentation separate cleanly.**
Two changes, and the second was the user's: isolate rotation from scale, and
**stop evaluating the extractor on 26 patients when 217 more are labelled.**

`strong` augmentation (v4) had widened rotation AND scale AND exposure together,
buying robustness (0 failures, IoU 0.883) while destroying accuracy
(p 0.013 -> 0.165, mask area +112%). Rotation does not change apparent size, so
it was not the plausible cause of over-prediction — scale was. `--augment
rotation` (v5) widens rotation alone, everything else identical to v3.

**Confirmed.** Local R² nearly doubled, 0.148 -> **0.294**, p 0.013 -> **0.001**,
and v5 has the **highest redness on both cohorts** — wider rotation improved mask
purity rather than merely preserving it. Likely because rotation is a true
invariance of the anatomy while apparent size carries real information, so
teaching scale-invariance discards signal.

**Evaluating on 243 patients rather than 26 settled the extractor question.**
Cohorts reported separately, never pooled (experiment 14: features 3+ sd apart,
transfer R² −1.9):

| extractor | local R² | local p | ext R² | ext p |
|---|---|---|---|---|
| **v5** | **0.294** | **0.001** | 0.336 | 0.000 |
| v3 | 0.148 | 0.013 | 0.320 | 0.000 |
| v4 | −0.072 | 0.165 | **0.337** | 0.000 |
| classical | −0.098 | 0.211 | 0.116 | 0.000 |

At n=26 the classical chain scored p = 0.211 — worse but not demonstrably so,
and the recommendation rested on placement evidence. At n=217 it is **R² 0.116
against 0.336**: the learned extractors nearly **triple** explained variance,
p < 0.0001. The 26-patient comparison was not merely noisy, it was structurally
unable to resolve the effect it was asked about.

**Augmentation damage is domain-specific.** v4 collapsed locally (−0.072) while
remaining best externally (0.337) — scale augmentation made it worse on *one
camera*, not globally. Another reason not to tune against a single cohort.

**Robustness: improved where aimed, worse elsewhere, and partly unexplained.**
All three rotation rows improved and failures halved 18 -> 9, but median IoU fell
0.752 -> 0.723 as zoom/blur rows degraded. v3 and v5 have *identical* scale
augmentation, so this is not a direct consequence of what changed. Plausibly
capacity spent on rotation invariance came out of scale cues; equally consistent
with single-run variance. **Recorded as unexplained — a second seed is needed
before claiming the mechanism.**

Default switched to v5 (`runs/segmenter/unet.pt`); v1–v5 all retained.
→ `experiments/13_learned_extractor/compare_extractors_full.py`, `RESULTS.md`

**16 Sep — Robustness reverses the recommendation; v3 is now the default**
The comparison that said v3 and the old bundle were "equivalent within noise"
scored masks on the 52 local captures — one camera, one operator, one setting. A
model that is more robust to *varied capture conditions* looks identical on that
test, because those conditions are not in it. **Accuracy on a narrow slice is
not robustness**, and the question of whether 547 varied smartphone images
bought anything was never actually asked.

Built the test that asks it: perturb each capture the way a real user would —
rotation, zoom, exposure, motion blur, JPEG, colour cast — and score
**self-consistency**, IoU between the mask on the perturbed image and the mask
on the clean image with the geometric transform undone. No ground truth needed.
Failures counted separately, because a perturbation that returns no mask is a
retake prompt to a patient.

| perturbation | old `unet` | **v3** | classical |
|---|---|---|---|
| rotate 20° | 0.727 (3) | 0.658 (1) | **0.925** (0) |
| **zoom out 0.75×** | 0.490 **(15)** | **0.725 (2)** | 0.677 (0) |
| **bright 140%** | 0.598 **(10)** | **0.684 (5)** | 0.666 (0) |
| median overall | 0.772 **(35)** | 0.752 **(18)** | **0.856 (0)** |

**v3 halves the failure rate, 35 → 18**, with the biggest gains exactly where
extra capture diversity should help: zoom-out failures 15 → 2, over-brightening
10 → 5. With downstream MAE 0.975 / R² 0.148 / p 0.013 against 0.988 / 0.111 /
0.017, **v3 is the better extractor.** Switched: `runs/segmenter/unet.pt` now
holds it, previous bundle preserved as `unet_v1.pt`, failed sequential
fine-tune kept as `unet_v2.pt` as the evidence for the forgetting result.

**The surprise: the classical chain is the most robust thing in the project** —
IoU 0.856, **zero failures**, and 0.925–0.944 under rotation against ~0.7 for
both networks. The reason is structural: it ranks pixels by redness *percentile
within the image*, and rotating a photograph does not change its distribution of
redness. **The property that makes it fragile across cameras — image-relative
decisions — is the same one that makes it stable under transformations of a
single image.** It remains the wrong choice because it reliably measures the
wrong pixels (sclera contamination, experiment 12's caveat), but no extractor
here dominates: classical wins robustness, the networks win accuracy.

Follow-up worth one training run: the networks' rotation weakness is a training
choice, not an intrinsic limit — augmentation used `A.Rotate(limit=20)`, and
pushing rotation and scale jitter harder should close most of that gap.

**Recording the error, not quietly amending it.** The previous entry recommended
keeping the old bundle on the strength of a test that could not measure the axis
in question. The failure mode — measuring the wrong thing and reporting it as if
it settled the question — is worth more than the number.
→ `experiments/13_learned_extractor/robustness.py`, `robustness.json`

**16 Sep — The annotators were never disagreeing about *where* — correction**
The earlier entry called Dice 0.736 "the human ceiling" and said exceeding it
would be fitting label noise. **That was wrong.** Measured across all 547 pairs,
the two annotations are **nested, not scattered**: annotator 2 is larger in 99%
of images, median area ratio **1.56×**, and **98.1%** of the smaller mask lies
inside the larger. Dice correlates with the area difference at **−0.859**.

They agree on *where* the conjunctiva is and disagree on **how far to extend
it** — anatomically expected, since the palpebral conjunctiva has no sharp upper
boundary and transitions continuously into the forniceal conjunctiva in the deep
fold behind the lid. How far you follow it depends on how far the lid was
everted. Thinness exaggerates the gap: median perimeter²/area is **37** against
12.6 for a circle, and Dice punishes thin shapes hard.

So 0.736 is a **systematic convention difference between two self-consistent
annotators**, not irreducible noise, and a model trained on one annotator can
legitimately exceed it against that annotator. Two consequences: the figure
bounds cross-annotator agreement rather than learnability, and — since
A1 ⊂ A2 almost everywhere — the `intersection` default is effectively
**annotator 1's mask**, the conservative convention, not a principled middle.

**16 Sep — Catastrophic forgetting, measured; and the new dataset's real ceiling**
Downloaded the Eye Conjunctiva Segmentation Dataset (Mendeley, CC BY 4.0): 547
smartphone captures with masks from **two independent annotators**. Two things
about it are not in its description.

**The masks are RGB {0, 128}, not {0, 255}.** A conventional `> 127` threshold
reads every mask as empty — and fails *silently*, because an empty prediction
against an empty target scores a perfect Dice. The first agreement run reported
Dice 1.0000 on 547 pairs with 0% coverage before that was caught.

**The advertised "~99.9% agreement" is pixel accuracy, not Dice.** Measured:
the annotators agree at **Dice 0.736 mean / 0.745 median**, 546 of 547 below
0.90. Pixel accuracy is uninformative here because the conjunctiva is a median
**0.76%** of the frame — predicting background everywhere already scores 99.24%.
The human ceiling for this task is ~0.74, which reframes what the dataset buys:
domain coverage, not a better Dice number.

**Then the training failed in an instructive way.** Fine-tuning on the 547
images with `--init` from the existing bundle — old data absent — produced a
model that returned an **empty mask on 42 of the 52 local captures** (against 2
for its parent) while scoring its **best ever validation Dice, 0.8709**. IoU
against its own parent: 0.070. Probability maps confirmed it was not
under-confidence — even at threshold 0.1 it activated on almost nothing.

That is catastrophic forgetting, and **no number on the new dataset warned of
it**. Re-running with *both* datasets in `--data` fixed it entirely: empties
42 → 2, IoU 0.070 → 0.557.

| extractor | median redness | empty | MAE | R² | F1 | p |
|---|---|---|---|---|---|---|
| old `unet.pt` | 16.08 | 2 | 0.988 | 0.111 | **0.710** | 0.017 |
| joint `unet_v3.pt` | 15.21 | 2 | **0.975** | **0.148** | 0.621 | **0.013** |
| classical | 9.64 | 0 | 1.062 | −0.098 | 0.688 | 0.211 |
| baseline | — | — | 1.083 | −0.082 | 0.722 | — |

**Not switching the default.** v3 wins on regression and loses on F1
(0.621 vs 0.710, both under the baseline's 0.722); at n=26 a 1.3% MAE difference
is within noise, and classification is what maps onto clinical consequence. The
two are equivalent, and the F1 gap favours the incumbent. What *is* decisive is
unchanged: both learned extractors clear their noise floor (p 0.013 / 0.017)
where the classical chain does not (p 0.211).

**Two mistakes recorded.** `train_segmenter.py` claimed `--init` "keeps the
general shape prior while adapting the appearance" — it does not, and the
docstring now carries the measurement instead of the assumption. And the
intersection-of-annotators default was chosen from the mask-geometry finding
that contamination costs more than omission; that holds for a fixed extractor on
a fixed domain but does not imply training toward a thinner target generalises
better. Median target coverage 0.76% taught the model to activate on very
little.
→ `scripts/prepare_conjunctiva_seg.py`, `experiments/13_learned_extractor/compare_bundles.py`, `DATASETS.md`

**16 Sep — Measuring against the sclera: portability bought with accuracy**
Tested the fix for cameras not transferring: measure the conjunctiva relative to
the sclera in the same photograph, so both pass through the same lens, sensor and
light. The sclera detector is the brightness-based mask originally rejected as a
conjunctiva extractor precisely because it selects sclera.

**Transfer improves from catastrophic to roughly baseline.** External → local R²
goes from −2.010 (absolute chromaticity) to **+0.032** for a single feature, conjunctiva
a\* minus sclera a\*, the first feature in the project to transfer positively in both
directions (+0.032 / +0.081). **It costs within-cohort accuracy** (external R² 0.346 →
0.221): dividing by a noisy reference adds its noise. Best cohort overlap
(redness difference, separation 0.68 sd) is not best transfer (−0.615) — overlap
achieved by discarding information is not portability.

**Many relative features, penalised, fail to transfer (−2.3 to −3.6).** 46 of 51
relative features are still 2+ sd apart between cohorts, and penalisation selects
for accuracy in the training cohort, not portability — so it weights exactly the
features that don't transfer. Portability has to be a constraint on feature
selection, not something regularisation discovers. Transfer numbers here use the
first learned extractor and have not been repeated with the deployed one.
→ `experiments/15_sclera_relative/`

**15 Sep — Throw everything in with L1: fails at n=26, works at n=217**
Tested the proposal directly — supply **38 colour statistics** (mean, median and
sd of R, G, B, chromaticity, redness, a\*, b\*, L\*, both log-ratios, plus
distribution tails and mask area) and let an L1 penalty zero the useless ones.
Penalty strength re-chosen inside every fold, so the held-out patient never
influences the model.

**At n=26 it fails outright.** L1 gives MAE 1.189 / R² −0.125 — worse than
predicting the cohort mean (1.083 / −0.082) — at a noise-floor **p = 0.516**.
The mechanism is in the selection counts: **only 1 feature survived all 26
folds**, and L1 kept ~25 of 38 rather than pruning. The penalty is chosen by
inner CV on 25 patients where the CV signal is itself noise and under-penalises;
and 45 feature pairs correlate above 0.95, so L1 keeps one member of each
correlated group essentially at random, a different one each fold.

**At n=217 it works.** L1 R² 0.413, ridge 0.421, elastic net 0.414, all beating
the two-feature model's 0.350; ridge also wins MAE (1.386 vs 1.499). Noise floor
**p = 0.000**. Improvement over baseline rises 25.2% → 30.9%. Crucially
**9 features survive all 217 folds** against 1 of 26 — the selection is now
reproducible rather than an artefact of fold assignment.

Two side findings. **Fixed alpha=1.0 and nested alpha are identical at n=217**
(both 1.499/0.350), so the project's hardcoded constant was sound; the
divergence at n=26 (0.988 vs 1.035) was the CV unable to find it, not the
constant being wrong — which partly rehabilitates the published number.
And **ridge slightly beat lasso**, so sparsity is not what helps; the extra
information is. Among correlated features ridge spreads weight where lasso picks
one arbitrarily.

**Caveat recorded:** this shows more features help *within* a cohort and says
nothing about transfer — and going from 2 absolute colour statistics to 38
multiplies exactly the quantities experiment 14 found do not survive a change of
camera.
→ `experiments/13_learned_extractor/kitchen_sink.py` (now takes `--cohort`)

**15 Sep — 217 more patients: the method works within a cohort and fails between them**
Eyes-Defy-Anemia carries haemoglobin, age and sex for 217 subjects (one of 218 rows
unparseable), measured here with our own extractor on the original photographs.

**Within the external cohort the signal is unambiguous:** R² +0.350, p < 0.0001 —
the first result in the project nowhere near the significance boundary. Not a
like-for-like gain over the local +0.111: the external Hb range is wider, which
inflates R²; improvement over each cohort's own baseline is 25% vs 9%.

**Between cohorts it fails completely:** fit external → test local R² −1.944, local →
external −1.160, far worse than predicting the mean. Red chromaticity sits 3.4
local standard deviations apart between cohorts, so a fitted line extrapolates. More
data from the same cameras does not help: the 217-subject fit transfers no better
than the 26. Chromaticity is invariant to light *intensity*, not to which camera
measured it.

**Data trap found:** 19 of 123 Italy haemoglobin values are text with a decimal comma
(`15,1`). A naive parse crashes; `to_numeric(errors="coerce")` would silently drop
those 19 patients. Parsed with a plausibility gate (3–22 g/dL).
→ `experiments/14_pooled_regression/`

**15 Sep — Datasets kept out of git; what's needed to rebuild them is committed**
The two segmentation datasets total ~4.9 GB, GitHub rejects files over 100 MB, and a
repository that size is impractical to clone. Committed instead: `DATASETS.md`
(sources, licences, download commands, contents, known traps),
`scripts/fetch_datasets.sh` (one command, resumable), and `datasets_manifests/`
recording every keep/reject decision, so the exact training sets can be rebuilt.
Data lives outside the synced folder behind a gitignored `datasets/` symlink.
→ `DATASETS.md`, `scripts/fetch_datasets.sh`, `datasets_manifests/`

**15 Sep — A mean is enough; and the median test says the mask is already clean**
Every model here reduces the conjunctiva to a **mean**, discarding the shape of
the colour distribution. Tested thirteen alternatives on pipeline C's masks, all
≤ 3 features because n = 26: robust centres (median, trimmed mean), spread terms
(IQR, std), and distribution shape (quartiles).

**Nothing clearly beats the mean.** `chroma mean` holds at MAE 0.988 / R²
+0.111; `erythema median` edges the MAE by 0.002 g/dL, which is a rounding
difference, while scoring worse on R². **Every three-feature set is worse on
MAE than the two-feature baseline** — the n = 26 overfitting penalty appearing
exactly where experiment 07 said it would.

One result is genuinely ambiguous and is recorded rather than resolved:
`chroma mean + std` has the **highest R² (+0.156) and best noise-floor
p (0.011)** in the set while being **worse on MAE (1.025)**. The divergence is
real — R² penalises large errors quadratically, MAE measures the typical one, so
a feature that rescues a few bad patients at slight cost to the rest moves them
opposite ways. Not treated as a win: thirteen sets tested with no multiplicity
correction, at k = 3 on 26 patients. A lead to re-test on the public datasets.

**The finding that matters for the extractor choice:** a robust centre does
*not* beat the mean — `chroma median` (+0.082) and `chroma trimmed mean`
(+0.103) both score below `chroma mean` (+0.111) on identical masks. A median
exists to resist outliers, so if sclera or lash pixels were still contaminating
the measurement, trimming the tails would help. It does not. **The pixels inside
C's mask are already clean**, which supports the extractor from a completely
different direction than the placement metrics, and says the mask-geometry gains
have been banked — there is no residual contamination left to remove.
→ `experiments/13_learned_extractor/feature_ablation.py`, `feature_ablation.json`

**15 Sep — Contamination costs more than omission, measured**
The learned masks cover a median 1.8% of frame against the classical chain's
10.7%, which raises two opposite worries with opposite predictions. Too few
pixels makes a colour mean noisy → widening should help. But the feature *is* a
mean, so sclera inside the mask shifts it by an amount set by how much sclera
got in, which varies per capture with framing and lid eversion — noise
uncorrelated with haemoglobin. Omitting conjunctiva only shrinks the sample, and
a consistent offset from always measuring the same part of the tissue is
absorbed by the intercept. **Contamination is bias that varies; omission is
variance plus a constant.**

Swept mask geometry, 26 patients, `chroma`, LOO. The result is **monotonic in
both directions**: R² runs from **+0.103** at 1.04% of frame (erode 6px) down to
**+0.007** at 6.83% (dilate 16px), degrading at every single step of widening.
The threshold sweep reproduces the ordering through a different mechanism — no
morphology at all, p ≥ 0.9 gives +0.100 and p ≥ 0.3 gives +0.066 — which is what
makes the trend credible rather than an artefact of one operation. It also
explains the classical chain from the other end: at 10.7% of frame it sits
beyond the worst variant tested.

**Negative result kept:** probability-weighted pooling, proposed as a way to
drop the threshold entirely, was the **worst** variant at R² −0.039. It weights
every pixel in the frame, so background contributes at low weight — which is
contamination by another name. Squaring the weights recovers most of it
(+0.062), still short of a plain tight threshold.

**Deliberately did not change the default.** Picking the best of 13 variants on
the same 26 patients is selection on the test set, and the top rows are within
noise of each other. The trend is the finding; the optimum is not. If a change
is made before more data arrives, raising the decision threshold states a
principle, where a tuned erosion radius just fits a constant.
→ `experiments/13_learned_extractor/mask_geometry.py`, `mask_geometry.json`

**15 Sep — Three-way head-to-head: the learned extractor against both pipelines**
Repeated experiment 11's protocol with a third arm, **importing** its
measurement functions rather than reimplementing them, so any difference in the
numbers is a difference in the pipelines and not in how they were measured.
26 patients, all 52 captures, every capture required to be usable by all three
arms so none is scored on an easier set.

Held out (LOO), best representation each:

| metric | A (means) | B (lab) | **C (chroma)** | baseline |
|---|---|---|---|---|
| MAE | 1.040 | 1.100 | **0.988** | 1.083 |
| R² | −0.104 | −0.129 | **+0.111** | −0.082 |
| Pearson r | −0.094 | −0.197 | **+0.341** | — |
| Recall | **0.923** | 0.769 | 0.846 | 1.000 |
| Specificity | 0.385 | 0.231 | **0.462** | 0.231 |
| F1 | **0.727** | 0.606 | 0.710 | 0.722 |

Fitted on all 26: R² A 0.129, B 0.036, **C 0.277** (baseline 0.000);
MAE A 0.920, B 1.000, **C 0.881**.

Against a 20,000-permutation noise floor, **C is the only arm that clears it** —
in-sample p = 0.022, held out p = 0.015, against A's 0.314 / 0.225 and B's
0.650 / 0.274. B's in-sample R² of 0.036 is *below* its own null mean of 0.080:
shuffled labels fit B better than real ones.

**The result that keeps this honest: C vs A paired is p = 0.554.** C is the
closer estimate on only 12 of 26 patients — its positive mean advantage of
+0.052 g/dL comes from winning by more when it wins, and the CI spans zero. That
does not contradict the noise floor. The floor asks *"does this beat chance?"*
(C yes, A no); the paired test asks *"is C better than A on these patients?"*
(too close to call). A failing its floor does not make C significantly better
than A. **C is the only arm demonstrably extracting signal; the direct C-vs-A
margin is unresolved at n = 26** — the same conclusion experiment 11 reached
about A vs B, for the same reason.

Two further caveats recorded rather than buried. **A keeps the better recall
(0.923 vs 0.846) and F1**, and for a screening tool a missed anaemic patient
costs more than a false alarm — a real argument against switching on regression
metrics alone. And **the best representation per arm was picked off the held-out
table**, which is selection on the test set; it biases all three equally so the
comparison stands, but the absolute values are optimistic. Experiment 11's
nested CV loop has not been replicated here, and is the obvious next addition.

Also reconciled a discrepancy: the earlier two-arm script reports A's erythema
held-out R² as −0.083 where the three-way gives −0.026. The cause is the
inclusion rule — the two-arm run let each extractor use every capture it could
segment, while the three-way drops a capture from *all* arms unless all three
can segment it. The stricter rule is experiment 11's, so **the three-way
supersedes**; the conclusion is unchanged either way.
→ `experiments/13_learned_extractor/three_way.py`, `three_way.json`

**15 Sep — The learned extractor beats the classical chain, and shows why**
Trained the U-Net on Eyes-Defy-Anemia and compared it against the classical
seeded-grabCut chain on the same 26 patients, same `erythema` features, same
ridge, same α. Only the pixels being measured differ.

| Extractor | in-sample R² | noise floor | p | **held-out R²** | p |
|---|---|---|---|---|---|
| Classical (refined) | 0.132 | 0.075 | 0.175 | **−0.083** | 0.194 |
| **Learned (U-Net)** | **0.280** | 0.078 | **0.020** | **+0.112** | **0.015** |

**The first arm in this project to clear its permutation noise floor held out.**
The classical chain's held-out −0.083 sits on its −0.0816 baseline to three
decimals — on unseen patients it carries nothing beyond the cohort mean.

Two traps avoided. The obvious comparison — `fit-linear` with each extractor —
gives +0.007 vs +0.048, and is **wrong**: the thinner learned masks trip the
blur gate more often, so those are 26 and 21 patients with different baselines
(−0.082 vs −0.103). Restricted to patients usable under both, all 26 survive.
And every R² is quoted against a 20,000-permutation noise floor, because k = 2
on n = 26 scores ≈ +0.076 on pure noise.

**The finding that matters beyond the metric:** the learned masks score redness
**+15.89** against the classical chain's **+9.36** on the same pixels, at a
fifth of the area, with mask IoU of only 0.127. Visual review confirms the
classical masks extend across sclera and lid skin. Experiment 12 asked whether
*mean* redness inside the mask was positive and A passed 52/52 — but a mask that
is half sclera still averages positive. **That test detects a mask that is off
the tissue, not one diluted by it.** A's colour features have been measured over
a mixture all along, which is a plausible contributor to the weak R² throughout.
Caveat now recorded on experiment 12's row.

Limitations kept: n = 26 with many comparisons already run against it; learned
masks are thin (0.3–3.1% of frame) and 2 of 52 come back empty; trained on 101
masks from two sites, neither the deployment hospital. The deployed model is
**not** switched over — that waits on a decision about thin masks.
→ `experiments/13_learned_extractor/`

**15 Sep — Stage traces for the learned pipeline, and a scare that did not hold**
The stage tracer only ever traced the classical chain — `anemia stages` took a
`--segmenter` directory but no `--extractor`, so the third pipeline had no
step-by-step walkthrough while the other two did. Added `--extractor`, and made
panel 03 adapt: the classical chain scores redness, the learned one scores a
per-pixel probability, and the panel should show whatever the segmentation
decision is actually made from. They are read differently and the documentation
now says so — the redness map is stretched per image, so its colours mean
something only within one capture, while the probability map is absolute 0–1
and comparable across captures.

Generated all 52 sheets into `experiments/01_stage_traces/learned_pipeline/`,
matching the layout of the other two pipelines.

Looking at them turned up **two captures where the network returns no mask at
all** — `left_eye/26.jpeg` and `right_eye/11.jpeg`. The first is one of the
three already-known blurred captures. The second is **sharp** (focus 133) but
very pale, which raised an obvious worry: if extraction fails preferentially on
pale conjunctivae, it fails on exactly the anaemic patients the system exists to
find, and mask size would then vary with the quantity being predicted.

**Tested it; it does not hold.** Mask area against Hb across all 52 captures
gives r = −0.047 (p = 0.738) for the learned extractor — no relationship. Mean
Hb among the nine smallest masks is 11.0 against 11.3 for the rest, well inside
noise at that n. Worth recording that the *classical* extractor shows the
stronger trend of the two, r = −0.253 (p = 0.070), though at n = 52 that is a
trend and not a finding either.

Both failures are caught by QC as "conjunctiva too small", so they surface as a
retake prompt rather than a confident wrong number.
→ `src/anemia/cli.py` (`cmd_stages`), `experiments/01_stage_traces/learned_pipeline/`

**15 Sep — Half of Eyes-Defy-Anemia's annotations are bounding boxes, not masks**
The dataset downloaded and does ship recoverable masks: each annotation is the
frame downscaled ~3.7× with everything outside the region blacked out, and on
foreground pixels it differs from the downscaled original by a mean of 0.06–0.60
out of 255 — resampling noise, so registration is exact.

But the annotations are not uniformly traced. **About 45% of `_palpebral`
annotations are axis-aligned rectangles**, several containing sclera, iris and
pupil; `_forniceal_palpebral` is 95% rectangles. Training on the mixture would
teach the network to predict boxes containing an eye, which is worse than the
extractor it replaces. Rectangles are detectable without opening files — a
rectangle fills its own bounding box, extent ≈ 1.0, against a median of 0.19 for
traced crescents — so `scripts/prepare_eyes_defy.py` filters on that and writes
a manifest of every keep/reject decision. **101 usable pairs from 101 distinct
subjects** survive, one capture each, so patient leakage is impossible rather
than merely avoided. Held-out Dice **0.841**.
→ `scripts/prepare_eyes_defy.py`

**15 Sep — A trainable extractor, split so that serving cannot rewrite what it serves**
Extraction has been classical throughout: a seeded grabCut driven by redness
percentiles. That design has a ceiling built into it. `np.percentile(red, 92)`
is an *image-relative* cut, so it assumes every capture devotes roughly the same
share of its pixels to conjunctiva — but a wide shot where the tissue is 2% of
the frame and a macro crop where it is 60% get very different thresholds from
the identical rule. This is the same class of defect as the percentile darkness
gate that once deleted the healthy patients from the cohort, and no amount of
tuning removes it; only learning the decision from examples does.

Added a fine-tuned U-Net backend (`segmentation_models_pytorch`, ImageNet
encoder) behind the existing `(rgb) -> (mask, name)` contract, so it drops into
the `EXTRACTORS` registry beside the classical chain with nothing downstream
changing. Two files, deliberately asymmetric:

- `src/anemia/segment_learned.py` **reads** a parameter bundle. It has no
  optimiser, no dataset code, and no way to write one.
- `scripts/train_segmenter.py` **reads and writes**. It is the only writer.

The asymmetry is the point: a bundle sitting on the server is then always the
recorded output of a training run, never something a serving process mutated.
A bundle carries its own training-time conditioning (encoder, input size,
normalisation, threshold) so a model trained at 384px still serves correctly
after the default changes, plus provenance — which datasets, how many images,
which subjects were held out — because in six months that is the only way to
answer *which images produced the model currently estimating haemoglobin*.
Incomplete or foreign bundles refuse to load rather than segmenting nonsense,
the same rule `LinearModel.load` already follows.

Two decisions worth recording. **Splits are by subject, not by image** — the
same eye photographed twice is not two independent samples, and an image-level
split would measure recall rather than generalisation. **Photometric
augmentation is aggressive and that is safe here**, which it would not be
elsewhere in this project: the output of this model is a *mask*, not a colour,
so teaching it that conjunctiva stays conjunctiva under a warm bulb is exactly
the invariance wanted — and the colour statistics that feed haemoglobin
regression are measured afterwards on the original untouched pixels.

`--init` warm-starts from an existing bundle, which is how the hospital cohort
gets folded in when it arrives: continue training from public-dataset weights on
the union, keeping the general shape prior while adapting the appearance. Better
than either training from scratch on a small local set or freezing a model that
never saw this domain.

Verified end to end on synthetic phantoms: train → save → load through
`get_extractor` → segment, warm-start from the saved bundle, and three classes
of malformed bundle correctly refused. **25 → 32 tests.**
→ `src/anemia/segment_learned.py`, `scripts/train_segmenter.py`,
`scripts/inspect_dataset.py`, `segment.py` (`get_extractor`, `TRAINED_EXTRACTORS`)

**Not yet trained on real data.** Eyes-Defy-Anemia needs a Kaggle API token to
download, which is a credential and had to wait.

**15 Sep — Added a dataset inspector, because "segmented" is ambiguous**
Most published anaemia datasets distribute *cropped conjunctiva patches*, not
masks. A crop cannot train a segmenter at all: the input a segmenter needs is
the whole eye, and the crop has discarded it. The distinction is invisible on a
download page and costs an afternoon to discover by hand, so
`scripts/inspect_dataset.py` measures it — a mask is near-binary, a photograph
has a broad histogram, and a cutout on black is a mask in disguise that can be
recovered. Run it on every new dataset before planning around one.
→ `scripts/inspect_dataset.py`

**15 Sep — Moved the Python environment off iCloud**
Training stalled with `errno=60` — a timeout mmapping numpy's C extension. The
cause was iCloud eviction, not a bad install: Desktop is synced, the disk is 88%
full, and macOS had evicted **266 `.so` files** out of the virtual environment.
`brctl download` would not materialise them. Rebuilt the environment at
`~/btp-local/venv` (Python 3.12, outside any synced folder), which also gets
MPS/Apple-GPU acceleration for training. The repository itself stays on Desktop;
only the environment and the datasets move, and `datasets/` is now gitignored
alongside the existing `runs/` and `*.pt` rules.

---

## Week of 18–24 Aug 2026

**20 Aug — Quality control now gates on everything it measures**
`assess_quality` computed focus, mask ratio, clipping and exposure, but only
the first two could reject a capture: `QualityConfig` had no clipping threshold
at all, so a flash-blown photo returned a confident number while `PIPELINE.md`
claimed three enforced checks. The same shape as B.1, one layer down. Added a
clipping gate and the illuminant-neutrality check that had been proposed since
the lighting stress test, so **all four statistics are now enforced** and each
has a test that a failing capture is rejected (19 → 25 tests).

Both thresholds were calibrated by measurement first. The cast ratio separates
by an order of magnitude on each side — the 52 cohort captures span 1.07–1.50
and a neutral test capture 1.7, against 22.6–206.0 for the five unusable
coloured illuminants — so the threshold sits at 3.0, and it is measured
**before** white balance, which exists to remove the cast and would otherwise
drive the ratio to ~1.0. Clipping is set generously at 5% because the cohort's
worst capture clips 0.3%, and it is recorded as **not calibrated against a real
blown-out capture**: the repository contains none.

Verified the change costs nothing: refitting the deployed model under the new
gates returns byte-identical results (26 patients, 3 captures skipped, same
equation, same metrics). Also added `QualityConfig` to the crop-cache key —
without it, tightening a gate would silently reuse verdicts computed under the
old thresholds, which is the B.8 trap applied to quality rather than extraction.
→ `src/anemia/config.py`, `preprocess.py`, `PIPELINE.md` §3.4

**20 Aug — "50 of 52" was wrong; the measured figure is 46, and the argument got stronger**
The claim that pipeline B's masks land off-tissue on 50 of 52 captures was the
decisive non-statistical argument for deploying A, quoted in the report and four
documents, and rested on visual review with no artifact behind it. Built one.
Measuring redness index inside each mask across all 52 captures gives **46 of
52**, not 50 — corrected everywhere.

Two findings came out of it that are stronger than the original claim.
**Pipeline B's placement is statistically indistinguishable from the brightness
baseline** (median redness −3.67 against −3.66; 46 off-tissue against 47) — the
baseline measured at Dice 0.000 against ground truth, which selects sclera. And
**the two masks overlap by a median of exactly 0.000**: on more than half the
captures they share no pixels, so the pipelines are reading different parts of
the photograph rather than treating the same tissue differently. That reframes
experiment 11's paired test, which compares two models one of which was largely
not looking at conjunctiva.

A follow-up question — *are the six on-tissue captures meaningful, given the
model was fitted on them?* — sharpened it further, and corrected a claim made
in the first write-up. They are not an existence proof. They are marginal
(median redness +0.87 against A's +8.49 on the same six images) and vanish as
the threshold tightens: 3 survive at +1.0 and none at +5.0, while A holds 52 of
52 across that whole range. More importantly, **no patient has both eyes
on-tissue**, and features are averaged per patient before fitting — so of B's
26 feature vectors, 20 average two off-tissue captures and 6 average one
on-tissue with one off-tissue. **Not one is a clean conjunctiva measurement.**
That is worse than being uniformly wrong: a consistently misplaced mask at
least measures the same thing every time, whereas this gives each patient a
different mixture of tissue types. B's count is now always quoted with its
threshold attached.
→ `experiments/12_mask_placement/`

**20 Aug — Serving path completed: linear model servable, extractor honoured, provenance exposed**
Three defects in the serving path, all of which would have surfaced as silently
wrong answers rather than errors:

- **`LinearPredictor` ignored the extractor recorded in the model file**, always
  using `refined`. `predict_folder.py` was the only consumer of that field.
  Serving a cielab-fitted model applied refined masks, and features measured
  inside a different mask are not comparable. It now reads the name back, as
  the documentation had claimed all along.
- **`serve/app.py` could only load a neural checkpoint**, and no network exists,
  so the API was unrunnable. `ANEMIA_LINEAR_MODEL` now serves the fitted linear
  model over the identical contract, with no PyTorch dependency.
- **`/health` reported nothing about which model produced a number.** It now
  returns extractor, representation and the equation, so the client can display
  provenance — the placeholder `UI_INTEGRATION.md` was holding open.

Verified end to end against the real service: `/health` 200 with provenance, a
good capture scoring 11.05 g/dL, a blurred capture returning `usable: false`
with its reason, and 400 on both an empty upload and a non-image.
→ `src/anemia/predict.py`, `serve/app.py`, `UI_INTEGRATION.md`

**20 Aug — Training no longer selects its model on the test fold**
`train_fold` scored the test fold every epoch, kept the best-scoring weights,
early-stopped on it, and then reported that fold's metrics. Every prediction was
out-of-fold, so nothing looked wrong, but the *model* had been selected using
the evaluation data and the reported figures were optimistically biased. A
patient-grouped validation set is now carved out of the training fold and drives
both early stopping and best-epoch selection; the test fold is scored exactly
once, after training ends. When a fold is too small to spare a validation split
the run trains the full schedule with no early stopping rather than peeking.
No CNN has been trained on real data, so no published number changes — this
protects the first one.
→ `src/anemia/train.py`

**20 Aug — Diagnostics figure regenerated for the deployed model**
`linear_model_diagnostics.png` still plotted the channel-mean model while
appearing in the report as the current diagnostic. Regenerated from the
deployed configuration. Prediction spread rises from **23% to 33%** of the
actual spread, Bland–Altman limits narrow from −2.94/+2.98 to **−2.68/+2.77**,
and the residual slope falls from **+2.02 to +0.46** — regression toward the
mean is weaker, not absent. Propagated to the report caption, `METRICS.md`,
the personal study notes and the meeting notes.

**20 Aug — Housekeeping across shared files**
`matplotlib` and `reportlab` added to `requirements.txt` — both were imported by
scripts and absent from the install, so `plot_results.py` and the report
generator failed on a clean environment. Cleaned 16 stray citation tokens and
repaired three collapsed tables in `README_cielab_pipeline.md`, and replaced two
absolute paths that only resolve on one machine with an explicit note that both
must be supplied. Fixed unescaped pipes in the Dice/IoU and `mean(|e|)` formula
cells of `GLOSSARY.md` and `METRICS.md`, which were splitting those table rows
on render. Corrected `LIGHTING_STRESS_TEST.md` §6, which described its own
spread ratios as "3× and 1.5×" where §2 reports 1.9× and 1.7×.

**20 Aug — Documentation audit: three generations of numbers separated, and the deployed model finally documented**
An audit found **three** different models in circulation across the documents,
all described as "fitted on all 26 patients": channel means under leave-one-out
(in-sample +0.103 / held-out −0.166), the head-to-head erythema model
(+0.133 / −0.094), and the model that actually ships. The last of these —
erythema index with the **QC gate applied before fitting** — appeared in no
shared document at all:

```
Hb = 9.8308 + 15.101390·log(R/G) − 12.790747·log(R/B)     [g/dL]
```

in-sample MAE 0.918 / R² **+0.195**, held out MAE 1.033 / R² **+0.007**,
against a leave-one-out baseline of 1.083 / −0.082. Applying the gate is what
lifted held-out R² above zero, making this **the only configuration in the
project that beats its baseline on unseen patients**. Reproduced from the images
independently; matches `runs/linear_model.json` to four decimals. It is now the
headline in `README.md`, the report §6.8, and `experiments/README.md`.

The report was the worst affected: §6.8 presented the *channel-mean* equation
under the heading "fitted on all 26 patients", four pages after §6.4's table for
a different model with the same caption. §6.9's physiological-check conclusion
and §6.10's protocol table were both channel-mean results read as current.
All three rewritten.

Added supersession banners to experiments **08**, **09** and **10**, matching
the ones already on 05 and 07, each naming the representation it used. Rebuilt
`experiments/README.md`, whose index table had also fragmented into four
separate tables. Retired the stale figures from the meeting notes' crib sheet
and the personal study notes §10.

**The physiological check now passes.** The "coefficients are backwards"
finding is true of channel means only; the deployed model's dominant
standardised coefficient is **+1.049 on log(R/G)** — red over green, positive.
The check was specified while it was failing and reversed when re-applied
unchanged, which is what makes it a test rather than a narrative.
→ `RESEARCH_LOG.md` A.9d

**20 Aug — Fitted-on-all comparisons corrected: measured against noise, not against zero**
The fitted table was being read as "pipeline A is clearly ahead of the
baseline (R² 0.133 vs 0)". That inference does not hold: in-sample the baseline
is the full-sample mean, whose R² is 0 by definition, and a fitted model with a
free intercept essentially cannot fall below it — the comparison is unlosable.
Measured against a permutation null instead (Hb shuffled against the same
features, refitted 20,000 times), **two free parameters on 26 patients score
R² ≈ +0.075 on pure noise**. A's +0.133 gives p = 0.172; B's +0.010 is *below*
the one-parameter noise mean of +0.041; the deployed model's +0.195 gives
p = 0.074, and p = 0.067 held out — the best in the project and still short of
significance.

Two confounds found alongside: **A carries one more free parameter than B**
(erythema 2 features vs `a_only` 1), so adjusted R² is +0.058 vs −0.031 and the
matched-at-k=1 pair is +0.060 vs +0.010; and **B's fitted classification is
identical to the baseline's**, patient for patient, while A's entire
accuracy/F1/specificity margin rests on **one net patient** of 26. A still leads
on every matched framing, so the deployment recommendation is unchanged — the
claim is what narrowed. Propagated to the report §1, §6.4, §6.7, §6.10,
`experiments/11_head_to_head/`, `README.md`, the personal study notes,
the meeting notes, the working notes.
→ `RESEARCH_LOG.md` A.9c

**20 Aug — Report generator was not the one producing the report**
`scripts/make_progress_report.py` in the deliverable had gone stale: 17 KB and
eight sections, with no `--diagnostics-figure` argument, so the documented
regeneration command **failed outright** and would in any case have produced a
report predating the colour-normalisation and quantitative-evaluation sections.
The working copy held the real generator (56 KB, ten sections). Synced, then
corrected and re-rendered. Report now 14 pages. Worth checking `diff` between
the two folders on this file before trusting any future regeneration.

**20 Aug — Client integration guide written; API contract documented and frozen**
Added `UI_INTEGRATION.md` so the app can be built without reading the research
code: the five files that matter, the `POST /predict` and `GET /health`
contract, both success shapes field by field, the three quality-control
rejection strings, screen-by-screen requirements, and a checklist. Two points
are stated as requirements rather than suggestions — the client branches on
`usable` and not on the HTTP status (a rejected capture is a 200), and it never
renders a number when `usable` is false. Also documented that the pipeline is
deliberately **not** a user-facing choice: extraction method and colour
representation are fixed when the model is fitted and recorded in the model
file, because measurements taken inside a different mask are not comparable, so
the client displays provenance rather than offering a selector.

Included a runnable mock server in the guide so client work is not blocked on a
fitted model file. Its threshold function was checked against
`anemia_threshold()` across all six WHO bands plus the unknown-demographics
fallback (13 cases, no mismatches), and its payload keys against
`Prediction.to_dict()`. Recorded three gaps the guide has to work around for
now: the service loads a neural checkpoint only, so the linear model cannot yet
be served over HTTP; `LinearPredictor` does not read the extractor name back
out of the model file the way `predict_folder.py` does; and `/health` does not
report which extractor and representation are active. Propagated to
`README.md`, `CODE_GUIDE.md` and `CONTRIBUTING.md` §6.
→ `UI_INTEGRATION.md`

**20 Aug — Report made reproducible from a cold checkout**
The report generator was being pointed at a figure living in a temporary
session directory, which has since been deleted — meaning the report could not
be regenerated by anyone, including us. Regenerated that figure into
`figures/phantom_contact_sheet.png` so all three report figures are now
repo-local, and recorded the full working command. Report re-rendered from
repo files only.

**20 Aug — Baseline corrected to match the framing**
The fitted-on-all comparison was reporting a **leave-one-out** baseline
alongside models fitted on every patient — two different framings in one table.
Corrected: the fitted comparison now uses the full-sample mean, whose
**R² is exactly 0 by definition** (predicting the mean makes numerator equal
denominator), giving MSE 1.877, RMSE 1.370, MAE 1.041. The previous −0.082 is
now shown to be exactly `1 − (n/(n−1))² = 1 − (26/25)²`, an artefact of the
leave-one-out protocol rather than a property of the data, and is reported only
alongside leave-one-out results. **This changes the reading of pipeline B**: its
margin over the correct baseline is negligible (MSE 1.858 vs 1.877, MAE 1.038
vs 1.041, R² 0.010), so only pipeline A is meaningfully ahead. Propagated to
report §6.4, `experiments/11_head_to_head/`, the personal study notes,
`METRICS.md`, `README.md`, the meeting notes, the working notes.

**20 Aug — All reported metrics recomputed with models fitted on every patient**
Both pipelines refitted on all 26 patients and the full suite recomputed, then
propagated to every document. Fitted on all: **pipeline A MSE 1.627, RMSE
1.275, MAE 0.944, R² 0.133, Pearson r 0.381, accuracy 0.654, precision 0.600,
recall 0.923, specificity 0.385, F1 0.727**; pipeline B MSE 1.858, MAE 1.038,
R² 0.010, F1 0.722; baseline MSE 2.030, MAE 1.083, R² −0.082, F1 0.722.
**A wins 10 of 11 metrics and both pipelines now show positive R²**; B's only
lead is recall, obtained by flagging every patient (specificity 0.231). The
held-out figures are retained alongside in every document (A: MAE 1.062,
R² −0.094) because the gap between the two is the overfitting, and the fitted
numbers describe the model while the held-out ones describe a new patient.
Updated: `Progress_Report.pdf` §6.4–6.5, `experiments/11_head_to_head/`,
the personal study notes, `METRICS.md`, `README.md`, the meeting notes,
the working notes.

**20 Aug — Deployable linear model: fit once, predict without retraining**
Separated fitting from prediction. `anemia fit-linear` fits on **all 26
patients** and writes `runs/linear_model.json` carrying coefficients, feature
specification, extractor name, standardisation constants and both metric sets;
`predict_folder.py` loads that file and scores a folder of new patients with no
training (26 patients in ~20 s). Metrics now reported as in-sample beside
held-out: **in-sample MAE 0.918, R² 0.195**; held-out MAE 1.033, R² +0.007;
baseline MAE 1.083, R² −0.082. Applying the QC gate before fitting is what
moved held-out R² above zero. Extraction is now swappable via an
`EXTRACTORS` registry (`refined`, `redness`, `brightness`, `grabcut`,
`cielab`) selectable with `--extractor`; the choice is recorded in the model
file and read back at prediction time, since features measured inside a
different mask are not comparable. Confirmed the provision works: fitting with
`--extractor cielab` gives held-out R² −0.159 against refined's +0.007.
Added `CONTRIBUTING.md` with recipes for adding an extractor, a feature
representation, a metric, a dataset, or a QC rule.
→ `src/anemia/linear_model.py`, `predict_folder.py`, `CONTRIBUTING.md`

**20 Aug — Documentation synced to the head-to-head results**
Propagated the two-pipeline comparison across the set: the personal study notes
§11 rewritten with the current metric table, the paired-test result and the
explanation of why R² stays negative while MAE beats baseline; `README.md`
status line updated to "statistically indistinguishable from predicting the
cohort mean"; experiments 05 and 07 marked as superseded in part, with pointers
to 11; `METRICS.md` flagged that its real-cohort figures predate the
illumination-invariant representations. Added `RESEARCH_LOG.md` entry A.9b
recording a partial miss of our own — concluding "feature richness is not the
constraint" from two data points was premature, and physically motivated
representations (chromaticity, erythema index) later improved both pipelines.

**20 Aug — HEAD-TO-HEAD PIPELINE COMPARISON (principal work of the week)**
Both pipelines trained and scored as models on identical patients under an
identical protocol: five feature representations each, the full metric suite
(MSE, RMSE, MAE, R², bias, Pearson r, accuracy, precision, recall, specificity,
F1, confusion), nested cross-validation, and a paired test of the difference.
Added two illumination-invariant representations not previously tried —
chromaticity coordinates R/(R+G+B) and the erythema index log(R/G) — which
**materially improved both pipelines**: A's R² rose from −0.166 to −0.094 and
F1 from 0.688 to 0.727; B's R² from −0.260 to −0.128. Best configurations:
A = gray-world + erythema (MAE 1.062, R² −0.094, F1 0.727, accuracy 0.654);
B = CIELAB + a\* alone (MAE 1.118, R² −0.128, F1 0.686). **A wins 9 of 11
metrics**, and beats the baseline on MAE, accuracy, precision, specificity and
F1. However the paired permutation test gives **p = 0.408** with a bootstrap CI
of [−0.067, +0.181] spanning zero: **the pipelines are not statistically
separable at n = 26**, and A wins on only 15 of 26 patients. Under nested
selection neither beats baseline. Also documented why R² stays negative while
MAE beats baseline: the leave-one-out baseline cannot use the full-sample mean,
so the practical floor is ≈ −0.08, not 0 — the correct statement is
"indistinguishable from predicting the mean", not "catastrophically wrong".
Recommendation: A as provisional default on non-statistical grounds (leads in
all five earlier experiments; higher specificity at equal recall; B's
segmentation lands off-tissue in 46 of 52 captures), with the comparison to be
repeated on ~900 patients once the segmenter is trained.
→ `experiments/11_head_to_head/`, report §6.3–6.6

**20 Aug — Fitted equation extracted; uncertainty quantified; coefficients fail a physiological check**
Wrote out the actual linear model rather than only its scores:
`Hb = 12.5311 + 0.010191·(mean R) − 0.058732·(mean G) + 0.035552·(mean B)`,
with the standardised form, inputs and cohort statistics, and a worked example
(patient 1 → 11.142 g/dL against a laboratory 12.0). **The coefficients are
physiologically backwards**: haemoglobin makes tissue red, so red should
dominate positively, but it is the weakest term (+0.137 standardised) while the
model runs on green negatively (−0.977) and blue positively (+0.647) — a
blue-minus-green contrast closer to residual colour cast than to blood. This
corroborates the negative metrics from an independent direction and gives a
concrete check for the next iteration: re-fit after training the segmenter and
inspect coefficient signs before the metrics. Also added bootstrap confidence
intervals over patients (MAE [0.708, 1.490], R² [−0.512, **+0.021**], F1
[0.462, 0.850]) — the R² interval reaches above zero, so the defensible claim
is "no better than the mean" rather than a precise −0.166. Documented why
leave-one-out yields one pooled metric rather than 26, and verified training
size is not driving results (2-fold 1.063, 5-fold 1.034, 13-fold 1.070, LOO
1.073).
→ `experiments/10_fitted_equation/`

**20 Aug — Training protocol verified; negative R² explained**
Confirmed the model does train and characterised why held-out R² is negative.
Fitting on all 26 and scoring in-sample gives R² **+0.103**; scoring only on
unseen patients gives **−0.166** — the gap is the overfitting, and it answers
"was training actually run" directly. Checked the finding is not a
leave-one-out artefact: 5-fold (−0.175) and 13-fold (−0.168) agree. Also
measured why no train/validation/test split is used at this size — over 200
random 60/20/20 splits of identical data, test R² ranges from **−2.43 to
+0.78**, so a single split reports which patients landed where rather than
model quality. Documents when a three-way split becomes correct (~900 images)
and why no hyperparameter is tuned (no validation budget; tuning on the
evaluation set would leak).
→ `experiments/09_train_test_protocol/`

**20 Aug — Documentation set brought into sync**
Audit found four documents stale after the research-log, glossary and
diagnostics work: they had not been updated to reference the new material.
Fixed across the set — the personal study notes gained an abbreviations
preamble, a section reading the diagnostic plots, a compressed per-stage
buys/costs table and a document map; the meeting notes gained the figure as a
visual aid plus the failure-documentation point and four new reference numbers;
`CODE_GUIDE.md` gained an experiments section (all eight studies and their
scripts, previously undocumented) and a documentation map; the study plan
gained a table of in-repo study material and a revised first exercise (read the
six failed extraction attempts before rebuilding it); `PIPELINE.md` gained
pointers to the glossary and research log. Also fixed a stale path in
the study plan left over from the experiments reorganisation.

**20 Aug — Research log, glossary and model diagnostics added**
Created `RESEARCH_LOG.md`: a complete record of every approach tried including
those that failed — six method attempts rejected with measured reasons, ten
defects found and fixed, six environment problems, four null results, and seven
approaches deliberately not attempted, each with justification. Includes
predictions that turned out wrong (the gray-world + a\*/b\* hybrid was recorded
as expected favourite and came third). Created `GLOSSARY.md` defining all
abbreviations across clinical, imaging, machine-learning and metric domains,
and added an abbreviations table to the report before first heavy use. Added
three-panel diagnostic plots for the linear model (predicted vs actual,
Bland–Altman, residual structure): prediction spread is 23% of actual spread,
making the absence of signal visible rather than only numerical. Report now 9
pages.
→ `RESEARCH_LOG.md`, `GLOSSARY.md`, `experiments/06_metrics_evaluation/`

**20 Aug — Advantages and limitations documented for every stage and test**
Audit found trade-off coverage was patchy: the colour-method comparison existed
only in its experiment file and had never reached the report, and the metrics
write-up had no limitations section at all. Added: a side-by-side advantages /
limitations table for the two colour-normalisation approaches in the report
(§5.2), a table of what the lighting stress test can and cannot establish
(§5.3), strengths and limitations of the evaluation itself (§6.4), per-stage
trade-off tables for all seven pipeline stages in `PIPELINE.md`, a per-metric
advantage/limitation table in `METRICS.md`, and a per-regulariser comparison in
the regularisation results. Report now 8 pages.
→ `Progress_Report.pdf`, `PIPELINE.md`, `experiments/04`, `06`, `08`

**20 Aug — Regulariser choice justified empirically**
Compared OLS, ridge, lasso and elastic net on the cohort with leave-one-out,
measuring coefficient stability as well as error. The features are
near-collinear (max |r| = 0.993 between channel means), which is exactly where
lasso becomes unstable — it keeps one of a correlated group arbitrarily and the
choice shifts between folds. Unregularised OLS wobbles ~300x more than ridge
(stability 11.499 vs 0.042) and posts the worst MAE (1.282). Lasso at
alpha = 0.5 zeroes every coefficient and lands exactly on the baseline,
independently confirming that no feature earns its keep. Ridge is the steadiest
and best (1.068 at alpha = 10), still only at baseline. Documents why alpha is
left at 1 rather than tuned: tuning on the evaluation set would leak.
→ `experiments/08_regularisation/`

**20 Aug — Stage traces consolidated into single contact sheets**
Both tracers now emit one captioned contact sheet per capture instead of nine
or ten separate files, so consecutive stages sit side by side and can be
compared directly. Panels are letterboxed rather than stretched — the stages
differ in shape (full frame, square working image, crescent crop) and
stretching would misrepresent the geometry the pipeline exists to preserve. A
header strip carries filename, backend, mask ratio, focus and QC verdict.
`--separate` restores per-file output; `--panel` / `--per-row` adjust layout.
All 104 traces regenerated (52 captures × 2 pipelines), 19M → 17M.
→ `src/anemia/cli.py`, `cielab_stages.py`, `experiments/01_stage_traces/`

**20 Aug — Progress report expanded to cover all findings**
Report grown from 5 to 7 pages with two new sections: colour normalisation and
lighting robustness (cohort correlation, six-illuminant stress test, the
dead-channel physics and the proposed illuminant-neutrality check), and
quantitative evaluation (how the Hb figure is produced, why each metric is
reported, synthetic results, and the real-cohort linear floor including the
24-feature overfitting demonstration). Summary, verification status and next
steps updated to match.
→ `Progress_Report.pdf`

**20 Aug — Both pipelines compared end-to-end; linear floor established**
Scored pipeline A (gray-world + refined extractor, RGB) against pipeline B
(CLAHE-on-L + own segmentation, a\*/b\*) two ways: end-to-end, and with a
shared mask to isolate colour representation from segmentation. **A wins on
every metric in both framings** (end-to-end MAE 1.073 vs 1.135, F1 0.688 vs
0.647) — a fourth independent result in the same direction, though the margin
sits inside the noise at n=26. Models are deliberately **linear only** (OLS and
ridge, no neural network) to fix the floor a CNN must clear. **Every arm is at
or below the predict-the-mean baseline** (all R² negative, all F1 ≤ 0.722).
Feature richness makes it worse, not better: 24 features on 25 training points
sends plain OLS to MAE 11.4 g/dL and R² −118.9 — a concrete overfitting warning
for the network stage, where ~8.4M parameters will be trainable.
→ `experiments/07_pipeline_comparison/`

**20 Aug — Hb mechanism documented and full metric suite computed**
Wrote up exactly how a crop becomes a g/dL value (ImageNet normalisation →
ResNet-18 with a `Linear(512→1)` head → de-standardisation by the checkpoint's
stored mean/std → per-patient WHO threshold), including what the libraries do
and do not provide: no haemoglobin library exists, the mapping is trained.
Added an evaluation script computing MSE, RMSE, MAE, median AE, bias, R²,
Pearson r and Bland–Altman limits, plus accuracy, precision, recall,
specificity, NPV, F1 and balanced accuracy, each against a baseline.
Synthetic phantoms: R² 0.888, MAE 0.666, recall 1.000, F1 0.986 (trivial task
by construction). **Real cohort: R² −0.166, Pearson r −0.238, F1 0.688 vs
baseline 0.722 — the channel-mean model does not work on real data**, with 2
false negatives among 13 anaemic patients. Bounds what mean colour alone can
achieve; a CNN on the full datasets is the next test.
→ `experiments/06_metrics_evaluation/`

**19 Aug — Workspace split into pipeline vs experiments**
All exploratory output moved under `experiments/` (stage traces, extraction
check, colour-normalisation study, lighting stress, representation ablation),
each folder self-contained with its own findings; `experiments/README.md`
indexes them. Top level now holds only the pipeline, docs, data and papers.

**19 Aug — Representation ablation, including the gray-world + a\*/b\* hybrid**
Four colour representations scored on Hb prediction over the 26-patient cohort
(shared masks, channel-mean features, leave-one-patient-out ridge). **No arm
beats the predict-the-mean baseline** (best: gray-world RGB, +0.010 g/dL —
noise at n=26), so mean colour alone carries no usable Hb signal at this scale.
Gray-world did beat its uncorrected counterpart in both pairs (~0.08 g/dL),
consistent with earlier findings. The hybrid did not win; downgraded from
"favourite" to "an arm to test with a CNN".
→ `experiments/05_representation_ablation/`

**19 Aug — Colour-management correction to the lighting stress test**
Captures are Display P3; the first export preserved that profile, which OpenCV
read as sRGB — a systematic shift against the cohort's true-sRGB images.
Re-exported with explicit P3 → sRGB conversion. This strengthened the findings:
correctly converted, four of six illuminants leave a colour channel effectively
dead (blue: R = 0.9/255; red and purple: G ≈ 1/255). JPEG compression verified
negligible (< 0.5 a\* units vs lossless PNG). All six frames recorded as the
same right eye in `capture_metadata.csv`.

**19 Aug — Lighting stress test (one subject, six illuminants)**
Photographed one volunteer's conjunctiva under white/red/green/blue/magenta/
purple light — same eye, so true Hb is constant and all reported variation is
lighting error. Both pipelines segment correctly only under white light; under
strong casts the mask lands on eyebrow or lashes, so the spread figures
(gray-world SD 9.51 a\* / 1.64 g/dL vs CIELAB 30.68 / 2.42) measure extraction
collapse as much as colour robustness — directionally consistent with the
cohort result, not a clean comparison. Root cause is physical: under blue light
the red channel averages 3/255, so tissue redness was never captured, and
gray-world's ×20.8 red gain amplifies noise. Proposed follow-up: add an
illuminant-neutrality check to QC (cast ratio cleanly separates acceptable
1.6 from reject 5.3–53.7).
→ `experiments/04_lighting_stress/`

**19 Aug — Colour-normalization comparison (CIELAB vs white balance)**
Measured gray-world-corrected vs uncorrected colour against the local cohort's
laboratory Hb (26 patients, identical masks): gray-world improves both the
redness–Hb correlation (ρ +0.30 vs +0.02) and left-vs-right-eye consistency.
Key structural finding: CLAHE on L\* leaves a\*/b\* unchanged, so the two
methods correct orthogonal things (colour cast vs brightness) and compose
naturally — recommended hybrid: gray-world → a\*/b\* input.
→ `experiments/03_colour_normalisation/`

**19 Aug — CIELAB pipeline integrated into the workspace**
`cielab_pipeline.py` (CLAHE-on-L\* normalization, 2-channel a\*/b\* ResNet with
pretrained conv1 weight-averaging) copied into this folder with its README, and
exercised on all 52 local images end to end. Companion tracer
`cielab_stages.py` added (imports the pipeline unmodified) — per-image folders
of every intermediate, including the a\*/b\* heatmaps the regressor would see.
Traces for the full cohort in `experiments/01_stage_traces/cielab_pipeline/`. Tracing surfaced one discussion
point: masked-out background in LAB is not neutral (a=0 ⇒ a\*=−128), so
background pixels carry extreme colour values in the a\*/b\* input.

**19 Aug — Reading library assembled**
`papers/` created: `conjunctiva-anemia/` (three open-access papers on
smartphone/CNN conjunctiva anemia detection, incl. one using CP-AnemiC),
`methods/` (ResNet, U-Net, Mask2Former, deep imbalanced regression), and
earlier psychiatry-LLM reading moved to `papers/psychiatry-llm/`. Papers behind
publisher blocks are listed with links in `papers/README.md` rather than stored.

**19 Aug — Work log and time tracking started**
This file, plus a personal daily time log. Every suggested change gets an entry
from now on; weeks run Thursday 6pm to Thursday 6pm, matching the 4pm meeting.

**18 Aug — Stage-by-stage pipeline tracer**
New `python -m anemia stages` command: one folder per image with every
intermediate (resized, white-balanced, redness map, mask, overlay, masked,
crop, letterbox, 224×224 model input, quality JSON), numbered to sort in
pipeline order. Generated for all 52 local images (now `experiments/01_stage_traces/`).
→ `src/anemia/cli.py`, README "Stage-by-stage walkthrough"

---

## Week of 11–17 Aug 2026

**Pipeline restructured as a package** (`src/anemia/`): separate modules for
config, imaging, segmentation, preprocessing, data loading, splits, metrics,
model, training, and serving; the inference path imports nothing from the
training stack. 19 automated tests; synthetic eye phantoms allow development
without datasets. → `CODE_GUIDE.md` for design rationale.

**Conjunctiva extraction developed and validated on real images.** Redness-based
prior in CIELAB (a\* − 0.5·b\*) with Otsu thresholding; then, after screening 52
real captures exposed bleeding onto lashes/lid skin, a refined seeded-grabCut
extractor (texture, sclera, and darkness gates; specular-hole filling). Clean
masks across all 52 on visual review; median mask area halved. Measured against
a brightness-based baseline: redness-in-mask +9.2 vs −3.9 median; the two
distributions do not overlap. → `segment.py`, `experiments/02_extraction_check/`

**Clinical labelling per WHO.** Anemia threshold resolved per patient from age
and sex (11.0/11.5/12.0/13.0 bands). Validated on the local cohort: a fixed
11.0 cutoff would mislabel 4 of 26 patients, all anemic children read as
normal. → `data.py::anemia_threshold`

**Local cohort integrated.** Loader for `DATASAMPLE.csv` + `left_eye/` +
`right_eye/` (26 children, both eyes, Hb + full blood count); both eyes share a
patient id so grouped splits cannot leak. Quality control enforced: 49/52 pass;
3 blur rejections. → `data.py::load_local_cohort`

**Evaluation scaffolding.** Patient-grouped, Hb-stratified k-fold; out-of-fold
predictions only; predict-the-mean baseline reported by construction;
sensitivity/specificity and Bland–Altman in metrics. Self-describing
checkpoints (weights + label standardisation + preprocessing settings).

**End-to-end smoke run on synthetic phantoms** (3-fold CV: MAE 0.66 g/dL,
R² 0.88 — plumbing validation only, no clinical meaning), serving API
(`serve/app.py`), datasets identified (Eyes-defy-anemia 218 img with masks;
CP-AnemiC 710 img), progress report produced. → `Progress_Report.pdf`

---

*Convention: entries state what changed, why, and where it lives. Results on
synthetic data are always labelled as such.*
