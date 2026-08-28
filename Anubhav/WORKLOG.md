# Work Log

Running record of work on the anemia-detection BTP, newest week first.
Every substantive change gets an entry: what was done, and where it lives.

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
`EVERYTHING_EXPLAINED.md` and `TALKING_POINTS.md`.

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
separate tables. Retired the stale figures from `TALKING_POINTS.md`'s crib sheet
and `EVERYTHING_EXPLAINED.md` §10.

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
`experiments/11_head_to_head/`, `README.md`, `EVERYTHING_EXPLAINED.md`,
`TALKING_POINTS.md`, `CLAUDE.md`.
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
report §6.4, `experiments/11_head_to_head/`, `EVERYTHING_EXPLAINED.md`,
`METRICS.md`, `README.md`, `TALKING_POINTS.md`, `CLAUDE.md`.

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
`EVERYTHING_EXPLAINED.md`, `METRICS.md`, `README.md`, `TALKING_POINTS.md`,
`CLAUDE.md`.

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
Propagated the two-pipeline comparison across the set: `EVERYTHING_EXPLAINED.md`
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
Fixed across the set — `EVERYTHING_EXPLAINED.md` gained an abbreviations
preamble, a section reading the diagnostic plots, a compressed per-stage
buys/costs table and a document map; `TALKING_POINTS.md` gained the figure as a
visual aid plus the failure-documentation point and four new reference numbers;
`CODE_GUIDE.md` gained an experiments section (all eight studies and their
scripts, previously undocumented) and a documentation map; `STUDY_PLAN.md`
gained a table of in-repo study material and a revised first exercise (read the
six failed extraction attempts before rebuilding it); `PIPELINE.md` gained
pointers to the glossary and research log. Also fixed a stale path in
`STUDY_PLAN.md` left over from the experiments reorganisation.

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
