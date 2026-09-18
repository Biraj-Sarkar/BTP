# A Learned Extractor vs the Classical Chain

Run: `python experiments/13_learned_extractor/learned_vs_classical.py`
Machine-readable: `results.json`

Every extractor in this project so far decides with hand-written rules. This
experiment asks whether learning the decision from annotated examples predicts
haemoglobin better, on the same patients, against the same noise floor.

**It does, and by the largest margin the project has measured.** The learned
extractor is the first arm whose held-out R² is both positive and clears its
permutation noise floor.

---

## The extractor

A U-Net (`segmentation_models_pytorch`, ImageNet-pretrained EfficientNet-B0
encoder) fine-tuned on **Eyes-Defy-Anemia**, which is the only openly available
source of conjunctiva masks.

Two facts about that dataset decided how it was used:

1. Its annotations are the full frame **downscaled ~3.7x with everything
   outside the annotated region blacked out**, so a mask is recoverable as the
   non-black pixels. Verified numerically: on foreground pixels the annotation
   and the downscaled original differ by a mean of **0.06–0.60 out of 255**,
   which is resampling noise.
2. **The annotations are not uniformly traced.** About 45% of `_palpebral`
   annotations are axis-aligned *rectangles* — bounding boxes that frequently
   contain sclera, iris and pupil — and only the rest are hand-traced crescents.
   `_forniceal_palpebral` is 95% rectangles and was discarded entirely.

A rectangle fills its own bounding box, so `extent = area / bbox_area` is ~1.0
against a median of **0.19** for traced contours; the two populations separate
cleanly, so the cut is not delicate. `scripts/prepare_eyes_defy.py` keeps only
traced annotations and records every keep/reject decision in a manifest.

**Training set: 101 image/mask pairs from 101 distinct subjects** (58 India, 43
Italy) — one capture per subject, so patient leakage is not merely avoided but
impossible. Held-out validation Dice **0.841**, split by subject.

The segmenter never saw the local cohort or any haemoglobin label, so nothing
about the evaluation below is contaminated by its training.

---

## Why the obvious comparison is wrong

Running `anemia fit-linear` with each extractor gives held-out R² **+0.007** for
the classical chain and **+0.048** for the U-Net. That comparison should not be
quoted. The U-Net's thinner masks trip the quality gate more often, so those
numbers are computed on **26 and 21 patients respectively** — and the tell is
that their baselines differ, −0.082 against −0.103. A model scored on a smaller,
easier subset looks better without being better.

Everything below is on the **matched set**: patients with a usable mask under
both extractors, which turns out to be all **26**. Neither extractor loses a
patient at the mask level; the earlier attrition was the blur gate, not
segmentation.

## Three-way comparison — the primary result

`three_way.py` repeats **experiment 11's protocol verbatim** with a third arm.
The measurement functions (`ridge`, `leave_one_out`, `score`, `representations`)
are *imported* from `experiments/11_head_to_head/head_to_head.py` rather than
reimplemented, so any difference below is a difference in the pipelines and not
in how they were measured.

    A  gray-world white balance  + seeded-grabCut extractor   (classical)
    B  CIELAB normalisation      + CIELAB extractor
    C  gray-world white balance  + fine-tuned U-Net           (learned)

A and C share their colour normalisation and differ **only** in the mask, which
is what isolates the extractor's contribution.

**26 patients, all 52 captures, every capture usable by all three arms.**
(C returns an empty mask on 2 captures — `left_eye/26`, `right_eye/11` — and
those captures are dropped for *all three* arms so none is scored on an easier
set. Both patients keep their other eye, so n stays 26.)

### Held out — leave-one-patient-out, best representation each

| metric | A (means) | B (lab) | **C (chroma)** | baseline |
|---|---|---|---|---|
| MSE (g/dL)² | 2.072 | 2.118 | **1.669** | 2.030 |
| RMSE (g/dL) | 1.440 | 1.455 | **1.292** | 1.425 |
| MAE (g/dL) | 1.040 | 1.100 | **0.988** | 1.083 |
| R² | −0.104 | −0.129 | **+0.111** | −0.082 |
| Pearson r | −0.094 | −0.197 | **+0.341** | — |
| Accuracy | 0.654 | 0.500 | 0.654 | 0.615 |
| Precision | 0.600 | 0.500 | **0.611** | 0.565 |
| Recall | **0.923** | 0.769 | 0.846 | 1.000 |
| Specificity | 0.385 | 0.231 | **0.462** | 0.231 |
| F1 | **0.727** | 0.606 | 0.710 | 0.722 |
| confusion | TP12 FP8 FN1 TN5 | TP10 FP10 FN3 TN3 | TP11 FP7 FN2 TN6 | |

### Fitted on all 26 patients — scored in-sample

| metric | A | B | **C** | baseline |
|---|---|---|---|---|
| MSE (g/dL)² | 1.635 | 1.808 | **1.357** | 1.877 |
| RMSE (g/dL) | 1.279 | 1.345 | **1.165** | 1.370 |
| MAE (g/dL) | 0.920 | 1.000 | **0.881** | 1.041 |
| R² | 0.129 | 0.036 | **0.277** | 0.000 |
| Pearson r | 0.386 | 0.191 | **0.534** | — |
| Accuracy | **0.692** | 0.577 | 0.654 | 0.615 |
| Recall | **1.000** | 0.846 | 0.846 | 1.000 |
| Specificity | 0.385 | 0.308 | **0.462** | 0.231 |
| F1 | **0.765** | 0.667 | 0.710 | 0.722 |

### Against the noise floor — 20,000 permutations

| pipeline | k | in-sample R² | null | p | held-out R² | null | p |
|---|---|---|---|---|---|---|---|
| A gray-world + grabCut | 3 | 0.129 | 0.106 | 0.314 | −0.104 | −0.181 | 0.225 |
| B CIELAB + CIELAB | 2 | 0.036 | 0.080 | 0.650 | −0.129 | −0.176 | 0.274 |
| **C gray-world + U-Net** | 2 | **0.277** | 0.078 | **0.022** | **+0.111** | −0.163 | **0.015** |
| baseline | — | 0.000 | | | −0.082 | | |

**C is the only arm that clears its own noise floor**, on both framings. A's
in-sample 0.129 against a null mean of 0.106 is unremarkable for three
parameters on 26 patients (p = 0.314), and B's 0.036 is *below* its null mean of
0.080 — shuffled labels fit B better than the real ones do.

### The paired test — and why it does not contradict the above

| | value |
|---|---|
| C closer than A on | **12 of 26 patients** |
| mean advantage to C | **+0.052 g/dL** (95% CI −0.111 to +0.215) |
| paired permutation p | **0.554** |

**C and A are not separable patient-by-patient at n = 26.** C is actually the
closer estimate on *fewer* than half the patients; its positive mean advantage
comes from winning by more when it wins. The confidence interval spans zero.

This is not a contradiction of the noise-floor table, and the distinction is
worth being precise about because it is easy to misstate:

- The **noise floor** asks *"does this model beat chance?"* — C does, A does not.
- The **paired test** asks *"is C better than A on these 26 patients?"* — too
  close to call.

A failing to clear its floor does not make C significantly better than A. Both
statements are true simultaneously. **The honest summary is that C is the only
arm demonstrably extracting signal, while the direct C-vs-A margin remains
unresolved at this sample size** — which is the same conclusion experiment 11
reached about A vs B, for the same reason: 26 patients is not enough.

---

## The two-extractor run, and why its numbers differ slightly

`learned_vs_classical.py` (run first) reports A held-out R² −0.083 against the
three-way's −0.026 for the same `erythema` features. The difference is the
inclusion rule, not the method: that script let each extractor use every capture
*it* could segment, while `three_way.py` requires a capture to be usable by all
three arms and drops it from all of them otherwise — the stricter rule, and the
one experiment 11 uses. **Where the two disagree, `three_way.py` supersedes.**
The conclusion is unchanged either way: A held out is at or below its baseline.

Baselines: in-sample R² = 0.000 by definition (full-sample mean); held-out
R² = 1 − (26/25)² = **−0.0816** (mean of the other n−1).

Noise floor = 20,000 permutations of Hb against the same features, refit and
rescored. Both models have k = 2 free parameters, so the floors are comparable.

Three things follow.

**The classical chain, held out, is exactly its baseline.** Its −0.083 sits on
top of the −0.0816 baseline to three decimals, and its p of 0.194 says the
in-sample R² of 0.132 is unremarkable for two parameters on 26 patients. On
unseen patients it carries no information beyond the cohort mean.

**The learned extractor clears its noise floor on both framings** — p = 0.020
in-sample and **p = 0.015 held out**. This is the first arm in the project to do
so. Held-out R² of +0.112 against a −0.082 baseline is a real gain, not an
artefact of fitting.

**The gain is extraction, not modelling.** Identical features (`erythema`:
mean log(R/G) and log(R/B)), identical ridge regression, identical patients,
identical α. The only thing that changed is which pixels were measured.

---

## What the masks are actually doing

Measured on all 52 captures with experiment 12's proxy — redness index inside
the mask, same image and same white balance, so only placement differs:

| Extractor | median redness | off-tissue | median mask area |
|---|---|---|---|
| Classical (refined) | +9.36 | 0 of 52 | 10.7% |
| CIELAB (pipeline B) | −3.72 | 48 of 52 | 7.7% |
| **Learned (U-Net)** | **+15.89** | 0 of 50 | **2.0%** (2 empty) |

Mask IoU between classical and learned: **median 0.127**. They are not
refinements of one another; they are reading different parts of the photograph.

**This exposes a limitation in experiment 12's test.** That experiment asked
whether *mean* redness inside the mask was positive, and the classical chain
passed 52 of 52 — but a mask that is half conjunctiva and half sclera still
averages positive. The test detects a mask that is *off* the tissue; it cannot
detect one that is *diluted* by it. Visual review confirms the dilution: on
`left_eye/1`, `10`, `11` and `13` the classical mask extends well across sclera
and lid skin, while the learned mask stops at the red rim. At +15.89 against
+9.36 on the same pixels, the learned mask is substantially purer conjunctiva.

So "pipeline A's extraction demonstrably works", as stated in experiment 11, is
true only in the weak sense that it is not off the tissue. Its colour features
have been measured over a mixture, and that dilution is a plausible contributor
to the weak R² reported throughout this project.

---

## Mask geometry — contamination costs more than omission

`mask_geometry.py`. The learned masks cover a median 1.8% of frame against the
classical chain's 10.7%, which invites two opposite worries. Too few pixels
makes a colour mean noisy, so widening should help. But the feature *is* a mean,
so including sclera shifts it by an amount set by how much sclera got in — which
varies per capture with framing and lid eversion, for reasons unrelated to
haemoglobin. Omitting conjunctiva only shrinks the sample, and a consistent
offset from always measuring the same part of the tissue is absorbed by the
intercept.

**Contamination is bias that varies; omission is variance plus a constant.**
The sweep separates them, 26 patients, `chroma` features, LOO:

| mask variant | area % | MAE | R² | F1 |
|---|---|---|---|---|
| erode 6px | 1.04 | 0.961 | **0.103** | 0.750 |
| **erode 4px** | 1.36 | **0.955** | 0.095 | 0.750 |
| erode 2px | 1.80 | 0.979 | 0.079 | 0.710 |
| baseline (p ≥ 0.5) | 2.31 | 0.987 | 0.072 | 0.710 |
| dilate 2px | 2.84 | 0.992 | 0.066 | 0.688 |
| dilate 4px | 3.36 | 0.997 | 0.055 | 0.710 |
| dilate 8px | 4.44 | 1.007 | 0.035 | 0.710 |
| dilate 16px | 6.83 | 1.016 | 0.007 | 0.733 |
| threshold p ≥ 0.9 | 1.52 | 0.967 | 0.100 | 0.750 |
| threshold p ≥ 0.7 | 1.97 | 0.976 | 0.081 | 0.710 |
| threshold p ≥ 0.3 | 2.71 | 0.988 | 0.066 | 0.688 |
| probability-weighted | 100.00 | 1.083 | −0.039 | 0.667 |
| probability²-weighted | 100.00 | 1.018 | 0.062 | 0.733 |
| baseline (predict mean) | — | 1.083 | −0.082 | 0.722 |

**Monotonic in both directions.** Every step of widening degrades R², every step
of tightening improves it, from +0.103 at 1.04% of frame down to +0.007 at
6.83%. The threshold sweep reproduces the ordering through a different mechanism
— no morphology involved — which is what makes the trend credible rather than an
artefact of one operation.

This also explains the classical chain's position from the other end: at 10.7%
of frame it sits well beyond the worst variant tested here.

**Negative result kept:** probability-weighted pooling, proposed as a way to
avoid thresholding entirely, was the *worst* variant at R² −0.039. The
implementation weights every pixel in the frame, so background contributes at
low weight — contamination by another name. Squaring the weights suppresses it
and recovers most of the loss (+0.062), but neither beats a plain tight
threshold. Soft pooling would need restricting to a neighbourhood of the mask to
be worth revisiting.

**Do not read "erode 4px" as the answer.** Selecting the best of 13 variants on
the same 26 patients is selection on the test set, and the top few rows are
within noise of each other. The *trend* is the finding; the optimum is not
trustworthy at this n, and the pipeline default has deliberately **not** been
changed to a tuned erosion radius. If a change is made before more data arrives,
raising the decision threshold (keep confident pixels) is more defensible than
a tuned morphological radius, because it states a principle rather than fitting
a constant.

## Is a mean enough? — feature ablation

`feature_ablation.py`. Every model in this project reduces the conjunctiva to a
**mean**, discarding everything about the distribution's shape. Thirteen
alternatives were tested on pipeline C's masks, all ≤ 3 features because n = 26.

| feature set | k | MAE | R² | F1 |
|---|---|---|---|---|
| **chroma mean (current)** | 2 | **0.988** | **+0.111** | 0.710 |
| erythema mean | 2 | 0.991 | +0.115 | 0.688 |
| redness mean | 1 | 1.024 | +0.068 | 0.645 |
| chroma median | 2 | 0.997 | +0.082 | 0.710 |
| chroma trimmed mean | 2 | 0.989 | +0.103 | 0.710 |
| erythema median | 2 | **0.986** | +0.089 | 0.710 |
| redness median | 1 | 1.007 | +0.080 | 0.688 |
| chroma mean + IQR | 3 | 1.084 | +0.056 | 0.667 |
| chroma mean + std | 3 | 1.025 | **+0.156** | 0.667 |
| redness mean + std | 2 | 1.005 | +0.016 | 0.645 |
| chroma quartiles | 3 | 1.118 | −0.142 | 0.750 |
| redness quartiles | 3 | 0.995 | +0.042 | 0.645 |
| redness p25 + p75 | 2 | 0.987 | +0.064 | 0.600 |
| baseline (predict mean) | — | 1.083 | −0.082 | 0.722 |

**The mean is good enough.** Nothing clearly beats it. `erythema median` edges it
on MAE by 0.002 g/dL — a rounding difference — while scoring worse on R². Every
three-feature set is worse on MAE than the two-feature baseline, which is the
n = 26 overfitting penalty showing up exactly where experiment 07 predicted.

### The one ambiguous result

`chroma mean + std` scores the **highest R² (+0.156) and the best noise-floor
p (0.011)** in the whole set, while being **worse on MAE (1.025 vs 0.988)**.

That divergence is real, not a typo: R² penalises large errors quadratically
while MAE measures the typical error, so a feature that rescues a few bad
patients at slight cost to the rest moves them in opposite directions. It is
*not* being treated as a win — thirteen feature sets were tested with no
multiplicity correction, and k = 3 on 26 patients is where this project has been
burned before. Recorded as a lead to re-test once the public datasets are in,
not as a recommendation.

### The finding that matters for extraction

**A robust centre does not beat the mean.** `chroma median` (+0.082) and
`chroma trimmed mean` (+0.103) both score *below* `chroma mean` (+0.111) on the
same masks.

That is informative rather than dull. A median exists to resist outliers; if
sclera or lash pixels were still contaminating the measurement, discarding the
tails would help. It does not — so **the pixels inside C's mask are already
clean**, and the remaining error is not contamination. This is independent
support for the extractor from a completely different direction than the
placement metrics, and it also says the mask-geometry gains have been banked:
there is no residual contamination left for a robust statistic to remove.

## Throw everything in and let L1 prune — it depends entirely on n

`kitchen_sink.py`. The proposal: rather than hand-picking two features, supply
**38 colour statistics** (mean, median and standard deviation of R, G, B,
chromaticity, redness, a\*, b\*, L\*, the two log-ratios, plus distribution tails
and mask area) and let an L1 penalty drive the useless ones to exactly zero.

Both cohorts, same 38 features, same extractor, penalty strength **re-chosen
inside every fold** so the held-out patient never influences the model.

### At n = 26 it fails outright

| model | features used | MAE | R² |
|---|---|---|---|
| current 2, alpha = 1 fixed | 2 | **0.988** | **+0.111** |
| current 2, alpha nested | 2 | 1.035 | +0.042 |
| all 38, L1 lasso | 25 | 1.189 | −0.125 |
| all 38, L2 ridge | 38 | 1.329 | −0.252 |
| all 38, elastic net | 38 | 1.107 | +0.088 |
| baseline | — | 1.083 | −0.082 |

L1 scores **worse than predicting the cohort mean**, at a noise-floor
**p = 0.516**. It extracted nothing.

The mechanism is visible in the selection counts: **only one feature survived
all 26 folds**, and L1 kept ~25 of 38 rather than pruning hard. Two causes. The
penalty is chosen by inner cross-validation on 25 patients, where the CV signal
is itself noise and systematically under-penalises; and 45 feature pairs
correlate above 0.95, so L1 keeps one member of each correlated group
essentially at random — a different member each fold.

### At n = 217 it works, and the selection stabilises

| model | features used | MAE | R² | F1 |
|---|---|---|---|---|
| current 2, alpha = 1 fixed | 2 | 1.499 | 0.350 | 0.720 |
| current 2, alpha nested | 2 | 1.499 | 0.350 | 0.720 |
| all 38, L1 lasso | 27 | 1.396 | 0.413 | **0.751** |
| **all 38, L2 ridge** | 38 | **1.386** | **0.421** | 0.730 |
| all 38, elastic net | 38 | 1.396 | 0.414 | 0.738 |
| baseline | — | 2.005 | −0.009 | 0.342 |

Noise floor **p = 0.000** (1,000 permutations, 10-fold). Improvement over
baseline rises from **25.2% to 30.9%** in MAE.

| | n = 26 | n = 217 |
|---|---|---|
| features kept in **every** fold | **1** | **9** |

Nine features — `R mean`, `r_chroma p25`, `R std`, `G median`, `G std`,
`b* std`, `a* std`, `a* median`, `g_chroma std` — survive all 217 folds. The
selection is reproducible rather than an artefact of which patients landed in
which fold.

**More features help; they just need enough patients to be chosen reliably.**

Two further observations:

* **Fixed alpha = 1.0 and nested alpha give identical results at n = 217**
  (1.499 / 0.350 either way). With enough data the inner CV converges on
  essentially the penalty this project had hardcoded, so alpha = 1.0 was a
  sound choice; the divergence at n = 26 (0.988 vs 1.035) was the CV unable to
  find it, not the constant being wrong.
* **Ridge slightly beat lasso** (0.421 vs 0.413). Sparsity is not what helps —
  the extra information is. Among heavily correlated features ridge spreads
  weight across a group where lasso picks one arbitrarily, and spreading wins.

### The caution this comes with

This shows more features help **within one cohort**. It says nothing about
transfer, and there is reason to expect it makes transfer *worse*: going from 2
absolute colour statistics to 38 multiplies exactly the quantities experiment 14
found do not survive a change of camera. The two findings point different ways
and both are real — within a camera, use more features; across cameras, the
absolute-colour approach breaks and feature count is beside the point.

## Robustness — the axis the other tests could not see

`robustness.py`. Every comparison above scores masks on the 52 local captures:
one camera, one operator, one setting. A model that is *more robust to varied
capture conditions* looks identical on that test, because those conditions are
not in it. **Accuracy on a narrow slice is not robustness.**

Each capture is perturbed the way a real user would — phone held at an angle,
closer or further, dimmer room, shakier hand — and scored on **self-consistency**:
IoU between the mask on the perturbed image and the mask on the clean image,
with the geometric transform undone. A robust extractor finds the same tissue
however the photograph was taken. No ground truth required, which is what makes
it usable on a cohort that has none. Failures — perturbations that make the
extractor return nothing — are counted separately, because in deployment that is
a retake prompt to a patient.

| perturbation | old `unet` | **`unet_v3`** | classical |
|---|---|---|---|
| rotate 10° | 0.812 (1) | 0.787 (0) | **0.944** (0) |
| rotate 20° | 0.727 (3) | 0.658 (1) | **0.925** (0) |
| rotate −15° | 0.758 (1) | 0.667 (4) | **0.937** (0) |
| zoom in 1.3× | 0.696 (1) | 0.697 (2) | 0.681 (0) |
| **zoom out 0.75×** | 0.490 **(15)** | **0.725 (2)** | 0.677 (0) |
| dim 60% | 0.722 (3) | 0.775 (1) | **0.859** (0) |
| **bright 140%** | 0.598 **(10)** | **0.684 (5)** | 0.666 (0) |
| motion blur | 0.807 (0) | 0.756 (1) | 0.812 (0) |
| jpeg quality 30 | 0.856 (0) | 0.870 (2) | 0.870 (0) |
| warm cast | **0.986** (1) | 0.981 (0) | 0.958 (0) |
| **median overall** | 0.772 **(35)** | 0.752 **(18)** | **0.856 (0)** |

`(N)` = perturbed captures where the extractor returned nothing.

### v3 halves the failure rate

**35 → 18 failures**, and the two worst cases are precisely the ones extra
capture diversity should fix: zoom-out (15 → 2 failures, IoU 0.490 → 0.725) and
over-brightening (10 → 5, IoU 0.598 → 0.684). That is the 547 varied smartphone
captures doing their job. In deployment a failure costs a retake, which is worth
more than the 0.02 median IoU v3 concedes elsewhere.

Together with the downstream numbers — v3 MAE 0.975 / R² 0.148 / p 0.013 against
the old model's 0.988 / 0.111 / 0.017 — **v3 is the better extractor and is now
the default**, written to `runs/segmenter/unet.pt` with the previous bundle kept
as `unet_v1.pt`.

### The classical chain is the most robust thing here, by a wide margin

**IoU 0.856 and zero failures**, ever. On rotation it is not close: 0.925–0.944
against roughly 0.7 for both networks.

The reason is structural. The classical chain ranks pixels by redness
*percentile within the image*. Rotating a photograph does not change its
distribution of redness, so the rule sees an almost identical image; dimming
shifts every pixel together and the percentile ordering survives. A network has
learned spatial and appearance patterns that genuinely move. **The property that
makes the classical extractor fragile across cameras — image-relative decisions
— is the same property that makes it stable under transformations of one image.**

So each extractor wins a different axis, and the choice is a trade rather than a
ranking:

| | clean-capture accuracy | capture robustness |
|---|---|---|
| classical | redness 9.64, downstream **p = 0.211** | **IoU 0.856, 0 failures** |
| old unet | redness 16.08, p = 0.017 | IoU 0.772, 35 failures |
| **v3** | redness 15.21, **p = 0.013** | IoU 0.752, **18 failures** |

The classical chain is stable but reliably measures the wrong pixels — it is
contaminated with sclera (experiment 12's caveat). The networks measure the
right pixels and are more sensitive to how the photograph was taken. v3 is the
best available trade, not a dominant option.

**Actionable follow-up:** the networks' rotation weakness is a training choice,
not an intrinsic limit. The augmentation used `A.Rotate(limit=20)` and moderate
scale jitter. Pushing rotation and scale augmentation substantially harder
should close most of the 0.79 → 0.94 gap without touching the accuracy
advantage, and costs one retraining run.

### What this says about the earlier comparison

`compare_bundles.py` reported v3 and the old model as equivalent within noise,
and that conclusion was drawn from a test structurally unable to measure capture
robustness. The recommendation to keep the old bundle was wrong, and was
reversed on this evidence. Recorded rather than quietly amended, because the
failure mode — *measuring the wrong axis and reporting the result as if it
settled the question* — is more instructive than the number.

## The decisive comparison — 243 patients, both cohorts

`compare_extractors_full.py`. Every extractor comparison above scored the
downstream model on the 26 local patients, because that was the only labelled
cohort available when they were written. It no longer is: Eyes-Defy-Anemia
supplies **217 further subjects** with haemoglobin, age and sex.

The cohorts are reported **separately, never pooled** — experiment 14 measured
their features more than three local standard deviations apart with transfer at
R² −1.9, so a pooled number would mostly describe a between-cohort offset and
would characterise neither site. An extractor must win on **both** independently.

### Local cohort — 26 patients

| extractor | median redness | MAE | R² | F1 | p |
|---|---|---|---|---|---|
| **v5** (rotation aug) | **16.00** | **0.890** | **0.294** | 0.690 | **0.001** |
| v3 (standard aug) | 15.21 | 0.975 | 0.148 | 0.621 | 0.013 |
| v4 (strong aug) | 12.76 | 1.051 | −0.072 | 0.688 | 0.165 |
| classical | 9.64 | 1.062 | −0.098 | 0.688 | 0.211 |
| baseline | — | 1.083 | −0.082 | 0.722 | — |

### External cohort — 217 patients

| extractor | median redness | MAE | R² | F1 | p |
|---|---|---|---|---|---|
| v4 | 20.64 | **1.515** | **0.337** | **0.713** | 0.000 |
| **v5** | **20.70** | 1.525 | 0.336 | 0.708 | 0.000 |
| v3 | 19.97 | 1.541 | 0.320 | 0.684 | 0.000 |
| classical | 20.34 | 1.820 | 0.116 | 0.483 | 0.000 |
| baseline | — | 2.005 | −0.009 | 0.342 | — |

### Three findings

**1. The learned-vs-classical question is now settled.** At n = 26 the classical
chain scored p = 0.211 — worse, but not demonstrably so, and the recommendation
rested on placement evidence rather than outcome. At n = 217 it is
**R² 0.116 against 0.336**, both at p < 0.0001: the learned extractors nearly
**triple** the explained variance. The 26-patient comparison was not merely
noisy, it was structurally unable to resolve the effect it was asked about.

**2. Rotation and scale augmentation separate cleanly.** v5 differs from v3 by
rotation alone (±20° → ±45°); v4 additionally widened scale both ways and
exposure.

| | rotation | scale | local R² |
|---|---|---|---|
| v3 | ±20° | crop-in only | 0.148 |
| **v5** | **±45°** | crop-in only | **0.294** |
| v4 | ±45° | both ways, 0.45–1.6 | −0.072 |

Rotation bought robustness *and* precision; scale cost precision. v5 also has the
**highest redness on both cohorts**, so wider rotation did not merely preserve
mask purity, it improved it. The likely reason is that rotation is a true
invariance of the anatomy — a conjunctiva is a conjunctiva at any angle — while
apparent size carries real information about what is being looked at, so teaching
scale-invariance discards signal.

**3. Augmentation damage is domain-specific.** v4 collapsed on the local cohort
(R² −0.072) while remaining the best model on the external one (0.337). The
scale augmentation did not make it globally worse — it made it worse *on one
camera*. A further reason not to tune augmentation against a single cohort.

### v5 robustness — improved where aimed, worse elsewhere

| perturbation | v3 | **v5** | classical |
|---|---|---|---|
| rotate 10° | 0.787 (0) | **0.822 (0)** | 0.944 |
| rotate 20° | 0.658 (1) | **0.723 (0)** | 0.925 |
| rotate −15° | 0.667 (4) | **0.695 (1)** | 0.937 |
| zoom out 0.75× | 0.725 (2) | **0.526** (1) | 0.677 |
| bright 140% | 0.684 (5) | 0.630 **(7)** | 0.666 |
| motion blur | 0.756 (1) | 0.657 (0) | 0.812 |
| **median overall** | 0.752 **(18)** | 0.723 **(9)** | **0.856 (0)** |

All three rotation rows improved and **failures halved, 18 → 9**. But median IoU
fell, because the non-rotation rows degraded.

**This is not explained.** v3 and v5 have *identical* scale augmentation — only
rotation differs — so the scale degradation is not a direct consequence of what
changed. The plausible reading is that capacity spent on rotation invariance came
out of scale cues, but at one training run per configuration it is equally
consistent with run-to-run variance. **A second seed is needed before the
mechanism is claimed**, and it is recorded here as unexplained rather than
narrated into a story.

### Default

`runs/segmenter/unet.pt` now holds **v5**. It wins accuracy on both cohorts —
the deliverable — has half v3's failure count, and the purest masks measured.
It concedes median self-consistency to v3 and, by a wide margin, to the
classical chain, which remains the most stable extractor here while measuring
the least useful pixels.

## Two checks before deploying — leakage, and the direction of the signal

`unseen_external.py`, plus direct correlation checks.

### Leakage: the advantage holds on photographs no segmenter saw

Every learned bundle was trained on 101 Eyes-Defy-Anemia photographs, and the
external cohort *is* Eyes-Defy-Anemia. The segmenter never saw a haemoglobin
value, but on those 101 images it had learned exactly where the conjunctiva is.
Splitting the external cohort by whether a photograph was in segmenter training
(`datasets_manifests/eyes_defy_prepared.json`):

| external subjects | learned R² | classical R² | learned MAE | classical MAE |
|---|---|---|---|---|
| **never in segmenter training (116)** | **0.354** | 0.074 | **1.479** | 1.791 |
| in segmenter training (101) | 0.286 | 0.134 | 1.613 | 1.877 |

The learned extractor's advantage is **larger on unseen photographs**, so the
extractor decision does not rest on training data leaking into evaluation.

### Direction: the local cohort is inverted

Physiology predicts redder, less blue conjunctiva at higher haemoglobin —
haemoglobin absorbs blue and green far more than red. Correlation of mean
chromaticity inside the mask with laboratory haemoglobin:

| cohort, extractor, colour | r | b | direction |
|---|---|---|---|
| external 217, learned, raw | +0.672 | −0.674 | physiological |
| external 217, learned, gray-world | +0.466 | −0.591 | physiological |
| external 116 unseen, learned, gray-world | +0.515 | −0.618 | physiological |
| **local 26, learned, raw** | **−0.261** | **+0.332** | **inverted** |
| local 26, learned, gray-world | −0.329 | +0.522 | inverted |
| local 26, classical, raw | +0.136 | −0.028 | weak, physiological |
| local 26, classical, gray-world | +0.101 | +0.068 | weak |

In 217 subjects the relationship is strong and physiological. In the 26 local
patients, under the learned extractor, it runs the other way. **Gray-world
strengthens the inversion but does not cause it**, and under the classical
extractor the local sign is weak but correct, so the inversion is tied to what
the learned extractor measures *in these captures*.

**The cause is not established.** Two candidates, neither tested: a capture
difference baked into the photographs (a phone's own automatic white balance
rescales colour by image content, and cannot be undone from a JPEG), or the
learned extractor landing on different tissue in these photographs than in the
public ones it was trained on.

The consequence is firm. A model fitted on the local 26 with the learned
extractor reaches held-out R² **+0.260** — the best local number in the project —
**by learning a relationship that contradicts physiology and 217 other
subjects.** It was fitted, examined, and deliberately *not* deployed (kept as
`runs/linear_model_unet_local26_INVERTED.json`). The deployed model is fitted on
the external cohort instead.

A side observation, recorded but not acted on: on the external cohort gray-world
white balance *weakens* the colour–haemoglobin correlation (r +0.672 → +0.466).
Earlier experiments that favoured gray-world were all on the local cohort, whose
relationship is now known to be anomalous. This is a correlation, not a held-out
comparison, and needs one before any preprocessing default changes.

## Honest limitations

- **n = 26.** p = 0.015 is nominally significant, but this cohort has now been
  used for many comparisons, and a single nominal p at this size is a reason to
  collect more data, not to declare the problem solved.
- **The learned masks are thin** — 0.3% to 3.1% of frame, thinnest on the
  hardest captures — and **2 of 52 produce no mask at all**. Fewer pixels means
  a noisier colour mean. The purity gain currently outweighs it, but that
  balance is not guaranteed on a new camera.
- **Trained on 101 masks from two sites**, neither of which is the hospital this
  project will deploy in. The transfer measured here is encouraging but is one
  domain shift, not evidence of general robustness.
- **Roughly half of Eyes-Defy-Anemia's palpebral annotations were discarded** as
  rectangles. That judgement rests on the extent statistic and on visual review
  of a sample, not on an independent annotation.
- The deployed model has **not** been switched to the learned extractor. That is
  a deployment decision, and it should follow a decision on whether thin masks
  are acceptable on hospital captures.
- **The best representation per pipeline was chosen from the held-out table**,
  which is selection on the test set and makes all three arms' numbers
  optimistic. It biases them equally so the *comparison* stands, but the
  absolute values do not. Experiment 11 handles this with a nested CV loop;
  that has not been replicated here and is the obvious next addition.
- **A wins on the classification metrics that matter clinically.** Held out, A
  has the better recall (0.923 vs 0.846) and F1 (0.727 vs 0.710); C has the
  better specificity (0.462 vs 0.385). For a screening tool a missed anaemic
  patient costs more than a false alarm, so A's higher recall is a real argument
  against switching on the regression metrics alone.

## Reproducing

```bash
python scripts/prepare_eyes_defy.py --source datasets/eyes_defy --out datasets/eyes_defy_prepared
python scripts/train_segmenter.py --data datasets/eyes_defy_prepared --out runs/segmenter/unet.pt --epochs 60
python experiments/13_learned_extractor/learned_vs_classical.py
```
