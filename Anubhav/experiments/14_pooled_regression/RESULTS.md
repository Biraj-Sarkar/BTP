# Training on 218 More Patients — the Method Works, and It Does Not Transfer

Run: `python experiments/14_pooled_regression/pooled.py`
Machine-readable: `pooled.json`

Every predictive number in this project had rested on 26 patients.
Eyes-Defy-Anemia carries haemoglobin, age and sex for **217 further subjects**
(one row is unparseable), which allows two things that were impossible before: a
fit on a sample where the statistics mean something, and a genuine **external
validation** — fit on one population and camera, test on a completely different
one.

Features come from running **our** extractor on the original photographs, not
from the dataset's own annotation files. Both cohorts are therefore measured by
the same instrument, which is the whole point of a transfer test, and it
sidesteps the fact that ~45% of that dataset's annotations are bounding boxes.

---

## Result

| arm | n | MAE | R² | F1 | Accuracy |
|---|---|---|---|---|---|
| local only (LOO) | 26 | 0.988 | +0.111 | 0.710 | 0.654 |
| *baseline* | 26 | 1.083 | −0.082 | 0.722 | 0.615 |
| **external only (LOO)** | **217** | **1.499** | **+0.350** | 0.720 | 0.756 |
| *baseline* | 217 | 2.005 | −0.009 | 0.342 | 0.327 |
| **external → local** | 26 | **2.130** | **−1.944** | 0.684 | 0.538 |
| **local → external** | 217 | **2.894** | **−1.160** | 0.568 | 0.396 |
| pooled (LOO) | 243 | 1.527 | +0.318 | 0.654 | 0.695 |
| *baseline* | 243 | 1.972 | −0.008 | 0.323 | 0.346 |

Permutation tests, 5,000 shuffles: external only **p < 0.0001**, pooled
**p < 0.0001**.

## 1. The method works. That is now settled.

At n = 217 the signal is unambiguous: **R² +0.350, p < 0.0001**. Conjunctival
colour predicts haemoglobin. Every previous result in this project was a
marginal p at n = 26 that could not rule out chance; this one is not close to
the boundary.

**But do not quote +0.350 against the local +0.111 as a like-for-like gain.**
R² is `1 − MSE/var(y)`, so a cohort with a wider spread of haemoglobin scores
higher for the same absolute error. The external cohort spans 7.0–17.4 g/dL
against the local 8.1–14.3, and its baseline MAE is 2.005 against 1.083 — nearly
double. The fair comparison is improvement over each cohort's own baseline:

| | MAE vs baseline | relative improvement |
|---|---|---|
| local (n = 26) | 0.988 vs 1.083 | **8.8%** |
| external (n = 217) | 1.499 vs 2.005 | **25.2%** |

Still a clear win for the larger sample, but ~3× in relative error reduction,
not the ~3× in R² that the headline numbers suggest.

## 2. The model does not transfer between cohorts. At all.

**External → local: R² −1.944. Local → external: R² −1.160.** Both are far
*worse* than predicting the mean, in both directions. A model fit on one cohort
is actively harmful on the other.

The cause is visible in the feature distributions:

| | r chromaticity | g chromaticity |
|---|---|---|
| external mean ± sd | 0.4589 ± 0.0318 | 0.2658 ± 0.0179 |
| local mean ± sd | 0.3881 ± 0.0206 | 0.2927 ± 0.0118 |
| separation | **2.2 external sd / 3.4 local sd** | 1.5 external sd |

In `r_chroma` the two cohorts are **essentially disjoint** — the gap between
their means is more than three local standard deviations. A model fitted on one
is extrapolating far outside its training range when applied to the other, and
linear extrapolation outside the data is exactly where a fitted slope does most
damage. The differing cohort means (12.80 vs 11.20 g/dL) put the intercept wrong
as well.

Gray-world white balance removes an overall illuminant cast. It does **not**
remove the camera's spectral response, its colour-processing pipeline, or
population differences in baseline conjunctival pigmentation. Chromaticity is
invariant to *intensity*, not to *which camera measured it*.

### Why this is the most important finding here

The deployment plan is a hospital photographing patients with a phone. That is
a different camera and a different population from the 26-patient cohort the
deployed equation is fitted on. **This experiment is a direct measurement of
that scenario, and it fails.** Absolute colour coordinates are not portable
between imaging setups, and no amount of extra training data fixes it — the
217-subject fit transfers no better than the 26-patient one.

The corollary is uncomfortable but worth stating plainly: the deployed model's
coefficients are calibrated to one camera and one cohort, and there is currently
no evidence they mean anything on another.

## 3. What follows

Three routes, in increasing order of cost:

1. **Per-site recalibration.** Treat the coefficients as site-specific and refit
   the intercept, or intercept and scale, on a small labelled sample from each
   new deployment. Cheapest, and the honest framing of what the current model
   is: a site-calibrated instrument, not a universal one.
2. **A colour reference in frame.** A printed calibration patch photographed
   beside the eye makes absolute colour recoverable across cameras. This is the
   standard solution in tele-dermatology and it is what would make the numbers
   genuinely portable.
3. **Features designed to be camera-invariant.** Ratios within the mask relative
   to a reference tissue in the same frame — sclera is the obvious candidate,
   being approximately white and always adjacent. That turns an absolute
   measurement into a relative one and cancels much of the camera response.
   Untested, and the natural next experiment.

Route 3 is testable immediately with data already on disk, and it is the one
that would rescue the existing architecture.

## Honest limitations

- **One external cohort, two sites.** A single transfer failure demonstrates the
  problem exists; it does not bound how bad it is in general.
- The external features come from one photograph per subject, where local
  patients contribute both eyes averaged. Fewer measurements per subject means
  noisier features on the external side.
- Neither cohort's captures were quality-gated here, so some of the external
  error is capture quality rather than method.
- The pooled model is reported for completeness, but pooling two cohorts whose
  features barely overlap mostly teaches the model a between-cohort offset. Its
  +0.318 should not be read as a deployable result.
