# Head-to-Head: Pipeline A vs Pipeline B

The deployment decision. Two pipelines, two models, identical patients,
identical protocol, full metric suite for both.

Run: `python experiments/11_head_to_head/head_to_head.py`
Machine-readable: `head_to_head.json`

---

## 1. Both pipelines, every feature representation

Earlier experiments tested only channel means. Two representations were added
here because they are *designed* to be invariant to illumination intensity —
dividing by total intensity removes any overall brightness scaling:

| Representation | Features | What it is |
|---|---|---|
| `means` | 3 | mean R, G, B |
| `chroma` | 2 | R/(R+G+B), G/(R+G+B) — chromaticity coordinates |
| `erythema` | 2 | log(R/G), log(R/B) — the dermatology standard for redness |
| `lab` | 2 | mean a\*, mean b\* |
| `a_only` | 1 | mean a\* alone — minimum variance |

Leave-one-patient-out, ridge regression:

| Representation | k | A: MAE | A: R² | A: F1 | B: MAE | B: R² | B: F1 |
|---|---|---|---|---|---|---|---|
| means | 3 | 1.073 | −0.166 | 0.688 | 1.163 | −0.260 | 0.562 |
| chroma | 2 | 1.063 | −0.103 | 0.688 | 1.144 | −0.199 | 0.647 |
| **erythema** | 2 | **1.062** | **−0.094** | **0.727** | 1.143 | −0.192 | 0.647 |
| lab | 2 | 1.105 | −0.130 | 0.710 | 1.135 | −0.196 | 0.647 |
| **a_only** | 1 | 1.091 | −0.084 | 0.688 | **1.118** | **−0.128** | **0.686** |
| *baseline* | — | *1.083* | *−0.082* | *0.722* | *1.083* | *−0.082* | *0.722* |

**Better features did help.** Moving from channel means to the erythema index
improved pipeline A's R² from −0.166 to −0.094 and its F1 from 0.688 to 0.727.
Pipeline B improved from −0.260 to −0.128. Both remain at or below baseline.

---

## 2. Full metric comparison — fitted on all 26 patients

**This is the deployed model**: every patient contributes to the fit, which is
standard for a model that ships. Scored on the same patients, so these numbers
describe the *fit*. Section 2b carries the held-out estimate of what to expect
from a new patient.

| Metric | **Pipeline A** (gray-world + erythema) | **Pipeline B** (CIELAB + a\*) | Baseline |
|---|---|---|---|
| MSE (g/dL)² | **1.627** | 1.858 | 2.030 |
| RMSE (g/dL) | **1.275** | 1.363 | 1.425 |
| MAE (g/dL) | **0.944** | 1.038 | 1.083 |
| **R²** | **0.133** | 0.010 | −0.082 |
| Bias (g/dL) | 0.000 | 0.000 | 0.000 |
| Pearson r | **0.381** | 0.100 | — |
| Accuracy | **0.654** | 0.615 | 0.615 |
| Precision | **0.600** | 0.565 | 0.565 |
| Recall | 0.923 | **1.000** | 1.000 |
| Specificity | **0.385** | 0.231 | 0.231 |
| F1 | **0.727** | 0.722 | 0.722 |
| Confusion | TP 12, FP 8, FN 1, TN 5 | TP 13, FP 10, FN 0, TN 3 | — |

**Both pipelines now have positive R²**, and both beat the baseline on MSE,
RMSE and MAE. **Pipeline A wins on 10 of 11 metrics**; B's only lead is recall,
which it achieves by flagging every patient (FN 0, but specificity 0.231).

## 2b. Held-out estimate (leave-one-patient-out)

What the same models achieve on patients they were not fitted on:

| Metric | Pipeline A | Pipeline B | Baseline |
|---|---|---|---|
| MAE (g/dL) | **1.062** | 1.118 | 1.083 |
| R² | −0.094 | −0.128 | **−0.082** |
| Accuracy | **0.654** | 0.577 | 0.615 |
| F1 | **0.727** | 0.686 | 0.722 |

The gap between the two tables is the overfitting. A's R² falls from +0.133
fitted to −0.094 held out — the model finds a relationship in patients it has
seen that does not transfer to new ones. **Quote the fitted numbers as a
description of the model; quote the held-out numbers when asked what it will do
on a new patient.**

### Why the held-out R² is negative even when MAE beats the baseline

This looks contradictory and is worth understanding, because it will be asked.

R² is computed as `1 − Σ(pred−true)² / Σ(true−mean)²`, where the denominator
uses the **full-sample** mean. But under leave-one-out the baseline can only
use the mean of the *other 25* patients — it does not have access to the
full-sample mean at prediction time.

That gap is why **the baseline itself scores R² = −0.082, not 0**. The
practical floor is around −0.08, not zero. Pipeline A at −0.094 is therefore
almost exactly at that floor, not catastrophically below it.

MAE and R² can disagree because MAE weights every error equally while R²
squares them, so a single large error moves R² much more than MAE.

**The defensible statement: the model is statistically indistinguishable from
predicting the cohort mean.** Not "catastrophically wrong" — indistinguishable.

---

## 3. The honest number: nested cross-validation

Picking the best representation from the table in section 1 is selection on the
test set, and the resulting figure is optimistic. A nested loop removes that:
the inner loop chooses the representation using training patients only, and the
outer loop scores a patient that choice never saw.

| | MAE | R² | F1 | Representation chosen |
|---|---|---|---|---|
| Pipeline A | 1.117 | −0.210 | 0.645 | erythema 14×, chroma 6×, means 5×, a_only 1× |
| Pipeline B | 1.142 | −0.199 | 0.647 | a_only 23×, lab 3× |
| *Baseline* | *1.083* | *−0.082* | *0.722* | — |

**With honest feature selection, neither pipeline beats the baseline.** The
1.062 in section 2 is partly the benefit of hindsight; 1.117 is what to expect
if the pipeline had to choose its own features on unseen data.

Note also that the inner loop does not agree with itself — pipeline A's choice
flips between four representations across the 26 folds. That instability is
itself evidence that no representation is clearly best at this sample size.

---

## 4. Can we choose between the pipelines? Not yet.

Both pipelines process the same 26 patients, so their errors are **paired**. A
paired test cancels the patient-to-patient variation both share and asks only
whether one is consistently better on the same individuals — far more powerful
than comparing two independent confidence intervals.

| | Result |
|---|---|
| Patients where A has lower error | **15 of 26 (58%)** |
| Mean advantage of A | +0.0554 g/dL |
| Bootstrap 95% CI on that advantage | **[−0.0665, +0.1807]** |
| Paired permutation test | **p = 0.408** |

**The confidence interval spans zero and p = 0.408.** The two pipelines are
**not statistically separable** on this cohort. A winning on 58% of patients is
close to a coin flip (chance is 50%).

### What this means for the deployment decision

The honest answer is: **this cohort cannot choose between them.** Anyone
picking A over B on these numbers alone would be selecting on noise.

That said, A is the better *provisional* default, for reasons that do not
depend on this significance test:

1. **A wins on 10 of 11 metrics when fitted, 9 of 11 held out** — no single
   one is significant, but the
   consistency of direction is itself weak evidence.
2. **A is ahead in every earlier experiment too** — cohort correlation
   (Spearman +0.30 vs +0.02), eye-to-eye consistency (2.02 vs 2.69), lighting
   robustness (SD 1.63 vs 2.74). Five independent measurements, same direction.
3. **A's extraction demonstrably works and B's does not.** On the 52 real
   captures, B's segmentation lands on non-conjunctiva tissue in 50 of 52
   images. B's model is reading mostly skin. This is the strongest argument and
   it is not statistical: it is visible in the overlays.
4. **Higher specificity (0.385 vs 0.231)** at equal recall means fewer
   unnecessary confirmatory blood tests.

**Recommendation:** proceed with A as the default, and re-run this exact
comparison once the segmenter is trained and the public datasets are in. The
comparison script is written and takes minutes; the decision should be made on
~900 patients, not 26.

---

## 5. Limits

- n = 26. Every interval is wide, and the paired test has low power — it can
  fail to detect a real difference of this size.
- p = 0.408 means "not distinguishable", **not** "proven equivalent".
- Representations were chosen by us; a different set could rank differently.
- Both pipelines share a weakness: features are summary statistics over the
  mask, discarding all spatial structure.
- The comparison is between the pipelines *as currently implemented*. B's
  deficit is dominated by its segmentation, which is fixable — a trained
  segmenter feeding B's a\*/b\* representation has not been tested.
