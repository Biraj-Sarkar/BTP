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
| MSE (g/dL)² | **1.627** | 1.858 | 1.877 |
| RMSE (g/dL) | **1.275** | 1.363 | 1.370 |
| MAE (g/dL) | **0.944** | 1.038 | 1.041 |
| **R²** | **0.133** | 0.010 | 0.000 |
| Bias (g/dL) | 0.000 | 0.000 | 0.000 |
| Pearson r | **0.381** | 0.100 | — |
| Accuracy | **0.654** | 0.615 | 0.615 |
| Precision | **0.600** | 0.565 | 0.565 |
| Recall | 0.923 | **1.000** | 1.000 |
| Specificity | **0.385** | 0.231 | 0.231 |
| F1 | **0.727** | 0.722 | 0.722 |
| Confusion | TP 12, FP 8, FN 1, TN 5 | TP 13, FP 10, FN 0, TN 3 | — |

**Pipeline A leads B on 10 of these 11 metrics.** B's only lead is recall,
achieved by flagging every patient (FN 0, but specificity 0.231).

**But do not read this table as A beating the baseline.** It cannot show that,
for two structural reasons, and both are easy to miss:

*The comparison is unlosable.* An in-sample baseline is the full-sample mean,
whose R² is 0 by definition; a fitted model with a free intercept essentially
cannot score below 0. The model is guaranteed to win before any data is
involved. The right question is not "does it beat 0" but "does it beat what
*noise* would score", and that has to be measured.

*A has one more free parameter than B.* A is erythema (2 features), B is
`a_only` (1). More parameters fit better in-sample mechanically, so part of
A's margin here is arithmetic rather than signal.

### 2a. The noise floor — what these fitted R² values are worth

Shuffling Hb against the same real features, refitting, and rescoring
in-sample gives the distribution of R² under "these features carry nothing".
20,000 permutations:

| | features | observed R² | null mean | null 95th | p |
|---|---|---|---|---|---|
| Pipeline A | 2 | +0.133 | +0.075 | +0.217 | **0.172** |
| Pipeline B | 1 | +0.010 | +0.041 | +0.153 | **0.630** |

**Neither fitted R² clears its own noise floor.** Two free parameters on 26
patients score ≈ +0.075 on pure noise, so A's +0.133 sits inside the null.
And B's +0.010 is *below* the noise mean for one parameter — random numbers
would have fitted better, which is a sharper statement than "B is at the
baseline".

Correcting for the parameter mismatch two ways:

| | adjusted R² (in-sample) | both at k = 1 (`a_only`), fitted | both at k = 1, held out |
|---|---|---|---|
| Pipeline A | **+0.058** | **+0.060** | **−0.084** |
| Pipeline B | −0.031 | +0.010 | −0.128 |

**A still leads B on every matched framing**, so the recommendation in §4
stands — but the honest size of the gap is +0.060 vs +0.010, not +0.133 vs
+0.010.

> **The deployed model does better than either row above.** `anemia fit-linear`
> applies the QC gate before fitting, which this comparison deliberately does
> not (both pipelines must see identical patients). With the gate, pipeline A's
> erythema model reaches in-sample R² **+0.195** (p = 0.074) and held-out
> **+0.007** (p = 0.067) — the only configuration in the project that beats its
> baseline on unseen patients. See `runs/linear_model.json` and
> `../README.md`.

**Why the baseline R² is exactly 0 here.** The baseline is the full-sample
mean, and R² is defined as `1 − Σ(pred−true)² / Σ(true−mean)²`. When the
prediction *is* the mean, numerator and denominator are identical, so R² = 0 by
construction. This is the correct comparison for a model also fitted on all
patients.

The leave-one-out baseline in §2b is different: it must predict patient *i*
from the mean of the *other* 25, which is systematically pulled away from
`y_i`. Its error is inflated by exactly `n/(n−1)`, giving

```
R²_baseline(LOO) = 1 − (n/(n−1))² = 1 − (26/25)² = −0.0816
```

That −0.082 is therefore an artefact of the protocol, not a property of the
data — and it is why a fitted model and a leave-one-out baseline must never be
put in the same table.


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

1. **A leads B on 10 of 11 metrics fitted, 9 of 11 held out, and on every
   parameter-matched framing in §2a** — no single one is significant, but the
   consistency of direction is itself weak evidence.
2. **A is ahead in every earlier experiment too** — cohort correlation
   (Spearman +0.30 vs +0.02), eye-to-eye consistency (2.02 vs 2.69), lighting
   robustness (SD 1.63 vs 2.74). Five independent measurements, same direction.
3. **A's extraction demonstrably works and B's does not.** Measured across the
   52 real captures (`../12_mask_placement/`), the redness index inside A's
   mask has a median of **+9.53** and never falls below the on-tissue
   threshold; inside B's mask it is **−3.67**, off-tissue on **46 of 52**.
   B's placement is statistically indistinguishable from the brightness
   baseline that scores Dice 0.000 against ground truth, and the two masks
   overlap by a **median of 0.000** — they are reading different parts of the
   photograph. This is the strongest argument and it is not statistical.
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
- **The paired test in §4 is weaker than it looks**, because
  `../12_mask_placement/` shows the two masks are essentially disjoint. It
  compares two models, one of which was largely not looking at conjunctiva, so
  it measures the combined effect of extraction and colour treatment rather
  than isolating either.
- **B's features are not even consistently wrong.** No patient has both eyes
  on-tissue, so of its 26 per-patient vectors, 20 average two off-tissue
  captures and 6 average one on-tissue with one off-tissue. Each patient is
  therefore measured on a different mixture of tissue types. A uniformly
  misplaced mask would at least be a stable measurement; this is not, which is
  a further reason not to read B's numbers as a colour-representation result.
