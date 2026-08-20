# The Fitted Equation, and Honest Uncertainty

Run: `python experiments/10_fitted_equation/fit_equation.py`
Machine-readable coefficients: `fitted_equation.json`

---

## 1. Why only one R², when leave-one-out fits 26 models?

Because each of those 26 models predicts **exactly one** held-out patient. The
run therefore produces 26 predictions in total — one per patient — not 26
separate result sets.

R² of a single data point is undefined (the variance of one number is zero), so
the 26 out-of-fold predictions are **pooled and scored once**. That is why
there is one R², one F1, one MAE.

This differs from k-fold, where each fold holds out several patients and can be
scored separately, giving a mean ± spread across folds.

### The single number needs an interval

A point estimate from 26 patients is not precise, and reporting it alone
overstates certainty. Bootstrap resampling over patients, 2000 draws:

| Metric | Point estimate | 95% confidence interval |
|---|---|---|
| MAE | 1.073 g/dL | **[0.708, 1.490]** |
| R² | −0.166 | **[−0.512, +0.021]** |
| F1 | 0.688 | **[0.462, 0.850]** |

**The R² interval includes values just above zero.** So the defensible claim is
*"no better than predicting the mean"*, not *"precisely −0.166"*. Quoting the
point estimate to three decimals implies a precision we do not have.

Repeated 5-fold, which does give a genuine per-fold spread:

```
fold MAE: mean 1.079, sd 0.402, range 0.359 – 2.076
```

An individual fold can look excellent (0.36) or terrible (2.08) purely by which
patients it contains — the same instability seen in experiment 09.

---

## 2. Is 25-train / 1-test too extreme?

It is the **low-bias, high-variance** end of a trade-off, and it is deliberate.

- **Low bias:** the model is trained on 25 of 26 patients, so it is almost
  exactly the model you would deploy on all 26. Training on fewer (say 13)
  gives a systematically *pessimistic* estimate, because a smaller training set
  produces a worse model than the one you would ship.
- **High variance:** each individual test is a single patient, so any one of
  them is noisy. Pooling all 26 is what recovers a usable estimate.

Measured across training sizes — every patient is still tested exactly once in
all four protocols; only the training size differs:

| Protocol | Train size | Pooled MAE | Pooled R² |
|---|---|---|---|
| 2-fold | 13 | 1.063 | −0.137 |
| 5-fold | 21 | 1.034 | −0.119 |
| 13-fold | 24 | 1.070 | −0.142 |
| Leave-one-out | 25 | 1.073 | −0.166 |

They agree. Training size is **not** driving the conclusion, so the choice of
25/1 is safe here. It would matter more with a model that improves steeply with
data — a CNN, for instance, where 5-fold is also the only affordable option
(leave-one-out would mean training the network once per patient).

---

## 3. The fitted equation

### Inputs

Three numbers per patient. Each is the average over **conjunctiva pixels only**
(inside the segmentation mask) of the gray-world white-balanced image at
512×512:

| Input | Cohort mean | Cohort sd |
|---|---|---|
| mean R | 163.41 | 13.48 |
| mean G | 152.53 | 16.64 |
| mean B | 167.80 | 18.21 |

Both eyes of a patient are averaged into one feature vector before fitting.

### The equation, standardised form

This is what the solver actually optimises — features are centred and scaled so
the ridge penalty treats them comparably:

```
Hb = 11.2038 + 0.1373·z(mean R) − 0.9771·z(mean G) + 0.6473·z(mean B)

where  z(x) = (x − cohort mean) / cohort sd
```

The intercept 11.2038 is the cohort mean haemoglobin: with all inputs at their
average, the model predicts the average.

### The equation, raw pixel units

Algebraically identical, directly usable on raw channel means:

```
Hb = 12.5311 + 0.010191·(mean R) − 0.058732·(mean G) + 0.035552·(mean B)     [g/dL]
```

### Worked example — patient 1

```
inputs:  R = 156.13,  G = 164.55,  B = 188.01

Hb = 12.5311 + (+0.010191 × 156.13)
             + (−0.058732 × 164.55)
             + (+0.035552 × 188.01)
   = 11.142 g/dL

laboratory value: 12.0 g/dL     error: −0.86 g/dL
```

### What the coefficients say

| Input | Effect of +1 sd | Share of Hb sd (1.37 g/dL) |
|---|---|---|
| mean G | **lowers** predicted Hb by 0.977 g/dL | 71% |
| mean B | **raises** predicted Hb by 0.647 g/dL | 47% |
| mean R | raises predicted Hb by 0.137 g/dL | 10% |

---

## 4. The coefficients fail a physiological sanity check

This is the most informative part of the exercise, and it is independent
evidence for the project's central negative result.

**What we would expect if the model had found real physiology.** Haemoglobin
makes tissue red. More haemoglobin should mean a *higher* red channel, so the
**R coefficient should dominate and be positive**.

**What we actually get.** The red channel is nearly irrelevant (+0.137, the
smallest of the three). The model is driven by **green, negatively** (−0.977)
and **blue, positively** (+0.647) — effectively computing a *blue-minus-green*
contrast.

Blue-minus-green is not a haemoglobin signal. It is much closer to a
lighting-and-white-balance artefact: residual colour cast the gray-world
correction did not fully remove. The model has latched onto whatever varied
between captures, and in this cohort that was illumination, not physiology.

**This matters for the report** because it is a check that does not depend on
any metric. Even if R² had come out mildly positive, coefficients pointing the
wrong way physiologically would be reason to distrust it. The metrics and the
coefficients agree: nothing real was learned.

**A concrete test for the next iteration:** once the segmenter is trained and
the corpus is ~900 images, re-fit this same model and inspect the coefficients
first. If the R coefficient becomes dominant and positive, the model is
plausibly reading tissue. If blue-minus-green persists, the problem is upstream
in colour handling, not in the model.

---

## 5. Limits

- 26 patients; all confidence intervals are wide, and the coefficients would
  move substantially with more data.
- Coefficients come from fitting on all 26 (the deployment fit). The
  out-of-fold metrics come from 26 separate fits, so the two are related but
  not identical.
- Ridge shrinks coefficients toward zero, so their magnitudes are conservative;
  their *signs* and *relative order* are the interpretable part.
- Channel means discard all spatial information — the equation cannot represent
  vessel patterns or the distribution of pallor.
