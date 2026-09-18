# Three Screening Categories Instead of a Number

Run: `python experiments/17_screening_categories/categories.py`
Outputs: `categories.json`, `unet_patient_categories.csv` (one row per patient),
cached features `features_unet.csv`, `features_refined.csv`.

The app is a screening aid, not a replacement for a blood test. So instead of a
haemoglobin number or a hard verdict, it would say one of:

| app says | when |
|---|---|
| **Likely anaemic — get a blood test** | P(anaemic) ≥ 0.50 |
| **Possibly anaemic — consider a blood test** | lower cut ≤ P < 0.50 |
| **Anaemia unlikely from this photo** | P < lower cut |
| Couldn't read this photo — please retake | quality check fails (already implemented) |

## Method

- **P(anaemic) comes from the model's own track record**, not a bell curve: the
  share of the model's past errors e for which *estimate + e* falls below the
  patient's WHO cutoff. Works unchanged for any model, including a network.
- **The lower cut is chosen so the top two categories catch 90% of anaemic
  patients** — an example target; the real one is a clinical decision.
- **Nothing is tuned on the patients being scored.** Outer 10-fold loop; inside
  each fold the model is fitted, its errors measured by an inner 10-fold loop, and
  the lower cut chosen, all on the other patients. Repeated over 5 random splits.
- 217 external subjects (Eyes-Defy-Anemia), per-patient WHO cutoff from age and
  sex, chromaticity features, ridge regression. **42% anaemic.**
- A **no-information reference** (every patient gets the training anaemia rate
  as their probability) shows what a useless model looks like on these tables.

## Table 1 — What the app would say, and who is actually anaemic

| | U-Net (deployed) | | Classical | | No information | |
|---|---|---|---|---|---|---|
| category | share | truly anaemic | share | truly anaemic | share | truly anaemic |
| **Likely** | 43% | **69%** | 35% | 51% | 0% | — |
| **Possibly** | 22% | 31% | 47% | 46% | 100% | 42% |
| **Unlikely** | 35% | **15%** | 18% | 15% | 0% | — |

Against a 42% base rate, the deployed extractor sorts people meaningfully: a
"Likely" patient is anaemic 69% of the time, an "Unlikely" one 15%.

## Table 2 — Anaemic patients told "Unlikely"

| | U-Net | Classical | No information |
|---|---|---|---|
| missed | **11 of 91 (12%)** | 6 of 91 (7%) | 0 |
| caught by top two categories | 88% (88–90% over 5 splits) | 93% | 100% |
| missed by more than 2 g/dL below their cutoff | **5** | 3 | 0 |

The deployed extractor's misses, by true haemoglobin: **7.0**, 8.5, 9.7, 9.7, 9.8,
10.3, 10.7, 11.0, 11.4, 11.8, 12.9. **A patient at 7.0 g/dL — severe anaemia — was
estimated at 13.6 and told "unlikely".** The wording "from this photo" and a
symptom prompt are not optional.

The 90% target is met on training patients and lands at 88–90% on held-out ones:
a small, expected shortfall from choosing a cut on one set and applying it to
another.

## Table 3 — Does the model commit?

| | U-Net | Classical | No information |
|---|---|---|---|
| **"Possibly" (no clear answer)** | **22%** | 47% | 100% |
| given a clear answer | **78%** | 53% | 0% |
| "Likely" truly anaemic | **69%** | 51% | — |
| "Unlikely" truly not anaemic | 85% | 85% | — |

**This table is why sensitivity alone is a trap.** The no-information model catches
100% of anaemic patients by telling everyone "possibly". The classical extractor
catches more than the U-Net (93% vs 88%) almost entirely by putting nearly half of
all patients in the middle, and its "Likely" group is barely more anaemic than
the population. **The U-Net gives a clear answer to 78% of patients** with a
better "Likely" group and the same "Unlikely" reliability. On the question that
matters for an app — can it sort people? — the learned extractor wins clearly.

## The finding to raise: the misses fall on women

| | cutoff | anaemic | told "Unlikely" | mean estimate vs true Hb (anaemic) |
|---|---|---|---|---|
| **women** | 12.0 | 53 | **9 (17%)** | 11.7 vs 9.9 |
| men | 13.0 | 38 | 2 (5%) | 12.1 vs 11.4 |

The model's estimates pull toward the cohort average, **12.8 g/dL** — typical of a
model with R² ≈ 0.3. That average sits **above the women's cutoff (12.0) and below
the men's (13.0)**. So hedging pushes anaemic women toward "not anaemic" and
anaemic men toward "anaemic". Anaemic women here are also more severely anaemic
(mean 9.9 against 11.4), so they are pulled further.

The counts are small, but the mechanism is structural: **any model that shrinks
toward a single pooled average will under-flag the group whose cutoff sits below
that average**, and that includes a neural network. Options, not yet tested: give
the model sex as an input, shrink toward sex-specific averages, or choose the
category cuts separately for men and women.

## Follow-up: does giving the model sex fix the misses? (`sex_feature.py`)

Same nested protocol and category rule; only the model inputs, or how the lower
category cut is chosen, change. Mean over 5 random fold splits.

| | colour only | colour + sex | colour, per-sex cut | **colour + sex, per-sex cut** | sex only, no photo |
|---|---|---|---|---|---|
| held-out R² | 0.335 | 0.482 | 0.335 | **0.482** | 0.295 |
| held-out MAE (g/dL) | 1.53 | 1.33 | 1.53 | **1.33** | 1.63 |
| anaemic **women** told "unlikely" | **17%** | 4% | 10% | **4%** | 0% |
| anaemic **men** told "unlikely" | 4% | **19%** | 11% | **12%** | 0% |
| anaemic caught, overall | 89% | 90% | 89% | **93%** | 100% |
| share in "Possibly" | 20% | 13% | 14% | **12%** | 60% |
| "Likely" truly anaemic | 68% | 70% | 68% | **70%** | 62% |
| "Unlikely" truly not anaemic | 87% | 89% | 89% | **93%** | — |

(An interaction model, letting colour's effect differ by sex, matched colour + sex
on every row and is omitted.)

### Adding sex as an input moves the misses rather than fixing them

It improves the estimate substantially (R² 0.335 → 0.482, MAE 1.53 → 1.33), and
women's misses fall from 17% to 4% — but **men's rise from 4% to 19%**. Once the
model knows sex, it shrinks toward each sex's own average instead of the pooled
one:

| | mean Hb | WHO cutoff | mean vs cutoff | anaemic |
|---|---|---|---|---|
| women | 11.17 | 12.0 | below | 62% |
| men | **13.86** | 13.0 | **above** | 29% |

Men's average now sits above their cutoff, so hedging pushes anaemic men toward
"fine" — the same mechanism, relocated. The general rule: **a model that hedges
toward an average under-flags whichever group's average sits above its own
cutoff.** A better model reduces hedging but cannot remove it, so the fix has to
be in how the categories are drawn.

### Choosing the category cut per sex does fix it

Setting the lower cut separately for women and men (each on its own training
patients, each to the 90% target) brings both groups near target together with
sex as an input: **women 4%, men 12% missed**, 93% caught overall, the smallest
"Possibly" group (12%), and "Unlikely" correct 93% of the time. This is the best
configuration tested on every row. Men's 12% sits slightly past the 10% target,
consistent with estimating a cut from only ~34 anaemic men per training fold.

### Is the photograph still earning its place?

Sex alone, with no photograph, reaches R² 0.295 — close to colour alone (0.335),
because men have higher haemoglobin. But it cannot sort anyone: 60% of patients
land in "Possibly" and nobody is ever told "unlikely". Colour + sex reaches 0.482
and gives a clear answer to 88% of patients. **The photograph adds substantially
on top of sex**, though less than colour-only R² suggested in isolation.

### One patient no rule rescues — and what her photo shows

A woman with **Hb 7.0** (severe anaemia, `eyesdefy-italy:105`) is told "unlikely" in
every arm and every split; her photograph reads as healthy (estimate 13.6). Photo:
`patient_italy105_hb7.jpg`.

- **The mask is correct.** The U-Net traces the lower-lid conjunctiva cleanly.
- **She is the most anaemic Italian subject**: nobody else from that site is at or
  below 8.5 g/dL, so she is far outside the range the model saw from it.
- **Her raw photo has the coolest cast of 40 Italians checked**: the white of her
  eye has the lowest red share (0th percentile) and highest blue share (100th) at
  normal brightness. It looks lavender as photographed.
- **Gray-world over-corrects it.** The frame is a close-up dominated by bluish sclera
  and a blue-grey iris, violating gray-world's assumption that the scene averages
  grey; after correction the sclera turns visibly mint-cyan and the conjunctiva is
  pushed toward orange-red.

Her colour cues, against the 121 other Italian patients:

| cue | gray-world (model input) | no white balance | relative to own sclera |
|---|---|---|---|
| blue | reads healthy | **reads anaemic** | reads healthy |
| red | reads anaemic | **reads anaemic** | reads healthy (marginal) |

**Without white balance, both cues correctly indicate anaemia**; gray-world flips the
blue cue the model relies on most. Sclera-referencing fails for the same reason —
her sclera is the bluest measured.

**Not only her.** Across all 122 Italian patients colour tracks haemoglobin more
strongly *without* gray-world (blue r −0.60 vs −0.56, red +0.66 vs +0.59), and on all
217 external patients red chromaticity falls from +0.67 raw to +0.47 after
gray-world. Gray-world was adopted on evidence from the 26-patient local cohort,
since shown to be anomalous (experiment 16). **It is costing signal within a
consistent capture setup and can actively mislead on sclera-heavy frames.** The
trade-off is that uncorrected colour depends on the camera, which is what gray-world
was meant to absorb. A held-out comparison of gray-world against no correction,
within site and across sites, is needed before any default changes.

## Honest limitations

- **Calibrated on Eyes-Defy cameras.** Error distributions measured here do not
  carry to another camera (experiment 14); categories would be overconfident at a
  new site until re-calibrated on its own labelled photos.
- **42% of this cohort is anaemic.** In a community setting with fewer anaemic
  people, "Likely" would contain a smaller share of true cases and "Unlikely" a
  larger share of true negatives. Table 1's percentages are specific to this mix.
- The 90% target and the 0.50 upper cut are examples. Where to put them trades
  missed cases against the size of the "Possibly" group — a clinical decision.
- Linear regression on two colour features stands in for the final model; the
  method, not these numbers, is what carries forward.
