# Gray-World White Balance Versus No Correction

Run: `python experiments/18_white_balance/compare.py`
Outputs: `results.json`, cached per-patient features `features.csv`.

Gray-world was adopted early on evidence from the 26-patient local cohort
(experiment 03: Spearman +0.30 corrected vs +0.02 uncorrected), since shown to be
anomalous (experiment 16). On the 217 external patients the evidence pointed the
other way, and one severely anaemic patient was missed because gray-world
over-corrected her sclera-dominated close-up (experiment 17). The argument *for*
gray-world is portability across cameras. Both sides are measured here.

**Only the colour reading changes.** Every patient gets one mask — the learned
extractor run on the gray-world image, exactly as deployed — and each arm reads
chromaticity inside that same mask from either the uncorrected or the balanced
pixels. Ridge regression on red and green chromaticity; 10-fold cross-validation
repeated over 5 random splits; 95% confidence intervals by patient-level bootstrap.

## 1. Within a site

| site | n | R², no correction | R², gray-world | red vs Hb, none / gray-world | gray-world minus none: MAE (95% CI) |
|---|---|---|---|---|---|
| India | 95 | **0.251** | 0.192 | +0.53 / +0.44 | +0.03 [−0.02, +0.09] |
| Italy | 122 | **0.409** | 0.301 | +0.65 / +0.58 | **+0.10 [+0.03, +0.16]** |
| Both | 217 | **0.467** | 0.335 | +0.68 / +0.47 | **+0.16 [+0.09, +0.23]** |

Gray-world is worse at both sites. The difference is significant for Italy and
for both sites together; for India alone it points the same way with an interval
that includes zero (the R² interval, [−0.09, −0.003], does exclude it).

## 2. Across sites — the job gray-world exists to do

| fit → test | MAE, no correction | MAE, gray-world | MAE, predict training-site mean |
|---|---|---|---|
| India → Italy | 2.84 | **4.00** | 2.82 |
| Italy → India | **2.39** | 2.87 | 2.63 |

Cross-site R² is strongly negative for every arm, including the baseline, because
the two sites' average haemoglobin differs; the fair reference is the baseline
column. **Gray-world is worse than guessing the training site's average in both
directions.** No correction matches the guess one way and beats it the other.
The portability gray-world was adopted for does not appear.

## 3. The screening outcome

Best configuration from experiment 17: colour + sex, category cut chosen per sex.

| | no correction | gray-world |
|---|---|---|
| held-out MAE (g/dL) | **1.247** | 1.329 |
| held-out R² | **0.544** | 0.482 |
| anaemic caught | 93% | 93% |
| anaemic women / men told "unlikely" | 3% / 13% | 4% / 12% |
| share in "Possibly" | **9%** | 12% |
| "Likely" truly anaemic | **74%** | 70% |
| "Unlikely" truly not anaemic | 93% | 93% |
| patient at Hb 7.0 (`eyesdefy-italy:105`) | **caught** | missed |

No correction gives the best configuration measured in the project: R² 0.544,
a clear answer for 91% of patients, and the severe case no longer missed.

## Why gray-world hurts here

Gray-world rescales each photo so the whole frame averages grey. How much of the
frame is sclera, iris, skin and lashes varies from photo to photo — close-up or
wide, eye open or looking down — so the correction applied to the conjunctiva
depends on **framing**, not only on lighting. That injects photo-to-photo colour
noise that has nothing to do with haemoglobin. Within one study's capture setup,
where lighting and camera are fairly consistent, uncorrected colour is the more
faithful measurement. Across the two sites, framing differs as much as cameras do,
so gray-world does not equalise them either.

## Decision and caveats

**Evidence favours measuring colour on uncorrected pixels**, while still running
the extractor on the balanced image (the mask was held fixed that way here). **The
deployed pipeline has not been changed.** Before changing it:

- **Both "sites" belong to one study** (Eyes-Defy-Anemia), and likely share a
  protocol. A genuinely different camera — the hospital's — has not been tested.
  Uncorrected colour is camera-dependent; per-site recalibration or a printed
  colour reference would still be required for a new site.
- The local 26-patient cohort, where gray-world looked better, lacks the colour
  signal altogether (experiment 16), so it cannot arbitrate.
- Bootstrap intervals use one fold assignment; the 5-split ranges agree in
  direction throughout.
- The earlier gray-world recommendation (experiment 03) stands as an accurate
  record of the 26-patient evidence, and is superseded by this measurement.
