# Pre-checks for the colour-correction screen

Run: `python experiments/26_correction_prechecks/run.py` (add `--remeasure` to
run the extractor again, `--sheets` to rewrite the grading sheets), then
`python experiments/26_correction_prechecks/figure.py`.
Machine-readable: `results.json`, per photograph `measurements.csv`.

> **Later (17 Nov 2026):** the brightest 1% of unclipped pixels above the lower
> lid is now the pipeline's default colour correction
> (`PreprocessConfig.white_balance = "white_patch"`,
> `imaging.white_patch_white_balance`). On 20 photographs drawn at random, the
> pipeline's estimate of the light equals this experiment's `bright_eye`
> estimate exactly. Its effect on haemoglobin error is not yet tested; that
> remains the screen this experiment prepares for.

## Result

- **Plain brightest-pixel correction does nothing here.** In 96% of the 217
  photographs every channel reaches 255 somewhere, so "scale each channel so its
  maximum is white" leaves the image as it is. A usable version must ignore
  clipped pixels, and enough unclipped ones remain: the clipped area is
  0.05% of the frame in the median photograph.
- **Half the brightest unclipped pixels are not on the eye.** They fall on the
  white plastic of the capture device, below the lower lid. Hospital photographs
  may have no such object, so the brightest pixels have to be taken above the
  lower lid, where they land on the sclera.
- **Both new candidates estimate the light far better than gray-world.** The
  device's white plastic gives a reference for the light in 211 photographs.
  Measured against it:
  - the brightest pixels above the lid are 0.8–1.5° away;
  - the sclera is 1.0–2.3° away;
  - gray-world is about 5° away, further than no correction at all (2.8–4.7°).
- **The sclera is not white.**
  - Its b* rises with age by about 0.5–0.6 units per decade once the light is
    removed. India, Italy and the cleaner Italian batch alone all agree.
  - Its redness (a*) follows haemoglobin (ρ +0.28 to +0.51), as expected of
    tissue covered by the vessels of the bulbar conjunctiva.
  - Its yellowness shows no link with haemoglobin. Jaundice cannot be tested:
    there is no bilirubin.
- **Effect of dividing by the sclera, in India:**
  - It would remove about a fifth of the conjunctiva's haemoglobin slope, with a
    wide interval.
  - It would make older patients read lower by about 0.4 g/dL per decade, with an
    interval of 0.1 to 1.0.
  - Gray-world removes about 0.3 of the slope, and the plastic (the control)
    about 0.1.
- **Found along the way: the Italian set is two capture batches.**
  - Patients 1–31 were photographed on five days between late May and mid July
    2019. They are older (mean age 66) and more anaemic (mean Hb 11.5).
  - Patients 32–123 were photographed on eight days between 6 and 14 June. Mean
    age 44, mean Hb 14.6.
  - The plastic shows that the light differed between the two batches, so part
    of Italy's colour–haemoglobin link is the light. Correcting for the light
    (a batch term, or dividing by the plastic) lowers the Italian slope by
    11–16%.
  - This tilts Section 7.7's Italian results toward "no correction". In India,
    which is one batch, the same comparison was small and not significant.

Nothing here fits a haemoglobin model. These checks shape the screen in report
Section 12, step 3; they do not replace it.

## The questions

The screen will extend Section 7.7 (gray-world against no correction, mask held
fixed, 217 external patients) to two more corrections: the brightest pixels as
white, and the sclera as white. Each rests on an assumption that can be checked
first:

1. **Brightest pixel.** It assumes the brightest thing in the photograph is white
   and records the colour of the light. That fails if the brightest pixels are
   clipped at 255: they then record nothing, and if all three channels clip, the
   correction is no correction.
2. **Sclera.** It assumes the sclera is white in everyone. It yellows with age and
   with jaundice. If its colour follows age or haemoglobin, dividing by it adds
   or removes something other than the light.

## Method

**Data.** The 217 Eyes-defy photographs with haemoglobin: Italy 122, India 95.
All are 3984 × 2988, from a Samsung Galaxy S6 (SM-G920F in Italy, SM-G920I in
India) with white balance set to manual and no flash (EXIF). The dataset
describes a magnifying device with its own white LED that shuts out room light.

**Conjunctiva mask.** The deployed U-Net in the 512-pixel frame, on raw pixels,
the condition it was trained on.

**Sclera.** No sclera annotations exist, so a simple rule was written and
checked by eye:
- Every photograph has the everted lower lid at the bottom; the conjunctiva's
  centroid is in the lower half in all 217.
- The sclera is searched for in a band from 20 to 220 pixels above the top edge
  of the conjunctiva.
- Within the band, the brightest 30% of pixels are kept. Near-clipped pixels
  (any channel ≥ 250) and clearly red pixels (a* > 20: vessels, tissue) are
  dropped, and the largest connected piece is kept.
- Selection is on lightness, not on colour, so a yellow sclera is not discarded
  for being yellow.

The rule was fixed from overlays alone, before any sclera colour was compared
with age or haemoglobin.
- **Grading:** all 217 overlays were then graded by eye from index-labelled sheets
  in shuffled order, so site, age, haemoglobin and colour were hidden
  (`sclera_grading.json`; sheets in `grading/`).
- **Result:** 216 pass. One fails: Italy 45, a close-up with almost no white of
  the eye in view. It is excluded from every sclera analysis.

**The device's white plastic.** It sits in the bottom of the frame, below the
lower lid, in almost every photograph.
- **Rule:** the bottom 15% of the frame, more than 30 pixels below the lid, with
  L* ≥ 75, chroma ≤ 30 and no near-clipped pixels.
- **Coverage:** it is used where at least 3,000 such pixels exist (211 of 217);
  it is out of frame or mostly clipped in the other six.
- **Why it matters:** with manual white balance, the plastic's colour should
  record only the light reaching it and the phone's fixed processing, not the
  patient. That makes it a reference to compare each estimate of the light
  against.

**Estimates of the light**, each a mean RGB in the 512 frame, which is where
`prepare()` computes the correction:
- gray-world: the frame mean;
- max-RGB: the per-channel maximum;
- brightest 1% of unclipped pixels, over the whole frame or only above the
  lower lid;
- sclera: the mean inside the sclera mask.

Each is compared with the plastic by the angle between the two colours, the
standard error measure for an estimate of the light, which ignores brightness.

**Sclera colour:**
- **Measures:** the median CIELAB b* (yellowness) and a* (redness) inside the
  sclera mask. Each is also taken minus the plastic's, which removes the light
  and leaves the sclera itself.
- **Within each site:** Spearman correlations with age and haemoglobin, and a
  linear fit of b* on age (per decade), haemoglobin and sex, with a 2,000-sample
  patient bootstrap interval.
- **Primary test,** fixed in advance: sclera b* against age, one per site,
  Holm-corrected.

**Added after the first results:**
- **What each correction would do.** The sclera's redness turned out to follow
  haemoglobin, so the cost was measured directly.
  - A per-channel correction by an estimate E changes the conjunctiva's log(R/G)
    by exactly −log(E_R/E_G).
  - The estimate's log(R/G) and the conjunctiva's are each fitted on
    haemoglobin, age and sex within a site.
  - The ratio of the two haemoglobin slopes is the share of the conjunctiva's
    slope the correction removes.
  - The estimate's age slope, divided by the conjunctiva's haemoglobin slope, is
    how much the correction moves an older patient's reading, in g/dL per decade.
  - The plastic is the control, since a white object should move with neither.
- **The Italian batches.** The plastic itself followed haemoglobin in Italy,
  which a fixed white object should not. File names and EXIF dates were checked.

## Results

### 1. Clipping

| | India (95) | Italy (122) | All (217) |
|---|---|---|---|
| All three channels reach 255 somewhere, original resolution | 92% | 99% | **96%** |
| The same in the 512 frame, after downscaling blurs the glare | 55% | 85% | 72% |
| Pixels with a channel at 255: median photograph | 0.06% | 0.04% | 0.05% |
| 90th percentile / largest | 3.0% / 6.2% | 0.18% / 1.1% | 0.56% / 6.2% |
| Clipped pixels on the conjunctiva | 1% | 4% | 3% |
| Clipped pixels below the lower lid (device plastic, lid skin) | 43% | 13% | 26% |
| Clipped pixels elsewhere (corneal and scleral highlights, upper lid) | 57% | 83% | 71% |
| Photographs with any clipped pixel on the conjunctiva | 39% | 48% | 44% |
| Share of the conjunctiva clipped, 90th percentile | 0.16% | 0.41% | 0.25% |
| **Brightest 1% of unclipped pixels: below the lower lid** | 59% | 46% | **51%** |
| … on the sclera | 40% | 51% | 46% |
| … on the conjunctiva | 0% | 0% | 0% |

**Clipping is nearly universal but tiny.**
- **Where:** almost every photograph has a few saturated pixels in all three
  channels, so max-RGB is no correction. A median of 0.05% of the frame is
  clipped.
- **India:** the exception is 17 photographs where more than 1% is clipped,
  mostly on the plastic.
- **Downscaling:** shrinking to 512 averages clipped pixels with their
  neighbours. Max-RGB in the 512 frame then reads blurred glare, not the light.
- **The conjunctiva:** glare there is too small to matter for the colour reading
  (at most 0.41% of the mask in nine photographs out of ten).

**The brightest unclipped pixels are the plastic or the sclera, never the
conjunctiva.** Over the whole frame, "brightest" means "the device plastic" in
half the photographs. Restricted to above the lower lid, it means the brightest
parts of the sclera.

### 2. Estimates of the light, against the plastic

Angle in degrees, median [interquartile range]. "Size" is the angle between
the estimate and neutral grey, that is, how large a correction it applies.

| Estimate | India (90) | Italy (120) | Size, India / Italy |
|---|---|---|---|
| No correction | 4.73 | 2.76 | 0 / 0 |
| Gray-world (frame mean) | 4.90 [4.51, 5.54] | 5.15 [4.60, 5.79] | 5.2 / 5.6 |
| Max-RGB (512 frame) | 4.62 [4.37, 4.86] | 2.73 [2.37, 3.11] | 0.0 / 0.0 |
| **Brightest 1% unclipped, above the lower lid** | **1.47** [0.82, 2.27] | **0.81** [0.52, 1.13] | 3.5 / 2.9 |
| **Sclera mean** | 2.26 [1.55, 3.10] | 0.95 [0.50, 1.45] | 3.5 / 2.8 |
| Brightest 1% unclipped, whole frame | 0.43 [0.28, 0.88] | 0.46 [0.21, 0.82] | 4.3 / 2.7 |

**The plastic itself is 2.8–4.7° from neutral.** That is the cast left after the
phone's fixed processing, and it is what a correction should remove.
- **The new candidates:** the brightest pixels above the lid and the sclera
  remove most of it.
- **Gray-world:** its correction is larger than either, and points the wrong
  way. It ends further from the plastic than no correction did: slightly further
  in India, much further in Italy.
- **Why:** the frame is dominated by pink conjunctiva and skin, so its average is
  not grey.

The whole-frame brightest pixels score best, but that is circular: half of
those pixels are the plastic.

### 3. Sclera colour against age and haemoglobin

| | India (95) | Italy (121) | Italy, patients 32–123 (90) |
|---|---|---|---|
| **Sclera b* against age, ρ** | **+0.35** (p 0.0005, Holm 0.001) | **+0.19** (p 0.035, Holm 0.035) | +0.23 (p 0.03) |
| b* per decade, Hb and sex held | +0.85 [+0.57, +1.14] | +0.31 [+0.08, +0.57] | +0.34 [+0.01, +0.71] |
| **Light removed (minus plastic b*): per decade** | **+0.64** [+0.33, +0.93] | **+0.62** [+0.35, +0.86] | **+0.47** [+0.13, +0.84] |
| … ρ with age | +0.21 (p 0.04) | +0.48 (p 4 × 10⁻⁸) | +0.32 (p 0.002) |
| Sclera b* against Hb, ρ | +0.16 (p 0.12) | +0.05 (p 0.60) | −0.00 (p 0.99) |
| … light removed | +0.17 (p 0.12) | −0.15 (p 0.11) | +0.13 (p 0.22) |
| **Sclera a* against Hb, ρ** | **+0.51** (p 1 × 10⁻⁷) | **+0.44** (p 5 × 10⁻⁷) | +0.28 (p 0.008) |
| … light removed | +0.47 (p 4 × 10⁻⁶) | +0.18 (p 0.05) | +0.19 (p 0.07) |
| Spread of sclera b* (SD) | 2.37 | 1.99 | 1.99 |
| Sclera b* against plastic b*, ρ | +0.20 (p 0.06) | +0.35 (p 8 × 10⁻⁵) | +0.41 (p 6 × 10⁻⁵) |

**The sclera yellows with age.**
- **Agreement:** with the light removed, the three groups agree at 0.5–0.6 b*
  units per decade.
- **Size:** across 50 years that is about 3 units, 1.3–1.5 standard deviations
  of the sclera's yellowness.
- **Removing the light brings the groups together:** India's slope falls from
  0.85 to 0.64 per decade, and Italy's rises from 0.31 to 0.62.

**The sclera's redness follows haemoglobin.** The white of the eye is covered by
the bulbar conjunctiva, whose vessels pale with anaemia like those of the
palpebral conjunctiva. Its yellowness shows no link with haemoglobin.

**Jaundice is a separate question this cannot answer.** The dataset has no
bilirubin, and probably few jaundiced patients.

The sclera does follow the light (ρ +0.20 to +0.41 with the plastic), which is
what makes it a candidate at all.

### 4. What each correction would do to the conjunctiva

Share of the conjunctiva's haemoglobin slope removed, and the shift added with
age. The shift is in g/dL per decade; positive means older patients read
lower. Both are on log(R/G) with 95% bootstrap intervals.

India: conjunctiva slope +0.022 per g/dL [+0.009, +0.035], n = 90.
Italy: +0.034 [+0.025, +0.044], n = 120.

| Correction by | India: share removed | India: age shift | Italy: share removed | Italy: age shift |
|---|---|---|---|---|
| Plastic (control) | 0.10 [−0.04, 0.30] | +0.05 [−0.03, 0.22] | 0.16 [0.09, 0.25] | −0.14 [−0.22, −0.06] |
| Gray-world | **0.29** [0.09, 0.61] | +0.14 [−0.04, 0.48] | **0.35** [0.23, 0.48] | −0.19 [−0.33, −0.07] |
| Max-RGB | 0.03 [−0.05, 0.16] | +0.07 [−0.01, 0.24] | 0.00 [−0.01, 0.01] | 0.00 [−0.01, 0.02] |
| Brightest 1%, above the lower lid | 0.15 [−0.10, 0.48] | **+0.47** [+0.19, +1.07] | 0.18 [0.10, 0.28] | +0.04 [−0.05, 0.13] |
| Sclera | 0.18 [−0.05, 0.57] | **+0.41** [+0.11, +1.01] | 0.21 [0.11, 0.33] | +0.05 [−0.06, 0.15] |

**India is the cleaner test.** It is one batch, and its plastic follows neither
haemoglobin nor age.
- **Sclera and brightest pixels:** each removes somewhat more of the slope than
  the plastic does, with wide intervals. Each also makes older patients read
  lower by about 0.4–0.5 g/dL per decade. Over India's age range (19–75) that
  is 2.3–2.6 g/dL.
- **Why:** the Indian sclera's red-to-green ratio rises with age along with its
  yellowness, which reads as browning; its a* does not change. The brightest pixels above the lid lie mostly on the sclera, so they
  inherit this. Age-related pigmentation of the sclera, common in darker-skinned
  populations, is one possible cause; this check cannot tell.
- **Gray-world:** it removes 0.29 of the slope, about three times what the plastic
  removes. Its frame mean contains the conjunctiva and pink skin, so it divides
  out part of the signal. That is a reason for Section 7.7's result that does not
  depend on the batches below.

**Italy's control is not clean** (section 5). Italy's numbers mix the batch
difference in light with the corrections' own effects. In the Italian batch of
patients 32–123 alone, the conjunctiva's own haemoglobin slope is too weak
(interval spanning zero) for these ratios to be informative.

### 5. The Italian set is two capture batches

| | Patients 1–31 | Patients 32–123 |
|---|---|---|
| File names | `1.jpg` … `31.jpg` | `T_<n>_<date>_<time>.jpg` |
| Capture dates (EXIF) | 5 days: 28 May, 5 and 11 June, 11–12 July 2019 | 8 days, 6–14 June 2019 |
| Mean Hb (g/dL) | 11.5 | 14.6 |
| Mean age (years) | 66 | 44 |
| Plastic log(R/G), mean (SD) | −0.001 (0.025) | +0.033 (0.014) |
| Conjunctiva log(R/G), mean | 0.540 | 0.761 |
| Plastic against Hb, ρ | +0.44 | +0.01 |
| Conjunctiva redness (r chromaticity) against Hb, Pearson | +0.55 | +0.32 |

Haemoglobin, age and the plastic's colour each differ between the batches
(Mann–Whitney p < 10⁻⁸). The same phone model and settings were used, and the
dataset describes a fixed white LED, yet the light reaching the plastic was
less red for patients 1–31. A different device unit, a fading LED or a different
exposure could each explain this; the photographs cannot say which.

**Why it matters.** The more anaemic batch was photographed under a less red
light, so its conjunctivas look less red partly because of the light. Pooled
over both batches, colour tracks haemoglobin more closely than within either
batch, partly for that reason:
- **Correlation:** r +0.64 pooled, against +0.55 and +0.32 within the batches.
- **Slope:** the conjunctiva's haemoglobin slope falls from 0.0344 to 0.0307 with
  a batch term (−11%), and to 0.0289 with the light removed (−16%).
- **Within each batch:** the slopes are 0.0367 for patients 1–31 and 0.0213 for
  patients 32–123. India's is 0.0220.

**Consequences for Section 7.7:**
- **What it measured:** it pooled both batches within Italy. Any correction that
  tracks the light removes the batch difference in light along with everything
  else, so "no correction" gains an advantage that is not about colour
  correction.
- **Size of that advantage:** not measured here.
- **Cross-site tests:** they also train or test on pooled Italy.
- **India:** one batch, where Section 7.7's difference was small and not
  significant (MAE +0.03 [−0.02, +0.09] g/dL).
- **Gray-world:** the case against it still stands in direction. It removes about
  three times the plastic's share of the slope in India.

Within patients 1–31 the plastic still follows haemoglobin (ρ +0.44, n = 31).
Those photographs span five days, and the batch may itself be several sessions.

## What this changes

For the screen (report Section 12, step 3):

1. **Brightest pixels:** use only unclipped pixels (no channel ≥ 250, rims
   excluded), and only above the lower lid. Otherwise the method is either no
   correction or a reading of the device plastic.
2. **Sclera as white:** keep it, with age in the model, and report India
   separately. Expect it to add an age shift there.
3. **Add the plastic as an arm.** It is the closest thing to a measured white in
   this dataset, so it shows what correcting the light can achieve here. It does
   not transfer to photographs without it.
4. **Model the Italian batch:** with a batch term, or by reporting India and
   patients 32–123 separately. Otherwise every correction that works is
   penalised for removing the batch difference in light.
5. **Judge on haemoglobin error,** as planned. The shares above use one feature
   and a linear fit, and the regressor may use others.

Elsewhere:
- **Section 7.7, and every Italian result that pools patients 1–31 with the rest,
  carries the batch difference.** The direction against gray-world is supported
  for its own reason (section 4). The size of the no-correction advantage in
  Italy is uncertain.
- **A printed white reference** (report Section 12, step 9) is supported in
  principle. The plastic tracks the light where the batch does not interfere.
  A reference card would need to sit where light reflected from the face cannot
  tint it.
- **Glare on the conjunctiva needs no action.**

## Limits

- **The plastic is not a calibrated white.**
  - Its true colour is unknown, and it is out of focus and close to the lens.
  - Light reflected from the face could tint it.
  - Supporting it as a reference: in India and in patients 32–123 it follows
    neither haemoglobin nor age.
- **The sclera detector is a rule, graded by one rater by eye.** Grading hid site,
  age and colour, but it is not an independent annotation.
- **Colour is measured on gamma-encoded sRGB values, as `prepare()` applies its
  correction, not in linear light.** Angles in linear light would differ
  somewhat.
- **No bilirubin, so jaundice cannot be tested.**
- **Section 4 uses one colour feature and a linear model.** Its intervals are wide
  at n = 90–120.
- **Not all of this was planned.** The pre-planned checks were the clipping census
  and sclera b* against age. The analysis of what each correction would do and
  the batch check were added after the first results showed the sclera's redness,
  and the plastic, following haemoglobin.
- **Six photographs have no usable plastic, and one has no usable sclera.**

## Files

- `run.py`: detectors, measurement, analysis; `--sheets` writes the grading
  sheets.
- `figure.py` → `report_figure.png`. Left: estimates of the light against the
  plastic. Middle: sclera b* minus plastic b* against age, with simple fitted
  lines; the per-decade numbers in the legend hold Hb and sex. Right: the
  plastic's colour against haemoglobin for the two Italian batches.
- `measurements.csv`: one row per photograph.
- `results.json`: every number above.
- `sclera_grading.json`: how the sclera overlays were graded, and the one
  exclusion.
- `grading/`: the eleven index-labelled sheets, with `index_to_patient.json`.
