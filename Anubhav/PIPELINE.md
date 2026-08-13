# Non-Invasive Anemia Detection — Pipeline Explained

How a photograph of the inner eyelid becomes a haemoglobin estimate and a
screening verdict, why each stage is built the way it is, and what is still
unproven.

**Status: the pipeline runs end to end and is validated on synthetic phantoms.
No model has been trained on real patient data yet. No accuracy number in this
document is a clinical result.**

---

## 1. The clinical idea

Haemoglobin carries oxygen and makes blood red. When it drops, perfused tissue
loses colour. The **palpebral conjunctiva** — the moist inner surface of the
lower eyelid — is one of the best places on the body to see this, because:

- it has no melanin, so its colour is not confounded by skin tone;
- it is thin and densely vascularised, so blood colour shows through directly;
- it can be exposed by any clinician (or the patient) simply by pulling the
  lower lid down.

Clinicians already use conjunctival pallor as an informal anemia check. The
project's goal is to make that judgement quantitative, from a phone camera.

The output is deliberately two-part:

| Output | Why |
|---|---|
| Estimated Hb in g/dL | A continuous number is more informative than a label, and lets the threshold change without retraining. |
| Anemic / not anemic | What a patient actually needs to be told. |

---

## 2. Pipeline overview

```
photo
  │
  ├─ 1. white balance ─────────── neutralise the illuminant
  │
  ├─ 2. segmentation ──────────── find the conjunctiva, discard everything else
  │
  ├─ 3. crop + letterbox ──────── square the ROI without distorting it
  │
  ├─ 4. quality control ───────── reject unusable captures (gate, not a note)
  │
  ├─ 5. Hb regression ─────────── ResNet → a continuous g/dL value
  │
  └─ 6. thresholding ──────────── WHO cutoff for this patient's age and sex
        │
        └─ Hb estimate + anemic / not anemic
```

The same code path runs during training and inside the API. That is a design
constraint, not a coincidence: if the server conditions images even slightly
differently from how the model was trained, accuracy degrades in a way that is
invisible to tests.

---

## 3. Stage by stage

### 3.1 Lighting normalisation — `imaging.gray_world_white_balance`

Ward lighting varies wildly (fluorescent, tungsten, daylight through a window),
and every phone's image processor applies its own colour correction. Both shift
the image's white point, and that shift is large enough to swamp the pallor
signal we are trying to measure.

**Gray-world assumption:** averaged over a whole scene, the world is grey. So
compute each channel's mean, and scale the channels until those means agree.

```
scale_c = mean(all channels) / mean(channel c)
```

This is crude but robust and needs no calibration card. A better long-term
option is a physical colour reference in frame (a printed card, or a sticker on
the phone case), which is what several published systems do.

### 3.2 Segmentation — `segment.py`

**Why segment at all?** The regressor must see conjunctiva and nothing else. If
skin, eyelashes, or sclera enter the crop, the network can learn to predict Hb
from skin tone — which correlates with ethnicity, not haemoglobin. That is both
scientifically wrong and an equity failure.

Two backends, one interface. Any segmenter is a callable
`(rgb) -> (mask, name)`, so backends swap without touching anything downstream.

**Primary: Mask2Former (Swin-Tiny), fine-tuned.** Trained on the hand-drawn
masks in Eyes-defy-anemia — the only dataset with pixel-level ground truth.
Trained once, then applied to every other dataset to produce crops.

**Fallback: a colour prior.** The conjunctiva is *red*. We score each pixel by

```
redness = a* − 0.5·b*        (CIELAB)
```

`a*` is the green–red axis, so high `a*` is red. Subtracting half of `b*` (the
blue–yellow axis) pushes down peri-orbital **skin**, which is red *and* strongly
yellow. On phantoms this separates cleanly: conjunctiva ≈ +14 to +26, skin ≈ −2,
sclera ≈ −5.

Two details in this stage were learned the hard way and are worth understanding,
because both are the same kind of mistake:

> **The lightness gate must be absolute, not a percentile.**
> Healthy conjunctiva is *darker* than pale conjunctiva — deeper red absorbs
> more light. Measured on phantoms, lightness falls from 174 at Hb 8 to 129 at
> Hb 14. A percentile-based "drop the darkest pixels" rule therefore deletes the
> **high-haemoglobin patients specifically**. The gate exists only to remove
> lashes, pupil, and deep shadow, which are near-black in absolute terms.

> **The threshold must adapt, not be fixed.**
> How much of the frame the conjunctiva fills depends entirely on how close the
> phone was held. A fixed percentile either floods the mask with skin — after
> which "keep the largest connected component" returns the *skin blob* — or
> misses the tissue entirely. Otsu's method finds the split point from the
> image's own histogram.

> **Measured on real photographs, the plain colour prior is not good enough.**
> Screened across 52 real captures it bled onto eyelashes and lower-lid skin —
> both reddish-brown enough to survive the redness test. Dice 0.999 on
> flat-colour phantoms, visibly poor on real tissue: exactly the gap a phantom
> score cannot reveal.

**The refined extractor** (`refined_conjunctiva_mask`) fixes this with a
seeded grabCut built on three observations about what actually distinguishes
the tissue:

1. **Conjunctiva is smooth; the lash zone is high-frequency.** A local
   standard-deviation map separates them, and high-texture pixels are seeded
   as definite background.
2. **Sclera is bright, desaturated, and not red; iris and pupil are dark.**
   Both are seeded as definite background explicitly rather than left to the
   colour model — a brown iris is dark *red*, and without an absolute darkness
   gate it wins the redness contest on wide shots.
3. **Specular highlights sit on the tissue.** The wet surface reflects the
   light source as white blobs inside the region we want; they are seeded as
   probable foreground and remaining holes are filled afterwards.

The seeds drive a mask-initialised grabCut whose edge-aware model stops the
region growing through the lash line. A wide morphological opening then shears
off any lash tendrils, the largest component is kept, and holes are filled.

Re-screened over all 52 real captures, the refined extractor produces clean,
tissue-hugging masks on visual review; the median mask ratio halves (0.21 →
0.10), consistent with the bleed being removed rather than the tissue. The one
residual failure mode is a **barely-everted lid**, where almost no conjunctiva
is exposed — a capture-quality problem, not a segmentation one. There is still
no ground-truth Dice on real photographs; that requires the Eyes-defy-anemia
masks.

The mask is then morphologically opened and closed, and the largest connected
component kept, to remove speckle.

### 3.3 Crop and letterbox — `imaging.pad_to_square`

The conjunctiva is a **crescent**: wide and short, often 3:1 or worse. Resizing
that straight to 224×224 stretches it vertically, which elongates the
micro-vasculature and corrupts the spatial density of vessels — one of the cues
the network can use.

So: tight bounding box around the mask (with 8% padding), **pad symmetrically
with black to 1:1**, *then* resize. Aspect ratio is preserved throughout.

**CLAHE is deliberately not applied.** Local histogram equalisation next to the
black letterbox creates severe boundary artefacts, and — more importantly — it
normalises away the very redness difference that encodes haemoglobin. Enhancing
contrast would destroy the signal.

### 3.4 Quality control — `preprocess.assess_quality`

Three checks:

| Check | Method | Catches |
|---|---|---|
| Focus | variance of the Laplacian | motion blur, misfocus |
| Mask area | fraction of frame segmented | lid not everted, camera too far or too close |
| Clipping | fraction of pixels at 0 or 255 | flash blowout on wet tissue |

**This gate is enforced.** A capture that fails is dropped from training, and at
serve time the API returns `usable: false` with a retake prompt rather than a
number. A confident-looking Hb value derived from a blurred photo is worse than
no answer at all — it is the failure mode most likely to cause harm in the
field.

Clipping deserves note: a flash fired at wet conjunctiva blows out exactly the
region we need, and the result still looks *sharp* to the focus check. Blur
detection alone would pass it.

### 3.5 Haemoglobin regression — `model.py`, `train.py`

A **ResNet-18** pretrained on ImageNet, with a scalar regression head.

- **Why regression, not classification?** The threshold varies by patient (see
  §3.6). Predicting Hb and thresholding afterwards means the same model serves
  a 3-year-old and an adult man. A binary classifier would need retraining per
  population.
- **Why ResNet-18?** ~11M parameters. On ~900 training images, anything larger
  memorises. Depth is not the bottleneck here; data is.
- **Transfer learning.** Layers up to `layer3` are frozen; `layer4` and the head
  fine-tune. Low-level edge and colour filters transfer fine from ImageNet;
  only the high-level features need to specialise.

Three details that matter more than they look:

**ImageNet normalisation is applied inside `forward()`.** Using pretrained
weights while feeding raw `[0,1]` inputs mismatches the distribution the weights
expect. Putting it inside the module means the serving path cannot forget it.

**Frozen BatchNorm is set to `eval()`.** Setting `requires_grad = False` freezes
the *weights* but BatchNorm keeps updating its running mean and variance in
training mode. On batches of 16 that drifts the very features the freezing was
meant to preserve. This is a genuinely subtle bug and it is easy to ship.

**Label standardisation constants are saved with the model.** The network emits
a standardised value; recovering g/dL needs the training mean and std. A
checkpoint without them is unusable — the weights alone cannot be served.

#### Handling imbalanced Hb

Most patients in any cohort are near-normal. A network minimising average error
learns it can do well by ignoring the image and predicting the mean — and
severe anemia, the cases that matter most, is exactly what it stops detecting.

Countermeasure: bin the continuous Hb values, weight each sample by inverse bin
frequency, and weight the loss accordingly. Rare severe cases then carry
proportionally larger gradients. `strength` interpolates between uniform and
full inverse-frequency; weights are normalised to mean 1 so the effective
learning rate does not shift.

**The honest check on this is the mean baseline.** The pipeline always reports
what a model that always predicts the training mean would score. If the trained
model does not clearly beat it, the model has learned nothing, regardless of how
respectable the MAE looks.

### 3.6 Diagnosis — `data.anemia_threshold`

Anemia is not one number. WHO thresholds:

| Population | Hb below (g/dL) |
|---|---|
| Children 6–59 months | 11.0 |
| Children 5–11 years | 11.5 |
| Children 12–14 years | 12.0 |
| Non-pregnant women 15+ | 12.0 |
| Pregnant women | 11.0 |
| Men 15+ | 13.0 |

A single hardcoded 11.0 labels an adult man at 12.0 g/dL as **normal** when he
is anemic. That is a false negative in a
screening tool: the error direction that sends an unwell patient home.

> **Measured on the local cohort, this is not a hypothetical.** Applying a
> fixed `Hb < 11.0` rule to the 26 real patients misclassifies **4 of them
> (15%)**, and every one of the four is a **false negative** — an anemic child
> labelled normal:
>
> | Patient | Hb | Age | WHO threshold | Correct | Fixed 11.0 rule |
> |---|---|---|---|---|---|
> | 10 | 11.0 | 11.1 | 11.5 | anemic | normal |
> | 14 | 11.4 | 6.6 | 11.5 | anemic | normal |
> | 16 | 11.3 | 8.4 | 11.5 | anemic | normal |
> | 21 | 11.4 | 7.2 | 11.5 | anemic | normal |
>
> The errors cluster just below 11.5 because most of this cohort falls in the
> 5–11 year band, where the correct cutoff is half a gram above the fixed
> value. A screening tool that misses 15% of anemic children before the model
> makes a single prediction is not fit for purpose, and no amount of
> regression accuracy would recover it.

When age or sex is unknown the code falls back to 12.0 rather than the lowest
cutoff, so an unknown patient is not assumed healthy.

---

## 4. Evaluation: how not to fool yourself

With roughly 900 images available, it is very easy to produce an impressive
number that means nothing. The pipeline is built to make that harder.

**Patient-grouped splits.** Eyes-defy-anemia has several captures per patient.
Splitting on *images* puts near-duplicate photos of the same eye in both train
and test, and the model scores well by recognising the patient. Grouping is on
`patient_id`.

**Stratified cross-validation, not one holdout.** With few severe cases, a
random split can hand you a fold with no anemic patients at all. Folds are
stratified by the patient's Hb, and results are reported as mean ± std across
folds. A single split's MAE has a confidence interval too wide to defend.

**Out-of-fold predictions only.** The results CSV contains only predictions made
on data the model did not train on. Mixing train-set predictions into the same
file would silently inflate headline results — with an 80/20 split, four of
every five rows would be training-set predictions.

**Sensitivity over accuracy.** For screening, a missed anemic patient costs far
more than a false alarm. Accuracy is also the metric most inflated by class
imbalance. Sensitivity, specificity, PPV, and NPV are all reported.

**Bland–Altman limits of agreement.** The standard way a new non-invasive device
is compared against a lab reference in clinical literature. Expect this to be
asked about in your viva.

---

## 5. Efficiency

The app is meant to be deployed, so two things are structural:

**Preprocessing is cached.** Segmentation is deterministic but expensive — a
Mask2Former forward pass per image. Caching keyed on image content plus
preprocessing settings means it runs once, not once per epoch. Augmentation
still happens per-epoch on the cached crop, so no augmentation diversity is
lost. (Without the cache, segmentation would re-run inside `__getitem__` — a
40-epoch run would segment every image 40 times.)

**The serving path is isolated.** `predict.py` imports nothing from `train.py`.
The API loads one model at startup and does one segmentation pass plus one
ResNet forward pass per request. Preprocessing settings are read *from the
checkpoint*, so a server running an older model still conditions images the way
that model was trained.

---

## 6. Data

| Source | Images | Patients | Hb labels | Masks | Population |
|---|---|---|---|---|---|
| Eyes-defy-anemia | 218 | ~218 | yes | yes | Adults, India + Italy |
| CP-AnemiC | 710 | 710 | yes | no | Children 6–59 months, Ghana |
| Local cohort | 52 | 26 | yes | no | Children 3–17 years, India |

The segmenter can only be trained on Eyes-defy-anemia — it is the only source
with pixel-level ground truth. It is then applied to the other sources to
produce crops for the regressor.

### The local cohort

26 subjects, both eyes photographed, so 52 close-up captures of the everted
lower lid. Labels come from `DATASAMPLE.csv`: haemoglobin, date of birth,
gender, height, weight, socio-economic status, and a full blood count (HCT, RBC,
MCV, MCH, MCHC, RDW, platelets, MPV, TLC).

| | |
|---|---|
| Hb range | 8.1 – 14.3 g/dL (mean 11.2, sd 1.37) |
| Anemic by WHO threshold | 13 of 26 patients (50%) |
| Severe (Hb < 9.0) | 3 patients |
| Age range | 3.4 – 16.7 years |
| Captures per patient | exactly 2 (one per eye) |

This is a small but genuinely usable cohort, and it has three properties that
make it more valuable than its size suggests.

**It is balanced.** A 50/50 anemic split is unusually favourable; most cohorts
are heavily skewed toward normal, which is what makes the mean baseline hard to
beat.

**Two captures per patient make grouping mandatory.** The left and right eye of
one child are highly correlated. Split on images and the model scores well by
recognising the subject rather than reading pallor. `load_local_cohort()`
assigns both eyes the same `patient_id`, so the grouped splitter keeps them
together. Under an image-level split this dataset would produce badly inflated
results.

**The ages straddle three WHO threshold bands** (11.0 under 5, 11.5 for 5–11,
12.0 for 12–14), which makes the per-patient threshold rule directly
consequential rather than theoretical — see §3.6.

The full blood count is not currently used. RDW and MCV distinguish iron-
deficiency anemia from other types, so there is a possible extension here:
predicting *which* anemia, not just whether. That is beyond the current scope
but worth recording.

Provenance and licensing are unconfirmed — worth establishing before any figure
derived from these images appears in a published write-up.

Two consequences worth stating in the report:

- **The datasets are not interchangeable.** Different ages, ethnicities,
  cameras, and lighting. A model trained on Ghanaian toddlers and evaluated on
  Italian adults is measuring domain transfer, not just pallor. Report
  per-source results as well as pooled.
- **~900 images is small.** This constrains model size, makes cross-validation
  mandatory, and means confidence intervals belong on every number.

---

## 7. Reading the code

```
src/anemia/
  config.py           all tunables; serialised next to every checkpoint
  imaging.py          colour ops, QC scores, letterbox crop (no torch import)
  segment.py          segmentation backends behind one callable contract
  preprocess.py       capture → model-ready crop, plus the QC gate and cache
  data.py             dataset loaders + WHO thresholds
  splits.py           patient-grouped, Hb-stratified cross-validation
  metrics.py          regression / screening / segmentation metrics
  model.py            regressor + self-describing checkpoint format
  train.py            regressor training
  train_segmenter.py  Mask2Former fine-tuning
  predict.py          serving path
serve/app.py          FastAPI service
scripts/              segmenter benchmark, synthetic data generator
tests/                runs without the real datasets
```

Suggested reading order: `data.py` (what a sample is) → `preprocess.py` (what
the model sees) → `model.py` (what it does) → `train.py` (how it learns) →
`predict.py` (what the app calls).

**See `CODE_GUIDE.md`** for what every file does in detail, and for the design
rationale behind each stage.

---

## 8. What is verified, and what is not

**Verified**
- All stages run end to end; 19/19 tests pass.
- The redness-based extractor reaches Dice 0.999 on phantoms where a
  brightness-based mask reaches 0.000 — confirming brightness selects sclera,
  not conjunctiva.
- Cross-validated training on phantoms: MAE 0.664 ± 0.096 g/dL, R² 0.880,
  1.45 g/dL better than the mean baseline.
- A saved checkpoint loads and predicts; QC correctly rejects blurred captures.
- **The pipeline runs on real, labelled photographs.** 52 captures from 26
  patients screened end to end without error; 49 passed QC (94%). Median focus
  85.3, median mask ratio 0.211. The three rejections were all blur, at 25.4,
  34.2 and 34.9 against a threshold of 35.
- **The per-patient WHO threshold is validated on real data.** A fixed 11.0
  rule misclassifies 4 of 26 real patients, all false negatives (§3.6).

**Measured and found wanting**
- **The colour heuristic does not segment real conjunctiva reliably.** It bleeds
  onto eyelashes and lower-lid skin on a substantial fraction of the 52 real
  captures (§3.2). This is a real negative result, and the most useful thing
  learned so far.
- **Two of the three QC rejections are marginal** (34.2 and 34.9 against a
  threshold of 35.0) on captures that are usable to the eye. The blur threshold
  was chosen on phantoms and looks slightly too aggressive for real images; it
  should be recalibrated once masks allow the downstream effect to be measured.

**Not verified — do not claim any of this yet**
- Any *accuracy* on real photographs. The local cohort is labelled and could in
  principle produce one, but at 26 patients it is far too small to give an
  error figure with a defensible confidence interval, and segmentation on these
  images is known to be unreliable. Train on it only after the segmenter works.
- The Mask2Former training path has not been run (needs the real masks).
- The CP-AnemiC loader has not been checked against a real download. Confirm it
  parses 710 images before trusting a run.

## 9. Next steps

1. Download both datasets; run `python -m anemia inspect` on each and confirm
   the image counts match the published figures.
2. Train the segmenter on Eyes-defy-anemia.
3. Run `scripts/benchmark_segmenters.py` — this produces the first real number
   in the project, and the before/after table that justifies the rework.
4. Train the regressor on the pooled data; report per-source and pooled.
5. Only then build the app UI around the API.

---
