# Lighting Stress Test — One Subject, Six Illuminants

A single volunteer's conjunctiva photographed under six coloured lights
(white, red, green, blue, magenta, purple). Because it is the same eye within
minutes, **the true haemoglobin is identical in all six frames** — so any
variation the pipeline reports is pure measurement error caused by lighting.
No blood test is needed for this to be informative.

All six frames are the **same right eye** of one volunteer, cropped to the
everted lower lid. Images and per-capture details: `images/`,
`capture_metadata.csv`.

**Colour management note.** The captures are iPhone HEIC in Display P3. An
initial export preserved the P3 profile, which OpenCV then read as if it were
sRGB — a systematic colour shift against the cohort images, which are true
sRGB. The images here were re-exported with an explicit P3 → sRGB conversion so
they match the dataset. This *strengthened* the findings below: correctly
converted, the colour casts are more extreme than they first appeared. JPEG
compression was separately verified as negligible (a\* differs by < 0.5 units
between quality-95 JPEG and lossless PNG).

---

## 1. Headline result

**Both pipelines segment correctly only under white light.** Under strong
coloured illumination the extractor lands on the eyebrow, the lashes, or
nothing coherent — see `lighting_stress_masks.png` (top row: gray-world
pipeline; bottom row: CIELAB pipeline).

This reframes the variance numbers below. They are **not** a clean measure of
colour robustness, because the mask is not on the same tissue between frames.
They largely measure how badly extraction collapses.

## 2. Numbers as measured

Redness (mean a\* inside the mask) and the Hb estimate, per illuminant:

| Light | Gray-world a\* | Gray-world Hb | CIELAB a\* | CIELAB Hb |
|---|---|---|---|---|
| white | 12.99 | 9.78 | 14.70 | 10.04 |
| red | 21.84 | QC reject | 65.02 | 13.44 |
| green | 16.18 | 12.85 | −14.23 | 11.01 |
| blue | 1.54 | 14.30 | 52.75 | 5.68 |
| magenta | 56.56 | QC reject | 72.60 | 14.15 |
| purple | 34.18 | 14.03 | 73.21 | 10.70 |

Spread across the six (lower is better — the truth is constant):

| | Gray-world | CIELAB |
|---|---|---|
| a\* standard deviation | **17.58** | 32.76 |
| a\* range | **55.02** | 87.44 |
| Hb standard deviation (g/dL) | **1.63** | 2.74 |
| Hb range (g/dL) | **4.52** | 8.46 |

Gray-world is roughly **1.9× tighter on redness** and **1.7× tighter on the Hb
estimate**, consistent in direction with the cohort result in
`../03_colour_normalisation/CIELAB_vs_WhiteBalance.md`. Both are far too wide
to be usable: a 4.5 g/dL swing from lighting alone spans the entire clinical
range from severe anaemia to normal.

**The Hb values are not clinical.** The only trained checkpoint was fitted on
synthetic phantoms, so absolute predictions are meaningless; only the spread is
being read, and only as a sensitivity indicator.

## 3. Why it fails — a physical limit, not a coding bug

Mean channel values of each capture:

Mean channel values in true sRGB:

| Light | R | G | B | Cast ratio (max/min) | Dead channel |
|---|---|---|---|---|---|
| white | 98.5 | 64.7 | 56.8 | 1.7 | — |
| green | 124.4 | 136.5 | 80.1 | 1.7 | — |
| magenta | 164.2 | 7.3 | 125.0 | 22.5 | green |
| red | 185.3 | 1.3 | 29.8 | 138.5 | green |
| purple | 145.9 | 1.1 | 161.0 | 147.6 | green |
| blue | **0.9** | 24.9 | 171.8 | **199.6** | red |

Four of the six lights leave a colour channel effectively **dead** — under blue
light the red channel averages 0.9 of 255, under red and purple the green
channel averages ~1. The sensor recorded nothing there because the lamp emitted
nothing there. Gray-world then divides by that near-zero mean, amplifying pure
sensor noise, which is the speckle visible in the figure.

This is the important lesson: under strongly non-neutral light the information
is *absent from the file*, not merely distorted. Neither normalization strategy
can fix a measurement that was never made.

## 4. What this implies for the pipeline

**Quality control should reject non-neutral illumination.** The cast ratio
above separates the cases cleanly and costs three channel means to compute:

- acceptable: white 1.7, green 1.7
- reject: magenta 22.5, red 138.5, purple 147.6, blue 199.6

A threshold near 3 divides them by an order of magnitude. This is a cheap,
well-motivated addition to the existing blur/exposure checks, and it converts a
silent failure (a confident number from an unusable photo) into an actionable
retake prompt.

**Capture protocol matters more than normalization.** The strongest fix is a
physical colour-reference card in frame, which turns illuminant correction from
an assumption into a measurement.

## 5. Advantages and limitations of this test design

| Advantages | Limitations |
|---|---|
| The subject is constant, so true Hb is identical across all six frames and every reported difference is measurement error | One subject, one eye — characterises failure modes, not accuracy |
| Needs **no laboratory value**, which is what makes it runnable on any volunteer | Cannot produce an error figure against ground truth |
| Illumination is the only varying factor, so attribution is clean | Illuminants are far harsher than any clinic; it maps where the method breaks, not whether it breaks in practice |
| Exposed a failure that neither synthetic images nor the labelled cohort revealed | Segmentation collapse and colour drift are confounded, so the spread ratios are directional, not precise |
| Cheap and repeatable — six photos and a script | Captures were hand-cropped, adding a small unquantified operator effect |
| Produced an actionable fix (the illuminant-neutrality check) | The proposed threshold is derived from six images and needs confirming on more |

## 6. Honest limits of this test

- **n = 1 subject, one eye.** It shows failure modes, not accuracy.
- **These lights are far harsher than any clinic.** Wards are fluorescent or
  daylight; nobody screens patients under a magenta lamp. The test establishes
  where the method breaks, not that it breaks in practice.
- **Segmentation failure confounds the variance comparison**, so the 1.9× and
  1.7× ratios should be read as directional, not precise.
- The white-light capture — the only realistic one — extracted correctly and
  produced a sane mask, which is the reassuring part.
