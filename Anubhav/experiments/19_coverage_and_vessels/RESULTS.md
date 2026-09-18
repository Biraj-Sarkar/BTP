# Wider Masks, and Whether the Visible Vessels Carry Signal

Run: `python experiments/19_coverage_and_vessels/sweep.py`
Outputs: `results.json`, `features.csv`, `sharpness_control.csv`.

Two questions on the 217 external patients, measured at 1024px (vessels are a few
pixels wide and vanish at 512), with **no colour correction** (experiment 18) and
sex among the inputs (experiment 17). Ridge, 10-fold, 5 splits.

## 1. Coverage: tighter is better, monotonically

| mask | coverage of frame | R² | MAE |
|---|---|---|---|
| erode 8px | 3.65% | **0.544** | **1.241** |
| p ≥ 0.9 | 4.50% | 0.542 | 1.247 |
| p ≥ 0.7 | 4.92% | 0.540 | 1.248 |
| **p ≥ 0.5 (deployed)** | 5.16% | 0.539 | 1.249 |
| p ≥ 0.3 | 5.42% | 0.538 | 1.251 |
| dilate 8px | 6.79% | 0.533 | 1.259 |
| dilate 16px | 8.53% | 0.528 | 1.270 |
| dilate 32px | 11.82% | 0.520 | 1.285 |

**Lowering the threshold does not help.** Every step of widening dilutes the
measurement with neighbouring tissue, and every step of tightening helps slightly.
The ordering is perfectly monotone across eight variants, which is what makes a
set of small differences credible.

This reproduces experiment 13's mask-geometry finding, which mattered because that
one was measured on the 26 local patients — later shown to lack the colour signal
(experiment 16) and therefore unable to settle it. **The deployed threshold is not
changed:** the gain from eroding is 0.005 R², chosen from eight variants on the
same patients, and a tighter mask returns an empty mask more often on hard
captures.

## 2. Vessels: a large apparent gain that is entirely a site effect

The conjunctiva's vasculature is visible and physiological, so texture features
were tested beside colour: variability inside the mask (`green_sd`, `redness_sd`),
edge strength, and a vessel-like share of pixels darker than their local background.

Pooled over both sites, this looked convincing:

| features | R² | MAE |
|---|---|---|
| colour + sex | 0.539 | 1.249 |
| **+ edge strength** | **0.563** | **1.234** |
| + vessel fraction & depth | 0.551 | 1.254 |
| + all structure | 0.552 | 1.239 |
| structure only, no colour | 0.434 | 1.403 |

Edge strength alone correlates **+0.403** with haemoglobin, and structure without
any colour reaches R² 0.434.

### It does not survive the controls

**Control 1 — is it just sharpness?** Edge strength on the tissue *surrounding*
the conjunctiva also correlates with haemoglobin (**+0.328**). Whole-frame blur
does not (+0.094, p = 0.17). Controlling for the surrounding tissue leaves
+0.289.

**Control 2 — is it the site?** The two sites differ in both haemoglobin and image
detail:

| | n | mean Hb | edges inside mask |
|---|---|---|---|
| India | 95 | 11.47 | 14.26 |
| Italy | 122 | 13.83 | 20.06 |

So "more edges" partly means "Italian patient", and Italian patients have higher
haemoglobin. Within each site the correlation collapses: **+0.180 (p = 0.081)** in
India, **+0.143 (p = 0.117)** in Italy. Controlling for site *and* surrounding
tissue: **r = −0.021 (p = 0.76)**.

**Control 3 — does it predict within a site?**

| | colour + sex | + edge strength | + all structure |
|---|---|---|---|
| both sites pooled | 0.539 | **0.563** | 0.552 |
| India only | 0.434 | 0.439 | 0.404 |
| Italy only | 0.425 | 0.432 | 0.393 |

Within a site the gain is +0.005 to +0.007 — nothing — and the full structure set
is clearly *worse*, as extra parameters usually are at this sample size. **The
pooled gain was the model using texture as a label for which hospital took the
photograph.**

**Conclusion: texture features are not adopted.** The idea was sound and the
measurement does not support it on this data. A fairer test needs a site where
resolution and capture distance are held constant.

## 3. An open observation worth keeping

Edge strength on the tissue *around* the conjunctiva — skin, lashes, sclera —
correlates with haemoglobin **within** each site: **+0.319 (p = 0.002)** in India
and **+0.309 (p = 0.0005)** in Italy. That is not conjunctival tissue, so it
cannot be pallor. Candidates: patients photographed at slightly different
distances or focus, or age-related skin texture (age correlates −0.213 with it).
Not investigated further.

It carries a warning for the network work: **a model trained end-to-end on whole
photographs can pick up photo-level cues like these and score well without ever
looking at the conjunctiva.** Held-out splits by *site*, and ideally an evaluation
on masked-out images, are the way to catch it.

## Limitations

- Eight mask variants and six feature sets were compared on the same patients;
  individual differences of 0.005 R² should not be read as real.
- "Vessel" features are simple proxies (local darkness, edge energy), not a
  vesselness filter or a traced vascular map. A stronger method might find
  something these miss, but it would face the same site confound.
- Experiment 16 found that pulling the eyelid harder engorges the vessels, so even
  a clean within-site vessel signal would need a capture-technique control.
