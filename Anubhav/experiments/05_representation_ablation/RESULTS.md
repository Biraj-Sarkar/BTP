# Representation Ablation — Which Colour Space Carries the Hb Signal?

Tests the hybrid proposal from
`../03_colour_normalisation/CIELAB_vs_WhiteBalance.md`: gray-world corrects
illuminant *colour*, the CIELAB a\*/b\* input discards illuminant *brightness*,
so combining them should beat either alone.

Run with `python experiments/05_representation_ablation/run_ablation.py`.

## Method

26 patients, both eyes, laboratory Hb. Segmentation is computed **once per
image and shared by all four arms**, so only the representation differs.
Channel means inside the mask form a deliberately tiny feature vector (2–3
numbers, appropriate for n = 26), and ridge regression predicts Hb under
leave-one-patient-out cross-validation. Every arm is scored against
predict-the-mean.

## Result

| Representation | Features | LOO MAE (g/dL) | vs baseline |
|---|---|---|---|
| raw RGB | 3 | 1.152 | −0.069 |
| gray-world RGB | 3 | **1.073** | +0.010 |
| raw a\*/b\* | 2 | 1.192 | −0.109 |
| gray-world a\*/b\* (hybrid) | 2 | 1.104 | −0.021 |
| *predict-the-mean baseline* | — | *1.083* | — |

## Reading it honestly

**No representation meaningfully beats the baseline.** Gray-world RGB is ahead
by 0.010 g/dL, which on 26 patients is indistinguishable from noise. At this
feature scale, mean conjunctival colour does not predict haemoglobin in this
cohort.

That is a real result and it should temper expectations, but it is **not**
evidence the method fails. A channel mean discards everything a CNN uses:
vessel density and pattern, spatial distribution of pallor, texture. The
ablation tests the crudest possible summary, and the crudest summary is not
enough.

**The consistent finding is the ordering.** Gray-world beats its uncorrected
counterpart in both pairs — RGB 1.073 vs 1.152, a\*/b\* 1.104 vs 1.192 — a
~0.08 g/dL improvement in each case. Colour correction helps regardless of the
representation it feeds, which agrees with the cohort correlation result and
the lighting stress test.

**The hybrid did not win.** It beat raw a\*/b\* but not gray-world RGB. Dropping
L\* removes information (pale tissue really is lighter, not only less red), and
at two features that loss outweighs the brightness invariance gained. The
hybrid may still win with a CNN, where invariance matters more and spatial
structure compensates — but on this evidence it is not the automatic choice,
and the earlier recommendation is downgraded from "favourite" to "worth an arm
in the CNN ablation".

## What this changes

1. Keep gray-world RGB as the default; it is at worst equal and consistently
   the better of each pair.
2. Re-run all four arms as CNN training arms once a segmenter and a larger
   dataset are available. The conclusion here binds only to channel means.
3. Do not report any Hb accuracy from this cohort alone. 26 patients cannot
   distinguish a real 0.01–0.08 g/dL effect from noise.

## Limits

- n = 26, leave-one-out; confidence intervals comfortably exceed the
  differences between arms.
- Features are channel means only — by design, and the main limitation.
- One masking method throughout: a better segmenter could change every row.
- Ridge alpha fixed at 1.0, not tuned; tuning inside the loop on 26 patients
  would leak.
