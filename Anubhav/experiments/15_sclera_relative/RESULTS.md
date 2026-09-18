# Measuring Against the Sclera — Portability Bought With Accuracy

Run: `python experiments/15_sclera_relative/sclera_relative.py`
Machine-readable: `sclera_relative.json`

Experiment 14 found the problem that matters most for deployment: a model
fitted on one cohort scores **R² −1.944** on another. Chromaticity is invariant
to *how much light there was* but not to *which camera measured it*, and the
two cohorts' features sit more than three standard deviations apart, so a fitted
line is extrapolating far outside the range that set it.

This tests the fix: stop asking "how red is the conjunctiva" and ask **"how red
is it compared to the sclera in the same photograph"**. Both tissues pass
through the same lens, sensor, white balance and illumination, so dividing or
subtracting one from the other should cancel most of what the camera did.

The sclera detector is `legacy_bright_neutral_mask` — the function this project
originally evaluated as a *conjunctiva* extractor and rejected because it
selects bright, low-chroma pixels, i.e. sclera. What made it a bad extractor
makes it the right reference detector. The conjunctiva mask is subtracted from
it so the reference can never contain the tissue being measured against it.

---

## Result

26 local patients, 217 external subjects.

| feature set | k | separation | local R² | external R² | **ext → loc** | **loc → ext** |
|---|---|---|---|---|---|---|
| absolute chroma (current) | 2 | **2.77** | 0.090 | **0.346** | **−2.010** | −1.112 |
| absolute redness | 1 | 1.22 | 0.068 | −0.008 | −0.902 | −0.152 |
| ratio RGB | 2 | 2.27 | −0.172 | 0.202 | −0.320 | −0.163 |
| log-ratio RGB | 2 | 2.16 | −0.150 | 0.211 | **−0.004** | −0.017 |
| chroma difference | 2 | 2.20 | −0.019 | 0.362 | −1.134 | −0.645 |
| redness difference | 1 | **0.68** | 0.014 | 0.058 | −0.615 | −0.180 |
| **a\* difference** | 1 | 1.60 | −0.061 | 0.221 | **+0.032** | **+0.081** |

`separation` = distance between the two cohorts' feature means, in pooled
standard deviations. Below ~1 the clouds overlap; above ~2 a model fitted on one
is extrapolating on the other.

## 1. Sclera-referencing fixes the transfer catastrophe

**External → local goes from R² −2.010 to +0.032.** `a* difference` — the mean
a\* of the conjunctiva minus the mean a\* of the sclera, a single number — is
**the only feature set in this project that transfers positively in both
directions** (+0.032 and +0.081). `log-ratio RGB` reaches −0.004, which is to
say exactly baseline: no better than guessing the mean, but no longer actively
harmful.

The mechanism shows up in the separation column. Absolute chromaticity puts the
cohorts 2.77 pooled standard deviations apart; `redness difference` brings that
to **0.68**, where the two clouds genuinely overlap. Referencing against tissue
in the same frame does cancel a large part of the inter-camera difference,
exactly as predicted.

## 2. And it costs most of the accuracy

Within a single cohort the relative features are clearly worse:

| | external R² | ext → loc R² |
|---|---|---|
| absolute chroma | **0.346** | −2.010 |
| a\* difference | 0.221 | **+0.032** |

Absolute chromaticity extracts substantially more signal *when the camera is
held constant*. Sclera-referencing throws some of that away — unsurprisingly,
since dividing by a noisy reference adds the reference's noise to every
measurement, and the sclera mask is itself an imperfect detector.

**So this is a trade, not a free improvement.** The two properties are in
tension, and which one matters depends entirely on the deployment question:

* Fitting and predicting **on one camera and population** — absolute
  chromaticity, and it is not close.
* Fitting on one setup and predicting on **another** — only the relative
  features function at all, and the absolute ones are worse than useless.

For a screening tool intended for a hospital that did not supply the training
data, the second is the real scenario. A model with R² 0.346 in the lab and
−2.010 in the clinic is worth less than one with 0.221 and +0.032.

## 3. Overlap is necessary but not sufficient

`redness difference` has the best cohort overlap of all (separation 0.68) and
still transfers at only −0.615, worse than `a* difference` at separation 1.60.
Making the distributions coincide is not enough — the feature also has to carry
haemoglobin signal once it has been made portable. `redness difference` achieves
portability largely by discarding information (external R² 0.058, essentially
nothing).

That rules out the lazy reading of this experiment, which would be "minimise
separation". The target is the joint condition: overlapping *and* predictive.

## Honest limitations

- **+0.032 is barely above zero.** This converts catastrophic failure into
  roughly baseline performance. It is a large improvement on −2.010 and it is
  the first transfer this project has achieved, but it is **not yet a working
  cross-camera model**. Do not present it as one.
- The `ext → loc` MAE for `a* difference` is 1.015 against a local baseline of
  1.083 — a 6% improvement on guessing the mean, on 26 patients.
- Only single-feature relative sets were tested beyond the pairs. A penalised
  model over many *relative* features is untested, and experiment 13 showed
  more features help at n = 217 — that combination is the obvious next step.
- The sclera detector was built as a conjunctiva extractor and repurposed. It
  has never been validated as a *sclera* detector against ground truth, and the
  quality of the reference bounds everything here.
- One external cohort. A single successful transfer is weaker evidence than a
  single failed one.

## Where this leaves the project

The deployed model uses absolute chromaticity fitted on 26 local patients. This
experiment says that model is **camera-specific**, and gives the first concrete
route to something that is not. Neither option is currently good enough to
deploy unsupervised; the honest framing is a site-calibrated instrument, with
sclera-referencing as the path toward a portable one.
