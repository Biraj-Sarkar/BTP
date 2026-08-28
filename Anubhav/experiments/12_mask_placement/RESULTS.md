# Where Each Pipeline's Mask Actually Lands

Run: `python experiments/12_mask_placement/measure_placement.py`
Machine-readable: `placement.json`

This experiment exists to replace a recollection with a number. The claim
"pipeline B's segmentation lands on non-conjunctiva tissue in **50 of 52**
captures" was being quoted as the decisive, non-statistical argument for
deploying pipeline A — in the progress report and four other documents — while
resting on visual review alone. Every other predictive claim in this project is
reported against a measurement, and this one was not.

**The measured figure is 46 of 52, not 50.** The claim was overstated and has
been corrected everywhere it appeared.

---

## Method

The local cohort has no ground-truth masks, so Dice cannot be computed. What
can be measured is whether the tissue *inside* each mask looks like conjunctiva
at all, using the redness index the extractor is built on:

```
redness = a* − 0.5·b*        (CIELAB)
```

high on the red mucosal band, low on sclera and peri-orbital skin. All three
masks are scored on the **same** white-balanced image, so only placement
differs.

The threshold comes from the earlier extraction work rather than from this
run: masks sitting on conjunctiva scored a median around +9, and a
brightness-based mask known to select sclera scored around −4. A cut at **0.0**
sits between those two populations.

## Result

| Mask | median redness | min | max | off-tissue |
|---|---|---|---|---|
| **Pipeline A** (refined seeded grabCut) | **+9.53** | +5.36 | +19.80 | **0 of 52** |
| **Pipeline B** (CIELAB extractor) | −3.67 | −9.48 | +4.56 | **46 of 52** |
| Brightness baseline (known to select sclera) | −3.66 | −8.06 | +2.27 | 47 of 52 |

Fraction of B's mask that also falls inside A's: **median 0.000**, mean 0.026.

## Reading it

**Pipeline A lands on tissue in every capture.** Median +9.53 matches the +9.2
measured during the extraction work, and no capture falls below the threshold.
That is the strongest evidence available that the refined extractor works on
real photographs, though it is still not a Dice score — it says the mask is on
something red, not that it is on the whole conjunctiva and nothing else.

**Pipeline B's placement is statistically indistinguishable from the
brightness baseline.** Median −3.67 against −3.66; 46 off-tissue against 47.
That baseline is the one measured at **Dice 0.000** against ground truth on
phantoms — it selects sclera. B's extractor is not merely imprecise; it is
finding the same wrong structure, which is consistent with it being a
brightness-and-chroma rule rather than a redness rule.

**The two masks are essentially disjoint.** The median overlap is exactly zero:
on more than half the captures, B's mask and A's mask share no pixels at all.
The two pipelines are not two measurements of the same tissue with different
colour handling — they are measuring different parts of the photograph. That
reframes the head-to-head comparison in `../11_head_to_head/`: its paired test
asks whether two models disagree, and one of them was never looking at
conjunctiva.

**The six captures where B lands on tissue are not an existence proof.** Two
reasons, and the second is the more serious.

*They are marginal, not clean.* Their redness values are +0.06, +0.23, +0.66,
+1.09, +2.18 and +4.56 — median **+0.87**, against **+8.49** for pipeline A on
those same six images. They scrape past a cut of 0.0 and disappear as it
tightens:

| on-tissue cut | A | B |
|---|---|---|
| −2.0 | 52 | 15 |
| **0.0** | **52** | **6** |
| +1.0 | 52 | 3 |
| +2.0 | 52 | 2 |
| +5.0 | 52 | 0 |

A holds 52 of 52 at every cut through +5.0. The "6" is a property of a lenient
threshold, not a set of successes.

*No patient gets a clean measurement out of them.* Features are averaged across
both eyes before fitting, and **not one patient has both eyes on-tissue**. The
six on-tissue captures belong to six different patients, each of whose other
eye is off-tissue:

| B's 26 per-patient feature vectors | count |
|---|---|
| average of two off-tissue measurements | 20 |
| average of one on-tissue and one off-tissue measurement | **6** |
| clean conjunctiva measurement | **0** |

This is worse than being uniformly wrong. A model fitted on a *consistently*
misplaced mask still measures the same thing on every patient, and could in
principle find a real correlation if that wrong tissue happened to track
haemoglobin. A model fitted on a per-patient blend of two different tissue
types has a different measurement basis for each patient, and no amount of
regularisation repairs that.

It also means the six cannot be read as evidence about B's extractor at all,
because every one of them is inside the fit — the models in
`../11_head_to_head/` are fitted on all 26 patients. Scoring well on training
data is not a claim about anything.

## What this changes

1. **The corrected figure is 46 of 52 (88%).** Propagated to the report,
   `../11_head_to_head/RESULTS.md`, `EVERYTHING_EXPLAINED.md`,
   `TALKING_POINTS.md` and `CLAUDE.md`.
2. **The argument for A survives and is stronger than it was**, because it now
   rests on a regenerable measurement plus three facts that were not previously
   stated: B matches the sclera-selecting baseline, the masks are disjoint, and
   **no patient in B's fit has a clean conjunctiva measurement** — 20 are
   averages of two off-tissue captures and 6 are on/off blends.
3. **The head-to-head result should be read accordingly.** B's deficit is
   dominated by extraction, not by the a\*/b\* representation. A trained
   segmenter feeding B's colour treatment remains untested and is the fair
   version of that comparison.

## Limits

- **This is not Dice.** Without ground-truth masks it cannot say how much of
  the conjunctiva a mask covers, only whether what it covers is red. A mask
  hugging a small corner of the tissue scores as well as one covering all of
  it. Eyes-defy-anemia is still the only route to a real segmentation score.
- The 0.0 threshold is carried over from the extraction study, not re-derived
  here. It does not change the *conclusion* — A is 52 of 52 at every cut from
  −2.0 to +5.0 — but it does change B's count, which falls from 15 to 0 across
  that range. Quote B's figure with the cut attached, never bare.
- Redness is measured on the gray-world image for all three masks. Pipeline B
  normally operates on its own CLAHE-normalised image; scoring it on the shared
  image is what makes the comparison like-for-like, and CLAHE on L\* leaves
  a\*/b\* unchanged, so the redness index is nearly unaffected either way.
- One cohort, one device, one operator.
