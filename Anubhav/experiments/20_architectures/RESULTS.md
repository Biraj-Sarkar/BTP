# Choosing the Extraction Architecture

Run: `python scripts/select_architecture.py --archs unet unetplusplus attention`
then `python scripts/select_architecture.py --ship`
Outputs: `leaderboard.json`, `bundles/<arch>.pt`, `results.json`.

**Result: Attention U-Net is selected.** It is the only architecture whose
advantage over the current U-Net is statistically resolved (+0.0105 Dice,
95% CI [+0.003, +0.020], p = 0.007). It is not distinguishable from UNet++.

## The question

The learned extractor is a U-Net. U-Net was chosen because it is the standard
encoder-decoder segmentation architecture, not because anything was measured
against it. Two candidates target the specific weakness this task has.

**UNet++** replaces each plain skip connection with a grid of intermediate
convolution blocks, each fed by the block to its left and an upsampled one from
below. The stated motivation is that a deep encoder feature map and a freshly
upsampled decoder feature map are semantically far apart, and concatenating them
directly asks the decoder to bridge that gap unaided. Relevant here because the
conjunctiva boundary is fine detail that a plain skip must carry across a large
semantic gap. Costs decoder parameters: 6.57M against 6.25M.

**Attention U-Net** (Oktay et al., 2018) keeps the plain skip and gates it. The
coarser decoder feature map acts as context: both are projected to a common
width, summed, passed through ReLU and a 1-channel convolution, and squashed to
[0, 1], giving one weight per spatial position which multiplies the skip before
concatenation. A position the decoder has no use for is scaled toward zero.
Relevant here because the skips carry high-resolution eyelash and lid-margin
edges sitting immediately beside the tissue being segmented, and gating is the
mechanism for ignoring them. Adds almost nothing: 6.29M against 6.25M.

`segmentation_models_pytorch` provides UNet++ but has no Attention U-Net, so the
gates and decoder are implemented in `src/anemia/architectures.py` against the
same encoder, with decoder widths matched to the library's U-Net defaults. Its
`scse` decoder attention is a different mechanism — it recalibrates features
*after* concatenation rather than gating the skip before it — and is available as
a fourth option, `unet-scse`, but is not part of this comparison.

All three share an ImageNet-pretrained EfficientNet-B0 encoder, so a difference
between them is the decoder design rather than the backbone or the model size.

## How these models are evaluated

Earlier extractor comparisons scored each mask by fitting a ridge regression on
the colour inside it and reporting R² against laboratory haemoglobin. That
measures the extractor and a stand-in model together, and the stand-in is far
weaker than the network that will actually read these crops. Segmentation models
have their own established evaluation, which measures only the mask. **This
comparison uses segmentation metrics alone.**

The model emits one probability per pixel; threshold at 0.5 and compare the
predicted mask against the annotator's.

| Metric | Definition | What it tells us |
|---|---|---|
| **Dice** | 2\|A∩B\| / (\|A\|+\|B\|) | Region overlap. The medical-imaging convention, so the headline |
| **IoU** | \|A∩B\| / \|A∪B\| | Same idea, stricter. Ranks identically to Dice, so it is a consistency check |
| **Precision** | \|A∩B\| / \|A\| | Of the pixels called conjunctiva, how many were. Low = the mask spilled onto lashes, lid or sclera |
| **Recall** | \|A∩B\| / \|B\| | Of the true conjunctiva pixels, how many were found. Low = the mask is too small |
| **HD95** | 95th percentile of boundary-to-boundary distance, px | Overlap is dominated by a region's interior; this catches an outline that wanders |
| **Distribution** | 10th percentile, worst case, empty-mask count | A model that ties on average but fails on a few captures is worse in deployment |

Two notes on why this set rather than a single number:

- **Plain accuracy is useless here.** Conjunctiva is a few percent of the frame,
  so predicting background everywhere scores about 99%.
- **Precision and recall are not interchangeable for this task.** Dice hides the
  direction of the error. A mask that spills onto eyelid skin contaminates the
  colour read inside it; a mask slightly too small does not. Spill was the
  measured failure of the rule-based extractor, and Dice alone understates it.

## What makes the comparison trustworthy

1. **Split by subject, never by image.** Both eyes of one person are nearly the
   same photograph, so an image-level split puts one eye in training and the
   other in test and measures memorisation. Already enforced by the trainer.
2. **A third split, untouched.** The epoch is chosen on validation Dice, so
   validation Dice is the maximum of a noisy sequence and is optimistic by
   construction. `--test-fraction` was added for this: 20% of subjects are held
   back from both training and selection, and the selected checkpoint is scored
   on them once.
3. **Paired comparison, per image.** All three are scored on identical images, so
   differences are tested per image with Wilcoxon signed-rank and a paired
   bootstrap interval, rather than by comparing two means from unrelated samples.
   The count of images each architecture wins is reported beside the mean.

Split: 648 annotated images from 648 subjects — 389 train, 130 validation, 129
test. Identical seed, augmentation (`rotation`), encoder and 40 epochs
throughout, so the decoder is the only variable.

## Two stages: select, then ship

**Stage 1** trains every candidate on the training subjects, selects each one's
epoch on the validation subjects, and scores it once on the test subjects.

**Stage 2** retrains the winner on *every* subject, including the held-out ones,
for the epoch count selection found — with no validation set there is nothing to
early-stop on, so that count is fixed rather than guessed. Output goes to
`runs/segmenter/<arch>_full.pt`; the deployed bundle is **not** overwritten
automatically.

> **The caveat that must travel with the shipped model.** Once it is retrained on
> everything, nothing is held back, so those weights have no score of their own.
> The number to quote is the stage-1 test score *for that architecture* — an
> estimate of what it reached when trained on 60% of the subjects. Training on
> all of them is expected to help, but that is an expectation, not a measurement,
> and it cannot be measured without holding data back again. `leaderboard.json`
> stores this note beside the shipped entry.

## Adding an architecture later

The leaderboard exists so a fourth candidate does not require retraining the
first three:

```bash
python scripts/select_architecture.py --archs unet-scse
```

This is sound **only against the same split**, so the leaderboard stores a
fingerprint of the data roots, seed, split fractions, epoch count, augmentation
and the resulting test subjects. A later run whose fingerprint differs — a
dataset grew, a seed changed — is refused with an explanation, because a
leaderboard mixing two splits would rank architectures by which subjects they
happened to be tested on.

A bundle already on disk is adopted rather than retrained, but only after
checking the subjects it actually held out, not its filename.

## Results

129 held-out images from 129 subjects, at 320px, threshold 0.5. No architecture
produced a single empty mask.

### Overlap

| architecture | params | Dice | IoU | precision | recall | HD95 px | empty |
|---|---|---|---|---|---|---|---|
| U-Net | 6.30M | 0.8372 | 0.7288 | 0.8239 | 0.8637 | 2.8 | 0 |
| UNet++ | 6.61M | 0.8446 | 0.7379 | 0.8266 | 0.8746 | **2.2** | 0 |
| **Attention U-Net** | 6.33M | **0.8476** | **0.7427** | **0.8320** | 0.8737 | 2.5 | 0 |

Dice and IoU rank the three identically, which is the consistency check passing.
Both new architectures beat the current U-Net on every overlap measure.

**The gain is mostly in precision.** Attention U-Net improves precision by 0.008
over U-Net and recall by 0.010; UNet++ gains almost entirely in recall (+0.011)
with precision nearly unchanged (+0.003). Precision is the direction that matters
here — a mask that spills onto lid skin contaminates the colour read inside it —
so the two architectures reach a similar Dice by different routes, and Attention
U-Net's route is the one this task prefers.

HD95 does not agree with Dice: UNet++ places its boundary most accurately (2.2px
against 2.5 and 2.8). Boundary accuracy and region overlap are genuinely
different questions, and a 0.3px median difference at 320px is not worth acting
on, but it is the one measure on which the chosen architecture does not lead.

### Distribution

| architecture | mean | sd | median | 10th pct | worst |
|---|---|---|---|---|---|
| U-Net | 0.8372 | 0.0894 | 0.8598 | 0.7304 | 0.3634 |
| UNet++ | 0.8446 | 0.0755 | 0.8660 | 0.7374 | 0.5410 |
| **Attention U-Net** | 0.8476 | 0.0771 | 0.8617 | 0.7426 | 0.5436 |

**The worst case improves more than the mean does.** U-Net's worst image scores
0.363; both new architectures lift that to about 0.54, and the spread narrows
(sd 0.089 → 0.077). For a screening tool the floor matters more than the average,
because the floor is what produces an unusable crop. Note that U-Net has the
*highest* median of the three on one reading — UNet++ leads at 0.8660 — so the
mean advantage is driven by the bad cases rather than the typical one.

### Paired comparison, per image

| pair | mean diff | 95% CI | p | wins | distinguishable |
|---|---|---|---|---|---|
| U-Net − UNet++ | −0.0075 | [−0.017, +0.000] | 0.0881 | 55/74 | no |
| U-Net − Attention U-Net | −0.0105 | [−0.020, −0.003] | 0.0074 | 54/75 | **yes** |
| UNet++ − Attention U-Net | −0.0030 | [−0.008, +0.002] | 0.1546 | 61/68 | no |

**Only one comparison resolves.** Attention U-Net beats U-Net by 0.0105 Dice,
with an interval excluding zero (p = 0.007). UNet++ beats U-Net by a similar
margin but its interval touches zero, and the two new architectures cannot be
told apart from each other at this sample size.

So the defensible statement is narrow: **Attention U-Net is better than the
current U-Net; it is not demonstrably better than UNet++.** It is selected
because it leads on Dice, IoU and precision, costs 0.03M parameters against
U-Net's 6.30M where UNet++ costs 0.31M, and is the only one whose advantage over
the incumbent is statistically resolved.

The win counts are worth reading beside the means: Attention U-Net wins 75 of 129
images and loses 54. This is a consistent small edge, not a large gain on a few
images.

## Shipped model

Stage 2 retrained Attention U-Net on all 648 images for 36 epochs — the epoch
selection chose — and wrote `runs/segmenter/attention_full.pt`.

**Estimated performance: Dice 0.8476 on held-out subjects**, carried over from
stage 1. Those weights saw 60% of the subjects; the shipped weights saw all of
them and have no score of their own. See the caveat above.

The deployed bundle at `runs/segmenter/unet.pt` is **not** replaced. Swapping it
changes every cached crop and every feature measured downstream, so it should be
a deliberate step taken with the extractor's robustness question (experiment 13)
settled, not a side effect of this experiment.

## Limitations known before the result arrives

- **These metrics measure agreement with the annotator, who is a proxy.** The
  annotations are imperfect: 117 of 218 Eyes-Defy-Anemia masks were bounding
  rectangles rather than traced outlines and were discarded (`PIPELINE.md`). A
  model can match annotations more closely without producing a mask that is
  better for reading colour.
- **One seed per architecture.** A difference smaller than run-to-run variance
  would not be distinguishable. The existing extractor already has an unresolved
  stability question for the same reason.
- **Dice on a held-out split says nothing about the hospital camera**, which no
  measurement in this repository covers.
