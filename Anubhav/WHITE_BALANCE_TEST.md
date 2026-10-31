# Does White Balance Help or Hurt Haemoglobin Prediction?

A controlled test to run on the hospital data with the ResNet regressor. Two
extractors are provided that are identical except for gray-world white balance;
the regressor is trained once on each one's crops, with everything else held
fixed, and the two are compared on the same patients.

## Why this test

On the segmentation side the question is already answered: an Attention U-Net
trained and served on raw pixels matches the annotators' masks better than one
trained and served on white-balanced images (Dice 0.8476 against 0.8407, paired
difference +0.0069, 95% CI [+0.0012, +0.0118], p 0.0015 —
`experiments/20_architectures/RESULTS.md`). Earlier work also found white balance
costs accuracy when colour is read with a simple model.

Neither of those is the question that matters, which is whether white balance
changes **haemoglobin error from the regressor that will actually be used**, on
**the hospital's camera**. Only this test answers it.

## What you need

1. The repository, pulled. The two extractor files come with it, in
   `runs/segmenter/` (about 25 MB each):

   | alias | file | segmenter trained on | crop colour |
   |---|---|---|---|
   | `raw` | `attention_raw_full.pt` | raw pixels | raw |
   | `gw` | `attention_gw_full.pt` | gray-world balanced | gray-world balanced |

   Both are Attention U-Nets trained on all 648 annotated images, with the same
   seed, encoder and augmentation. The only difference is white balance.

   Check they arrived as real files, not a few hundred bytes each:
   `ls -lh runs/segmenter/attention_*_full.pt` should show ~24M for both.

## Step 1 — extract crops, once

```bash
python scripts/extract_crops.py --images /path/to/hospital/photos --out crops/
```

This writes:

```
crops/raw/<same relative path as the photo>.png
crops/gw/<same relative path as the photo>.png
crops/manifest.csv       one row per photo: accepted by each extractor? usable_in_all?
crops/extractors.json    which bundle (and its hash) produced which folder
```

Each crop is 224×224 RGB, background black outside the mask, saved as PNG so
colour is not altered. Crop colour follows the extractor: the `gw` folder is
white-balanced end to end, the `raw` folder is not. Use `--limit 20` for a quick
trial first.

## Step 2 — train the regressor twice, changing only the crop folder

The rules that make the comparison valid:

1. **Same photographs in both arms.** Keep only manifest rows where
   `usable_in_all` is `True`. Quality control can accept a photo under one
   extractor and reject it under the other; using each arm's own accepted set
   would compare them on different patients.
2. **Split by patient, never by image,** and use the **same folds and seeds** in
   both arms. Both eyes of one patient are nearly the same photograph.
3. **Same everything else** — architecture, pretrained weights, learning rate,
   epochs or early-stopping rule, batch size, loss, normalisation.
4. **Turn colour augmentation off in both arms** — `ColorJitter`, hue, saturation,
   brightness or gamma shifts, random grayscale, and any white balance or
   per-image colour normalisation of your own. These teach the network to ignore
   exactly the colour difference being tested, and would hide the effect.
   Geometric augmentation (flips, rotations, crops) is fine. ImageNet mean/std
   normalisation is fine — it is the same fixed transform in both arms.
5. **Load crops as RGB.** `cv2.imread` returns BGR; convert it, or use PIL.
6. **Do not choose epochs or hyperparameters on the test fold.** Use an inner
   validation split or a fixed epoch count, identically in both arms.
7. **Repeat**: five patient-grouped folds, and at least three seeds. A difference
   smaller than the seed-to-seed spread is not a difference.

## Step 3 — compare, paired per patient

From out-of-fold predictions, for each arm: MAE, RMSE, R², Bland–Altman bias and
95% limits of agreement, and anaemia sensitivity/specificity at the WHO cutoffs
for each patient's age and sex.

Then the comparison itself, **paired per patient**: the difference in absolute
error between arms, with a bootstrap 95% confidence interval and a Wilcoxon
signed-rank test.

- **White balance hurts** if the raw arm's error is lower and the interval
  excludes zero.
- **White balance helps** if the gw arm's error is lower and the interval
  excludes zero.
- **No detectable effect** if the interval includes zero — which is a valid and
  useful result, and should be reported as such rather than as a tie-break.

## What to send back

The metrics table for both arms, the paired difference with its interval and p,
the number of patients used and how many were dropped by `usable_in_all`, folds
and seeds, the augmentation actually used, and `crops/extractors.json`.

## Optional follow-up: is it the mask or the colour?

If white balance does change the result, two more runs separate why:

```bash
python scripts/extract_crops.py --images ... --out crops_rawcolour/ --crop-colour raw
python scripts/extract_crops.py --images ... --out crops_gwcolour/  --crop-colour gw
```

These fix the crop colour for both extractors, so the only difference between
the two folders inside each output is the mask.

---

## Prompt for your coding assistant

Two plain-text files in `handoff/` are written to be sent on their own — for
example over a messaging app — to someone using ChatGPT or Gemini, which cannot
see this repository:

- `handoff/WhiteBalanceTest_1_Steps.txt` — what to do, for the person running the
  test: pull, extract crops, which files to upload to the assistant and which
  never to upload, what to send back, and how to read the result.
- `handoff/WhiteBalanceTest_2_Prompt.txt` — the prompt itself, self-contained,
  for the assistant: background, the eight rules, the code changes wanted, the
  analysis, the WHO cutoffs, the decision rule, and a fixed results block to fill
  in so the answer comes back in a comparable form.

The paired difference there is defined as d = |error_gw| − |error_raw|, so a
positive value means white balance made the prediction worse. Keep this file and
the prompt in step if either changes; there is deliberately only one copy of the
prompt.
