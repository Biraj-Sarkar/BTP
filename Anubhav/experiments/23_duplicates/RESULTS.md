# Experiment 23: repeat photographs in the Mendeley set

**Result.** The Mendeley set is not 547 independent images. At least 463 of them fall
into 115 groups of 2–10 shots of the same subject, taken in quick succession (most
often four). So the 547 images come from **at most 199 distinct subjects**. The real
number is lower still, because the matcher misses some repeats, and because
neighbouring groups are often the same person's other eye (see below). The number
of people may be nearer half of 199.

Because the trainer had no subject ids for this set, it treated each image as its
own subject. In every split of experiment 21, **83–93 of the 129 test images (64–72%)
had a same-subject photograph in the training set**.

The leak is real. The evidence that it inflated our scores is weak. Within
Mendeley, test images with a partner in training scored only 0.003–0.010 Dice
higher for five of the six architectures (LinkNet scored lower), and the
per-split differences point both ways. Experiment 24 retrained all six
architectures on splits that keep each group, and its neighbours, together.
Mendeley scores fell by 0.006–0.021 while the Eyes-defy control did not, giving
a leak estimate of +0.013 to +0.025 Dice. Every interval spans zero, and the
architecture order held.

- **No exact copies.**
- **No photograph shared between Mendeley and Eyes-defy.**

## The question

The Mendeley files are numbered 1–547 and carry no subject identifier and no
metadata (no EXIF, no timestamps). The trainer's `subject_of` therefore makes every
image its own subject: "648 subjects" for 648 images. If one eye was photographed
several times, those photographs can sit on both sides of a split. A model that
trained on one of them partly recognises the others, and its test score then
partly measures recognition rather than generalisation.

## Method

1. **Exact copies.** A hash of the decoded pixels.
2. **Near copies.** A 64-bit perceptual hash (DCT of a 32×32 thumbnail),
   compared under all eight flips and quarter-turns. This catches resized,
   recompressed, mirrored or rotated copies.
3. **Same subject, different shot.** SIFT keypoints at 1024 px, matched between
   **every** pair: 149,331 Mendeley pairs plus 55,247 Eyes-defy × Mendeley pairs.
   Each pair then gets a RANSAC homography fit. Two shots of one eye share
   vessels, lashes, skin and lid texture in a consistent arrangement, so many
   matches survive the fit. Different people leave only chance matches. The
   score is the number of surviving matches (inliers).

The cut-off was not fixed in advance. It was read from two independent sources:
the file numbers, and looking at the pairs.

**File numbers.** If the set was saved session by session, true repeats should
have neighbouring numbers, while chance matches between strangers should not.
Two random images have numbers within 3 of each other about 1% of the time.

| RANSAC inliers | Pairs | Numbers ≤ 3 apart | Numbers ≤ 10 apart |
|---|---|---|---|
| 1–10 | 32,129 | 1% | 4% |
| 15–20 | 363 | 12% | 17% |
| 20–25 | 125 | 26% | 32% |
| 25–30 | 40 | 47% | 55% |
| 30–40 | 48 | 75% | 79% |
| 40–50 | 38 | 87% | 89% |
| 50–100 | 108 | 81% | **100%** |
| 100–200 | 114 | 82% | 98% |
| ≥ 200 | 347 | 94% | 98% |

**By eye** (sheets in `review_sheets/`):

- **Top pairs** are burst shots of one eye, with the same finger and the same
  framing.
- **50–100 inliers, numbers 4–9 apart:** every pair inspected is the same person.
  Subjects were photographed in runs.
- **30–50 inliers, numbers far apart:** mostly different people. Image 231 matches
  several strangers, probably on hair texture.
- **20–30 inliers, neighbouring numbers:** essentially all the same person.
- **20–30 inliers, numbers far apart:** 12 of 12 inspected are different people.
- **Eyes-defy × Mendeley:** the strongest pairs (28–41 inliers) are plainly
  different photographs, macro close-ups against face shots. So chance alone can
  reach about 40–49 inliers on texture-heavy images.

**Rule adopted.** A pair counts as the same subject if it has at least 50 inliers,
or at least 20 inliers with file numbers no more than 10 apart. Linked pairs are
joined into groups: if A matches B and B matches C, all three are one subject.

The rule also catches some later shots filed far from their session. For example,
395 belongs with 103–105 (485 inliers), 400 with 308–310 (790), and 251 and 547
with 101–102 (121 and 183).

## Results

| | |
|---|---|
| Mendeley images in multi-shot groups | **463 of 547** |
| Groups | 115 |
| Group sizes | 2 (14), 3 (14), **4 (64)**, 5 (15), 6 (2), 7 (1), 8 (3), 9 (1), 10 (1) |
| Distinct subjects, at most | **199** (84 ungrouped + 115 groups) |
| Exact copies | 0 |
| Near copies (perceptual hash ≤ 4 bits) | 174 pairs, all inside groups (burst shots); 173 have ≥ 50 inliers, and the last (292/294, 23 inliers, 2 numbers apart) is linked by the neighbouring-numbers rule |
| Groups spanning Mendeley and Eyes-defy | 0 |

**The matcher misses some repeats, so 199 is an upper bound.**

- **Files 1–8** look like four pairs of shots of one person each (1–2, 3–4, 5–6,
  7–8), but score only 0–7 inliers.
- **Files 62 and 78** clearly belong to the runs around them, but score 0–14
  against their neighbours.

SIFT fails when the shots differ in angle, zoom or blur, or show the other eye.
Most of files 1–48 are ungrouped for this reason, and they were also taken on
several different phones.

**Appearance embeddings cannot recover the misses.** ImageNet ResNet-18
embeddings rate the missed same-person pairs (cosine similarity 0.88–0.91) no
higher than 1% of far-apart strangers (0.917). So for subjects whose shots don't
match geometrically, nothing automatic can be trusted. Neither do the sessions
follow a fixed block of file numbers: only 60% of links fall inside an aligned
block of four.

### Neighbouring groups are often one person

A group is a run of shots that match each other geometrically, which in practice
means one eye. Looking at where one group ends and the next begins (24 of the 97
places where two groups sit within two file numbers; sheets
`9_neighbouring_groups_a/b.jpg`):

- **At least 14 of 24 are plainly the same person.** It is either the other
  eye or the same eye re-framed, with the same nose stud, shirt, haircut,
  background or hand over the face. The framing is often mirrored, as when
  the photographer moves to the other eye.
- **4 are plainly different people.**
- **6 could not be called.**

So the usual session is one person, two runs of shots, and the SIFT groups
split people into eyes. No matcher links the two eyes: their vessels differ, and
appearance embeddings already failed to separate same-person pairs from
strangers. Splitting by group alone would therefore still put one eye in
training and the other in test. Experiment 24 handles this by dealing out
consecutive blocks of files, below.

### Leakage into experiment 21

For each of experiment 21's five splits:

| Split (seed) | Test images with a partner in training | Mendeley test images with no partner |
|---|---|---|
| 0 | 87 of 129 | 25 of 112 |
| 1 | 88 of 129 | 18 of 106 |
| 2 | 83 of 129 | 31 of 114 |
| 3 | 89 of 129 | 24 of 113 |
| 42 | 93 of 129 | 21 of 114 |

**Did those images score higher?** Every leaked image is from Mendeley, and Eyes-defy
images differ in difficulty, so the comparison is within Mendeley only:

| Architecture | Partner in training | No partner | Difference | Per split |
|---|---|---|---|---|
| U-Net | 0.8485 | 0.8390 | +0.0095 | +0.044, −0.027, −0.010, +0.018, +0.022 |
| UNet++ | 0.8496 | 0.8434 | +0.0062 | +0.025, −0.021, −0.005, +0.019, +0.014 |
| Attention U-Net | 0.8495 | 0.8468 | +0.0027 | +0.025, −0.018, −0.016, +0.008, +0.014 |
| FPN | 0.8382 | 0.8347 | +0.0036 | +0.028, −0.020, −0.010, +0.007, +0.013 |
| LinkNet | 0.8147 | 0.8301 | −0.0154 | +0.011, −0.049, −0.023, −0.009, −0.007 |
| PSPNet | 0.7656 | 0.7625 | +0.0030 | +0.020, −0.013, −0.033, +0.023, +0.018 |

Mean test Dice over the five splits.

**How to read this:**

- **The effect is small and inconsistent:** at most about 0.01 on average, with
  splits pointing both ways.
- **The "no partner" group is small** (18–31 images per split).
- **The "no partner" group is not clean.** It contains repeats the matcher
  missed, which pulls the difference towards zero.
- **The "no partner" images may also differ in kind.** Many are early files
  taken on other phones.

A plausible reason the leak matters little for segmentation: a repeat shot
differs in position, framing and eyelid pull, so the mask must still be found
afresh, not recalled. The comparison cannot rule out a larger effect, though.

**What it probably does not change:** the architecture ranking. Every architecture
was trained and tested on the same leaky splits. The bigger-capacity models show no
sign of exploiting the leak more (UNet++ +0.006, Attention U-Net +0.003).

**What it does change:**

- **Every figure that treated the 129 test images as independent was
  overconfident.** About 64–72% of them have a relative in training, and many
  have relatives among the other test images. This applies above all to the
  per-image tests of experiment 20.
- **The experiment-21 corrected test assumed independent subjects.** It is
  likewise somewhat optimistic about its own precision.

## How to handle it

This is now wired into the trainer (`scripts/train_segmenter.py`) and used by
experiment 24:

1. **Split by group, not by image.** `--groups groups.csv` keeps every group on
   one side of the split. Ungrouped images stay their own subject. The `subject`
   column is the id the trainer's `subject_of` gives each image.
2. **Deal out blocks of consecutive files.** `--block 24` cuts the Mendeley list
   into runs of about 24 files, never through a group, and deals out whole
   blocks. A person's two eyes can then straddle training and test only at a
   block edge.
3. **Close the edges.** `--buffer 4` leaves out of training and validation every
   group with a file within 4 numbers of a test file.

   A first version applied the buffer to test images drawn one by one from
   the whole list. Taking whole neighbouring groups with it then removed
   175–240 images, about half the training set. Blocks bring that down to
   28–57, leaving 339–366 training images against 390.
4. **Rerun the comparison on the new splits.** That is experiment 24, all six
   architectures over the same five seeds. It answers directly whether the leak
   moved the scores, instead of inferring it from the comparison above.
   **Outcome:** leak estimate +0.013 to +0.025 Dice, every 95% interval spanning
   zero; same order of architectures; LinkNet's deficits no longer significant.
   See `experiments/24_grouped_splits/RESULTS.md`.
5. **For the report: count subjects, not images.** The Mendeley set is "547
   images of at most 199 subjects", and probably far fewer people.

## Limits

- **SIFT matching under-counts repeats.** Files 1–8, 62 and 78 show it, and the
  grouping is a lower bound on duplication.
- **Left and right eyes of one person are not linked** unless they happen to
  match. Neighbouring groups are often one person (at least 14 of 24 sampled).
  The groups are therefore not a person-level split on their own; the blocked
  split of experiment 24 is what covers this.
- **The cut-off rule rests on the file-number pattern and on judgement by eye.**
  The numbers and sheets are kept so it can be checked.
- **The partner comparison is observational.** Experiment 24's rerun on grouped
  splits is the direct check. It agrees in sign and size (of order 0.01–0.02
  Dice), and neither excludes zero.

## Files

| File | What |
|---|---|
| `find_duplicates.py` | Hashes, perceptual hashes, SIFT + RANSAC on every pair (features cached outside the repo; about 5 minutes) |
| `review.py` | Contact sheets (`--sheet`), grouping and the experiment-21 leakage check (`--threshold 50 --near 20 10`) |
| `pairs.csv` | Every pair with any RANSAC inlier or a perceptual-hash distance ≤ 16 |
| `groups.csv` | The 115 groups, one row per image; `subject` is the trainer's id for it (read by `--groups`) |
| `leakage.json` | Groups, and per split the leakage counts and Dice by partner status for all six architectures |
| `summary.json` | Inlier and hash-distance distributions, and the top 60 pairs |
| `review_sheets/` | The contact sheets the cut-off was judged on; `9_neighbouring_groups_*` are the group boundaries |
