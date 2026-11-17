# Datasets

The image data is **not committed to this repository**, deliberately. The two
segmentation sources total ~4.9 GB, GitHub rejects any single file above 100 MB,
and a repository that size is impractical to clone. What is committed instead is
everything needed to reconstruct the exact training set: source, version,
licence, download command, expected file counts, and the manifest recording
which files were used and which were rejected.

Run `bash scripts/fetch_datasets.sh` to download both, or follow the per-dataset
commands below.

Everything lands under `datasets/`, which is gitignored. On a machine where the
repository sits in an iCloud-synced folder (Desktop, Documents), point
`datasets/` at a directory outside that sync — macOS evicts large files and
training then stalls with `mmap ... errno=60`. The layout used here is a
symlink:

    ln -s ~/btp-local/datasets datasets

---

## 1. Eye Conjunctiva Segmentation Dataset — *segmentation*

| | |
|---|---|
| Source | https://data.mendeley.com/datasets/yxwjgcndg2/1 |
| Authors | Rahman, Faruk, Ullah, Alam, Sadman — Ahsanullah University of Science and Technology |
| Licence | **CC BY 4.0** — redistribution permitted with attribution |
| Size | 4.26 GB |
| Contents | 547 smartphone eye images (~3000×4000) of **at most 199 subjects** (see below), plus masks from **two independent annotators** |
| Labels | none — no haemoglobin, no anaemia status |

```bash
curl -L -o datasets/conjunctiva_seg/yxwjgcndg2-1.zip \
  "https://data.mendeley.com/public-api/zip/yxwjgcndg2/download/1"
unzip datasets/conjunctiva_seg/yxwjgcndg2-1.zip -d datasets/conjunctiva_seg/
```

### Measured facts that its description does not tell you

**The masks are RGB with values {0, 128}, not {0, 255}.** A conventional
`> 127` threshold reads every mask as empty — and fails silently, because an
empty prediction scored against an empty target gives a perfect Dice. Use
`> 0`.

**The "~99.9% agreement" is pixel accuracy, not Dice.** Measured over all 547
pairs, the two annotators agree at **Dice 0.736 mean / 0.745 median**, with 546
of 547 below 0.90. The 99.9% figure is uninformative here because the
conjunctiva occupies a median **0.76%** of the frame — predicting "background
everywhere" already scores 99.24%.

**But 0.736 is not a noise floor, and must not be quoted as one.** The two
annotations are **nested, not scattered**:

| | |
|---|---|
| annotator 2 larger than annotator 1 | **99% of images** |
| median area ratio (A2 / A1) | **1.56×** |
| fraction of the smaller mask inside the larger | **0.981** |
| correlation of Dice with area difference | **−0.859** |

They agree on *where* the conjunctiva is and disagree on **how far to extend
it**. Anatomically that is expected: the palpebral conjunctiva has no sharp
upper boundary, transitioning continuously into the forniceal conjunctiva in the
deep fold behind the lid, so how far you follow it is a judgement call that
depends on how far the lid was everted. Annotator 2 consistently follows it
further.

Thinness exaggerates the gap. Median perimeter²/area is **37** against 12.6 for
a circle, and Dice punishes thin shapes hard — a one-pixel boundary shift on a
long crescent moves a large *fraction* of its area.

So this is a **systematic convention difference between two self-consistent
annotators**, not irreducible label noise. A model trained on one annotator can
legitimately exceed 0.736 against that annotator; the figure bounds
cross-annotator agreement, not learnability.

**Consequence for `--annotator`:** because A1 ⊂ A2 almost everywhere, the
`intersection` default is effectively **annotator 1's mask** — the more
conservative convention — rather than a principled middle between the two.

The annotations are otherwise excellent: median extent 0.282 and **zero
rectangles in 547**, i.e. genuinely traced contours throughout.

**The 547 images are repeat shots of far fewer subjects.** Files carry no subject
id and no metadata, but keypoint matching between every pair finds **463 images
in 115 groups** of 2–10 shots of one subject, most often four, taken in quick
succession with neighbouring file numbers. That makes **at most 199 distinct
subjects**, and fewer in fact, because the matcher misses shots taken at a
different angle or of the other eye. Neighbouring groups are themselves often one
person: at a sample of 24 places where one group ends and the next begins, at
least 14 showed the same person (the other eye, or the same eye re-framed) and 4
plainly different people. So a "group" is closer to one eye than to one person,
and the number of people may be nearer half of 199. Treated as 547 independent
subjects (as `subject_of` does), about two-thirds of every test split had a
same-subject photograph in training. `experiments/23_duplicates/groups.csv`
lists the groups. To split safely, pass it to the trainer with `--groups`, plus
`--block 24 --buffer 4` for the neighbouring runs (experiment 24). Details and
the leakage measurement are in `experiments/23_duplicates/RESULTS.md`. On those
splits Dice is 0.001–0.018 lower than with image-level splits, and the order
of architectures is unchanged (`experiments/24_grouped_splits/RESULTS.md`).

```bash
python scripts/prepare_conjunctiva_seg.py \
  --source "datasets/conjunctiva_seg/Eye Conjunctiva Segmentation Dataset" \
  --out datasets/conjunctiva_seg_prepared
```

Defaults to the **intersection** of the two annotators — the region both agree
is conjunctiva. Justified by this project's own measurement that contamination
costs more than omission: widening the mask degraded haemoglobin prediction
monotonically (`experiments/13_learned_extractor/`). `--annotator` selects
`1`, `2`, `union` or `intersection`.

The primary segmentation training source. Three properties make it preferable
to the alternative below:

* **Purpose-built.** It exists to be a segmentation benchmark, so the
  annotations are traced throughout rather than incidental to another study.
* **Smartphone captures.** The deployment target is a phone photographing an
  everted lid, so this is far closer to the serving domain than clinical-camera
  sources.
* **Two annotators.** Measured at Dice 0.736 against each other — but nested
  rather than scattered (see above), so this is a convention difference, not a
  noise floor. Its practical value is that it makes the *shape* of annotator
  disagreement visible, which no single-annotator dataset can.

**What this dataset measurably bought**, once trained jointly with Eyes-Defy
(`experiments/13_learned_extractor/robustness.py`): the extractor's failure rate
under realistic capture variation — rotation, zoom, exposure, blur — **halved,
35 → 18** across 520 perturbed captures, with the largest gains exactly where
extra capture diversity should help (zoom-out failures 15 → 2, over-brightening
10 → 5). Clean-capture accuracy was unchanged within noise. **Robustness, not
accuracy**, is what 547 varied smartphone images are worth here.

## 2. Eyes-Defy-Anemia — *segmentation + haemoglobin*

| | |
|---|---|
| Source | https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia |
| Licence | CC BY-SA 4.0, as listed on the Kaggle page (a re-upload; cite the original authors given there) |
| Size | 654 MB (635 MB compressed) |
| Contents | 218 subject folders across India and Italy; per subject a full-resolution photograph plus up to three annotation images |
| Labels | **haemoglobin**, in `India.xlsx` and `Italy.xlsx` |

Needs a Kaggle API token. The current CLI takes the token string from the
settings page in `~/.kaggle/access_token` — **not** the older `kaggle.json`:

```bash
kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia \
  -p datasets/eyes_defy --unzip
python scripts/prepare_eyes_defy.py --source datasets/eyes_defy \
  --out datasets/eyes_defy_prepared
```

### Two things to know before using it

**The annotations are the frame downscaled ~3.7× with everything outside the
region blacked out**, so a mask is recoverable as the non-black pixels.
Registration was verified numerically: on foreground pixels the annotation and
the downscaled original differ by a mean of 0.06–0.60 out of 255, which is
resampling noise.

**They are not uniformly traced.** Roughly **45% of `_palpebral` annotations are
axis-aligned rectangles** — bounding boxes that frequently contain sclera, iris
and pupil — and `_forniceal_palpebral` is 95% rectangles. Training on the
mixture teaches a network to predict boxes containing an eye. A rectangle fills
its own bounding box, so `extent = area / bbox_area` is ≈1.0 against a median of
0.19 for traced crescents; `prepare_eyes_defy.py` filters on that and keeps
**101 pairs from 101 distinct subjects**.

Its haemoglobin labels are usable independently of the mask problem.

**The Italian set is two capture batches** (experiment 26). Patients 1–31 have
files named `1.jpg` … `31.jpg` and were photographed on five days between 28 May
and 12 July 2019; patients 32–123 have files named `T_<n>_<date>_<time>.jpg` and
were photographed on eight days between 6 and 14 June 2019. The first batch is
older (mean age 66 against 44) and more anaemic (mean Hb 11.5 against 14.6), and
the device's white plastic shows its light was less red. Any Italian analysis
that pools the two should model the batch. All photographs, both sites, come
from a Samsung Galaxy S6 (SM-G920F in Italy, SM-G920I in India) with white
balance set to manual and no flash.

## 3. CP-AnemiC — *haemoglobin*

| | |
|---|---|
| Contents | ~710 cropped conjunctiva patches with haemoglobin |
| Status | **not yet downloaded**; `anemia.data.load_cp_anemic` is unverified against a real download |

Cropped patches, so it **cannot train a segmenter** — the input a segmenter
needs is the whole eye. Useful for the regression stage only.

---

## Committed provenance

| File | What it records |
|---|---|
| `datasets_manifests/eyes_defy_prepared.json` | every keep/reject decision made while preparing Eyes-Defy, with the extent and coverage that drove it |
| `datasets_manifests/conjunctiva_seg_prepared.json` | per-image coverage and inter-annotator Dice for all 547 Mendeley captures, plus the combination policy used |

A manifest is the reproducibility contract: given the source archive and the
manifest, the training set is reconstructible exactly, including which files
were excluded and why.

## Checking a new dataset before planning around it

Most published anaemia datasets distribute **cropped conjunctiva patches, not
masks**, and the distinction is invisible on a download page. Run this on
anything new before assuming it can train a segmenter:

```bash
python scripts/inspect_dataset.py datasets/<name>
```

It reports the directory shape, finds image/mask pairs, and classifies what the
"mask" side really is — a binary mask, a label map, a cutout on black (from
which a mask is recoverable), or an ordinary photograph.
