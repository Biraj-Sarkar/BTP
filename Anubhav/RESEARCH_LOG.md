# Research Log — Everything Tried, Including What Failed

A complete record of attempts, dead ends, wrong predictions and bugs, with the
reason each one failed and what it changed. Successes are documented elsewhere;
this file exists because **the failures are the part that usually goes
unrecorded, and they are where most of the learning is**.

Nothing here is embarrassing. An approach that was tested and rejected for a
measured reason is a result. An approach that was never tested is a gap.

---

## A. Method attempts that did not work

### A.1 Brightness-based segmentation — rejected

**Idea.** Select bright, low-chroma (pale, colourless) pixels, on the reasoning
that the conjunctiva region is pale.

**Why it fails.** Bright and colourless is the **sclera** — the white of the
eye — not the conjunctiva. Measured against ground truth it scores Dice
**0.000**: no overlap at all with the target tissue.

**What it changed.** Established that extraction must key on *redness*, not
brightness. Kept in the codebase as `legacy_bright_neutral_mask` so the
redness-based design can be justified by a measured comparison rather than an
assertion.

### A.2 Redness threshold with a percentile darkness gate — failed on healthy patients

**Idea.** Score pixels by redness in CIELAB, drop the darkest 20% to remove
lashes and pupil.

**Why it fails.** Healthy conjunctiva is *darker* than pale conjunctiva —
deeper red absorbs more light. Measured on phantoms, lightness falls from 174
at Hb 8 to 129 at Hb 14. A percentile darkness cut therefore deletes **the
high-haemoglobin patients specifically**. Dice was 0.998 at Hb 8 and ≈0 at
Hb 11 and 14.

**What it changed.** The gate became an *absolute* lightness cut (only
near-black pixels). Also a general lesson: any rule phrased as a percentile
adapts to the image, and if the thing you are measuring changes the
distribution, the rule silently changes with it.

**How it was caught.** A test that checked Dice across the Hb range rather than
at one value. A single-value test would have passed.

### A.3 Redness threshold at a fixed percentile — mask ran onto skin

**Idea.** Keep the top 28% reddest pixels.

**Why it fails.** The conjunctiva is only ~7% of a typical frame. Selecting 28%
floods the mask with skin, and "keep the largest connected component" then
returns the **skin blob** rather than the tissue. The failure is silent —
the mask is a plausible size and shape.

**What it changed.** Threshold selection moved to Otsu's method, which reads
the split point from the image's own histogram instead of assuming a fixed
fraction.

### A.4 Redness + Otsu on real photographs — bled onto lashes and cheek

**Idea.** The fixed method above, applied to real captures.

**Why it fails.** Eyelashes and peri-orbital skin are reddish-brown enough to
pass a redness test, and neither is dark enough to be caught by the absolute
lightness gate. On several of the 52 real images the mask extended below the
lash line onto the cheek.

**The trap.** It scored **Dice 0.999 on synthetic phantoms** and looked
finished. Synthetic validation could not reveal this, because phantoms have no
lashes and flat-coloured skin.

**What it changed.** Motivated the seeded grabCut extractor, and permanently
changed how synthetic results are reported — always labelled, never quoted as
evidence the method works.

### A.5 Seeded grabCut, first attempt — still leaked into lashes

**Idea.** Seed grabCut with a high-redness core as foreground and low-redness
as background.

**Why it fails.** Colour alone still cannot separate lash-zone skin from
tissue, so the colour model learned an overlapping distribution.

**What it changed.** Added a **texture gate** — conjunctiva is smooth, the lash
zone is high-frequency — computed as a local standard-deviation map. High
texture is seeded as definite background.

### A.6 Seeded grabCut, second attempt — segmented the iris

**Idea.** The above, with texture and sclera gates.

**Why it fails.** On a wide shot including the whole eye, the mask landed on
the **iris**. A brown iris is dark *red*, so it wins a redness contest outright.

**What it changed.** Added an absolute darkness gate (L\* < 80). Notably this
is the *same class of bug* as A.2 in reverse — the first time darkness was
over-excluded, here it was under-excluded.

**How it was caught.** Only by looking at all 52 overlays. The QC numbers
(mask ratio, focus) were entirely normal for that image.

### A.7 The gray-world + a\*/b\* hybrid — predicted to win, came third

**Idea.** Gray-world corrects the colour cast, the a\*/b\* input discards
brightness; they fix orthogonal halves of the lighting problem, so combining
them should beat either alone. This was recorded as the expected favourite.

**Why it fails.** Dropping the L\* channel discards real signal — pale
conjunctiva genuinely *is* lighter, not only less red. At channel-mean scale
that loss outweighs the brightness invariance gained. Measured MAE 1.104
against 1.073 for gray-world RGB alone.

**What it changed.** The recommendation was downgraded from "favourite" to "one
arm worth testing with a CNN", and both write-ups were corrected. Recorded
here because **the prediction was made before the measurement and was wrong** —
which is exactly the sort of thing that quietly disappears from a report.

### A.8 Richer features — made things worse

**Idea.** Channel means are crude; add standard deviations and percentiles for
R, G, B, L\*, a\*, b\* (24 features) so the model has more to work with.

**Why it fails.** With 24 features and 25 training points per fold, ordinary
least squares can nearly interpolate its training data, so it fits noise:
**MAE 11.4 g/dL, R² −118.9** — predictions wilder than the entire clinical
range. Ridge on the same features gained nothing (1.102 vs 1.073).

**What it changed.** Established that the constraint is **data quantity, not
feature poverty**, and produced the concrete warning for the network stage,
where ~8.4M parameters will be trainable.

### A.9b Channel means were not the best representation — a partial miss

**What happened.** Early experiments tested only mean R, G, B, and one
deliberately over-rich 24-feature set. Concluding from those two points that
"feature richness is not the constraint" was too quick: the space *between*
them — small, physically motivated representations — had not been tried.

**What fixed it.** Adding chromaticity coordinates, R/(R+G+B), and the erythema
index, log(R/G), both of which are invariant to illumination intensity by
construction. Measured improvement:

| Pipeline | with channel means | with erythema index |
|---|---|---|
| A | MAE 1.073, R² −0.166, F1 0.688 | **MAE 1.062, R² −0.094, F1 0.727** |
| B | MAE 1.163, R² −0.260, F1 0.562 | MAE 1.143, R² −0.192, F1 0.647 |

**What it changed.** The conclusion held — both remain at baseline — but the
margin narrowed materially, and pipeline A now beats the baseline on MAE,
accuracy, precision, specificity and F1. The lesson is that "we tried features
and it didn't help" was true of the features tried, not of features in general.
Choosing representations for a *physical* reason beat choosing them for
convenience.

### A.9 Lasso — zeroed every coefficient

**Idea.** Try L1 regularisation as an alternative to ridge; it might select the
few informative colour features.

**Why it fails.** At α = 0.5 it drove **all six coefficients to exactly zero**,
so the model degenerated to predicting the mean (MAE 1.083 — exactly the
baseline). At α = 0.1 it kept ~2.9 of 6 features, but *which* features changed
between folds, because the predictors are near-collinear (max |r| = 0.993) and
Lasso picks arbitrarily among correlated groups.

**What it changed.** Justified ridge empirically rather than by convention. The
zeroing is also an independent confirmation of the central negative result:
Lasso concluded that no colour feature earns its keep.

---

## B. Bugs found in our own work

### B.1 Quality control computed but never enforced
QC scores were calculated and written to the manifest, then ignored — blurred
captures trained the model exactly like good ones. **Fixed:** the gate now
drops failing samples and the API returns a retake prompt.

### B.2 Checkpoints could not be served
The network is trained on standardised targets, so its output needs the
training mean and standard deviation to become g/dL. Those constants were not
being saved. Every checkpoint written was unusable for inference. **Fixed:**
the checkpoint bundles weights, constants, preprocessing settings and metrics,
and refuses to load an incomplete file.

### B.3 Train and test predictions mixed in one results file
The headline predictions CSV combined training-set and test-set rows with no
column distinguishing them — with an 80/20 split, ~80% of "results" were
predictions on data the model had trained on. **Fixed:** out-of-fold
predictions only.

### B.4 Splits leaked between train and test
Splitting was done on images, sorted by patient number, unshuffled. Multiple
captures of the same eye landed on both sides. **Fixed:** patient-grouped,
Hb-stratified k-fold. This matters especially for the local cohort, where both
eyes of each child are photographed.

### B.5 Segmentation re-run every epoch
Segmentation ran inside the dataset's `__getitem__`, so a 40-epoch run
segmented every image 40 times. **Fixed:** disk cache keyed on image content
and settings.

### B.6 Frozen BatchNorm was not actually frozen
`requires_grad = False` freezes the *weights* but BatchNorm keeps updating its
running statistics in training mode, quietly drifting the "frozen" features.
**Fixed:** frozen BatchNorm layers are forced to eval mode.

### B.7 Missing ImageNet normalisation
Inputs were scaled `/255` only, while the pretrained weights expect
ImageNet-normalised inputs. **Fixed:** normalisation moved *inside* the model's
`forward()`, so the serving path cannot omit it.

### B.8 Cache key did not include the extractor version
The crop cache was keyed on the segmenter's function *name*. Changing the
extraction algorithm would have silently reused crops cut by the old one —
training on stale data with no warning. **Fixed:** an explicit `cache_key`
(`classical-v2`) to bump on any algorithm change.

### B.9 `drop_last` discarded real data
The training loader dropped every incomplete final batch, throwing away up to
15 of 47 images per fold on a small cohort. **Fixed:** only a trailing batch of
exactly one is dropped (which would break BatchNorm).

### B.10 Colour profile mismatch in the lighting test
The iPhone captures are Display P3. The first export preserved that profile,
which OpenCV then read as if it were sRGB — a systematic colour shift against
the cohort images, which are true sRGB. **Fixed:** explicit P3 → sRGB
conversion. This *strengthened* the findings: correctly converted, the colour
casts are far more extreme (worst cast ratio 53.7 → 199.6). JPEG compression
was separately verified as negligible (<0.5 a\* units vs lossless PNG).

---

## C. Environment and tooling problems

Recorded because they cost real time and will recur.

| Problem | Cause | Fix |
|---|---|---|
| "Corrupted" git repository, twice | Not corruption — iCloud had **evicted** the pack files' contents, leaving stubs with the right size and no data. Freeing disk space makes this *more* likely | Materialise the files; keep repos out of iCloud-synced folders |
| Every `.xlsx` read failed after installing torch | The install upgraded pandas 2.x → 3.0, which no longer infers the Excel engine | Pass `engine="openpyxl"` explicitly |
| `CERTIFICATE_VERIFY_FAILED` downloading pretrained weights | macOS Python ships without a usable CA bundle | `export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")` |
| Python imports hanging for minutes | Spotlight indexing a freshly created 750MB virtualenv on a nearly full disk | `.metadata_never_index` markers |
| `np.ndarray.ptp()` raised AttributeError | Removed in NumPy 2.x | `np.ptp(array)` |
| Two papers could not be downloaded | Publisher bot protection; not circumvented | Linked for manual download instead |

---

## D. Measurements that showed nothing

Null results, kept deliberately.

- **Mean colour does not predict haemoglobin** in this cohort. Across ten
  configurations (two pipelines × five representations) every one has negative
  R², and under nested selection neither pipeline beats the baseline.
- **The two pipelines cannot be separated at n = 26.** Paired permutation test
  p = 0.408, bootstrap CI on the difference [−0.067, +0.181] spanning zero. A
  wins on 15 of 26 patients, close to a coin flip.
- **Mask ratio cannot detect wrong-tissue segmentation.** Masks landing on
  skin had entirely normal area fractions. Only visual inspection of overlays
  caught it, which is why the overlay sheet exists.
- **Blur detection cannot detect flash blowout.** A blown-out capture is still
  *sharp*, so Laplacian variance passes it. Hence the separate clipping check.
- **The lighting stress test cannot separate two failures.** Segmentation
  collapse and colour drift are confounded in it, so its spread ratios are
  directional only. The clean colour measurement is the cohort correlation.

---

## E. Things not tried, and why

Recorded so the gaps are deliberate rather than accidental.

| Not tried | Why |
|---|---|
| SAM (Segment Anything) for extraction | ~600MB, slow on CPU, and needs a prompt strategy since it segments "things" without knowing which. Worth revisiting if the trained segmenter disappoints |
| U-Net instead of Mask2Former | Would probably work about as well and needs less machinery; kept as the fallback rather than the first choice |
| Larger backbones (ResNet-50, EfficientNet, ViT) | Parameter count already far exceeds what ~900 images supports; transformers especially are data-hungry |
| Tuning the ridge penalty | Tuning on the 26 patients used for evaluation would leak. Needs a nested loop or separate split, unaffordable at this n |
| Training a CNN on the local cohort | 26 patients against ~8.4M trainable parameters. A.8 shows what happens at 24 parameters |
| Using the full blood count (RDW, MCV) | RDW and MCV distinguish iron-deficiency anaemia from other types — a possible extension to predicting *which* anaemia, beyond current scope |
| Altitude and smoking threshold adjustments | WHO specifies them; not implemented because the cohort has neither recorded |

---

## F. What the failures collectively taught

1. **Synthetic validation cannot establish that a method works.** A.4 scored
   0.999 on phantoms and failed visibly on real tissue. Synthetic results now
   verify plumbing only, and are labelled as such everywhere.
2. **Silent failures are the dangerous ones.** B.1 through B.8 all produced
   plausible output while being wrong. Every one was caught by an explicit
   check, not by the code crashing.
3. **A rule that adapts to the image can adapt in the wrong direction.** A.2
   deleted exactly the patients it most needed to keep, because the quantity
   being measured changes the distribution the rule reads.
4. **Measure before predicting.** A.7 was reasoned carefully and was still
   wrong. The measurement took twenty minutes.
5. **More features is not more signal.** A.8 and A.9 both point at data
   quantity as the binding constraint.
6. **Numbers cannot replace looking at images.** D shows three failures
   invisible to every metric being computed.
