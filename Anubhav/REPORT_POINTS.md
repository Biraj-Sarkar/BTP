# Points for the IEEE report

Notes to write from, not text to paste: write each point in your own words.
Every number gives its source: a page, table or figure of the paper, or one of our
experiments in `experiments/`.

---

## Dhalla et al. 2023: the LinkNet comparison

S. Dhalla, J. Maqbool, T. S. Mann, A. Gupta, A. Mittal, P. Aggarwal, K. Saluja,
M. Kumar, S. S. Saini, "Semantic segmentation of palpebral conjunctiva using
predefined deep neural architectures for anemia detection", *Procedia Computer
Science* 218 (2023) 328–337 (ICMLDE), doi:10.1016/j.procs.2023.01.015.
PDF: `papers/conjunctiva-anemia/06_Dhalla_2023_...pdf`.

### What they did (Related Work)

- **Segmentation only.** No haemoglobin or anaemia evaluation; that is listed as
  future work (§5).
- **Data** (§3.1):
  - 162 children, both eyes → 324 photographs;
  - one tertiary hospital in North India, 11 Mar – 15 May 2022;
  - one smartphone app (64 MP main camera), 2–5 inches away, autofocus, PNG.
- **Masks:** CVAT polygons "labelled by the experts". The number of annotators and
  their agreement are not reported. The data are available on request only.
- **Augmentation:** "mirroring, horizontal and vertical flips" → 2592 images
  (= 324 × 8) (§3.1).
- **Models:** U-Net, UNet++, FPN, PSPNet and LinkNet, all on a **DenseNet-121**
  encoder pretrained on ImageNet, using the segmentation_models_pytorch library
  (§3.2–3.3).
- **Training** (§3.2):
  - Adam, learning rate 1e-3, weight decay 1e-5, cosine annealing;
  - batch 4, 200 epochs, RTX 2080 Ti;
  - the same hyperparameters for every model, final values set by "Bayesian
    optimisation".
- **Split:** 80 : 20, then the 80 split 80 : 20 again for per-epoch validation,
  so about 64 / 16 / 20. The paper calls the held-out 20% "validation" (§3.2).
- **Metrics:** pixel accuracy, IoU and Dice (§3.4).
- **Results** (Table 2):

  | | Acc % | IoU % | Dice % | Params | Size MB | s / epoch |
  |---|---|---|---|---|---|---|
  | U-Net | 94.12 | 90.05 | 93.76 | 12.9M | 204.27 | 209.41 |
  | UNet++ | 94.15 | 90.12 | 93.79 | 29.2M | 267.08 | 529.98 |
  | FPN | 93.34 | 89.95 | 92.34 | 8.6M | 228.46 | 187.02 |
  | PSPNet | 91.85 | 87.52 | 92.31 | 1.7M | 177.88 | 149.47 |
  | LinkNet | 94.17 | 90.14 | 93.87 | 9.7M | 192.14 | 201.83 |

- **Conclusion:** LinkNet is best. It was picked over U-Net partly on size and
  training time (UNet++ was dropped as largest and slowest), and called
  suitable for "real-time" segmentation (abstract, §4–5).

### Where the evidence is thin (Discussion; also motivates our protocol)

1. **The top three are separated by almost nothing.** The gaps are 0.05 points of
   accuracy, 0.09 of IoU and 0.11 of Dice (Table 2). That is from one split and
   one training run, with no spread, interval or test. For the same decoders,
   our Dice varies by 0.55–1.1 points (SD) from one split to the next
   (experiment 21), and by 1.2–2.9 points on person-safe splits (experiment 24).
   Their gap is a tenth to a fifth of the smaller figure.
2. **Possible train/test leakage.**
   - **Why we suspect it:** augmentation is described before the split, the split
     is applied to "the entire dataset", and Table 3 gives the dataset as 2592.
   - **If the split came after augmentation:** each test image has 7 flipped
     copies, and about 4–5 of them land in the 64% training share. The chance
     that none does is 0.36^7 < 0.1%.
   - **Children are not mentioned:** nothing says both eyes of a child were kept
     on one side.
   - The paper does not say which order was used. A sentence would settle it.
   - **Our own data show the same problem** without any augmentation: in a
     public set, repeat shots put a relative of 64–72% of test images in
     training (experiment 23).
   - **What it did to our scores** (experiment 24, the same 30 models retrained on
     person-safe splits):
     - Dice fell by 0.001–0.018. The leak estimate, with Eyes-defy as the
       control, is +0.013 to +0.025 Dice, with every 95% CI spanning zero.
     - More important, it made our comparison look **sharper** than it is.
       LinkNet's deficits were significant on the leaky splits (Holm p
       0.03–0.05) and are not on person-safe ones (Holm p 0.18–0.43).
     - An augment-before-split leak puts mirror images of the same photograph
       in training, which is a closer copy than our repeat shots, so its effect
       could be larger.
3. **The reported accuracy and Dice do not fit the masks shown.**
   - From the definitions, correctly found conjunctiva as a share of all pixels
     is TP/N = D(1 − A) / (2(1 − D)).
   - For LinkNet's 94.17% accuracy and 93.87% Dice, that is **at least 45% of
     the pixels**. Across the five models it is 40–49%.
   - The ground-truth masks in Fig. 3 cover **8–30% of the frame (median 13%)**,
     measured from the figure's embedded images.
   - At 13% coverage, a Dice of 93.9% would come with roughly **98% accuracy,
     not 94%**.
   - So either the scored images differ from those shown, or one metric was not
     computed as §3.4 defines it. The bound is exact for pooled counts and
     approximate for per-image averages.
4. **LinkNet's Dice is given two ways:** 93.78 (abstract, §4, Table 3) and 93.87
   (Table 2, §5).
5. **The evaluated epoch is not stated.**
   - Fig. 2: validation loss is flat from about epoch 50. For FPN it climbs after
     epoch ~100 (about 0.025 → 0.045), and a little for UNet++ and PSPNet, while
     training loss keeps falling.
   - If the epoch-200 weights were scored, part of the ranking is overfitting,
     not architecture.
   - The loss function is never named.
6. **Hyperparameter tuning is unclear.** All models are said to share the same
   hyperparameters, yet these were set by Bayesian optimisation. On which model?
   On which split? Tuning on the reported split would make every score
   optimistic.
7. **The comparison with earlier work (Table 3) spans different datasets.**
   - It sets 115 images, 92 patients and 608 images against their 2592, so the
     "better than state of the art by a genuine margin" claim is not supported.
   - Table 3 also credits their own LinkNet result to the LinkNet authors
     (Chaurasia et al. [33]).
8. **"Real-time" is not measured.** Only training time per epoch is reported; no
   inference time.
9. **The size column does not follow the parameter counts.**
   - PSPNet: 1.7M parameters but 177.9 MB.
   - UNet++: 29.2M parameters at 267.1 MB.
   - Float32 weights alone would be about 7 MB and 117 MB.
10. **The code repositories listed are not what was run.**
    - Table 2 links the original authors' implementations (milesial/Pytorch-UNet,
      MrGiovanni/UNetPlusPlus, …), while §3.2 says segmentation_models_pytorch.
    - The parameter counts fit the library. Rebuilding the five networks with a
      DenseNet-121 encoder gives 13.6, 30.1, 9.3 and 10.4M against the reported
      12.9, 29.2, 8.6 and 9.7M. PSPNet uses 2.1M of its 7.75M, since it only
      reads the early encoder stages, against 1.7M reported.
11. **Editorial and citation slips:**
    - The abstract and conclusion name FCN; the methods and tables use FPN.
    - DenseNet is cited as [34], which is Iandola et al. 2014 ("DenseNet:
      implementing efficient convnet descriptor pyramids"), not Huang et al. 2017.
    - Fig. 1(b) cites UNet++ as [30], a knee-arthroscopy paper; it should be [35].
    - The listed flips give 4 distinct versions of an image, not 8. Mirroring is
      the horizontal flip; reaching 8 needs 90° rotations.
    - Table 1 and Table 3 disagree on sizes: 94 vs 92 patients, and 135 + 500 vs
      608 images.
12. **Framing is very different from ours.**
    - Fig. 3 shows tight crops of the lower lid, with conjunctiva at 8–30% of
      the frame.
    - Our inputs are whole eye photographs. Conjunctiva is about 1.4% of the
      pixels, and predicting "background everywhere" already scores 0.986
      accuracy (experiment 21).
    - So their accuracy and Dice are not comparable with ours. Dice rises with
      object size for the same boundary error.
    - The paper does not say whether the crops were cut before segmentation. If
      they were, the pipeline has a cropping step that is not described.
13. **Generalisation is untested.** One site, one phone, children only, no
    external test set.
    - Inter-annotator agreement is not reported.
    - On the Mendeley set, two annotators agree at only Dice 0.736 because they
      draw the upper edge differently: one mask sits inside the other in 98% of
      images. A single annotation convention therefore bounds what a Dice above
      90% means.
14. *(Intro only.)* For a paediatric study, it quotes adult haemoglobin ranges
    (12–16 g/dL for women, 14–18 g/dL for men). WHO's child thresholds are
    11.0–12.0 g/dL.

### What our experiments show (Results / Discussion)

- **Same five decoders plus Attention U-Net.** EfficientNet-B0 encoder, 320 px
  full frames, 40 epochs, AdamW 3e-4, batch 8. 648 public images (Eyes-defy 101,
  Mendeley 547).
- **Five splits, image-level** (experiment 21), Dice mean ± SD:

  | | Dice | rank per split |
  |---|---|---|
  | Attention U-Net | 0.8464 ± 0.0069 | 1–3 |
  | UNet++ | 0.8462 ± 0.0072 | 1–3 |
  | U-Net | 0.8445 ± 0.0076 | 1–3 |
  | FPN | 0.8353 ± 0.0055 | 4 every time |
  | LinkNet | 0.8187 ± 0.0108 | 5 every time |
  | PSPNet | 0.7654 ± 0.0094 | 6 every time |

- **Five splits, person-safe** (experiment 24: repeat-shot groups kept together,
  Mendeley dealt out in blocks of consecutive files, a 4-file buffer around test
  files; 339–366 training images):

  | | Dice | rank per split |
  |---|---|---|
  | UNet++ | 0.8333 ± 0.0169 | 1–3 (first in 4) |
  | Attention U-Net | 0.8311 ± 0.0123 | 1–3 |
  | U-Net | 0.8265 ± 0.0179 | 2–5 |
  | FPN | 0.8206 ± 0.0181 | 3–5 |
  | LinkNet | 0.8099 ± 0.0206 | 4–5 |
  | PSPNet | 0.7640 ± 0.0287 | 6 every time |

- **The LinkNet claim is not reproduced.**
  - LinkNet is never first, in ten splits across both experiments, and is
    above only PSPNet.
  - Image-level splits (experiment 21): below U-Net (−0.026 Dice, Holm p
    0.030), UNet++ (−0.028, p 0.047) and FPN (−0.017, p 0.047), each in 5 of 5
    splits; above PSPNet (+0.053, p 0.019).
  - Person-safe splits (experiment 24): below U-Net (−0.017, Holm p 0.43, 4 of 5
    splits), UNet++ (−0.023, p 0.18, 5 of 5) and FPN (−0.011, p 0.43, 4 of 5);
    above PSPNet (+0.046, p 0.072, 5 of 5). **Not significant.**
  - All are corrected resampled t-tests (Nadeau–Bengio), Holm within the family.
  - Experiment 20 (one split) points the same way, with per-image p < 0.0001.
    Those per-image tests are overconfident because of the repeat shots
    (experiment 23).
- **What agrees with the paper:**
  - PSPNet is last.
  - The top group cannot be separated: ours at p > 0.5, theirs by 0.11 points
    on one split.
- **How LinkNet fails for us: it spills.** It has the highest recall (0.901) and
  the lowest precision of the top five (0.765). Its boundary error is HD95
  3.0 px, against 2.2–2.6 for the U-Net family. Spill matters here because it
  mixes non-conjunctiva colour into the haemoglobin features. The same pattern
  holds on person-safe splits (recall 0.880, precision 0.767, HD95 3.6 against
  3.3–3.5).
- **Why results may differ** (none established):
  - encoder: DenseNet-121 against EfficientNet-B0;
  - training length: 200 epochs against 40. LinkNet peaked late for us, at
    epochs 33–39, but would need about +0.026 Dice to draw level;
  - framing: tight crops against full frames;
  - data: children in hospital against adult public data;
  - annotation convention;
  - possible leakage, and a single split, in theirs.
- **Wording to use:** "not reproduced under our conditions", not "refuted".
  Their data are unavailable and our recipe differs.
  - Do **not** write "LinkNet is significantly worse": that held only on the
    leaky splits. Write "never best, lower than the U-Net family in most splits,
    not significantly so on person-safe splits".
- **Why five more splits would not settle LinkNet against U-Net:**
  - The corrected variance is σ²(1/J + n_test/n_train). The second term (0.36)
    does not shrink with J.
  - At the observed gap and spread, p would be about 0.14 with 10 splits and
    0.10 with 20.
  - It needs a lower-variance design (blocked k-fold) or more data.

### A precision-weighted score and the mask threshold (Methods / Results; experiment 25)

- **Why weight precision.** A pixel wrongly included (lid skin, sclera, lash,
  glare) pulls the measured colour toward the wrong tissue. A pixel wrongly left
  out only removes some conjunctiva. Dice is F1 on pixels and counts both
  equally.
- **The score:** F-beta with beta = 0.5, so precision weighs twice as much as
  recall.
  - F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP).
  - It equals the Tversky index with 0.8 on false positives and 0.2 on false
    negatives.
  - Fixed at the conventional 0.5 before rescoring, not tuned.
  - Candidate references, check before citing: van Rijsbergen 1979
    (*Information Retrieval*) for F-beta; Salehi et al. 2017 (MLMI) for the
    Tversky loss in segmentation.
- **Keep Dice as the headline**, because published work reports it. Put F0.5
  beside it.
- **Person-safe splits, test, mean of five:**

  | | F0.5 at 0.5 | threshold chosen (median) | F0.5 at chosen | Dice at chosen |
  |---|---|---|---|---|
  | UNet++ | 0.823 | 0.97 | 0.853 | 0.810 |
  | Attention U-Net | 0.825 | 0.98 | 0.846 | 0.809 |
  | U-Net | 0.824 | 0.96 | 0.846 | 0.802 |
  | FPN | 0.809 | 0.96 | 0.841 | 0.788 |
  | LinkNet | 0.782 | 0.97 | 0.832 | 0.778 |
  | PSPNet | 0.740 | 0.95 | 0.781 | 0.746 |

- **How the threshold was chosen:** each model's threshold maximises F0.5 on its
  own validation images; the test images were not used. It landed on average
  0.02 from the test optimum.
- **What the higher threshold does:**
  - test F0.5 rose in all 29 models whose threshold moved (+0.021 to +0.050 per
    architecture);
  - precision rises by 0.06–0.12 and recall falls to 0.71–0.77;
  - Dice falls by 0.018–0.033;
  - masks shrink to about 5/6 of the true area;
  - 6 of 3,750 masks become empty.
- **Control:** choosing the threshold for Dice instead gains nothing (−0.001 to
  +0.002). So 0.5 is fine for Dice, and the high threshold is the price of
  weighting precision.
- **Effect on the architecture comparison:**
  - At 0.5 the order is unchanged and LinkNet's gap doubles (−0.043 against
    Attention U-Net, Holm p 0.30).
  - At the chosen thresholds the middle closes up: LinkNet −0.014 (Holm 0.45),
    FPN −0.006.
  - Much of LinkNet's and FPN's shortfall was spill at 0.5.
  - Nothing beats Attention U-Net, and LinkNet is still never first.
- **One trained model never exceeds probability 0.53** (seed-0 Attention U-Net),
  so its threshold could not move. A threshold must be chosen per trained model
  on its own held-out images, not copied.
- **Why the best threshold sits near 1:** the training loss is cross-entropy plus
  soft Dice, which pushes probabilities toward 0 and 1.
- **Wording to use:**
  - "a validation-chosen threshold raised test F0.5 in every model whose
    threshold moved". Do not say it is significant per architecture: no single
    gain survives Holm over five splits.
  - "LinkNet over-segments at 0.5", not "LinkNet cannot find the conjunctiva".

### Checks before further colour corrections (Methods / Results / Limitations; experiment 26)

Points only; numbers from `experiments/26_correction_prechecks/RESULTS.md`.

- **Why:** further corrections for the regressor's input (brightest pixels as
  white, sclera as white) each rest on an assumption; checked on 217 Eyes-defy
  photographs before any haemoglobin model.
- **Clipping:** all three channels reach 255 somewhere in 96% of photographs, so
  plain max-RGB (white-patch) is no correction. Clipped area is tiny (median
  0.05%), so an unclipped variant is workable.
- **Brightest unclipped pixels:** about half lie on the capture device's white
  plastic, none on the conjunctiva; above the lower lid they lie on the sclera.
- **A reference for the light:** the device's white plastic, with white balance
  fixed manually (EXIF). Angle to it (degrees, India / Italy): no correction
  4.7 / 2.8; gray-world 4.9 / 5.2; brightest above the lid 1.5 / 0.8; sclera
  2.3 / 1.0.
- **The sclera is not white:**
  - b* minus the plastic's b* rises +0.64 (India) and +0.62 (Italy) per decade,
    with haemoglobin and sex held; +0.47 within one Italian batch.
  - Its a* follows haemoglobin: ρ +0.51 (India), +0.44 (Italy).
  - No link between its yellowness and haemoglobin; jaundice untestable (no
    bilirubin).
- **What dividing by an estimate does** (share of the conjunctiva's log(R/G)
  haemoglobin slope removed, India): plastic 0.10, gray-world 0.29, sclera 0.18.
  Sclera correction makes older Indian patients read about 0.4 g/dL per decade
  lower.
- **Dataset finding:** the Italian set is two capture batches (patients 1–31:
  mean age 66, Hb 11.5, less red light; 32–123: age 44, Hb 14.6). A batch term
  lowers Italy's colour–haemoglobin slope by 11%, dividing by the plastic by 16%.
- **Candidate references (check before citing):** von Kries (diagonal
  correction); Land and McCann 1971 (Retinex, white-patch); Buchsbaum 1980
  (gray-world); Finlayson and Trezzi 2004 (Shades of Gray); Hordley and
  Finlayson 2006 (angular error for illuminant estimates).
- **Wording rules:**
  - "the sclera yellows with age in these photographs", not "the sclera is
    yellow".
  - "part of the Italian colour–haemoglobin link is the light", not "the Italian
    results are invalid".
  - The leakage analysis and the batch check were added after the first results;
    say so if they are reported.

### White-patch as the default colour correction (Methods / Future Work)

Points only. Changed 17 Nov 2026; nothing about haemoglobin was measured.

- **What changed:** the pipeline's default correction is now white-patch, the
  brightest 1% of unclipped pixels above the lower lid taken as white
  (previously gray-world).
- **How it works (Methods):**
  - computed on the full 512×512 frame, never on the crop; crop cut afterwards;
  - pixels with any channel ≥ 250 (and a 5×5 neighbourhood) excluded;
  - pixels more than 10 px below the conjunctiva's lower edge excluded (device
    plastic);
  - gains = mean(white) / white per channel, so brightness is kept.
- **Order:** segmentation first, then correction, since the mask locates the
  lid. The segmenter's own input is unchanged, so masks are unchanged.
- **Why this one:** closest estimate of the light measured (0.8–1.5° from the
  device plastic; gray-world about 5°), experiment 26.
- **Saved models** record their correction and are served with it; the
  deployed model stays gray-world.
- **Check done:** on 20 random photographs the pipeline's estimate equals the
  experiment's exactly (0.000°).
- **Future Work:**
  - test it on haemoglobin error against no correction and gray-world, one
    extractor, same patients, paired per patient;
  - known risk to state: it lies mostly on the sclera, so in India older
    patients would read about 0.47 g/dL per decade lower, and it removes about
    0.15–0.18 of the conjunctiva's haemoglobin slope (plastic: 0.10–0.16);
  - add the device plastic as a reference arm and model the Italian batch.
- **Wording rules:** "chosen because it estimates the light most closely", not
  "improves accuracy"; say plainly that it is untested on haemoglobin.

### Still to do

- **Faithful reproduction of their recipe on our data:**
  - DenseNet-121, Adam 1e-3, weight decay 1e-5, cosine, batch 4, 200 epochs,
    flips;
  - on full frames and on lid crops;
  - plus one arm that augments before splitting, to measure how much that order
    could inflate Dice.
  - Not timed yet. DenseNet-121 at 200 epochs will take hours per model on this
    Mac, so it needs scoping (for example LinkNet against U-Net only, a few
    seeds).

### Where each part could go

- **Related Work:** what they did, plus one line on the single split.
- **Methods:** why we use repeated splits, a corrected test and person-safe
  splits (points 1, 2); how the person-safe split works.
  Why precision is weighted (F0.5), and how each model's threshold is chosen on
  validation images.
- **Results:** the person-safe table (experiment 24) as the headline, with the
  image-level one (experiment 21) beside it; LinkNet not reproduced. The F0.5
  table at 0.5 and at the chosen threshold (experiment 25), with the Dice cost.
  The colour-correction checks (experiment 26): clipping, the sclera against age
  and haemoglobin, the Italian batches.
- **Future Work:** testing white-patch, the new default, on haemoglobin error
  (see "White-patch as the default colour correction").
- **Discussion:**
  - why results can differ;
  - leakage risk;
  - Dice and accuracy are not comparable across framings (points 3, 12);
  - pixel accuracy is uninformative on full frames.
