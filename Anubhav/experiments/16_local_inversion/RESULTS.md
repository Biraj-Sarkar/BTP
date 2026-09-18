# Why the Local Cohort Does Not Show the Colour Signal

Files: `measure_local.py` → `local_measurements.csv`; `degrade_external.py` →
`degrade_external.json`; blinded eversion ratings in `local_measurements.csv`
(`eversion`, `blind_code`) with `blind_rating_key.json` and `blind_sheet_*.jpg`;
`lowest_vs_highest_hb.jpg`.

Under the learned extractor, 217 external subjects show a strong physiological
relationship between conjunctival colour and laboratory haemoglobin (red
chromaticity r = +0.67, blue −0.67). The 26 local patients do not. This
experiment asks why.

---

## 1. First, how strong is the "inversion"?

| local, learned extractor | r vs Hb | p |
|---|---|---|
| red chromaticity, raw | −0.26 | 0.198 |
| red chromaticity, gray-world | −0.33 | 0.101 |
| blue chromaticity, raw | +0.33 | 0.098 |
| **blue chromaticity, gray-world** | **+0.52** | **0.006** |
| local vs external difference, red | | **8 × 10⁻⁷** |

**"Inverted" overstated the red channel**, whose local correlation is not
distinguishable from zero. Two things are firm: the local cohort **lacks** the
physiological signal every external subset shows, and under gray-world its blue
channel correlates positively with haemoglobin. The second is one of many
correlations examined below at n = 26, and should be weighed accordingly.

## 2. What the local images are

| | local | external |
|---|---|---|
| EXIF, colour profile, capture date | none (52 of 52) | intact |
| dimensions | 52 different sizes, 695×414 to 1556×500 | uniform 2988×3984 |
| file size | 29–97 KB | ~3 MB |
| JPEG | quality ~75, 4:2:0, all files | camera output |

Hand-cropped, recompressed exports rather than camera files. Labels map to
images correctly (0 mismatches against `DATASAMPLE.csv`), and the two eyes of
each child agree in measured colour (r ≈ 0.6), so pairing is sound.

## 3. Ruled out

| suspect | test | result |
|---|---|---|
| **Image processing** (downscale, JPEG, re-save) | External photos downscaled until their crescents matched local thickness in pixels (×0.146), then JPEG q75 once, twice, and q40 twice | Red vs Hb stays **+0.673** in every variant. Processing cannot remove the signal |
| Illumination, phone white balance, colour cast linked to Hb | Sclera and whole-frame chromaticity vs Hb | all \|r\| < 0.08 |
| Capture session (dates span nine months) | days since first capture vs Hb | r +0.01 |
| Age, sex, socio-economic status | vs Hb | all p > 0.2 |
| Crop framing drives the colour | framing (sclera area, width, mask area) vs colour; colour–Hb partial on framing | framing does not predict colour (all \|r\| < 0.27); controlling it leaves colour–Hb unchanged (+0.51 → +0.50) |
| Sclera-relative measurement fixes it | conjunctiva minus sclera vs Hb | still +0.43 on blue (p 0.027) — not a global cast |

Framing *does* correlate with haemoglobin (sclera area r +0.53, p 0.006) but
does not act through measured colour.

## 4. Found: how far the lid is pulled down changes the colour

Overlays of the lowest- and highest-haemoglobin children
(`lowest_vs_highest_hb.jpg`) suggested lids were everted very differently, and
that the lowest-haemoglobin conjunctivae looked red rather than pale. Eversion
is a visual property, so it was **rated blind**: all 52 captures shuffled,
labelled with codes only, and rated on a fixed rubric (0 thin rim, 1 moderate
band, 2 broad sheet) before joining to haemoglobin.

**Eversion vs measured colour** — strong, and under *both* extractors:

| eversion vs | ρ | p |
|---|---|---|
| learned, red | +0.40 | 0.044 |
| learned, blue | −0.51 | 0.008 |
| classical, red (raw) | +0.42 | 0.033 |
| classical, blue | −0.44 | 0.026 |
| sclera area in frame (objective) | −0.42 | 0.033 |

A harder, broader eversion gives redder, less blue tissue whichever extractor
measures it, so this is a property of the tissue and not of the learned model.
The simplest physical reading is mechanical congestion — pulling and holding a
child's lid engorges the vessels — though exposing deeper, more vascular
conjunctiva would act the same way. The rating also tracks an objective
geometric quantity (more eversion, less sclera in frame), which argues it is
measuring something real.

**Eversion vs haemoglobin** — suggestive, not established:

| | ρ | p |
|---|---|---|
| patients, both eyes averaged | −0.46 | 0.018 |
| **right eyes only — fully blind** | **−0.24** | **0.234** |
| left eyes only — rater had seen 10 with their Hb | −0.55 | 0.004 |

The rater had seen ten left-eye images with their haemoglobin values while
forming the hypothesis. The fully blind right eyes show the same direction but a
weak, non-significant association, and the gap between eyes is exactly what
rater bias would produce. **The claim that lower-haemoglobin children were
everted harder is not supported to the standard of the rest of this project.**

**Does eversion explain the colour–haemoglobin association?** Partly:

| learned extractor vs Hb | raw r | controlling eversion |
|---|---|---|
| red, raw (patients) | −0.24 | −0.10 |
| blue, gray-world (patients) | +0.51 | +0.40 |
| red, gray-world (patients) | −0.36 | −0.23 |
| blue, gray-world (right eyes, blind) | +0.40 | +0.36 |

Associations move toward zero, and the classical extractor's red swings toward
the physiological direction (+0.02 → +0.21), but none is eliminated.

## 5. Conclusion

The local cohort lacks the colour signal for reasons in how the photographs were
taken, not in how they were processed, and not because of illumination, session
or demographics. The one mechanism demonstrated is that **lid eversion varies
widely across these captures and changes measured conjunctival colour** in the
direction that would mask pallor. Whether eversion also tracked haemoglobin —
which would turn masking into an apparent inversion — is suggestive but not
established, and eversion accounts for only part of the pattern. The remainder
may be chance at n = 26 after many comparisons, or factors not recorded:
conjunctivitis or allergy, and the time between blood draw and photograph.

## 6. What this means for the project

- **Capture protocol is part of the measurement.** Colour from lids everted to
  different degrees is not comparable. Hospital collection needs a fixed,
  gentle eversion, consistent framing, and ideally a colour reference.
- **Record what cannot be recovered afterwards:** eversion, time between blood
  draw and photograph, and any eye irritation.
- **Keep original camera files.** Cropped exports lost the metadata that would
  have ruled other explanations in or out.
- **A second, fully independent blind rater** is needed before eversion is
  quoted as linked to haemoglobin.

## Limitations

- One rater; inter-eye rating agreement ρ = 0.42, so the ratings are noisy.
- Blinding was imperfect for ten left-eye images, disclosed and separated above.
- Eversion was judged from the same pixels whose colour is measured; a more
  vivid conjunctiva may look more everted, which could inflate the
  eversion–colour correlation. The rubric was about extent of exposed tissue,
  and the objective sclera-area correlation partly guards against this.
- Around 60 correlations were examined in this experiment; individual p-values
  near 0.05 should not be read as confirmatory.
