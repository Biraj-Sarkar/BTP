# Attention U-Net extractor integration

The frozen candidate bundle was loaded explicitly through `load_attention_unet()` and through `ROIExtractor(checkpoint_path=...)`. The existing default `load_unet()` path and `Models/UNet_extractor/unet.pt` were not changed.

Regression tests: `python -m pytest -q extractor/test_inference.py` — **4 passed**. The tests cover candidate loading, finite/repeatable probability maps, legacy checkpoint loading, 224×224 uint8 RGB crop output, mask dimensions, threshold configuration, and empty-mask fallback behavior.

The full repository suite completed with 56 passed and 6 unrelated pre-existing failures: four Anubhav tests cannot import the removed `train_segmenter` helper, and two predictor verification tests require the absent `ROI/roi_manifest.csv`. These failures are outside the extractor integration changes.

The existing Anubhav API regression subset also passed: `algorithms/Anubhav/tests/test_api.py` — **4 passed** (deprecation warnings only).

A synthetic RGB contract check through `extract_from_path()` preserved a known `[11, 22, 33]` RGB pixel as `[11, 22, 33]` after BGR file decoding, conversion, masking, and crop generation.

The packaged candidate was reloaded beside the source checkpoint on the same RGB input; both probability maps were finite, had the expected dimensions, and had maximum absolute difference 0.0.

The extractor CLI was run on `segmentation_dataset/images/1731.jpg`; it produced a 224×224 crop and `cli_test_output/extraction_manifest.csv`. Candidate-versus-production comparisons were run on `1731.jpg`, `1942.jpg`, and `6-3004.jpg`. Side-by-side crops are under `comparisons/`; they are visual comparisons only, not ground truth. Full extraction timings were approximately 199–225 ms/image for the candidate and 176–184 ms/image for the legacy path on this CPU; the previously measured model-only candidate latency is approximately 92.9 ms/image.

Candidate artifact SHA-256: `c7c8eea36c2f162297e7dc1d5683051b8090faac881a5cd98ec62588593f1498`. The 27-image test set has no patient IDs, so patient-independent generalization is not established. No production promotion or upload was performed.

## Pre-production assessment

Saved candidate-versus-production crops were reviewed for `1731` (difficult/lowest-Dice case), `1942`, and `6-3004`. Both extractors localized the lower conjunctiva and produced usable square crops; differences in crop extent remain, so the old extractor output is not treated as ground truth.

For a downstream contract check, the existing ResNet18 hemoglobin predictor (`predictor/runs/resnet/best.pt`) was applied without retraining to the 27-image validation split only. Both paths produced 27 non-empty 224×224 uint8 RGB crops. Candidate crops yielded MAE 1.219 g/dL, RMSE 1.561, bias −0.783; production crops yielded MAE 1.100, RMSE 1.365, bias −0.383. Mean full-extraction runtime was 383.9 ms/image for the candidate versus 348.2 ms/image for production on this CPU. Detailed rows and quality reports are in `preproduction_assessment/rows.json` and `summary.json`.

Recommendation: **needs further fixes before deployment**. The candidate passes the segmentation/inference contract, but its downstream validation error is worse with the unchanged predictor and its end-to-end path is slower. This does not prove the candidate segmentation is intrinsically inferior—the predictor was trained on the existing ROI distribution—but it is evidence against replacing the production extractor without a controlled shadow study or predictor compatibility work. No test split was re-evaluated, and no production artifact was changed.

Reproducible commands:

```bash
python -m pytest -q extractor/test_inference.py
python -m extractor.preproduction_assessment
python -m pytest -q
```

## Paired error diagnosis (validation only)

Across the 27 paired labeled images, 11 candidate predictions improved absolute error, 16 worsened, and 0 were within 0.005 g/dL of unchanged. Mean candidate-minus-production absolute-error change was **+0.1185 g/dL**; the deterministic 20,000-resample paired bootstrap 95% interval was **−0.0993 to +0.3467 g/dL**. The median change was +0.0600 g/dL (bootstrap interval −0.0900 to +0.1900). Candidate signed error shifted from −0.3830 to −0.7830 g/dL, a −0.4000 g/dL change in bias.

Largest regressions were `1820.jpg` (+1.69 g/dL), `89H-544.jpg` (+1.25), `2745.jpg` (+0.93), and `2743.jpg` (+0.90). Largest improvements were `12-3011.jpg` (−1.26), `3267.jpg` (−0.90), and `2165.jpg` (−0.51). These are paired associations, not independent-patient estimates.

Candidate masks were slightly larger on average (mask fraction 0.007154 vs 0.007066) while candidate bounding boxes were tighter (mean width/height 123.9×52.4 vs 132.0×59.9 pixels). All 27 quality reports passed, with identical whole-image brightness/focus/clipping/cast measurements because both paths white-balance the same 512×512 source before segmentation. The three saved crop comparisons show that tighter geometry changes the amount of masked/background context; for example, `1731` candidate background fraction was 0.907 versus 0.944 for production. This supports crop-distribution shift as a plausible mechanism, but does not prove causality.

Predictor tracing found no verified channel or numeric mismatch: file decoding is BGR→RGB, crops remain uint8 RGB, `get_val_transforms()` resizes to 224×224 and converts to float [0,1], and the ResNet checkpoint applies ImageNet mean/std normalization internally. The predictor configuration and manifest builder identify the training extractor as `unet` (`ROI/roi_manifest.csv`), so the candidate is out-of-distribution relative to the predictor's training ROI generation. This is the main confirmed compatibility risk; effects of crop geometry versus masking/color statistics remain hypotheses.
