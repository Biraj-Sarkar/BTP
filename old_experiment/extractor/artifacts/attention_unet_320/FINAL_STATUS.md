# Final extractor status

Date: 2026-10-22

## Decision

**Keep the production U-Net; retain Attention U-Net as an experimental candidate; do not promote it yet.**

No model was retrained, tuned, uploaded, or deployed. The production default and checkpoint remain unchanged.

## What was built and verified

- Candidate artifact: `attention_unet_candidate.pt` (Attention U-Net, EfficientNet-B0 ImageNet initialization, 320×320 input, ImageNet normalization, threshold 0.80).
- Explicit loader: `load_attention_unet(path)`. The default `load_unet()` still resolves to `Models/UNet_extractor/unet.pt`.
- Candidate reload produced finite probability maps and matched the source checkpoint exactly (`max_abs_diff = 0.0`).
- End-to-end extraction preserved the downstream contract: 224×224, 3-channel RGB, `uint8`; empty-mask fallback and quality reports are covered by regression tests.
- Candidate-versus-production visual comparisons are saved under `comparisons/`; the difficult/failure examples were reviewed without treating production output as ground truth.

Current checkpoint hashes:

- Production `unet.pt`: `35b07d3c8bf7f36603d3b38f96ec95bfee8aac367089111046cc163d6983865c`
- Candidate artifact: `c7c8eea36c2f162297e7dc1d5683051b8090faac881a5cd98ec62588593f1498`

## Segmentation evidence

The frozen Attention U-Net held-out segmentation test result was Dice **0.865367**, IoU **0.765218**, precision **0.860222**, recall **0.881058**, specificity **0.998756**, and pixel accuracy **0.997708**. This is segmentation performance only; it is not a hemoglobin result. The 27-image test set has no patient identifiers, so patient-independent generalization is not established.

## Downstream hemoglobin evidence

On the same 27-image validation split, using the unchanged ResNet18 predictor:

| Extractor | MAE (g/dL) | RMSE | Bias | Empty masks |
|---|---:|---:|---:|---:|
| Production U-Net | 1.100 | 1.365 | −0.383 | 0/27 |
| Attention U-Net candidate | 1.219 | 1.561 | −0.783 | 0/27 |

The candidate improved absolute error on 11 images and worsened it on 16. Mean paired absolute-error change was **+0.1185 g/dL**; the paired bootstrap 95% interval was **−0.0993 to +0.3467 g/dL**. The predictor was trained using production U-Net crops, while the candidate produces tighter boxes and different masked/background context. This distribution shift is the confirmed compatibility risk; the small paired sample cannot establish a definitive causal mechanism.

## Tests and reproducibility

Relevant extractor integration tests passed:

```text
python -m pytest -q extractor/test_inference.py
4 passed in 21.63s
```

The existing Anubhav API subset previously passed 4 tests. Earlier full-suite results (not rerun for this closeout) were 56 passed and 6 pre-existing failures: four tests require the removed `algorithms/Anubhav/scripts/train_segmenter.py`, and two require the absent `ROI/roi_manifest.csv`. These failures are unrelated to the extractor integration.

Relevant artifact commands:

```bash
python -m pytest -q extractor/test_inference.py
python -m extractor.extract --input datasets/ --output ROI_attention \
  --checkpoint extractor/artifacts/attention_unet_320/attention_unet_candidate.pt
```

No segmentation test evaluation was rerun during this closeout, and no dataset split, threshold, predictor preprocessing, production configuration, or checkpoint was changed.

