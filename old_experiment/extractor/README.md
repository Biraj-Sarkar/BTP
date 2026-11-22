# Conjunctival ROI Extractor

Standalone segmentation and region-of-interest extraction module for eye photographs, targeting the palpebral and forniceal conjunctiva.

## Pipeline Architecture

```
Raw Eye Image
    ↓
Standardize Resolution (512×512)
    ↓
Gray-World White Balance (Illumination Invariance)
    ↓
U-Net Segmentation Inference (EfficientNet-B0 Backbone, 320×320)
    ↓
Morphological Cleanup (Largest Connected Component + Hole Filling)
    ↓
Quality Control Auditing (Focus Blur, Mask Ratio, Exposure, Color Cast)
    ↓
Tight Bounding Box (+8% Padding)
    ↓
Background Masking & Symmetrical Pad-to-Square
    ↓
Resize to Standard Predictor Dimensions (224×224)
```

## Checkpoint

The pre-trained U-Net weights are stored at:
`Models/UNet_extractor/unet.pt`

- **Architecture:** U-Net with `efficientnet-b0` encoder
- **Input:** 320×320 RGB normalized with ImageNet mean and std
- **Output:** Sigmoid probability map thresholded at 0.5
- **Minimum area fraction:** 0.002

## Usage

### In Python

```python
from extractor import ROIExtractor

extractor = ROIExtractor()
result = extractor.extract_from_path("path/to/eye.jpg")

print("Quality passed:", result.passed)
print("Bounding box:", result.bbox)
print("ROI Crop shape:", result.crop.shape)  # (224, 224, 3)
```

### CLI Batch Extraction

```bash
python -m extractor.extract --input datasets/ --output ROI/
```

### Explicit Attention U-Net candidate

The validated Attention U-Net is packaged separately at
`extractor/artifacts/attention_unet_320/attention_unet_candidate.pt`. It uses
320×320 input, ImageNet normalization, and threshold 0.80. Select it explicitly
so the existing production default is unchanged:

```bash
python -m extractor.extract --input datasets/ --output ROI_attention \
  --checkpoint extractor/artifacts/attention_unet_320/attention_unet_candidate.pt
```

The candidate's validation/test evidence and limitations are documented in its
`integration_report.md`; the 27-image test set has no patient identifiers.

