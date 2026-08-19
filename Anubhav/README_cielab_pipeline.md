# CIELAB Conjunctiva Preprocessing & Hemoglobin Regression Pipeline

This README documents `cielab_pipeline.py`, a CIELAB-based conjunctiva
image pipeline for anemia detection through hemoglobin (Hb/Hgb)
regression.

## 1. What the pipeline does

The complete pipeline is:

```text
Indian eye-image dataset
        |
        v
India.xlsx (Number, Hgb)
        |
        v
Select first N labeled samples
        |
        v
80/20 train / held-out split
        |
        +-----------------------------+
        |                             |
        v                             v
Segmentation stage              Hb regression stage
        |                             |
        v                             v
Resize 512x512                 CIELAB + CLAHE on L*
        |                             |
        v                             v
CIELAB + CLAHE on L*           Segment conjunctiva
        |                             |
        v                             v
Mask2Former fine-tuning        Apply mask to LAB
        |                             |
        v                             v
Dice / IoU evaluation          Crop + square padding
        |                             |
        v                             v
Mask2Former or fallback        Resize 224x224
segmenter                            |
                                      v
                                Keep only a* and b*
                                      |
                                      v
                                2-channel ResNet18
                                      |
                                      v
                                  Hb prediction
                                      |
                                      v
                              Anemic / Normal label
```

The important design choice is that the Hb regression network uses
**only the CIELAB a\* and b\* channels**. The L\* channel is used for
lighting normalization but is discarded before regression.

The current CIELAB version also **does not use Gray World White
Balance**; the implementation preserves the color channels and applies
CLAHE only to L\*. 

---

## 2. Dataset layout

The script expects an Indian dataset roughly like:

```text
India/
├── India.xlsx
├── 1/
│   ├── image.jpg
│   ├── image_forniceal_palpebral.png
│   ├── image_forniceal.png
│   └── image_palpebral.png
├── 2/
│   ├── image.jpg
│   └── ...
└── ...
```

`India.xlsx` must contain:

  Column     Meaning

---

  `Number`   Sample identifier
  `Hgb`      Ground-truth hemoglobin

The script sorts by `Number`, takes the first `--train-limit` records,
and then uses the first 80% for training and the remaining 20% as the
held-out set.

For segmentation training masks, the script searches for:

```text
<stem>_forniceal_palpebral.png
<stem>_forniceal.png
<stem>_palpebral.png
```

Multiple available masks are combined pixelwise.


---

## 3. CIELAB preprocessing

Each image is converted from RGB to LAB.

CLAHE is applied only to the L\* channel:

```text
RGB
 |
 v
LAB
 |
 +-- L* --> CLAHE
 |
 +-- a* --> unchanged
 |
 +-- b* --> unchanged
```

The implementation uses:

```text
clipLimit = 2.0
tileGridSize = (8, 8)
```

The normalized LAB image can also be converted back to RGB for
visualization. 

---

## 4. Conjunctiva segmentation

### Mask2Former path

The default checkpoint is:

```text
facebook/mask2former-swin-tiny-cityscapes-semantic
```

The segmentation model is configured for 2 labels and fine-tuned using
available image/mask pairs.

Default segmentation settings:

```text
Training image size: 512 x 512
Epochs:              3
Batch size:          2
Learning rate:       5e-5
Optimizer:           AdamW
```

The trained model and processor are saved as:

```text
<output-root>/mask2former_model/
<output-root>/mask2former_processor/
```


### Fallback path

Mask2Former is loaded with `local_files_only=True`. If it is unavailable
offline, the code falls back to heuristic segmentation.


The heuristic segmentation:

1. Converts the image to LAB.
2. Computes an L\* threshold.
3. Computes a*/b* chroma distance from their medians.
4. Builds a candidate mask.
5. Smooths it using morphology.
6. Keeps the largest connected component.

If the mask area is implausible, GrabCut is attempted as a fallback.


---

## 5. Segmentation evaluation

When Mask2Former is available, the held-out segmentation images are
evaluated using:

- **Dice score**
- **Intersection over Union (IoU)**

The metrics are saved to:

```text
<output-root>/evaluation_metrics.json
```

If Mask2Former is unavailable offline, learned segmentation evaluation
is skipped and the code records zero-valued segmentation metrics.


---

## 6. Mask post-processing

After segmentation, the mask is:

1. Morphologically opened.
2. Morphologically closed.
3. Reduced to its largest connected component.
4. Converted into a tight bounding box with padding.
5. Cropped and padded to a square.

This gives the regression stage a focused conjunctiva region.


---

## 7. Image quality checks

The demo stage computes:

### Focus score

The focus score is the variance of the grayscale Laplacian.

Default acceptance threshold:

```text
35.0
```

### Exposure score

The exposure score is the standard deviation of L\*.

### Mask ratio

```text
foreground pixels / total pixels
```

Default acceptable range:

```text
0.015 to 0.85
```

An image is marked `accepted_for_training` when the focus threshold and
mask-ratio limits are satisfied. 

---

## 8. Hb regression preprocessing

The Hb regression path is:

```text
RGB
 |
 v
Resize to 320x320
 |
 v
CIELAB + CLAHE on L*
 |
 v
Conjunctiva segmentation
 |
 v
Apply mask to LAB
 |
 v
Crop + square padding
 |
 v
Resize to 224x224
 |
 v
Discard L*
 |
 v
Keep a* + b*
 |
 v
2-channel tensor
```

The code explicitly extracts channels 1 and 2 from LAB, corresponding to
a\* and b*, and discards L*.

## 9. ResNet18 Hb model

The regression model is based on ResNet18.

Original:

```text
3-channel RGB input
```

Modified:

```text
2-channel a*/b* input
```

The first convolution is changed from 3 input channels to 2. When
pretrained weights are used, the original three-channel convolution
weights are averaged and duplicated into the two new channels.


The final classifier is replaced with:

```text
Dropout(0.3)
Linear(..., 1)
```

so the network outputs one continuous Hb value. When pretrained weights
are used, the implementation freezes most of the backbone while leaving
the new first convolution, `layer4`, and final `fc` trainable.


---

## 10. Hb normalization and training

Hb targets are standardized using:

```text
normalized_Hgb = (Hgb - mean) / std
```

Predictions are converted back to the original Hb scale for evaluation.


Default regression settings:

```text
Epochs:       25
Batch size:    8
Learning rate: 1e-4
Image size:   224 x 224
```

The model uses AdamW and a ReduceLROnPlateau scheduler.
fileciteturn0file0L73-L79 fileciteturn0file0L561-L569

The training loss is weighted Smooth L1 loss. Sample weights are derived
from Hb-value histogram frequencies to give more weight to less
represented Hb ranges. fileciteturn0file0L825-L836
fileciteturn0file0L574-L594

Training augmentation can include:

- Horizontal flip, probability 0.5
- Rotation, probability 0.35, between -8° and +8°
- Gaussian blur, probability 0.2

fileciteturn0file0L759-L781

---

## 11. Hb evaluation

The held-out regression set is evaluated with:

### MAE

```text
mean(abs(prediction - target))
```

### RMSE

```text
sqrt(mean((prediction - target)^2))
```

Results are written to:

```text
<output-root>/hb_regression_metrics.json
```

Per-image held-out predictions are written to:

```text
<output-root>/hb_regression_predictions.csv
```

The trained model is saved to:

```text
<output-root>/hb_regression_model.pt
```

fileciteturn0file0L619-L652

---

## 12. Anemia classification

The prediction package converts continuous Hb predictions into a binary
label using the threshold implemented in the script:

```text
predicted Hgb < 11.0  -> Anemic
predicted Hgb >= 11.0 -> Normal
```

The same threshold is applied to the ground-truth Hb values for
`actual_anemia`. 

---

## 13. Output structure

A typical output directory is:

```text
cielab_pipeline_outputs/
├── pipeline_config.json
├── mask2former_model/
├── mask2former_processor/
├── evaluation_metrics.json
├── hb_regression_model.pt
├── hb_regression_metrics.json
├── hb_regression_predictions.csv
├── raw/
├── masks/
├── overlays/
├── crops/
├── normalized/
├── final_prediction_package/
│   ├── original_images/
│   ├── masks/
│   ├── overlays/
│   ├── normalized_a_channel/
│   └── all_predictions.csv
├── manifest.csv
└── contact_sheet.png
```

Important outputs:

---

  File                                             Purpose

---

  `pipeline_config.json`                           Configuration used for the run

  `evaluation_metrics.json`                        Segmentation Dice/IoU

  `hb_regression_model.pt`                         Hb regression model weights

  `hb_regression_metrics.json`                     Held-out Hb MAE/RMSE

  `hb_regression_predictions.csv`                  Held-out Hb predictions

  `manifest.csv`                                   Demo preprocessing and quality
                                                   information

  `contact_sheet.png`                              Visual summary of demo
                                                   preprocessing

`final_prediction_package/all_predictions.csv`   Predictions for all regression
                                                   records
----------------------------------------------------------

The output paths are created and populated by the pipeline during
execution. fileciteturn0file0L689-L710
fileciteturn0file0L1103-L1118

---

## 14. Installation

Create a virtual environment:

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the packages imported by the script:

```bash
pip install numpy pandas opencv-python torch torchvision transformers
```

The script imports NumPy, pandas, OpenCV, PyTorch, torchvision, and
Hugging Face Transformers. fileciteturn0file0L30-L59

---

## 15. Run instructions

The script accepts these command-line arguments:

```text
--dataset-root
--output-root
--train-limit
--demo-limit
--segmentation-checkpoint
--train-epochs
--train-batch-size
--learning-rate
--blur-threshold
```

The defaults are defined in the script.
fileciteturn0file0L1121-L1163

### Basic run

```bash
python cielab_pipeline.py \
  --dataset-root "/path/to/India" \
  --output-root "./cielab_pipeline_outputs"
```

### Small smoke test

Before doing a longer training run, use:

```bash
python cielab_pipeline.py \
  --dataset-root "/path/to/India" \
  --output-root "./cielab_pipeline_outputs_test" \
  --train-limit 10 \
  --demo-limit 3 \
  --train-epochs 1
```

### Default configuration

```bash
python cielab_pipeline.py \
  --dataset-root "/path/to/India" \
  --output-root "./cielab_pipeline_outputs"
```

The script saves the effective configuration to `pipeline_config.json`.
fileciteturn0file0L689-L710

---

## 16. CLI defaults

---

  Argument                            Default

---

  `--dataset-root`                    `/Users/yatikajena/Desktop/AnemiaDetection/dataset anemia/India`

  `--output-root`                     `/Users/yatikajena/Desktop/AnemiaDetection/cielab_pipeline_outputs_v1`

  `--train-limit`                     `50`

  `--demo-limit`                      `5`

  `--segmentation-checkpoint`         `facebook/mask2former-swin-tiny-cityscapes-semantic`

  `--train-epochs`                    `3`

  `--train-batch-size`                `2`

  `--learning-rate`                   `5e-5`

`--blur-threshold`                  `35.0`
------------------

These are the command-line defaults currently implemented in
`parse_args()`. fileciteturn0file0L1121-L1146

Note that the Hb regression settings (25 epochs, batch size 8, learning
rate `1e-4`) are currently part of `PipelineConfig` and are **not
exposed as command-line arguments**. fileciteturn0file0L61-L79

---

## 17. Device behavior

The implementation uses:

```text
Mask2Former: CPU
Hb regression: MPS if available, otherwise CPU
```

This is explicitly selected in `run_demo()`.
fileciteturn0file0L1016-L1024

---

## 18. Hugging Face cache

The script creates a local Hugging Face cache next to the Python file:

```text
.hf_cache/
```

and sets:

```text
HF_HOME=.hf_cache
HUGGINGFACE_HUB_CACHE=.hf_cache/hub
```

This keeps the model cache inside the workspace.
fileciteturn0file0L26-L28

Because Mask2Former loading uses `local_files_only=True`, a suitable
local checkpoint or previously saved model must be available for the
learned segmentation path. Otherwise, the implemented fallback is
heuristic segmentation. fileciteturn0file0L400-L425

---

## 19. End-to-end execution checklist

When you run the script, it:

1. Loads `India.xlsx`.
2. Validates the `Number` and `Hgb` columns.
3. Locates the corresponding JPG images.
4. Selects up to `--train-limit` records.
5. Splits them into training and held-out sets.
6. Loads available segmentation masks.
7. Resizes images for segmentation training.
8. Applies CIELAB conversion and L\*-only CLAHE.
9. Fine-tunes Mask2Former if locally available.
10. Evaluates segmentation with Dice and IoU.
11. Uses Mask2Former or the implemented fallback segmenter.
12. Produces conjunctiva masks, overlays, crops, and normalized images.
13. Prepares LAB images for Hb regression.
14. Keeps only a\* and b\* channels.
15. Trains the modified 2-channel ResNet18.
16. Evaluates Hb prediction using MAE and RMSE.
17. Converts predictions to Anemic/Normal.
18. Saves models, metrics, predictions, manifests, and visual outputs.

---

## 20. Quick start

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas opencv-python torch torchvision transformers

# Smoke test
python cielab_pipeline.py \
  --dataset-root "/path/to/India" \
  --output-root "./cielab_pipeline_outputs_test" \
  --train-limit 10 \
  --demo-limit 3 \
  --train-epochs 1

# Full/default run
python cielab_pipeline.py \
  --dataset-root "/path/to/India" \
  --output-root "./cielab_pipeline_outputs"
```

## Source of truth

This README describes the behavior currently implemented in
`cielab_pipeline.py`; it does not assume additional pipeline stages that
are not present in the uploaded script.
