# Hemoglobin Predictor Pipeline

Reproducible, scientifically defensible Hemoglobin (Hb) regression framework supporting three interchangeable deep neural network architectures:

1. **ResNet** (ResNet-18 / ResNet-34)
2. **VGG** (VGG-16 / VGG-16 with BatchNorm)
3. **MobileNet** (MobileNetV3-Large / MobileNetV3-Small)

## Architecture & Dataflow

```
ROI Images (224×224) + Reference Hb Labels (g/dL)
                    ↓
     Manifest Validation & Quality Gate
                    ↓
 Patient-Level Grouped Splitting (Zero Patient Leakage)
                    ↓
 Target Standardization: z = (Hb - mean_train) / std_train
                    ↓
 Architecture-Agnostic Model Training (Huber/Smooth L1 Loss)
                    ↓
 Validation-Based Checkpoint Selection (Val MAE g/dL)
                    ↓
 Evaluation on Strictly Held-Out Test Patients
                    ↓
 Inverse Target Denormalization: Hb = z * std_train + mean_train
                    ↓
 Comparative Multi-Model Evaluation & Selection
                    ↓
 Hugging Face Hub / Backend Inference Export
```

## Quick Start Commands

Run all commands from the repository root:

### 1. Train ResNet
```bash
python -m predictor.train --config predictor/configs/resnet.yaml
```

### 2. Train VGG
```bash
python -m predictor.train --config predictor/configs/vgg.yaml
```

### 3. Train MobileNet
```bash
python -m predictor.train --config predictor/configs/mobilenet.yaml
```

### 4. Compare All Three Architectures
```bash
python -m predictor.compare_models
```

### 5. Run Single Inference
```bash
python -m predictor.predict --roi ROI/captured_123H-577.png --checkpoint predictor/runs/resnet/best.pt --sex male
```

### 6. Export Artifacts for Deployment
```bash
python -m predictor.export_huggingface --checkpoint predictor/runs/resnet/best.pt --output Models/selected_predictor
```

