"""Export trained Hemoglobin Predictor artifacts and publish to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

from .training import load_checkpoint


def generate_model_card(
    architecture: str,
    variant: str,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    repo_id: str,
) -> str:
    """Generates markdown Model Card adhering to clinical ML documentation standards."""
    mae = metrics.get("mae_g_dl", "N/A")
    rmse = metrics.get("rmse_g_dl", "N/A")
    r2 = metrics.get("r2", "N/A")
    pearson = metrics.get("pearson_r", "N/A")
    scr = metrics.get("screening", {})
    sens = scr.get("sensitivity", "N/A")
    spec = scr.get("specificity", "N/A")

    return f"""---
language: en
tags:
- medical
- computer-vision
- anemia-detection
- hemoglobin-prediction
- {architecture}
license: apache-2.0
datasets:
- clinical-conjunctiva-cohort
metrics:
- mae
- rmse
- r_squared
- pearson_r
---

# {architecture.upper()} ({variant}) Hemoglobin Predictor

Non-invasive hemoglobin (Hb) concentration estimation model predicting continuous Hb in g/dL from 224×224 palpebral/forniceal conjunctival region-of-interest (ROI) crops.

## Model Summary

- **Architecture:** {architecture.upper()} ({variant})
- **Task:** Continuous Hemoglobin Regression (g/dL)
- **Input:** 224×224 RGB conjunctival crop
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Target Denormalization:** Internal z-score converted via target mean and standard deviation

## Held-Out Test Set Performance

The evaluation was performed on strictly held-out patient splits with zero patient leakage:

| Metric | Value |
|---|---|
| **MAE** | {mae} g/dL |
| **RMSE** | {rmse} g/dL |
| **R²** | {r2} |
| **Pearson Correlation (r)** | {pearson} |
| **Clinical Sensitivity (Anemia)** | {sens} |
| **Clinical Specificity** | {spec} |

## Intended Use & Clinical Limitations

> [!IMPORTANT]
> **Research and investigational use only.** This model is not approved as a primary diagnostic device. Non-invasive conjunctival imaging provides supportive screening evidence and must be corroborated by standardized laboratory complete blood count (CBC) or clinical hematology analysis.

## Usage

```python
import torch
from predictor.training import load_checkpoint

model, bundle = load_checkpoint("best.pt")
# input_tensor: (1, 3, 224, 224) RGB float in [0, 1]
hb_g_dl = model.predict_hb(input_tensor)
print(f"Predicted Hb: {{hb_g_dl.item():.2f}} g/dL")
```
"""


def export_artifact(
    checkpoint_path: Path,
    export_dir: Path,
    repo_id: str | None = None,
    push_to_hub: bool = False,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    model, bundle = load_checkpoint(checkpoint_path)

    arch = bundle.get("architecture", "resnet")
    variant = bundle.get("variant", "resnet18")
    val_metrics = bundle.get("val_metrics", {})
    cfg = bundle.get("config", {})

    # Copy weights
    target_weights = export_dir / "best.pt"
    shutil.copy2(checkpoint_path, target_weights)

    # Save export config
    export_config = {
        "format": bundle.get("format"),
        "version": bundle.get("version"),
        "architecture": arch,
        "variant": variant,
        "input_size": bundle.get("input_size", [224, 224]),
        "image_mean": bundle.get("image_mean"),
        "image_std": bundle.get("image_std"),
        "target_mean": bundle.get("target_mean"),
        "target_std": bundle.get("target_std"),
        "validation_metrics": val_metrics,
        "parameters": model.count_parameters(),
    }
    (export_dir / "config.json").write_text(json.dumps(export_config, indent=2), encoding="utf-8")

    # Generate Model Card
    readme_content = generate_model_card(
        architecture=arch,
        variant=variant,
        metrics=val_metrics,
        config=cfg,
        repo_id=repo_id or f"anemia-{arch}-predictor",
    )
    (export_dir / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Successfully packaged model artifact to {export_dir}")

    # Optional Push to Hugging Face Hub
    if push_to_hub and repo_id:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("HF_TOKEN environment variable not found. Skipping remote push.")
            return export_dir
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(export_dir),
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Uploaded artifacts to Hugging Face Hub: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Failed to upload to Hugging Face Hub: {e}")

    return export_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictor model for deployment and Hugging Face.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--output", type=Path, default=Path("exported_model"), help="Export output directory")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face model repository ID")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    args = parser.parse_args()

    export_artifact(
        checkpoint_path=args.checkpoint,
        export_dir=args.output,
        repo_id=args.repo_id,
        push_to_hub=args.push,
    )


if __name__ == "__main__":
    main()

