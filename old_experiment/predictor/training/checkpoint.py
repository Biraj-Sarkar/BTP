"""Self-contained checkpoint saving and loading for Hemoglobin predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torchvision
from ..models import build_model, BaseHbRegressor

CHECKPOINT_FORMAT = "anemia-hb-predictor"
CHECKPOINT_VERSION = 1


def save_checkpoint(
    model: BaseHbRegressor,
    path: Path | str,
    architecture: str,
    variant: str,
    epoch: int,
    val_metrics: Dict[str, Any],
    config: Dict[str, Any],
    audit_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an inference-ready self-contained checkpoint bundle."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "architecture": architecture,
        "variant": variant,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "input_size": [224, 224],
        "image_mean": model.norm_mean.squeeze().cpu().tolist(),
        "image_std": model.norm_std.squeeze().cpu().tolist(),
        "target_mean": float(model.target_mean.item()),
        "target_std": float(model.target_std.item()),
        "val_metrics": val_metrics,
        "config": config,
        "audit": audit_summary or {},
        "env": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }

    torch.save(bundle, target_path)


def load_checkpoint(
    checkpoint_path: Path | str,
    device: Optional[torch.device | str] = None,
) -> Tuple[BaseHbRegressor, Dict[str, Any]]:
    """Loads model architecture and restores weights, normalization buffers, and metadata."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    bundle = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(f"Invalid checkpoint format at {path}")

    arch = bundle.get("architecture", "resnet")
    variant = bundle.get("variant", "resnet18")
    target_mean = float(bundle.get("target_mean", 0.0))
    target_std = float(bundle.get("target_std", 1.0))
    image_mean = tuple(bundle.get("image_mean", [0.485, 0.456, 0.406]))
    image_std = tuple(bundle.get("image_std", [0.229, 0.224, 0.225]))

    model = build_model(
        architecture=arch,
        variant=variant,
        pretrained=False,  # Weights loaded from state_dict
        target_mean=target_mean,
        target_std=target_std,
        image_mean=image_mean,
        image_std=image_std,
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()

    return model, bundle

