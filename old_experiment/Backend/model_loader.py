"""Resolve the approved production model artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNET_PATH = SUITE_ROOT / "Models" / "UNet_extractor" / "unet.pt"
DEFAULT_PREDICTOR_PATH = SUITE_ROOT / "predictor" / "runs" / "resnet" / "best.pt"


def resolve_unet_path() -> Path:
    """Return the approved production U-Net checkpoint."""
    if DEFAULT_UNET_PATH.is_file():
        return DEFAULT_UNET_PATH
    raise RuntimeError(f"Production U-Net checkpoint not found: {DEFAULT_UNET_PATH}")


def resolve_predictor_path() -> Path:
    """Return the approved production ResNet18 checkpoint."""
    if DEFAULT_PREDICTOR_PATH.is_file():
        return DEFAULT_PREDICTOR_PATH
    raise RuntimeError(f"Production predictor checkpoint not found: {DEFAULT_PREDICTOR_PATH}")
