"""Loading and initialization of the fine-tuned U-Net conjunctiva segmentation model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import segmentation_models_pytorch as smp

DEFAULT_UNET_PATH = (
    Path(__file__).resolve().parents[1]
    / "Models"
    / "UNet_extractor"
    / "unet.pt"
)

BUNDLE_FORMAT = "anemia-segmenter"
BUNDLE_VERSION = 1


@dataclass(frozen=True)
class UNetConfig:
    encoder: str = "efficientnet-b0"
    input_size: int = 320
    threshold: float = 0.5
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    min_area_fraction: float = 0.002

    @classmethod
    def from_dict(cls, data: dict) -> "UNetConfig":
        return cls(
            encoder=str(data.get("encoder", "efficientnet-b0")),
            # Training checkpoints historically stored this as ``size``;
            # accept both names so inference uses the trained resolution.
            input_size=int(data.get("input_size", data.get("size", 320))),
            threshold=float(data.get("threshold", 0.5)),
            mean=tuple(float(v) for v in data.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(v) for v in data.get("std", (0.229, 0.224, 0.225))),
            min_area_fraction=float(data.get("min_area_fraction", 0.002)),
        )


class UNetSegmenter:
    """Wrapper around trained EfficientNet-B0 U-Net for conjunctiva segmentation."""

    def __init__(self, checkpoint_path: Optional[Path | str] = None, device: Optional[torch.device | str] = None):
        if checkpoint_path is None:
            checkpoint_path = os.getenv("UNET_CHECKPOINT_PATH", str(DEFAULT_UNET_PATH))
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"U-Net checkpoint not found at {self.checkpoint_path}. "
                "Ensure Models/UNet_extractor/unet.pt exists."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        bundle = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._validate_bundle(bundle)

        bundle_config = bundle.get("config", {})
        self.config = UNetConfig.from_dict(bundle_config)
        self.provenance = bundle.get("provenance", {})
        self.architecture = str(bundle.get("architecture", "unet"))
        if self.architecture in {"unet", "unet++", "unetplusplus", "attention", "attention_unet"}:
            from .training.models import build_model
            self.model = build_model(self.architecture, self.config.encoder, pretrained=False)
        else:
            self.model = smp.Unet(encoder_name=self.config.encoder, encoder_weights=None, in_channels=3, classes=1)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _validate_bundle(self, bundle: object) -> None:
        if not isinstance(bundle, dict):
            raise ValueError(f"Invalid bundle at {self.checkpoint_path}: expected dictionary")
        if "state_dict" not in bundle:
            raise ValueError(f"Invalid bundle at {self.checkpoint_path}: missing state_dict")
        architecture = str(bundle.get("architecture", "unet"))
        supported = {"unet", "unet++", "unetplusplus", "attention", "attention_unet"}
        if architecture not in supported:
            raise ValueError(f"Unsupported segmentation architecture in {self.checkpoint_path}: {architecture}")

    @torch.inference_mode()
    def predict_probability(self, image_rgb: torch.Tensor | np.ndarray) -> np.ndarray:
        """Infers per-pixel conjunctiva probability at original image resolution."""
        import cv2
        import numpy as np

        if isinstance(image_rgb, torch.Tensor):
            img_np = image_rgb.cpu().numpy()
        else:
            img_np = image_rgb

        h, w = img_np.shape[:2]
        size = self.config.input_size

        resized = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - np.array(self.config.mean, dtype=np.float32)) / np.array(self.config.std, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(norm.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob


def load_unet(checkpoint_path: Optional[Path | str] = None, device: Optional[torch.device | str] = None) -> UNetSegmenter:
    """Factory function to load and initialize UNetSegmenter."""
    return UNetSegmenter(checkpoint_path=checkpoint_path, device=device)


def load_attention_unet(checkpoint_path: Path | str, device: Optional[torch.device | str] = None) -> UNetSegmenter:
    """Load an Attention U-Net candidate explicitly; never changes the default model."""
    segmenter = UNetSegmenter(checkpoint_path=checkpoint_path, device=device)
    if segmenter.architecture not in {"attention", "attention_unet"}:
        raise ValueError(f"Expected an Attention U-Net checkpoint, got {segmenter.architecture!r}")
    return segmenter

