"""Hb regression network and its self-describing checkpoint format.

The checkpoint carries the label standardisation constants. Without them a
saved model is useless: the network emits a standardised value, and recovering
g/dL needs the training mean and std. The inherited script saved the weights
alone, so its checkpoints could never be served.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

# ImageNet statistics. The inherited pipeline scaled to [0, 1] and stopped,
# leaving the input distribution mismatched with the pretrained weights it was
# fine-tuning from.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
}


class HbRegressor(nn.Module):
    """ResNet backbone with a scalar regression head.

    Normalisation lives inside `forward` so the serving path cannot forget it —
    a preprocessing mismatch between training and inference is invisible in
    tests and quietly destroys accuracy in production.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_until: Optional[str] = "layer3",
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")

        factory, weights_enum = BACKBONES[backbone]
        self.backbone_name = backbone
        net = factory(weights=weights_enum.DEFAULT if pretrained else None)

        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        self.net = net

        self.register_buffer("norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        if pretrained and freeze_until:
            self._freeze_through(freeze_until)

    def _freeze_through(self, layer_name: str) -> None:
        """Freeze everything up to `layer_name`, then unfreeze the rest.

        Also switches frozen BatchNorm layers to eval mode in `train()` below.
        Setting `requires_grad = False` alone does not stop BN from updating its
        running statistics, which on batches of 16 drifts the frozen features
        the transfer is supposed to preserve.
        """

        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
        if layer_name not in order:
            raise ValueError(f"freeze_until must be one of {order}")
        cutoff = order.index(layer_name)

        self.frozen_modules = []
        for name in order[: cutoff + 1]:
            module = getattr(self.net, name)
            for parameter in module.parameters():
                parameter.requires_grad = False
            self.frozen_modules.append(module)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in getattr(self, "frozen_modules", []):
            for submodule in module.modules():
                if isinstance(submodule, nn.BatchNorm2d):
                    submodule.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """`images` is float in [0, 1], shape (N, 3, H, W)."""

        normalised = (images - self.norm_mean) / self.norm_std
        return self.net(normalised)


@dataclass
class Checkpoint:
    """Everything needed to turn a photo into a g/dL number, in one file."""

    state_dict: dict
    target_mean: float
    target_std: float
    backbone: str
    model_size: Tuple[int, int]
    work_size: Tuple[int, int]
    gray_world: bool
    metrics: dict
    config: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "backbone": self.backbone,
                "model_size": list(self.model_size),
                "work_size": list(self.work_size),
                "gray_world": self.gray_world,
                "metrics": self.metrics,
                "config": self.config,
                "format_version": 1,
            },
            path,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"metrics": self.metrics, "config": self.config}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        missing = {"state_dict", "target_mean", "target_std"} - set(blob)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. Checkpoints written before the "
                "rework cannot be served — retrain to produce a complete bundle."
            )
        return cls(
            state_dict=blob["state_dict"],
            target_mean=float(blob["target_mean"]),
            target_std=float(blob["target_std"]),
            backbone=blob.get("backbone", "resnet18"),
            model_size=tuple(blob.get("model_size", (224, 224))),
            work_size=tuple(blob.get("work_size", (512, 512))),
            gray_world=bool(blob.get("gray_world", True)),
            metrics=blob.get("metrics", {}),
            config=blob.get("config", {}),
        )

    def build(self, device: Optional[torch.device] = None) -> HbRegressor:
        model = HbRegressor(self.backbone, pretrained=False, freeze_until=None)
        model.load_state_dict(self.state_dict)
        model.eval()
        if device is not None:
            model.to(device)
        return model

    def denormalise(self, standardised: np.ndarray) -> np.ndarray:
        return np.asarray(standardised) * self.target_std + self.target_mean


def to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float in [0, 1]."""

    return torch.from_numpy(np.ascontiguousarray(crop_rgb.transpose(2, 0, 1))).float() / 255.0
