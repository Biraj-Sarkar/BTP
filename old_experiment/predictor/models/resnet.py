"""ResNet architecture implementation for Hemoglobin regression."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

from .common import BaseHbRegressor


class ResNetRegressor(BaseHbRegressor):
    """Pretrained ResNet backbone with a continuous regression head for Hb estimation."""

    def __init__(
        self,
        variant: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_until: Optional[str] = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__(
            target_mean=target_mean,
            target_std=target_std,
            image_mean=image_mean,
            image_std=image_std,
        )
        self.variant = variant
        self.dropout_rate = dropout
        self.freeze_until = freeze_until

        if variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        elif variant == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

        if freeze_until:
            self._apply_freezing(freeze_until)

    def _apply_freezing(self, freeze_until: str) -> None:
        """Freeze early convolutional stages up to the specified layer name."""
        layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        if freeze_until not in layer_order:
            raise ValueError(f"Unknown freeze_until target: {freeze_until}")

        target_idx = layer_order.index(freeze_until)
        for idx, name in enumerate(layer_order[:target_idx + 1]):
            layer = getattr(self.backbone, name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(x)
        features = self.backbone(normalized)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        z = self.head(features)
        return z

