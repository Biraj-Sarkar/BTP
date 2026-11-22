"""MobileNet architecture implementation for Hemoglobin regression."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    mobilenet_v3_small,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

from .common import BaseHbRegressor


class MobileNetRegressor(BaseHbRegressor):
    """Pretrained MobileNetV3 backbone with a lightweight regression head for efficient Hb estimation."""

    def __init__(
        self,
        variant: str = "mobilenet_v3_large",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_features: bool = False,
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
        self.freeze_features = freeze_features

        if variant == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = mobilenet_v3_large(weights=weights)
        elif variant == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = mobilenet_v3_small(weights=weights)
        else:
            raise ValueError(f"Unsupported MobileNet variant: {variant}")

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # MobileNet classifier structure has linear -> hardswish -> dropout -> linear
        in_features = base_model.classifier[0].in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(x)
        feat_map = self.features(normalized)
        pooled = self.avgpool(feat_map)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(x)
        z = self.head(pooled)
        return z

