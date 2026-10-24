"""VGG architecture implementation for Hemoglobin regression."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights

from .common import BaseHbRegressor


class VGGRegressor(BaseHbRegressor):
    """Pretrained VGG-16 backbone with a lightweight regression head for Hb estimation."""

    def __init__(
        self,
        variant: str = "vgg16_bn",
        pretrained: bool = True,
        dropout: float = 0.3,
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

        if variant == "vgg16_bn":
            weights = VGG16_BN_Weights.DEFAULT if pretrained else None
            base_model = vgg16_bn(weights=weights)
        elif variant == "vgg16":
            weights = VGG16_Weights.DEFAULT if pretrained else None
            base_model = vgg16(weights=weights)
        else:
            raise ValueError(f"Unsupported VGG variant: {variant}")

        self.features = base_model.features
        # Use Global Adaptive Average Pooling to reduce spatial dimensions from 7x7 to 1x1,
        # preventing parameter explosion and overfitting on small clinical cohorts
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact regression head (512 -> 128 -> 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
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

