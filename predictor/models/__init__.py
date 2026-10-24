"""Model factory and registry for Hemoglobin predictors."""

from __future__ import annotations

from typing import Dict, Any, Type
from .common import BaseHbRegressor
from .resnet import ResNetRegressor
from .vgg import VGGRegressor
from .mobilenet import MobileNetRegressor

MODEL_REGISTRY: Dict[str, Type[BaseHbRegressor]] = {
    "resnet": ResNetRegressor,
    "resnet18": ResNetRegressor,
    "resnet34": ResNetRegressor,
    "vgg": VGGRegressor,
    "vgg16": VGGRegressor,
    "vgg16_bn": VGGRegressor,
    "mobilenet": MobileNetRegressor,
    "mobilenet_v3_large": MobileNetRegressor,
    "mobilenet_v3_small": MobileNetRegressor,
}


def build_model(
    architecture: str,
    variant: str | None = None,
    pretrained: bool = True,
    dropout: float = 0.3,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    **kwargs,
) -> BaseHbRegressor:
    """Instantiate a Hemoglobin regressor by architecture name."""
    arch_lower = architecture.lower()
    if arch_lower in {"resnet", "resnet18", "resnet34"}:
        return ResNetRegressor(
            variant=variant or ("resnet18" if arch_lower == "resnet" else arch_lower),
            pretrained=pretrained,
            dropout=dropout,
            target_mean=target_mean,
            target_std=target_std,
            **kwargs,
        )
    elif arch_lower in {"vgg", "vgg16", "vgg16_bn"}:
        return VGGRegressor(
            variant=variant or ("vgg16_bn" if arch_lower == "vgg" else arch_lower),
            pretrained=pretrained,
            dropout=dropout,
            target_mean=target_mean,
            target_std=target_std,
            **kwargs,
        )
    elif arch_lower in {"mobilenet", "mobilenet_v3_large", "mobilenet_v3_small"}:
        return MobileNetRegressor(
            variant=variant or ("mobilenet_v3_large" if arch_lower == "mobilenet" else arch_lower),
            pretrained=pretrained,
            dropout=dropout,
            target_mean=target_mean,
            target_std=target_std,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )


__all__ = [
    "BaseHbRegressor",
    "ResNetRegressor",
    "VGGRegressor",
    "MobileNetRegressor",
    "build_model",
    "MODEL_REGISTRY",
]

