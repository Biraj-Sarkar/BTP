"""Common regression interface and base classes for Hemoglobin predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class BaseHbRegressor(nn.Module, ABC):
    """Abstract base class for all Hemoglobin regression models.
    
    Ensures a standardized interface across ResNet, VGG, and MobileNet.
    Manages image input normalization and target z-score denormalization as registered buffers.
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        # Register normalization buffers
        self.register_buffer("norm_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))

    def set_target_stats(self, mean: float, std: float) -> None:
        """Set the target standardization statistics computed from training cohort."""
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(float(std))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet channel normalization."""
        return (x - self.norm_mean) / self.norm_std

    def denormalize_target(self, pred_z: torch.Tensor) -> torch.Tensor:
        """Convert standardized z-score back to physical Hemoglobin in g/dL."""
        return pred_z * self.target_std + self.target_mean

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial/pooled features from the backbone."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning standardized z-score predictions (batch_size, 1)."""
        pass

    @torch.inference_mode()
    def predict_hb(self, x: torch.Tensor) -> torch.Tensor:
        """Inference entry point returning continuous Hb predictions in g/dL."""
        self.eval()
        z = self.forward(x)
        return self.denormalize_target(z)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

