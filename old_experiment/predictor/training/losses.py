"""Loss functions for Hemoglobin regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss with optional sample reweighting for lower/anemic Hb values."""

    def __init__(self, beta: float = 1.0, severe_hb_weight: float = 1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")
        self.severe_hb_weight = severe_hb_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, raw_hb: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.loss_fn(pred, target)
        if self.severe_hb_weight > 1.0 and raw_hb is not None:
            # Upweight samples with severe/moderate anemia (Hb < 11.0 g/dL)
            weights = torch.where(raw_hb < 11.0, self.severe_hb_weight, 1.0).view_as(loss)
            loss = loss * weights
        return loss.mean()


def get_loss_fn(loss_name: str = "smooth_l1", beta: float = 1.0, severe_hb_weight: float = 1.0) -> nn.Module:
    name = loss_name.lower()
    if name in {"smooth_l1", "huber"}:
        return WeightedSmoothL1Loss(beta=beta, severe_hb_weight=severe_hb_weight)
    elif name in {"mse", "l2"}:
        return nn.MSELoss()
    elif name in {"l1", "mae"}:
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function '{loss_name}'")

