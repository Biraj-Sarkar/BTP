"""Shared, reproducible segmentation-training defaults."""
from dataclasses import dataclass
@dataclass(frozen=True)
class SegmentationConfig:
    input_size: int = 320
    encoder: str = "efficientnet-b0"
    pretrained: bool = False
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    threshold: float = 0.5
    min_area_fraction: float = 0.002
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
