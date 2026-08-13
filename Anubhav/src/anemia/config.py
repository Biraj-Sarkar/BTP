"""Central configuration for the anemia pipeline.

Everything tunable lives here so that experiments are reproducible from a single
serialised blob: `Config.to_dict()` is written next to every trained checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreprocessConfig:
    """Image conditioning applied identically at train time and serve time.

    Any change here invalidates cached crops and existing checkpoints, so the
    values are hashed into the cache key (see `anemia.cache`).
    """

    work_size: Tuple[int, int] = (512, 512)
    """Resolution the segmenter sees. Larger than the regressor input because
    thin conjunctiva bands lose definition when downsampled too early."""

    model_size: Tuple[int, int] = (224, 224)
    """Final square crop fed to the regressor."""

    bbox_pad_ratio: float = 0.08
    gray_world: bool = True


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for rejecting non-viable captures.

    These are *enforced*, not merely recorded: a sample failing QC is dropped
    from training, and at serve time the API returns a retake prompt rather
    than a confident number derived from a blurred photo.
    """

    min_focus: float = 35.0
    """Variance of the Laplacian. Below this the capture is too blurred."""

    min_mask_ratio: float = 0.015
    max_mask_ratio: float = 0.85
    enforce: bool = True


@dataclass(frozen=True)
class SegmentationConfig:
    checkpoint: str = "facebook/mask2former-swin-tiny-ade-semantic"
    """ADE20k-semantic init — a broader scene prior than street-scene
    checkpoints, and a closer match for close-up tissue."""

    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 5e-5
    mask_kind: str = "palpebral_forniceal"
    """Which ground-truth mask to train against: `palpebral`, `forniceal`, or
    `palpebral_forniceal` (the union)."""


@dataclass(frozen=True)
class RegressionConfig:
    backbone: str = "resnet18"
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    freeze_until: str = "layer3"
    """Layers up to and including this one stay frozen; the rest fine-tunes."""

    imbalance_bins: int = 10
    imbalance_strength: float = 1.0
    """0.0 disables inverse-frequency weighting; 1.0 is full inverse frequency."""

    augment: bool = True
    early_stopping_patience: int = 8


@dataclass(frozen=True)
class SplitConfig:
    folds: int = 5
    """Patient-grouped cross-validation. With <1000 images a single holdout
    gives an MAE whose confidence interval is too wide to report honestly."""

    holdout_fraction: float = 0.2
    stratify_bins: int = 4
    seed: int = 1337


@dataclass(frozen=True)
class Config:
    data_roots: Tuple[Path, ...] = ()
    output_root: Path = REPO_ROOT / "runs"
    cache_root: Path = REPO_ROOT / ".cache" / "crops"

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    seed: int = 1337

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["data_roots"] = [str(path) for path in self.data_roots]
        payload["output_root"] = str(self.output_root)
        payload["cache_root"] = str(self.cache_root)
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
