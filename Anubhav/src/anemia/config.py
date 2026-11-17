"""Central configuration for the anemia pipeline.

Everything tunable lives here so that experiments are reproducible from a single
serialised blob: `Config.to_dict()` is written next to every trained checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

WHITE_BALANCE_METHODS = ("white_patch", "gray_world", "none")


def white_balance_from_record(record: Optional[dict], default: str = "gray_world") -> str:
    """Read the colour correction a saved model or config was made with.

    Newer files say `white_balance`. Older ones carry only the `gray_world`
    switch (absent meant on), which maps to "gray_world" or "none".
    """

    record = record or {}
    if record.get("white_balance"):
        return str(record["white_balance"])
    if "gray_world" in record:
        return "gray_world" if record["gray_world"] else "none"
    return default


@dataclass(frozen=True)
class PreprocessConfig:
    """Image conditioning applied identically at train time and serve time.

    Any change here invalidates cached crops and existing checkpoints, so the
    values are hashed into the cache key (see `anemia.preprocess.CropCache`).
    Saved models record their own `white_balance`, and the predictors serve
    each model with that, not with this default.
    """

    work_size: Tuple[int, int] = (512, 512)
    """Resolution the segmenter sees. Larger than the regressor input because
    thin conjunctiva bands lose definition when downsampled too early."""

    model_size: Tuple[int, int] = (224, 224)
    """Final square crop fed to the regressor."""

    bbox_pad_ratio: float = 0.08

    white_balance: str = "white_patch"
    """Colour correction applied to the full frame before the crop is cut.

    - "white_patch": the brightest unclipped pixels above the lower lid are
      taken as white (`imaging.white_patch_white_balance`). The default since
      17 Nov 2026: of the estimates compared, it lay closest to the capture
      device's white plastic (median 0.8-1.5 degrees, gray-world about 5;
      `experiments/26_correction_prechecks/`). Its effect on haemoglobin
      error has not been tested yet.
    - "gray_world": the frame's mean colour is taken as gray. The default
      before 17 Nov 2026; every model saved before then uses it.
    - "none": the camera's own colour.

    White-patch needs the mask, so it is computed after segmentation. The
    segmenter's own input is set separately (see `preprocess.segmenter_input`)."""

    gray_world: InitVar[Optional[bool]] = None
    """The older switch, still accepted when constructing: True means
    "gray_world", False means "none". Not stored; read `white_balance`."""

    def __post_init__(self, gray_world: Optional[bool]) -> None:
        if gray_world is not None:
            object.__setattr__(self, "white_balance", "gray_world" if gray_world else "none")
        if self.white_balance not in WHITE_BALANCE_METHODS:
            raise ValueError(f"white_balance must be one of {WHITE_BALANCE_METHODS}, "
                             f"got {self.white_balance!r}")


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

    max_clipped_fraction: float = 0.05
    """Share of pixels crushed to black or blown to white before the capture is
    rejected. A flash fired at wet conjunctiva blows out exactly the region we
    need and the result still looks *sharp*, so the focus check passes it.

    Deliberately generous: the cohort's worst capture clips 0.003 of the frame,
    so 0.05 is ~17x headroom and will not cost a retake on a usable photo.
    Unlike the cast threshold below, this one is **not calibrated against a
    real blown-out capture** — the repository contains none. Tighten it once a
    genuine flash-blowout example exists."""

    max_cast_ratio: float = 3.0
    """Ratio of brightest to dimmest channel mean, measured *before* white
    balance. Rejects strongly coloured illumination, where the information is
    absent from the file rather than merely distorted: under blue light the red
    channel averages 0.9/255, so tissue redness was never captured, and
    gray-world then divides by that near-zero mean and amplifies sensor noise.

    Measured separation is an order of magnitude on each side — the 52 cohort
    captures span 1.07-1.50 and the neutral stress captures 1.7, against
    22.6-206.0 for the five unusable coloured illuminants (see
    `experiments/04_lighting_stress/`)."""

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
