"""Learned conjunctiva segmentation — inference only.

This module **reads** a trained parameter bundle and nothing else. It contains
no optimiser, no dataset code, and no way to write a bundle; retraining lives
in `scripts/train_segmenter.py`. The split is deliberate: serving must not be
able to mutate the parameters it is serving, and a bundle that appears on the
server is then always something a training run produced and recorded.

The backend satisfies the same contract as every other extractor —
`(rgb) -> (mask, name)`, mask uint8 {0, 255} at input resolution — so it drops
into `EXTRACTORS` beside the classical chain with nothing else changing.

Abbreviations
-------------
U-Net    encoder-decoder segmentation architecture with skip connections
CNN      convolutional neural network
RGB      red-green-blue colour channels
smp      `segmentation_models_pytorch`, the library providing the architecture
Dice     overlap score, 2|A∩B| / (|A|+|B|); 1.0 is a perfect mask
LCC      largest connected component
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .imaging import largest_connected_component
from .segment import Mask, SegmentResult, _fill_holes

BUNDLE_FORMAT = "anemia-segmenter"
BUNDLE_VERSION = 1

# Where `get_extractor("unet")` looks when no explicit path is given.
DEFAULT_BUNDLE = Path("runs/segmenter/unet.pt")

REQUIRED_KEYS = ("format", "version", "state_dict", "config", "provenance")


@dataclass(frozen=True)
class SegmenterConfig:
    """Everything needed to reproduce the training-time input conditioning.

    Inference must condition an image exactly as training did, so these values
    travel inside the bundle rather than being defaults in this file. A bundle
    trained at 384px still serves correctly after the default here changes.
    """

    arch: str = "unet"
    """Decoder architecture (see `anemia.architectures`). Defaults to `unet` so
    that bundles written before architectures were selectable still load."""

    encoder: str = "efficientnet-b0"
    input_size: int = 320
    threshold: float = 0.5
    # ImageNet statistics, the normalisation the pretrained encoder expects.
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    min_area_fraction: float = 0.002

    gray_world: bool = False
    """Whether the image was white-balanced before the network saw it during
    training. Serving must match: `preprocess.prepare()` reads this. False is
    how every bundle written before this field existed was trained."""

    @classmethod
    def from_dict(cls, raw: dict) -> "SegmenterConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


class LearnedSegmenter:
    """A fine-tuned U-Net, loaded once and reused for every call.

    Loading weights costs far more than a forward pass, so the model is built
    in `__init__` and kept; constructing one per request would dominate the
    latency budget.

    Two things this gives the pipeline that the classical extractor cannot:

    * **Absolute decisions.** The classical chain thresholds redness at
      `np.percentile(red, 92)`, which assumes every capture devotes the same
      share of its pixels to conjunctiva. A wide shot where the tissue is 2% of
      the frame and a macro crop where it is 60% get very different cuts from
      the same rule. A learned model scores each pixel on its own evidence.
    * **A probability, not a verdict.** `probability()` exposes the raw map, so
      colour features can be pooled *weighted by confidence* instead of over a
      hard mask — boundary pixels that are half eyelash stop counting fully.
    """

    def __init__(self, bundle_path: Path, device: Optional[str] = None):
        import torch  # imported lazily: the classical path needs no torch

        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"No trained segmenter at {bundle_path}. Train one with:\n"
                f"  python scripts/train_segmenter.py --data <dataset> "
                f"--out {bundle_path}"
            )

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        _validate(bundle, bundle_path)

        self._torch = torch
        self.config = SegmenterConfig.from_dict(bundle["config"])
        self.provenance = bundle["provenance"]
        self.metrics = bundle.get("metrics", {})
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        from .architectures import build

        # Weights come from the bundle, not the hub, so the encoder is built
        # empty. The architecture is read from the bundle too: a model served
        # by the wrong decoder would fail to load, which is the intent.
        self.model = build(self.config.arch, self.config.encoder, pretrained=False)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)
        # Frozen BatchNorm: in train mode the running statistics would drift
        # with whatever images happened to be served.
        self.model.eval()

        # Feature caches key off this. A retrained segmenter produces different
        # masks, so it must not silently reuse features cached under the old
        # one — the trained_at stamp changes on every run.
        self.cache_key = f"{self.config.arch}-{self.provenance.get('trained_at', 'unknown')}"

    @property
    def wants_gray_world(self) -> bool:
        """The kind of image this bundle was trained on.

        A segmentation network is as sensitive to a global colour shift as to
        anything else in its input, so serving it white-balanced images when it
        trained on raw ones (or the reverse) is a train/serve mismatch.
        `preprocess.prepare()` reads this and conditions the segmenter's input
        accordingly, independently of which pixels colour is finally read from.
        """

        return bool(self.config.gray_world)

    @property
    def min_mask_ratio(self) -> float:
        """The quality gate's mask-size floor for this extractor.

        `preprocess.prepare` reads this in place of the global
        `QualityConfig.min_mask_ratio`, which was calibrated for the classical
        chain's much larger masks. It is the same floor this segmenter already
        applies to its own output, so a capture is never "too small" for the
        gate while still being large enough for the extractor.
        """

        return self.config.min_area_fraction

    def probability(self, image_rgb: np.ndarray) -> np.ndarray:
        """Per-pixel conjunctiva probability at the input resolution, float32."""

        torch = self._torch
        height, width = image_rgb.shape[:2]
        size = self.config.input_size

        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        array = resized.astype(np.float32) / 255.0
        array = (array - np.array(self.config.mean, np.float32)) / np.array(
            self.config.std, np.float32
        )
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Back to the caller's resolution, so the mask indexes their image.
        return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

    def __call__(self, image_rgb: np.ndarray) -> SegmentResult:
        """The extractor contract: `(rgb) -> (mask, name)`.

        The cleanup after thresholding is the same shape as the classical
        chain's, and for the same reason — a network can leave a speck of
        probability on a red patch of cheek. Keeping the largest component and
        filling holes turns a scattered map into the single crescent the
        downstream colour statistics assume they are measuring.
        """

        probability = self.probability(image_rgb)
        mask: Mask = (probability >= self.config.threshold).astype(np.uint8) * 255

        if not mask.any():
            return mask, "unet-empty"

        mask = largest_connected_component(mask)
        mask = _fill_holes(mask)

        # Too small to be conjunctiva: report it rather than hand back a mask
        # whose colour mean is a handful of pixels.
        if np.count_nonzero(mask) / mask.size < self.config.min_area_fraction:
            return np.zeros_like(mask), "unet-empty"

        return mask, "unet"


def _validate(bundle: object, path: Path) -> None:
    """Refuse a bundle that cannot be served correctly.

    A partially written or foreign checkpoint that loads and then segments
    nonsense is far worse than one that refuses to load: the pipeline would
    keep returning confident haemoglobin numbers measured inside a meaningless
    mask. Mirrors `LinearModel.load`, which refuses incomplete model files.
    """

    if not isinstance(bundle, dict):
        raise ValueError(f"{path} is not a segmenter bundle")
    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"{path} cannot be loaded — missing {', '.join(missing)}")
    if bundle["format"] != BUNDLE_FORMAT:
        raise ValueError(
            f"{path} cannot be loaded — format {bundle['format']!r}, "
            f"expected {BUNDLE_FORMAT!r}"
        )
    if bundle["version"] > BUNDLE_VERSION:
        raise ValueError(
            f"{path} was written by a newer trainer (bundle version "
            f"{bundle['version']}, this code reads {BUNDLE_VERSION})"
        )
