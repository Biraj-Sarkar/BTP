"""Conjunctiva segmentation backends.

A segmenter is any callable `(rgb: np.ndarray) -> (mask: np.ndarray, name: str)`
where `mask` is uint8 {0, 255} at the input resolution. Keeping the contract
this narrow lets the trained Mask2Former, the colour heuristic, and any future
backbone be swapped without touching preprocessing, training, or serving.

Torch is imported lazily so the heuristic path stays importable in a stripped
serving image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, Tuple

import cv2
import numpy as np

from .imaging import lightness, redness_index, smooth_mask

Mask = np.ndarray
SegmentResult = Tuple[Mask, str]


class Segmenter(Protocol):
    def __call__(self, image_rgb: np.ndarray) -> SegmentResult: ...


def heuristic_conjunctiva_mask(
    image_rgb: np.ndarray,
    *,
    min_lightness: float = 30.0,
    max_lightness: float = 248.0,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.60,
) -> Mask:
    """Colour prior for the palpebral conjunctiva.

    The conjunctiva is the *red* mucosal band on the everted lid, and redness
    separates it cleanly: on calibrated phantoms it sits around +14 to +26 on
    `redness_index` while skin sits near -2 and sclera near -5.

    Two details matter more than they look:

    * **The lightness gate is absolute, not a percentile.** Healthy conjunctiva
      is *darker* than pale conjunctiva (deeper red absorbs more light), so a
      percentile-based darkness cut removes the high-haemoglobin patients
      specifically. The gate exists only to drop lashes, pupil, and deep
      shadow, which are near-black in absolute terms.
    * **The threshold is chosen by Otsu**, not a fixed percentile. The
      conjunctiva occupies a small and highly variable share of the frame
      depending on how close the phone was held; any fixed percentile either
      floods the mask with skin (and then `largest_connected_component` returns
      the skin blob) or misses the tissue entirely.

    This is a fallback and a bootstrap for the learned segmenter, not the final
    answer. Measure it against real ground truth with
    `scripts/benchmark_segmenters.py` before relying on it.
    """

    redness = redness_index(image_rgb)
    light = lightness(image_rgb)

    eligible = (light >= min_lightness) & (light <= max_lightness)
    if not np.any(eligible):
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    values = redness[eligible]
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-3:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    scaled = np.zeros(redness.shape, dtype=np.uint8)
    scaled[eligible] = np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(scaled[eligible], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidate = eligible & (scaled >= otsu_level)
    fraction = float(np.count_nonzero(candidate)) / float(candidate.size)

    # Otsu assumes a roughly bimodal histogram. When the frame is nearly all
    # tissue (or nearly none) that assumption breaks, so fall back to a strict
    # top-percentile cut rather than returning a mask covering half the photo.
    if not (min_area_fraction <= fraction <= max_area_fraction):
        strict = float(np.percentile(values, 95.0))
        candidate = eligible & (redness >= strict)

    return smooth_mask((candidate.astype(np.uint8)) * 255)


def legacy_bright_neutral_mask(image_rgb: np.ndarray) -> Mask:
    """The inherited heuristic, preserved verbatim for benchmarking.

    It keeps bright, low-chroma pixels, which selects the *sclera* rather than
    the conjunctiva. Retained so the rework can be justified with a Dice
    comparison in the report instead of an assertion.
    """

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_threshold = np.percentile(l_channel, 64)
    a_center = np.median(a_channel)
    b_center = np.median(b_channel)
    chroma = np.abs(a_channel.astype(np.int16) - int(a_center)) + np.abs(
        b_channel.astype(np.int16) - int(b_center)
    )
    chroma_threshold = np.percentile(chroma, 55)

    candidate = np.logical_and(l_channel >= l_threshold, chroma <= chroma_threshold)
    return smooth_mask((candidate.astype(np.uint8)) * 255)


def grabcut_mask(image_rgb: np.ndarray) -> Mask:
    """Deterministic region proposal used when the colour prior degenerates."""

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    rect = (int(width * 0.08), int(height * 0.08), int(width * 0.84), int(height * 0.84))

    try:
        cv2.grabCut(
            image_bgr,
            mask,
            rect,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            5,
            cv2.GC_INIT_WITH_RECT,
        )
        proposal = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    return smooth_mask(proposal)


def heuristic_segmenter(image_rgb: np.ndarray) -> SegmentResult:
    """Colour prior with a grabCut escape hatch when the mask is degenerate.

    The inherited code defined this fallback chain but bypassed it whenever a
    segmenter was supplied, so the escape hatch never actually ran. Here it is
    part of the segmenter itself and therefore always active.
    """

    mask = heuristic_conjunctiva_mask(image_rgb)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)

    if 0.01 <= ratio <= 0.90:
        return mask, "heuristic"

    fallback = grabcut_mask(image_rgb)
    if np.count_nonzero(fallback) > 0:
        return fallback, "grabcut"
    return mask, "heuristic"


class Mask2FormerSegmenter:
    """Fine-tuned Mask2Former wrapper.

    Loads once and is reused for every call, which is what makes it viable in
    the request path — the inherited code re-ran segmentation inside the
    training dataloader on every epoch instead.
    """

    def __init__(self, model_dir: Path, device: Optional[str] = None):
        import torch
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        self._torch = torch
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"No trained segmenter at {model_dir}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_dir / "processor")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir / "model")
        self.model.to(self.device).eval()

    def __call__(self, image_rgb: np.ndarray) -> SegmentResult:
        torch = self._torch
        with torch.inference_mode():
            encoding = self.processor(images=image_rgb, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
            semantic = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_rgb.shape[:2]]
            )[0]
        mask = (semantic.detach().cpu().numpy() == 1).astype(np.uint8) * 255
        return smooth_mask(mask), "mask2former"


def load_segmenter(model_dir: Optional[Path] = None) -> Callable[[np.ndarray], SegmentResult]:
    """Return the best segmenter available, falling back loudly rather than silently.

    The inherited pipeline swallowed a load failure and degraded to the colour
    heuristic with only a print, so a run could report "trained" results that
    were nothing of the sort. Here an explicitly requested model directory that
    fails to load raises.
    """

    if model_dir is None:
        return heuristic_segmenter
    return Mask2FormerSegmenter(model_dir)
