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

from .imaging import largest_connected_component, lightness, redness_index, smooth_mask

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
    """Brightness-based baseline, kept for benchmarking.

    It keeps bright, low-chroma pixels, which selects the *sclera* rather than
    the conjunctiva — retained so redness-based extraction can be justified
    with a measured Dice comparison instead of an assertion.
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


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes — specular reflections punch gaps in wet tissue."""

    height, width = mask.shape[:2]
    flood = mask.copy()
    border = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, border, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def refined_conjunctiva_mask(image_rgb: np.ndarray) -> Mask:
    """Seeded-grabCut extraction, tuned on real conjunctiva photographs.

    The plain redness threshold fails on real captures because eyelashes and
    peri-orbital skin are red enough to pass it. This version fixes that with
    three observations about what distinguishes the tissue:

    * **Conjunctiva is smooth; the lash zone is high-frequency.** A local
      standard-deviation map separates them cleanly, so high-texture pixels are
      seeded as definite background.
    * **Sclera is bright, desaturated, and not red** — seeded as definite
      background rather than left for the colour model to decide.
    * **Specular highlights sit on the tissue.** The wet surface reflects the
      light source as white blobs *inside* the region we want, so they are
      seeded as probable foreground and any remaining holes filled afterwards.

    The seeds drive a mask-initialised grabCut, whose edge-aware colour model
    stops the region growing through the lash line — the failure mode of the
    plain threshold. Redness percentiles are computed on a morphologically
    closed redness map so thin dark lashes crossing the tissue cannot split the
    core seed region.
    """

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    redness = (lab[:, :, 1] - 128.0) - 0.5 * (lab[:, :, 2] - 128.0)
    saturation = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 1]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mean = cv2.blur(gray, (15, 15))
    mean_sq = cv2.blur(gray * gray, (15, 15))
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    spread = float(np.ptp(redness))
    if spread < 1e-3:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    red_u8 = np.clip((redness - redness.min()) / spread * 255, 0, 255).astype(np.uint8)
    red_closed = cv2.morphologyEx(
        red_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    specular = (l_channel > 235) & (saturation < 40)
    sclera = (l_channel > 170) & (saturation < 60) & (redness < np.percentile(redness, 60))
    lash_zone = texture > np.percentile(texture, 80)
    # A brown iris is dark red — without an absolute darkness gate it wins the
    # redness contest on wide shots. Conjunctiva is mid-lightness; iris, pupil,
    # and deep shadow are not.
    dark = l_channel < 80

    core_threshold = np.percentile(red_closed, 92)
    background_threshold = np.percentile(red_closed, 55)

    seeds = np.full(image_rgb.shape[:2], cv2.GC_PR_BGD, np.uint8)
    seeds[(red_closed > background_threshold) & (red_closed < core_threshold)] = cv2.GC_PR_FGD
    seeds[red_closed <= background_threshold] = cv2.GC_BGD
    seeds[sclera] = cv2.GC_BGD
    seeds[lash_zone] = cv2.GC_BGD
    seeds[dark] = cv2.GC_BGD
    seeds[specular] = cv2.GC_PR_FGD

    core = (red_closed >= core_threshold) & ~lash_zone & ~sclera & ~dark
    seeds[core] = cv2.GC_FGD
    if not core.any():
        return heuristic_conjunctiva_mask(image_rgb)

    try:
        cv2.grabCut(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            seeds,
            None,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return heuristic_conjunctiva_mask(image_rgb)

    mask = np.where(
        (seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    # A wide opening shears off lash tendrils the colour model let through.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    mask = largest_connected_component(mask)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    return _fill_holes(mask)


def heuristic_segmenter(image_rgb: np.ndarray) -> SegmentResult:
    """Classical extraction chain: refined grabCut, then simpler fallbacks.

    The fallback chain lives inside the segmenter itself, so it is always
    active no matter how the segmenter is invoked.
    """

    refined = refined_conjunctiva_mask(image_rgb)
    ratio = float(np.count_nonzero(refined)) / float(refined.size)
    if 0.01 <= ratio <= 0.90:
        return refined, "refined"

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
    the request path.
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


# Bump when any classical-extraction algorithm changes: it namespaces the crop
# cache, so stale crops from an older extractor can never leak into training.
heuristic_segmenter.cache_key = "classical-v2"


def load_segmenter(model_dir: Optional[Path] = None) -> Callable[[np.ndarray], SegmentResult]:
    """Return the best segmenter available, falling back loudly rather than silently.

    Swallowing a load failure and quietly degrading to the colour heuristic
    would let a run report "trained" results that are nothing of the sort, so
    an explicitly requested model directory that fails to load raises.
    """

    if model_dir is None:
        return heuristic_segmenter
    return Mask2FormerSegmenter(model_dir)


# ---------------------------------------------------------------- registry

def _named(fn, name):
    """Wrap a bare mask function into the (mask, name) segmenter contract."""

    def segmenter(image_rgb: np.ndarray) -> SegmentResult:
        return fn(image_rgb), name

    segmenter.__name__ = f"{name}_segmenter"
    segmenter.cache_key = f"extractor-{name}-v1"
    return segmenter


def cielab_segmenter(image_rgb: np.ndarray) -> SegmentResult:
    """The CIELAB pipeline's own extractor, imported rather than reimplemented.

    Kept behind a lazy import so this module has no hard dependency on a file
    that lives outside the package.
    """

    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import cielab_pipeline as cp

    normalised, _ = cp.cielab_normalization(image_rgb)
    mask, backend = cp.segment_conjunctiva(normalised)
    return mask, f"cielab-{backend}"


cielab_segmenter.cache_key = "extractor-cielab-v1"


EXTRACTORS = {
    # the production chain: refined seeded grabCut, then simpler fallbacks
    "refined": heuristic_segmenter,
    # plain redness prior with Otsu, no seeding
    "redness": _named(heuristic_conjunctiva_mask, "redness"),
    # brightness / low-chroma baseline — selects sclera, kept for comparison
    "brightness": _named(legacy_bright_neutral_mask, "brightness"),
    # unseeded grabCut from a centre rectangle
    "grabcut": _named(grabcut_mask, "grabcut"),
    # the CIELAB pipeline's extractor
    "cielab": cielab_segmenter,
}


def get_extractor(name: str = "refined", model_dir: Optional[Path] = None):
    """Resolve an extractor by name, or load a trained segmenter directory.

    Recording *which* extractor produced a model's features matters as much as
    the coefficients: features measured inside a different mask are not
    comparable, so a model served with the wrong extractor is silently wrong.
    `LinearModel` therefore stores this name and the predictor reads it back.
    """

    if model_dir is not None:
        return Mask2FormerSegmenter(model_dir)
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name]
