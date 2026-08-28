"""Raw capture -> model-ready crop, with quality control that actually gates.

The same `prepare()` call runs at train time and in the request handler, which
is the only way to guarantee the server sees the distribution the model was
trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import PreprocessConfig, QualityConfig
from .imaging import (
    apply_mask,
    clipped_fraction,
    exposure_score,
    focus_score,
    gray_world_white_balance,
    pad_to_square,
    read_rgb,
    resize,
    tight_bbox,
)
from .segment import SegmentResult


@dataclass
class QualityReport:
    focus: float
    exposure: float
    clipped: float
    mask_ratio: float
    passed: bool
    reasons: list = field(default_factory=list)
    cast_ratio: float = float("nan")
    """Brightest/dimmest channel mean before white balance. NaN when the
    pre-balance image was not supplied, in which case the check is skipped."""

    def to_dict(self) -> dict:
        return {
            "focus": round(self.focus, 3),
            "exposure": round(self.exposure, 3),
            "clipped": round(self.clipped, 5),
            "mask_ratio": round(self.mask_ratio, 5),
            "cast_ratio": (
                None if np.isnan(self.cast_ratio) else round(self.cast_ratio, 3)
            ),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass
class Prepared:
    crop: np.ndarray
    """Square `model_size` RGB crop — the tensor source."""

    mask: np.ndarray
    balanced: np.ndarray
    backend: str
    quality: QualityReport


def cast_ratio(image_rgb: np.ndarray) -> float:
    """Brightest channel mean over dimmest — how far the illuminant is from neutral.

    Must be measured *before* gray-world white balance, which exists precisely
    to remove this and would drive the ratio to ~1.0 on any input.
    """

    means = image_rgb.reshape(-1, 3).mean(axis=0)
    return float(means.max() / max(float(means.min()), 1e-6))


def assess_quality(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: QualityConfig,
    raw_rgb: Optional[np.ndarray] = None,
) -> QualityReport:
    """Gate a capture. Every statistic computed here is also enforced.

    `raw_rgb` is the working image *before* white balance, needed for the
    illuminant-neutrality check. When it is omitted that one check is skipped
    rather than silently passing on a meaningless value.
    """

    focus = focus_score(image_rgb)
    exposure = exposure_score(image_rgb)
    clipped = clipped_fraction(image_rgb)
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    cast = cast_ratio(raw_rgb) if raw_rgb is not None else float("nan")

    reasons = []
    if focus < config.min_focus:
        reasons.append(f"blurred (focus {focus:.1f} < {config.min_focus})")
    if mask_ratio < config.min_mask_ratio:
        reasons.append(f"conjunctiva too small ({mask_ratio:.3f})")
    if mask_ratio > config.max_mask_ratio:
        reasons.append(f"mask implausibly large ({mask_ratio:.3f})")
    if clipped > config.max_clipped_fraction:
        reasons.append(
            f"over- or under-exposed ({100 * clipped:.1f}% of pixels clipped)"
        )
    if not np.isnan(cast) and cast > config.max_cast_ratio:
        reasons.append(f"strongly coloured lighting (cast ratio {cast:.1f})")

    return QualityReport(
        focus=focus,
        exposure=exposure,
        clipped=clipped,
        mask_ratio=mask_ratio,
        cast_ratio=cast,
        passed=not reasons,
        reasons=reasons,
    )


def prepare(
    image_rgb: np.ndarray,
    segmenter: Callable[[np.ndarray], SegmentResult],
    preprocess: PreprocessConfig,
    quality: QualityConfig,
) -> Prepared:
    """White balance -> segment -> tight crop -> letterbox -> resize.

    Padding to square happens *before* the resize so the crescent shape is never
    stretched, and no local contrast equalisation is applied: CLAHE next to the
    black letterbox creates boundary artefacts and flattens the very erythema
    difference the regressor reads.
    """

    working = resize(image_rgb, preprocess.work_size)
    balanced = gray_world_white_balance(working) if preprocess.gray_world else working

    mask, backend = segmenter(balanced)
    # `working` is pre-white-balance, which the illuminant check needs.
    report = assess_quality(balanced, mask, quality, raw_rgb=working)

    masked = apply_mask(balanced, mask)
    left, top, right, bottom = tight_bbox(mask, preprocess.bbox_pad_ratio)
    cropped = masked[top:bottom, left:right]
    if cropped.size == 0:
        cropped = masked

    crop = resize(pad_to_square(cropped), preprocess.model_size)
    return Prepared(crop=crop, mask=mask, balanced=balanced, backend=backend, quality=report)


def prepare_path(
    image_path: Path,
    segmenter: Callable[[np.ndarray], SegmentResult],
    preprocess: PreprocessConfig,
    quality: QualityConfig,
) -> Prepared:
    return prepare(read_rgb(image_path), segmenter, preprocess, quality)


class CropCache:
    """Disk cache of prepared crops, keyed by image content and settings.

    Segmentation is deterministic but expensive — a Mask2Former forward pass per
    image. Running it inside `__getitem__` would repeat it every epoch, so a
    40-epoch run would segment every image 40 times. Caching turns that into
    once, which is most of the training-time win.

    Augmentation still happens per-epoch on the cached crop, so this costs no
    augmentation diversity.
    """

    def __init__(
        self,
        root: Path,
        preprocess: PreprocessConfig,
        backend: str,
        quality: Optional[QualityConfig] = None,
    ):
        self.root = Path(root)
        # The cached QualityReport was produced under a specific set of
        # thresholds, so those thresholds belong in the key. Without them,
        # tightening a gate would silently reuse verdicts computed under the
        # old one — the same trap as reusing crops cut by an old extractor.
        signature = json.dumps(
            {
                "work_size": list(preprocess.work_size),
                "model_size": list(preprocess.model_size),
                "bbox_pad_ratio": preprocess.bbox_pad_ratio,
                "gray_world": preprocess.gray_world,
                "backend": backend,
                "quality": asdict(quality) if quality is not None else None,
            },
            sort_keys=True,
        )
        self.namespace = hashlib.sha256(signature.encode()).hexdigest()[:16]
        self.dir = self.root / self.namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, image_path: Path):
        stat = image_path.stat()
        key = hashlib.sha256(
            f"{image_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}".encode()
        ).hexdigest()[:24]
        return self.dir / f"{key}.png", self.dir / f"{key}.json"

    def get(self, image_path: Path) -> Optional[Prepared]:
        crop_path, meta_path = self._paths(image_path)
        if not (crop_path.exists() and meta_path.exists()):
            return None
        crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop_bgr is None:
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        quality_meta = dict(meta["quality"])
        quality_meta["reasons"] = quality_meta.get("reasons", [])
        # to_dict() writes JSON null for an unmeasured cast ratio; NaN is the
        # in-memory representation, and json cannot carry it.
        if quality_meta.get("cast_ratio") is None:
            quality_meta["cast_ratio"] = float("nan")
        report = QualityReport(**quality_meta)
        return Prepared(
            crop=cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            mask=np.empty((0, 0), dtype=np.uint8),
            balanced=np.empty((0, 0, 3), dtype=np.uint8),
            backend=meta["backend"],
            quality=report,
        )

    def put(self, image_path: Path, prepared: Prepared) -> None:
        crop_path, meta_path = self._paths(image_path)
        cv2.imwrite(str(crop_path), cv2.cvtColor(prepared.crop, cv2.COLOR_RGB2BGR))
        meta_path.write_text(
            json.dumps({"backend": prepared.backend, "quality": prepared.quality.to_dict()}),
            encoding="utf-8",
        )

    def prepare(
        self,
        image_path: Path,
        segmenter: Callable[[np.ndarray], SegmentResult],
        preprocess: PreprocessConfig,
        quality: QualityConfig,
    ) -> Prepared:
        cached = self.get(image_path)
        if cached is not None:
            return cached
        prepared = prepare_path(image_path, segmenter, preprocess, quality)
        self.put(image_path, prepared)
        return prepared
