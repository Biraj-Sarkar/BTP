"""Quality control gates for eye captures and extracted conjunctival ROIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np

from .preprocessing import cast_ratio


@dataclass
class QualityConfig:
    """Configurable quality thresholds for accepting conjunctiva captures."""
    min_focus: float = 30.0
    min_mask_ratio: float = 0.002
    max_mask_ratio: float = 0.85
    max_clipped_fraction: float = 0.08
    max_cast_ratio: float = 3.5
    enforce: bool = True


@dataclass
class QualityReport:
    """Detailed audit metrics and pass/fail gate decision."""
    focus: float
    exposure: float
    clipped: float
    mask_ratio: float
    cast_ratio: float
    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "focus": round(self.focus, 2),
            "exposure": round(self.exposure, 2),
            "clipped": round(self.clipped, 4),
            "mask_ratio": round(self.mask_ratio, 5),
            "cast_ratio": round(self.cast_ratio, 2) if not np.isnan(self.cast_ratio) else None,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def focus_score(image_rgb: np.ndarray) -> float:
    """Variance of Laplacian measuring image sharpness and blur."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_score(image_rgb: np.ndarray) -> float:
    """Mean luminance across the image (0-255)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def clipped_fraction(image_rgb: np.ndarray) -> float:
    """Fraction of pixels saturated at 0 or 255 across any channel."""
    clipped = (image_rgb == 0) | (image_rgb == 255)
    return float(clipped.any(axis=-1).mean())


def assess_quality(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: QualityConfig,
    raw_rgb: Optional[np.ndarray] = None,
) -> QualityReport:
    """Audit focus, illumination, clipping, and mask size against quality thresholds."""
    focus = focus_score(image_rgb)
    exposure = exposure_score(image_rgb)
    clipped = clipped_fraction(image_rgb)
    mask_ratio = float(np.count_nonzero(mask)) / float(max(mask.size, 1))
    cast = cast_ratio(raw_rgb) if raw_rgb is not None else float("nan")

    reasons = []
    if focus < config.min_focus:
        reasons.append(f"Blurred capture (focus {focus:.1f} < {config.min_focus})")
    if mask_ratio < config.min_mask_ratio:
        reasons.append(f"Conjunctival area too small ({mask_ratio:.4f} < {config.min_mask_ratio})")
    if mask_ratio > config.max_mask_ratio:
        reasons.append(f"Mask implausibly large ({mask_ratio:.3f} > {config.max_mask_ratio})")
    if clipped > config.max_clipped_fraction:
        reasons.append(f"Over- or under-exposed ({100 * clipped:.1f}% pixels clipped)")
    if not np.isnan(cast) and cast > config.max_cast_ratio:
        reasons.append(f"Strong color cast (cast ratio {cast:.1f} > {config.max_cast_ratio})")

    passed = (len(reasons) == 0) if config.enforce else True

    return QualityReport(
        focus=focus,
        exposure=exposure,
        clipped=clipped,
        mask_ratio=mask_ratio,
        cast_ratio=cast,
        passed=passed,
        reasons=reasons,
    )

