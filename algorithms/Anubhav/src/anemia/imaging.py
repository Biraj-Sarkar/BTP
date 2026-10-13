"""Colour operations, I/O, and capture-quality scores.

Deliberately free of torch imports: the serving path pulls these functions in
without dragging the training stack into the request handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(image_rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def write_gray(image_gray: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_gray)


def resize(image: np.ndarray, size: Tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interpolation)


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """Neutralise the illuminant so tissue colour is comparable across devices.

    Phone ISPs and ward lighting (fluorescent vs tungsten) shift the white point
    enough to swamp the pallor signal we are trying to measure.
    """

    image = image_rgb.astype(np.float32)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    target_mean = float(channel_means.mean())
    scale = target_mean / np.maximum(channel_means, 1e-6)
    return np.clip(image * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def focus_score(image_rgb: np.ndarray) -> float:
    """Variance of the Laplacian; low values indicate motion blur or misfocus."""

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_score(image_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].astype(np.float32).std())


def clipped_fraction(image_rgb: np.ndarray, low: int = 4, high: int = 251) -> float:
    """Share of pixels crushed to black or blown to white.

    A flash fired straight at wet conjunctiva blows out exactly the region we
    need, and the resulting image looks sharp to the focus check.
    """

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clipped = np.count_nonzero((gray <= low) | (gray >= high))
    return float(clipped) / float(gray.size)


def redness_index(image_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel redness in CIELAB, biased away from yellow skin tones.

    The palpebral conjunctiva is red (high a*) and comparatively neutral on the
    blue-yellow axis, whereas peri-orbital skin carries a strong yellow cast
    (high b*). Subtracting a fraction of b* separates the two far better than a*
    alone, which is what makes this usable as a segmentation prior.
    """

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0
    return a_star - 0.5 * b_star


def lightness(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)


def pad_to_square(image: np.ndarray, fill: int = 0) -> np.ndarray:
    """Symmetric letterbox to 1:1 *before* the square resize.

    Resizing a crescent-shaped crop straight to 224x224 stretches it, which
    elongates the micro-vasculature and corrupts the spatial density cues the
    regressor depends on.
    """

    height, width = image.shape[:2]
    size = max(height, width)
    top = (size - height) // 2
    left = (size - width) // 2
    value = [fill] * 3 if image.ndim == 3 else fill
    return cv2.copyMakeBorder(
        image,
        top,
        size - height - top,
        left,
        size - width - left,
        cv2.BORDER_CONSTANT,
        value=value,
    )


def tight_bbox(mask: np.ndarray, pad_ratio: float = 0.08) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    pad_x = max(int((x_max - x_min + 1) * pad_ratio), 4)
    pad_y = max(int((y_max - y_min + 1) * pad_ratio), 4)

    height, width = mask.shape[:2]
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(width, x_max + pad_x + 1),
        min(height, y_max + pad_y + 1),
    )


def apply_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_rgb, image_rgb, mask=(mask > 0).astype(np.uint8))


def build_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy()
    selected = mask > 0
    if not np.any(selected):
        return overlay
    tint = np.zeros_like(image_rgb)
    tint[:, :, 1] = 255  # green reads clearly against red tissue
    overlay[selected] = (
        (1 - alpha) * overlay[selected].astype(np.float32)
        + alpha * tint[selected].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8 * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return largest_connected_component(closed)
