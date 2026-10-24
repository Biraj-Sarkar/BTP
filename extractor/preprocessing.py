"""Preprocessing utilities for conjunctival image extraction and U-Net input preparation."""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """Standard gray-world white balance to compensate for lighting variations.
    
    Scales each color channel so its mean matches the overall grayscale mean.
    """
    work = image_rgb.astype(np.float32)
    channel_means = work.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1e-6)
    balanced = np.clip(work * scale, 0, 255).astype(np.uint8)
    return balanced


def cast_ratio(image_rgb: np.ndarray) -> float:
    """Ratio of brightest channel mean over dimmest channel mean.
    
    Measures illuminant cast before white-balance.
    """
    means = image_rgb.reshape(-1, 3).mean(axis=0)
    return float(means.max() / max(float(means.min()), 1e-6))


def pad_to_square(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Pad a rectangular crop symmetrically to square to prevent aspect-ratio distortion."""
    h, w = image.shape[:2]
    if h == w:
        return image
    max_side = max(h, w)
    pad_y = max_side - h
    pad_x = max_side - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    
    if image.ndim == 3:
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(fill_value, fill_value, fill_value)
        )
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value
    )


def tight_bbox(mask: np.ndarray, pad_ratio: float = 0.08) -> Tuple[int, int, int, int]:
    """Calculate tight bounding box [left, top, right, bottom] with contextual padding."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    
    pad = int(pad_ratio * max(x_max - x_min, y_max - y_min))
    h, w = mask.shape[:2]
    left = max(0, x_min - pad)
    top = max(0, y_min - pad)
    right = min(w, x_max + pad + 1)
    bottom = min(h, y_max + pad + 1)
    return left, top, right, bottom


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the binary mask."""
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask[:, :, None] > 0, image, 0).astype(np.uint8)

