"""Post-processing morphological filters for segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected 8-component in a binary mask."""
    if not mask.any():
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    # Background is label 0; find component with largest area
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes inside the segmented conjunctiva component."""
    if not mask.any():
        return mask
    h, w = mask.shape[:2]
    flood = mask.copy()
    border = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, border, (0, 0), 255)
    inverted = cv2.bitwise_not(flood)
    return mask | inverted


def clean_mask(mask: np.ndarray, min_area_fraction: float = 0.002) -> np.ndarray:
    """Apply LCC, fill holes, and reject if area is under the minimum fraction."""
    if not mask.any():
        return mask
    mask = largest_connected_component(mask)
    mask = fill_holes(mask)
    area_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if area_fraction < min_area_fraction:
        return np.zeros_like(mask)
    return mask

