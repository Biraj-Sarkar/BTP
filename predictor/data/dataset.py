"""PyTorch Dataset implementation for Hemoglobin estimation."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import SampleRecord


class HbDataset(Dataset):
    """Hemoglobin prediction dataset with in-memory caching and target standardization."""

    def __init__(
        self,
        records: List[SampleRecord],
        transform: Optional[Callable] = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        cache_in_memory: bool = True,
    ):
        self.records = records
        self.transform = transform
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        self.cache_in_memory = cache_in_memory
        self._image_cache: Dict[str, np.ndarray] = {}

        if self.cache_in_memory:
            for r in self.records:
                self._load_image(r.image_path)

    def _load_image(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._image_cache:
            return self._image_cache[key]

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Failed to load image from {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.cache_in_memory:
            self._image_cache[key] = rgb
        return rgb

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image_rgb = self._load_image(record.image_path)

        if self.transform is not None:
            image_tensor = self.transform(image_rgb)
        else:
            # Fallback simple tensor conversion
            image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        # Standardized continuous regression target
        target_z = (record.hb - self.target_mean) / self.target_std
        target_tensor = torch.tensor([target_z], dtype=torch.float32)

        return {
            "image": image_tensor,
            "target": target_tensor,
            "hb_raw": float(record.hb),
            "patient_id": record.patient_id,
            "image_id": record.image_id,
        }

