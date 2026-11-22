"""Data transforms and conservative medical augmentation policies."""

from __future__ import annotations

from typing import Tuple
import torchvision.transforms as T


def get_train_transforms(input_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Conservative data augmentation policy preserving physiological erythema and vascular cues."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        # Subtle brightness and contrast variations; hue is explicitly kept at 0 to preserve hemoglobin signal
        T.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06, hue=0.0),
        T.ToTensor(),
    ])


def get_val_transforms(input_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Deterministic validation and testing preprocessing without augmentation."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

