"""Full conjunctival ROI extraction pipeline with quality control and caching."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from .unet_loader import UNetSegmenter, load_unet
from .preprocessing import (
    resize,
    gray_world_white_balance,
    apply_mask,
    tight_bbox,
    pad_to_square,
)
from .postprocessing import clean_mask
from .quality_control import QualityConfig, QualityReport, assess_quality


@dataclass
class ExtractionResult:
    """Output bundle from conjunctival extraction."""
    crop: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    quality: QualityReport
    backend: str = "unet"

    @property
    def passed(self) -> bool:
        return self.quality.passed


class ROIExtractor:
    """Encapsulates the complete eye-image to standardized 224x224 conjunctiva crop pipeline."""

    def __init__(
        self,
        segmenter: Optional[UNetSegmenter] = None,
        checkpoint_path: Optional[Path | str] = None,
        target_size: Tuple[int, int] = (224, 224),
        pad_ratio: float = 0.08,
        gray_world: bool = True,
        quality_config: Optional[QualityConfig] = None,
    ):
        self.segmenter = segmenter or load_unet(checkpoint_path=checkpoint_path)
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.gray_world = gray_world
        self.quality_config = quality_config or QualityConfig()

    def extract(self, image_rgb: np.ndarray) -> ExtractionResult:
        """Extract ROI from an RGB image array."""
        # 1. Standardize working size (512x512) for consistent illuminant and morphology
        working_raw = resize(image_rgb, (512, 512))
        
        # 2. Gray-world white balance
        balanced = gray_world_white_balance(working_raw) if self.gray_world else working_raw

        # 3. Predict probability map via U-Net
        prob_map = self.segmenter.predict_probability(balanced)

        # 4. Threshold & Morphological cleanup
        binary = (prob_map >= self.segmenter.config.threshold).astype(np.uint8) * 255
        mask = clean_mask(binary, min_area_fraction=self.segmenter.config.min_area_fraction)

        # 5. Quality gate evaluation
        quality_cfg = self.quality_config
        quality_report = assess_quality(balanced, mask, quality_cfg, raw_rgb=working_raw)

        if not mask.any():
            # Fallback zero crop if segmentation produced no valid mask
            empty_crop = np.zeros((*self.target_size, 3), dtype=np.uint8)
            return ExtractionResult(
                crop=empty_crop,
                mask=mask,
                bbox=(0, 0, 0, 0),
                quality=quality_report,
                backend="unet-empty",
            )

        # 6. Tight bounding box with contextual padding
        left, top, right, bottom = tight_bbox(mask, pad_ratio=self.pad_ratio)

        # 7. Zero out background non-mask pixels and crop
        masked = apply_mask(balanced, mask)
        crop_raw = masked[top:bottom, left:right]
        if crop_raw.size == 0:
            crop_raw = masked

        # 8. Pad symmetrically to square and resize to target predictor dimensions
        square = pad_to_square(crop_raw)
        crop_final = resize(square, self.target_size)

        return ExtractionResult(
            crop=crop_final,
            mask=mask,
            bbox=(left, top, right, bottom),
            quality=quality_report,
            backend="unet",
        )

    def extract_from_path(self, image_path: Path | str) -> ExtractionResult:
        """Load image from disk and extract ROI."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not decode image at {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.extract(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract conjunctival ROIs from eye photos.")
    parser.add_argument("--input", type=Path, required=True, help="Input image file or folder")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ROI crops")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to U-Net checkpoint (.pt)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    extractor = ROIExtractor(checkpoint_path=args.checkpoint)

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted([p for p in args.input.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    rows = []
    print(f"Extracting ROIs from {len(files)} image(s)...")
    for idx, img_path in enumerate(files, 1):
        res = extractor.extract_from_path(img_path)
        out_name = f"roi_{img_path.stem}.png"
        out_path = args.output / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(res.crop, cv2.COLOR_RGB2BGR))
        rows.append({
            "image": img_path.name,
            "roi_file": out_name,
            "passed": res.passed,
            "backend": res.backend,
            "mask_ratio": res.quality.mask_ratio,
            "focus": res.quality.focus,
            "reasons": "; ".join(res.quality.reasons),
        })
        if idx % 20 == 0 or idx == len(files):
            print(f"Processed {idx}/{len(files)}")

    manifest_path = args.output / "extraction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extraction complete. Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

