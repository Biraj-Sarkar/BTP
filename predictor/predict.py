"""Predict Hemoglobin and Anemia from an eye photo or ROI crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from .training import load_checkpoint
from .data.transforms import get_val_transforms


def predict_from_crop(model, crop_rgb: np.ndarray, age: float | None = None, sex: str | None = None) -> dict:
    tf = get_val_transforms()
    tensor = tf(crop_rgb).unsqueeze(0)
    with torch.inference_mode():
        hb = float(model.predict_hb(tensor).item())

    # WHO cutoff rule: 13.0 for adult men, 12.0 for non-pregnant women
    is_male = (sex or "").strip().lower().startswith("m")
    threshold = 13.0 if is_male else 12.0
    anemic = hb < threshold

    return {
        "hb_g_dl": round(hb, 2),
        "anemic": bool(anemic),
        "threshold_g_dl": threshold,
        "input_crop_shape": list(crop_rgb.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Hemoglobin using a trained checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Raw eye photo (runs U-Net extraction first)")
    source.add_argument("--roi", type=Path, help="Pre-extracted 224x224 conjunctiva crop")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default=None)
    args = parser.parse_args()

    model, bundle = load_checkpoint(args.checkpoint)

    if args.roi:
        bgr = cv2.imread(str(args.roi))
        if bgr is None:
            raise ValueError(f"Could not read ROI at {args.roi}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = predict_from_crop(model, rgb, age=args.age, sex=args.sex)
        result["source"] = "roi"
    else:
        # Import extractor for end-to-end raw image inference
        from extractor import ROIExtractor
        extractor = ROIExtractor()
        ext_res = extractor.extract_from_path(args.image)
        if not ext_res.mask.any():
            result = {
                "hb_g_dl": None,
                "anemic": None,
                "error": "U-Net segmentation could not detect a valid conjunctival region.",
                "quality": ext_res.quality.to_dict(),
            }
        else:
            result = predict_from_crop(model, ext_res.crop, age=args.age, sex=args.sex)
            result["source"] = "raw_image"
            result["roi_bbox"] = list(ext_res.bbox)
            result["quality"] = ext_res.quality.to_dict()

    result["architecture"] = bundle.get("architecture")
    result["variant"] = bundle.get("variant")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

