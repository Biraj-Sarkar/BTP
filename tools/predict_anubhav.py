"""Predict Hb and anemia with an Anubhav checkpoint.

Use ``--image`` for a raw eye photo (the Anubhav extractor and quality gate run
first), or ``--roi`` for a crop already produced by ``extract_rois.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ANUBHAV = ROOT / "algorithms" / "Anubhav"
sys.path.insert(0, str(ANUBHAV / "src"))

from anemia.data import anemia_threshold  # noqa: E402
from anemia.imaging import read_rgb  # noqa: E402
from anemia.model import Checkpoint, to_tensor  # noqa: E402
from anemia.predict import AnemiaPredictor  # noqa: E402


def predict_roi(checkpoint_path: Path, roi_path: Path, age: float | None, sex: str | None, pregnant: bool) -> dict:
    checkpoint = Checkpoint.load(checkpoint_path)
    model = checkpoint.build(torch.device("cpu"))
    image = read_rgb(roi_path)
    with torch.inference_mode():
        standardised = model(to_tensor(image).unsqueeze(0)).cpu().numpy().reshape(-1)
    hb = float(checkpoint.denormalise(standardised)[0])
    threshold = anemia_threshold(age, sex, pregnant=pregnant)
    return {
        "hb_g_dl": round(hb, 2),
        "anemic": hb < threshold,
        "threshold_g_dl": threshold,
        "usable": True,
        "quality": {"passed": True, "source": "pre-extracted-roi"},
        "message": "Prediction from a pre-extracted Anubhav ROI.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="raw eye image; runs extraction and QC")
    source.add_argument("--roi", type=Path, help="224x224 ROI from extract_rois.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default=None)
    parser.add_argument("--pregnant", action="store_true")
    args = parser.parse_args()

    if args.image:
        result = AnemiaPredictor(args.checkpoint).predict_path(
            args.image, age_years=args.age, sex=args.sex, pregnant=args.pregnant
        ).to_dict()
    else:
        result = predict_roi(args.checkpoint, args.roi, args.age, args.sex, args.pregnant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
