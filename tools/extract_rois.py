"""Run Anubhav's extractor only and materialize 224x224 ROI crops."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
ANUBHAV = ROOT / "algorithms" / "Anubhav"
sys.path.insert(0, str(ANUBHAV / "src"))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.data import load_dataset  # noqa: E402
from anemia.imaging import read_rgb  # noqa: E402
from anemia.preprocess import prepare_path  # noqa: E402
from anemia.segment import EXTRACTOR_CHOICES, get_extractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", choices=EXTRACTOR_CHOICES, default="refined")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ROI output directory (default: ROI; use a separate directory when comparing extractors)",
    )
    args = parser.parse_args()
    dataset = ROOT / "datasets" / "captured_data.csv"
    output = args.output or (ROOT / "ROI")
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig()
    quality = QualityConfig()
    samples = load_dataset(dataset)
    extractor = get_extractor(args.extractor)
    rows = []
    for index, sample in enumerate(samples, start=1):
        prepared = prepare_path(sample.image_path, extractor, preprocess, quality)
        image_id = sample.patient_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        roi_path = output / f"{image_id}.png"
        cv2.imwrite(str(roi_path), cv2.cvtColor(prepared.crop, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "patient_id": sample.patient_id,
                "image_id": sample.image_path.stem,
                "roi_224x224": roi_path.name,
                "age": "" if sample.age_years is None else sample.age_years,
                "sex": sample.sex or "",
                "hb": sample.hb,
                "usable": prepared.quality.passed,
                "backend": prepared.backend,
                "extractor": args.extractor,
                "mask_ratio": prepared.quality.mask_ratio,
                "quality_reasons": "; ".join(prepared.quality.reasons),
            }
        )
        if index % 25 == 0 or index == len(samples):
            print(f"extracted {index}/{len(samples)}")
    with (output / "roi_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["usable"] is True for row in rows)
    print(f"Wrote {len(rows)} ROI crops to {output} ({passed} passed quality gate).")


if __name__ == "__main__":
    main()
