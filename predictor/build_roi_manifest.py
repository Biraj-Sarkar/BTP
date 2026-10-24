"""Build the predictor manifest by joining current dataset labels to extracted ROIs."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "captured_data.csv"
EXTRACTION = ROOT / "ROI" / "extraction_manifest.csv"
OUTPUT = ROOT / "ROI" / "roi_manifest.csv"


def main() -> None:
    labels: dict[str, dict[str, str]] = {}
    def keys_for(value: str) -> set[str]:
        stem = Path(value).stem.strip()
        return {stem, stem.removeprefix("captured_")}
    with DATASET.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                images = json.loads(row.get("Images", "[]"))
            except json.JSONDecodeError:
                continue
            for image in images:
                for name in keys_for(image.get("image_path", "")):
                    if name:
                        labels[name] = row
            for name in keys_for(row.get("RegistrationNo", "")):
                if name:
                    labels[name] = row

    output_rows = []
    with EXTRACTION.open(encoding="utf-8", newline="") as handle:
        for ext in csv.DictReader(handle):
            stem = Path(ext["image"]).stem
            label = labels.get(stem) or labels.get(stem.removeprefix("captured_"))
            if label is None:
                continue
            roi_path = ROOT / "ROI" / ext["roi_file"]
            if not roi_path.is_file():
                continue
            output_rows.append({
                "patient_id": label.get("RegistrationNo", stem),
                "image_id": stem,
                "roi_224x224": ext["roi_file"],
                "age": label.get("AgeInYears", ""),
                "sex": label.get("Gender", ""),
                "hb": label.get("Haemoglobin", ""),
                "usable": str(ext.get("passed", "")).lower() == "true",
                "backend": ext.get("backend", "unet"),
                "extractor": "unet",
                "mask_ratio": ext.get("mask_ratio", ""),
                "quality_reasons": ext.get("reasons", ""),
            })

    fields = ["patient_id", "image_id", "roi_224x224", "age", "sex", "hb", "usable", "backend", "extractor", "mask_ratio", "quality_reasons"]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} joined ROI records to {OUTPUT}")


if __name__ == "__main__":
    main()
