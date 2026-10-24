"""Build the auditable intersection of production and candidate ROI manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["image_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate image IDs in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    production = read_manifest(args.production)
    attention = read_manifest(args.attention)
    ids = sorted(set(production) | set(attention))
    fields = [
        "image_id",
        "source_image",
        "production_roi_path",
        "attention_roi_path",
        "hemoglobin",
        "label_valid",
        "age",
        "sex",
        "split",
        "production_passed",
        "attention_passed",
        "production_empty_mask",
        "attention_empty_mask",
        "production_roi_exists",
        "attention_roi_exists",
        "production_roi_shape",
        "attention_roi_shape",
        "production_roi_dtype",
        "attention_roi_dtype",
        "matched_training_eligible",
        "exclusion_reason",
    ]
    rows = []
    exclusions = []
    for image_id in ids:
        prod = production.get(image_id)
        att = attention.get(image_id)
        if prod is None or att is None:
            raise RuntimeError(f"Manifest ID mismatch for {image_id}")
        comparable = [
            ("source_image", prod["source_image"], att["source_image"]),
            ("hemoglobin", prod["hemoglobin"], att["hemoglobin"]),
            ("label_valid", prod["label_valid"], att["label_valid"]),
            ("split", prod["split"], att["split"]),
        ]
        mismatches = [name for name, left, right in comparable if left != right]
        eligible = (
            prod["training_eligible"].lower() == "true"
            and att["training_eligible"].lower() == "true"
        )
        reasons = []
        if mismatches:
            reasons.append("manifest_mismatch:" + ",".join(mismatches))
        if prod["label_valid"].lower() != "true":
            reasons.append("missing_or_invalid_label")
        if not prod["training_eligible"].lower() == "true":
            reasons.append("production_not_training_eligible")
        if not att["training_eligible"].lower() == "true":
            reasons.append("attention_not_training_eligible")
        reason = ";".join(reasons)
        row = {
            "image_id": image_id,
            "source_image": prod["source_image"],
            "production_roi_path": prod["roi_path"],
            "attention_roi_path": att["roi_path"],
            "hemoglobin": prod["hemoglobin"],
            "label_valid": prod["label_valid"],
            "age": prod["age"],
            "sex": prod["sex"],
            "split": prod["split"],
            "production_passed": prod["passed"],
            "attention_passed": att["passed"],
            "production_empty_mask": prod["empty_mask"],
            "attention_empty_mask": att["empty_mask"],
            "production_roi_exists": prod["roi_exists"],
            "attention_roi_exists": att["roi_exists"],
            "production_roi_shape": prod["roi_shape"],
            "attention_roi_shape": att["roi_shape"],
            "production_roi_dtype": prod["roi_dtype"],
            "attention_roi_dtype": att["roi_dtype"],
            "matched_training_eligible": str(eligible and not mismatches),
            "exclusion_reason": reason,
        }
        rows.append(row)
        if reason:
            exclusions.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    exclusions_path = args.output.with_name("exclusions.csv")
    with exclusions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exclusions)

    summary = {
        "records": len(rows),
        "matched_training_eligible": sum(
            row["matched_training_eligible"].lower() == "true" for row in rows
        ),
        "excluded": len(exclusions),
        "splits": {
            split: sum(
                row["matched_training_eligible"].lower() == "true"
                and row["split"] == split
                for row in rows
            )
            for split in ("train", "val", "test")
        },
        "excluded_by_reason": {
            "missing_or_invalid_label": sum(
                "missing_or_invalid_label" in row["exclusion_reason"] for row in exclusions
            ),
            "production_extraction_or_roi": sum(
                "production_not_training_eligible" in row["exclusion_reason"]
                for row in exclusions
            ),
            "attention_extraction_or_roi": sum(
                "attention_not_training_eligible" in row["exclusion_reason"]
                for row in exclusions
            ),
        },
        "production_manifest": str(args.production),
        "attention_manifest": str(args.attention),
        "exclusions_path": str(exclusions_path),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
