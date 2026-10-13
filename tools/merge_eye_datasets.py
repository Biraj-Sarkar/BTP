"""Merge labelled India/Italy eye captures into the shared datasets folder."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SOURCES = ("India", "Italy")
FIELDS = [
    "RegistrationNo",
    "Gender",
    "AgeInYears",
    "Haemoglobin",
    "CapturedAt",
    "CapturedBy",
    "HbValueEnteredAt",
    "HbValueEnteredBy",
    "Images",
]


def choose_eye_image(folder: Path) -> Path | None:
    """Choose the raw complete-eye JPG, preferring the largest capture."""

    images = sorted(folder.glob("*.jpg"), key=lambda p: p.stat().st_size, reverse=True)
    return images[0] if images else None


def parse_hb(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text == "_":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    csv_path = DATASETS / "captured_data.csv"
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        existing = list(csv.DictReader(handle))

    # The original export pointed at nested capture-storage paths. The suite
    # stores those reference images flat, so make every existing row resolve to
    # the actual file that is present beside this CSV.
    for row in existing:
        registration = str(row.get("RegistrationNo", "")).strip()
        flat_image = DATASETS / f"{registration}.jpg"
        if not flat_image.is_file():
            continue
        try:
            records = json.loads(row.get("Images", "[]"))
            for record in records:
                record["image_path"] = flat_image.name
            row["Images"] = json.dumps(records, ensure_ascii=False)
        except (TypeError, json.JSONDecodeError):
            row["Images"] = json.dumps([{"image_path": flat_image.name}])

    existing_ids = {row.get("RegistrationNo", "") for row in existing}
    additions: list[dict[str, str]] = []
    copied = 0
    skipped = 0

    for source in SOURCES:
        source_dir = ROOT / source
        table = pd.read_excel(source_dir / f"{source}.xlsx")
        for _, row in table.iterrows():
            number = str(row["Number"]).strip()
            if number.endswith(".0"):
                number = number[:-2]
            hb = parse_hb(row["Hgb"])
            image = choose_eye_image(source_dir / number)
            if hb is None or image is None:
                skipped += 1
                continue

            registration = f"{source}-{int(number):03d}"
            if registration in existing_ids:
                continue
            destination = DATASETS / f"{registration}.jpg"
            shutil.copy2(image, destination)
            gender = str(row["Gender"]).strip().upper()
            gender = "Male" if gender.startswith("M") else "Female" if gender.startswith("F") else gender
            note = "" if pd.isna(row.get("Note")) else str(row.get("Note")).strip()
            image_record = {"image_path": destination.name, "source_dataset": source}
            if note:
                image_record["note"] = note
            additions.append(
                {
                    "RegistrationNo": registration,
                    "Gender": gender,
                    "AgeInYears": str(float(row["Age"])).rstrip("0").rstrip("."),
                    "Haemoglobin": f"{hb:g}",
                    "CapturedAt": "",
                    "CapturedBy": f"Eyes-Defy-Anemia {source}",
                    "HbValueEnteredAt": "",
                    "HbValueEnteredBy": f"Eyes-Defy-Anemia {source}",
                    "Images": json.dumps([image_record], ensure_ascii=False),
                }
            )
            existing_ids.add(registration)
            copied += 1

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(additions)
    print(f"Copied {copied} labelled eye images; skipped {skipped} rows without valid Hb/image.")
    print(f"Updated {csv_path} with {len(existing) + len(additions)} total rows.")


if __name__ == "__main__":
    main()
