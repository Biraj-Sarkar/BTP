"""Manifest loading, verification, and data integrity auditing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class SampleRecord:
    patient_id: str
    image_id: str
    image_path: Path
    hb: float
    age: Optional[float] = None
    sex: Optional[str] = None
    usable: bool = True
    mask_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "hb": self.hb,
            "age": self.age,
            "sex": self.sex,
            "usable": self.usable,
            "mask_ratio": self.mask_ratio,
        }


def normalize_sex(val: Any) -> Optional[str]:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip().upper()
    if s.startswith("M") or s in {"1", "MALE"}:
        return "M"
    if s.startswith("F") or s in {"0", "2", "FEMALE", "W"}:
        return "F"
    return None


def load_and_validate_manifest(
    manifest_path: Path | str,
    filter_usable: bool = True,
    min_hb: float = 3.0,
    max_hb: float = 25.0,
) -> Tuple[List[SampleRecord], Dict[str, Any]]:
    """Loads manifest CSV, validates files on disk, verifies physiological labels, and returns clean records."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    total_raw_rows = len(df)
    parent_dir = path.parent

    # Detect column names across formats
    patient_col = "patient_id" if "patient_id" in df.columns else "RegistrationNo"
    image_col = "roi_224x224" if "roi_224x224" in df.columns else ("image_id" if "image_id" in df.columns else "Images")
    hb_col = "hb" if "hb" in df.columns else "Haemoglobin"
    age_col = "age" if "age" in df.columns else "AgeInYears"
    sex_col = "sex" if "sex" in df.columns else "Gender"
    usable_col = "usable" if "usable" in df.columns else None

    records: List[SampleRecord] = []
    missing_files = 0
    invalid_hb_count = 0
    rejected_quality = 0

    for _, row in df.iterrows():
        # Validate Hb
        raw_hb = row.get(hb_col)
        if pd.isna(raw_hb):
            invalid_hb_count += 1
            continue
        try:
            hb_val = float(str(raw_hb).replace(",", ".").strip())
        except ValueError:
            invalid_hb_count += 1
            continue

        if not (min_hb <= hb_val <= max_hb):
            invalid_hb_count += 1
            continue

        # Check usable gate
        usable = True
        if usable_col and usable_col in row:
            val_usable = str(row[usable_col]).strip().lower()
            usable = val_usable in {"true", "1", "yes"}

        if filter_usable and not usable:
            rejected_quality += 1
            continue

        # Resolve image path
        raw_img = str(row.get(image_col, "")).strip()
        img_file = parent_dir / raw_img
        if not img_file.is_file():
            # Try flat in parent directory or datasets directory
            alt = parent_dir / f"{raw_img}.png"
            if alt.is_file():
                img_file = alt
            else:
                missing_files += 1
                continue

        patient_id = str(row.get(patient_col, img_file.stem)).strip()
        image_id = img_file.stem

        raw_age = row.get(age_col)
        age_val = None
        if not pd.isna(raw_age) and str(raw_age).strip():
            try:
                age_val = float(raw_age)
            except ValueError:
                pass

        sex_val = normalize_sex(row.get(sex_col))
        mask_ratio = float(row["mask_ratio"]) if "mask_ratio" in row and not pd.isna(row["mask_ratio"]) else None

        records.append(
            SampleRecord(
                patient_id=patient_id,
                image_id=image_id,
                image_path=img_file,
                hb=hb_val,
                age=age_val,
                sex=sex_val,
                usable=usable,
                mask_ratio=mask_ratio,
            )
        )

    hb_arr = np.array([r.hb for r in records], dtype=np.float32) if records else np.array([])
    unique_patients = len({r.patient_id for r in records})

    audit_summary = {
        "manifest_path": str(path),
        "total_rows": total_raw_rows,
        "valid_samples": len(records),
        "unique_patients": unique_patients,
        "missing_files": missing_files,
        "invalid_hb_count": invalid_hb_count,
        "rejected_quality": rejected_quality,
        "hb_mean": float(hb_arr.mean()) if len(hb_arr) else 0.0,
        "hb_std": float(hb_arr.std()) if len(hb_arr) else 0.0,
        "hb_min": float(hb_arr.min()) if len(hb_arr) else 0.0,
        "hb_max": float(hb_arr.max()) if len(hb_arr) else 0.0,
    }

    return records, audit_summary

