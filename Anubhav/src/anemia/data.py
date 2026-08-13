"""Dataset loaders and the clinical labelling rules.

Two public datasets back this project and they do not share a layout, so each
gets an adapter that emits a common `Sample`. Adding the hospital data later
means writing one more adapter, not touching training code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MASK_SUFFIXES = {
    "palpebral_forniceal": ("_forniceal_palpebral", "_palpebral_forniceal"),
    "palpebral": ("_palpebral",),
    "forniceal": ("_forniceal",),
}


@dataclass
class Sample:
    """One labelled capture, normalised across source datasets."""

    patient_id: str
    """Globally unique and stable — `source:number`. Splits are grouped on this
    so multiple captures of the same eye can never straddle train and test."""

    source: str
    image_path: Path
    hb: float
    age_years: Optional[float] = None
    sex: Optional[str] = None
    mask_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def has_mask(self) -> bool:
        return bool(self.mask_paths)


def anemia_threshold(
    age_years: Optional[float],
    sex: Optional[str],
    *,
    pregnant: bool = False,
) -> float:
    """WHO haemoglobin cutoff in g/dL for the patient in front of you.

    A single fixed cutoff of 11.0 would be correct for young children and
    pregnant women only. Applied to adult men (13.0) it silently labels
    genuinely anemic patients as normal — the exact direction of error a
    screening tool must not make.

    Falls back to 12.0 when age or sex is unknown: the middle of the adult range,
    chosen so an unknown-demographic patient is not assumed healthy.
    """

    if age_years is None:
        return 12.0
    if age_years < 0.5:
        return 13.5
    if age_years < 5:
        return 11.0
    if age_years < 12:
        return 11.5
    if age_years < 15:
        return 12.0

    normalised = (sex or "").strip().upper()[:1]
    if normalised == "M":
        return 13.0
    if normalised == "F":
        return 11.0 if pregnant else 12.0
    return 12.0


def is_anemic(hb: float, age_years: Optional[float], sex: Optional[str]) -> bool:
    return float(hb) < anemia_threshold(age_years, sex)


def _normalise_sex(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    token = str(value).strip().upper()
    if token.startswith("M") or token in {"1", "MALE"}:
        return "M"
    if token.startswith("F") or token in {"0", "2", "FEMALE", "W"}:
        return "F"
    return None


def _find_sheets(directory: Path, recursive: bool = False) -> List[Path]:
    """Metadata sheets in a stable order, ignoring editor lock files.

    Excel leaves `~$Name.xlsx` stubs behind when a sheet is open. They are not
    valid workbooks, and picking one up produces an opaque pandas error rather
    than anything that points at the real cause.
    """

    glob = directory.rglob if recursive else directory.glob
    sheets = [
        path
        for pattern in ("*.xlsx", "*.xls", "*.csv")
        for path in glob(pattern)
        if not path.name.startswith(("~$", "."))
    ]
    return sorted(sheets)


def _read_table(path: Path) -> pd.DataFrame:
    """Read a metadata sheet, failing with something actionable."""

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    try:
        return pd.read_excel(path, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {path.name} needs openpyxl. Install it with "
            "`pip install openpyxl` (it is in requirements.txt)."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse {path} as a workbook ({exc}). If openpyxl is "
            "installed, confirm the file is not truncated or an Excel lock stub."
        ) from exc


def _first_column(table: pd.DataFrame, *candidates: str) -> Optional[str]:
    lowered = {str(col).strip().lower(): col for col in table.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for key, original in lowered.items():
            if candidate in key:
                return original
    return None


def _discover_masks(image_path: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    stem = image_path.stem
    for kind, suffixes in MASK_SUFFIXES.items():
        for suffix in suffixes:
            candidate = image_path.with_name(f"{stem}{suffix}.png")
            if candidate.exists():
                masks[kind] = candidate
                break
    return masks


def _is_raw_capture(path: Path) -> bool:
    """Reject the segmentation PNGs that sit beside the raw JPGs."""

    return path.suffix.lower() in {".jpg", ".jpeg"} and not any(
        marker in path.stem.lower() for marker in ("palpebral", "forniceal", "mask")
    )


def load_eyes_defy_anemia(root: Path, subsets: Sequence[str] = ("India", "Italy")) -> List[Sample]:
    """Loader for the Eyes-defy-anemia dataset (IEEE DataPort / Kaggle).

    Layout: `<root>/<Subset>/<Subset>.xlsx` plus numbered folders whose names
    match the sheet's `Number` column. Each folder holds the raw JPG and the
    hand-drawn conjunctiva masks.

    Keeps *every* capture in a patient folder rather than only the first, and
    does not cap the record count.
    """

    root = Path(root)
    samples: List[Sample] = []

    for subset in subsets:
        subset_dir = root / subset
        if not subset_dir.is_dir():
            subset_dir = root if root.name.lower() == subset.lower() else subset_dir
            if not subset_dir.is_dir():
                continue

        sheets = _find_sheets(subset_dir)
        if not sheets:
            continue

        # Prefer a sheet named after the subset (India.xlsx) when several exist.
        sheet_path = next(
            (p for p in sheets if p.stem.lower() == subset.lower()), sheets[0]
        )
        table = _read_table(sheet_path)

        number_col = _first_column(table, "number", "id", "patient")
        hb_col = _first_column(table, "hgb", "hb", "haemoglobin", "hemoglobin")
        age_col = _first_column(table, "age")
        sex_col = _first_column(table, "sex", "gender")
        if number_col is None or hb_col is None:
            raise ValueError(f"{sheet_path} lacks a patient-number or Hb column")

        for _, row in table.iterrows():
            if pd.isna(row[number_col]) or pd.isna(row[hb_col]):
                continue
            number = str(row[number_col]).strip()
            if number.endswith(".0"):
                number = number[:-2]

            patient_dir = subset_dir / number
            if not patient_dir.is_dir():
                continue

            age = float(row[age_col]) if age_col and not pd.isna(row[age_col]) else None
            sex = _normalise_sex(row[sex_col]) if sex_col else None

            for image_path in sorted(p for p in patient_dir.iterdir() if _is_raw_capture(p)):
                samples.append(
                    Sample(
                        patient_id=f"eyesdefy-{subset.lower()}:{number}",
                        source=f"eyes-defy-anemia/{subset.lower()}",
                        image_path=image_path,
                        hb=float(row[hb_col]),
                        age_years=age,
                        sex=sex,
                        mask_paths=_discover_masks(image_path),
                    )
                )

    return samples


def load_cp_anemic(root: Path) -> List[Sample]:
    """Loader for CP-AnemiC (Mendeley, Ghana, children 6-59 months).

    No segmentation masks ship with this dataset, so its samples train the Hb
    regressor only — the segmenter has to come from Eyes-defy-anemia.

    The exact column names in the distributed metadata sheet have not been
    verified against a real download yet, so lookup is tolerant: it matches on
    substrings and raises with the observed columns if nothing fits. Confirm the
    parsed count against the published 710 before trusting any run.
    """

    root = Path(root)
    sheets = _find_sheets(root, recursive=True)
    if not sheets:
        raise FileNotFoundError(f"No metadata sheet found under {root}")

    sheet_path = sheets[0]
    table = _read_table(sheet_path)

    hb_col = _first_column(table, "hb", "hgb", "haemoglobin", "hemoglobin")
    name_col = _first_column(table, "image", "file", "filename", "photo")
    age_col = _first_column(table, "age")
    sex_col = _first_column(table, "sex", "gender")
    if hb_col is None or name_col is None:
        raise ValueError(
            f"{sheet_path} lacks an Hb or image-name column. Observed columns: {list(table.columns)}"
        )

    index = {path.name.lower(): path for path in root.rglob("*") if _is_raw_capture(path)}
    index.update({path.stem.lower(): path for path in root.rglob("*") if _is_raw_capture(path)})

    samples: List[Sample] = []
    for row_number, row in table.iterrows():
        if pd.isna(row[hb_col]) or pd.isna(row[name_col]):
            continue
        key = str(row[name_col]).strip().lower()
        image_path = index.get(key) or index.get(Path(key).stem)
        if image_path is None:
            continue

        age_months = float(row[age_col]) if age_col and not pd.isna(row[age_col]) else None
        # The published cohort is 6-59 months; a value under 6 is already years.
        age_years = None
        if age_months is not None:
            age_years = age_months if age_months < 6 else age_months / 12.0

        samples.append(
            Sample(
                patient_id=f"cp-anemic:{row.get('ID', row_number)}",
                source="cp-anemic",
                image_path=image_path,
                hb=float(row[hb_col]),
                age_years=age_years,
                sex=_normalise_sex(row[sex_col]) if sex_col else None,
            )
        )

    return samples


def load_local_cohort(
    root: Path,
    sheet_name: str = "DATASAMPLE.csv",
    eye_dirs: Sequence[str] = ("left_eye", "right_eye"),
) -> List[Sample]:
    """Loader for the locally collected cohort.

    Layout: a metadata sheet keyed by `Image ID`, plus one folder per eye whose
    filenames are that same ID (`left_eye/7.jpeg`, `right_eye/7.jpeg`).

    Both eyes of a subject share a `patient_id`. This is the single most
    important detail in this loader: the two captures of one person are highly
    correlated, so if they land on opposite sides of a train/test split the
    model scores well by recognising the subject rather than reading pallor.
    Grouping is what prevents that.

    Age is computed at the date of capture rather than today, so the WHO
    threshold applied is the one that was correct when the blood was drawn.
    """

    root = Path(root)
    sheet = root / sheet_name
    if not sheet.exists():
        candidates = _find_sheets(root)
        if not candidates:
            raise FileNotFoundError(f"No metadata sheet under {root}")
        sheet = candidates[0]

    table = _read_table(sheet)
    id_col = _first_column(table, "image id", "id", "image")
    hb_col = _first_column(table, "haemoglobin", "hemoglobin", "hgb", "hb")
    dob_col = _first_column(table, "date of birth", "dob")
    sex_col = _first_column(table, "gender", "sex")
    taken_col = _first_column(table, "created on", "date")
    if id_col is None or hb_col is None:
        raise ValueError(
            f"{sheet} needs an image-ID and a haemoglobin column. Observed: {list(table.columns)}"
        )

    samples: List[Sample] = []
    for _, row in table.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row[hb_col]):
            continue
        image_id = str(row[id_col]).strip()
        if image_id.endswith(".0"):
            image_id = image_id[:-2]

        age_years = None
        if dob_col and not pd.isna(row[dob_col]):
            born = pd.to_datetime(row[dob_col], errors="coerce", format="mixed")
            taken = (
                pd.to_datetime(str(row[taken_col])[:24], errors="coerce")
                if taken_col and not pd.isna(row.get(taken_col))
                else None
            )
            if pd.notna(born):
                reference = taken if taken is not None and pd.notna(taken) else pd.Timestamp.now()
                age_years = float((reference - born).days) / 365.25

        for eye in eye_dirs:
            for suffix in (".jpeg", ".jpg", ".png"):
                image_path = root / eye / f"{image_id}{suffix}"
                if image_path.exists():
                    samples.append(
                        Sample(
                            # Both eyes share one id so grouping keeps them together.
                            patient_id=f"local:{image_id}",
                            source=f"local/{eye}",
                            image_path=image_path,
                            hb=float(row[hb_col]),
                            age_years=age_years,
                            sex=_normalise_sex(row[sex_col]) if sex_col else None,
                        )
                    )
                    break

    if not samples:
        raise RuntimeError(f"Metadata parsed but no images matched under {root}")
    return samples


def load_unlabelled(root: Path, recursive: bool = True) -> List[Sample]:
    """Load a folder of conjunctiva photographs that carry no Hb labels.

    Unlabelled images cannot train or evaluate the regressor — there is nothing
    to regress against. They are still worth a great deal for the things that
    do not need a label:

    * checking whether segmentation lands on conjunctiva in real photographs,
      which phantoms cannot tell you;
    * calibrating the quality-control thresholds against real captures;
    * producing honest figures for the report.

    `hb` is set to NaN so that any attempt to train on these fails loudly rather
    than silently learning from a placeholder value.
    """

    root = Path(root)
    glob = root.rglob if recursive else root.glob
    images = sorted(
        (p for p in glob("*") if _is_raw_capture(p)),
        key=lambda p: (p.parent.name, _natural_key(p.stem)),
    )

    return [
        Sample(
            patient_id=f"unlabelled:{path.parent.name}/{path.stem}",
            source=f"unlabelled/{path.parent.name}",
            image_path=path,
            hb=float("nan"),
        )
        for path in images
    ]


def _natural_key(stem: str):
    """Sort `2` before `10` rather than lexicographically."""

    return (0, int(stem)) if stem.isdigit() else (1, stem)


def has_labels(samples: Sequence[Sample]) -> bool:
    return bool(samples) and not all(np.isnan(s.hb) for s in samples)


LOADERS = {
    "eyes-defy-anemia": load_eyes_defy_anemia,
    "cp-anemic": load_cp_anemic,
    "unlabelled": load_unlabelled,
}


def load_dataset(root: Path) -> List[Sample]:
    """Dispatch on directory shape so `--data` can point at either dataset."""

    root = Path(root)
    if any((root / subset).is_dir() for subset in ("India", "Italy")):
        return load_eyes_defy_anemia(root)
    if any((root / eye).is_dir() for eye in ("left_eye", "right_eye")):
        return load_local_cohort(root)
    if re.search(r"(cp[-_]?anemic|ghana)", root.name, re.IGNORECASE):
        return load_cp_anemic(root)
    if list(root.glob("*.xlsx")) and any(p.is_dir() and p.name.isdigit() for p in root.iterdir()):
        return load_eyes_defy_anemia(root.parent, subsets=(root.name,))

    # No metadata sheet anywhere, but images present: an unlabelled collection.
    if not _find_sheets(root, recursive=True) and any(
        _is_raw_capture(p) for p in root.rglob("*")
    ):
        return load_unlabelled(root)

    raise ValueError(
        f"Could not infer dataset layout for {root}. "
        "Call load_eyes_defy_anemia, load_cp_anemic, or load_unlabelled directly."
    )


def summarise(samples: Sequence[Sample]) -> dict:
    """Cohort description for the report — and a sanity check on the loader."""

    if not samples:
        return {"images": 0, "patients": 0}

    if not has_labels(samples):
        return {
            "images": len(samples),
            "patients": len({s.patient_id for s in samples}),
            "with_masks": sum(1 for s in samples if s.has_mask),
            "sources": sorted({s.source for s in samples}),
            "labelled": False,
            "note": "No Hb labels — usable for segmentation and QC checks only, not training.",
        }

    hb_values = np.array([s.hb for s in samples], dtype=np.float32)
    anemic = [is_anemic(s.hb, s.age_years, s.sex) for s in samples]
    return {
        "images": len(samples),
        "patients": len({s.patient_id for s in samples}),
        "with_masks": sum(1 for s in samples if s.has_mask),
        "sources": sorted({s.source for s in samples}),
        "hb_min": float(hb_values.min()),
        "hb_max": float(hb_values.max()),
        "hb_mean": float(hb_values.mean()),
        "hb_std": float(hb_values.std()),
        "anemic": int(sum(anemic)),
        "anemic_fraction": float(np.mean(anemic)),
        "severe_under_9": int(np.count_nonzero(hb_values < 9.0)),
    }
