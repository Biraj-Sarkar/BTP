"""Predict haemoglobin for a folder of patients. No training.

This is the everyday entry point. It **loads a previously fitted model file**
and applies it — nothing is trained, so it runs in seconds and gives identical
answers every time.

    Two-step workflow
    -----------------
    1. FIT (rarely — only when new *labelled* data arrives):
           python -m anemia fit-linear --data . --out runs/linear_model.json
       Writes the equation, its coefficients, the feature specification and the
       standardisation constants into one JSON file.

    2. PREDICT (any time, on any number of new patients):
           python predict_folder.py --dir new_patients --out predictions.csv
       Loads that JSON and applies it. No labels needed, no training.

    Expected folder layout
    ----------------------
        new_patients/
            left_eye/   1.jpg  2.jpg  ...
            right_eye/  1.jpg  2.jpg  ...
            patients.csv          (optional: ID, Date Of Birth or Age, Gender)

    Filenames are the patient ID. A patient may have one eye or both; when both
    are present their features are averaged, exactly as during fitting.

    Age and sex are optional but recommended — without them the WHO anaemia
    threshold falls back to 12.0 g/dL rather than the value correct for that
    patient.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.data import _normalise_sex, anemia_threshold  # noqa: E402
from anemia.imaging import read_rgb  # noqa: E402
from anemia.linear_model import LinearModel, extract_features  # noqa: E402
from anemia.preprocess import prepare  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

EYE_DIRS = ("left_eye", "right_eye")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_metadata(folder: Path) -> dict:
    """Optional per-patient age and sex, keyed by patient ID.

    Tolerant about column names and about whether age is given directly or as a
    date of birth, because a field spreadsheet will not match a fixed schema.
    """

    sheets = [p for p in folder.glob("*.csv")] + [p for p in folder.glob("*.xlsx")]
    sheets = [p for p in sheets if not p.name.startswith(("~$", "."))]
    if not sheets:
        return {}

    sheet = sheets[0]
    table = pd.read_excel(sheet, engine="openpyxl") if sheet.suffix == ".xlsx" else pd.read_csv(sheet)
    lowered = {str(c).strip().lower(): c for c in table.columns}

    def column(*names):
        """Exact match first, then whole-word match.

        Whole-word matters: a bare substring test makes `"age"` match
        `"image id"`, which would silently use the patient ID as the age and
        produce wrong clinical thresholds.
        """

        for name in names:
            if name in lowered:
                return lowered[name]
        for name in names:
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            for key, original in lowered.items():
                if pattern.search(key):
                    return original
        return None

    id_col = column("image id", "patient", "id")
    age_col = column("age")
    dob_col = column("date of birth", "dob")
    sex_col = column("gender", "sex")
    if id_col is None:
        print(f"  note: {sheet.name} has no recognisable ID column — ignoring it")
        return {}

    meta = {}
    for _, row in table.iterrows():
        if pd.isna(row[id_col]):
            continue
        key = str(row[id_col]).strip()
        if key.endswith(".0"):
            key = key[:-2]

        age = None
        if age_col and not pd.isna(row[age_col]):
            age = float(row[age_col])
        elif dob_col and not pd.isna(row[dob_col]):
            born = pd.to_datetime(row[dob_col], errors="coerce", format="mixed")
            if pd.notna(born):
                age = float((pd.Timestamp.now() - born).days) / 365.25

        meta[key] = {"age": age,
                     "sex": _normalise_sex(row[sex_col]) if sex_col else None}
    print(f"  metadata: {len(meta)} patients from {sheet.name}")
    return meta


def gather_images(folder: Path) -> dict:
    """patient ID -> {eye: path}, from the left_eye/right_eye layout."""

    patients: dict = {}
    for eye in EYE_DIRS:
        directory = folder / eye
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                patients.setdefault(path.stem, {})[eye] = path

    if not patients:                                   # flat folder fallback
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                patients[path.stem] = {"single": path}
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True,
                        help="folder containing left_eye/ and right_eye/")
    parser.add_argument("--model", type=Path, default=ROOT / "runs" / "linear_model.json",
                        help="fitted model file (produced once by `anemia fit-linear`)")
    parser.add_argument("--out", type=Path, default=None, help="output CSV")
    parser.add_argument("--extractor", default=None,
                        choices=["refined", "redness", "brightness", "grabcut", "cielab"],
                        help="override the extractor recorded in the model file "
                             "(normally leave unset — a mismatch invalidates the model)")
    parser.add_argument("--segmenter", type=Path, default=None,
                        help="trained segmenter directory, if available")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"No model at {args.model}.\n"
            "Fit one first (needs labelled data, and only has to be done once):\n"
            "    python -m anemia fit-linear --data . --out runs/linear_model.json")

    model = LinearModel.load(args.model)
    # The extractor comes from the model file, not from a default: features
    # measured inside a different mask are not comparable, so serving with the
    # wrong extractor would be silently wrong rather than obviously broken.
    extractor_name = args.extractor or model.extractor
    segmenter = get_extractor(extractor_name, args.segmenter)
    preprocess = PreprocessConfig(work_size=tuple(model.work_size),
                                  gray_world=model.gray_world)
    quality = QualityConfig()

    print(f"Model:     {args.model}")
    print(f"Extractor: {extractor_name}" + ("  (OVERRIDDEN — may not match how the "
          "model was fitted)" if args.extractor and args.extractor != model.extractor else ""))
    print(f"Equation:  {model.equation_string()}")
    held = model.metrics.get("held_out") or model.metrics.get("leave_one_out")
    if held:
        print(f"Held-out MAE at fit time: {held.get('mae', float('nan')):.3f} g/dL "
              f"on {model.metrics.get('patients', '?')} patients "
              f"(NOT a clinical accuracy figure)")
    print()

    patients = gather_images(args.dir)
    if not patients:
        raise SystemExit(f"No images found under {args.dir}")
    print(f"Found {len(patients)} patients in {args.dir}")
    metadata = load_metadata(args.dir)
    print()

    rows = []
    for patient_id in sorted(patients, key=lambda k: (0, int(k)) if k.isdigit() else (1, k)):
        info = metadata.get(patient_id, {})
        age, sex = info.get("age"), info.get("sex")
        threshold = anemia_threshold(age, sex)

        features, used, rejected = [], [], []
        for eye, path in patients[patient_id].items():
            prepared = prepare(read_rgb(path), segmenter, preprocess, quality)
            if not prepared.quality.passed or not (prepared.mask > 0).any():
                rejected.append(f"{eye}: {'; '.join(prepared.quality.reasons) or 'empty mask'}")
                continue
            features.append(extract_features(prepared.balanced, prepared.mask,
                                             model.representation))
            used.append(eye)

        if not features:
            rows.append({"patient": patient_id, "hb_g_dl": "", "anemic": "",
                         "threshold_g_dl": threshold, "usable": False,
                         "eyes_used": 0, "age_years": age, "sex": sex,
                         "note": "; ".join(rejected) or "no usable capture"})
            print(f"  {patient_id:>6}   RETAKE  ({'; '.join(rejected)})")
            continue

        # Average both eyes before predicting — the same aggregation used when
        # the model was fitted, so serving matches training.
        hb = model.predict_features(np.mean(features, axis=0))
        anemic = hb < threshold
        rows.append({"patient": patient_id, "hb_g_dl": round(hb, 2),
                     "anemic": anemic, "threshold_g_dl": threshold, "usable": True,
                     "eyes_used": len(used), "age_years": age, "sex": sex,
                     "note": "; ".join(rejected)})
        flag = "ANAEMIC" if anemic else "normal "
        print(f"  {patient_id:>6}   {hb:6.2f} g/dL   {flag}  "
              f"(threshold {threshold:.1f}, {len(used)} eye{'s' if len(used) > 1 else ''})")

    out = args.out or args.dir / "predictions.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    usable = [r for r in rows if r["usable"]]
    flagged = [r for r in usable if r["anemic"]]
    print(f"\n{len(usable)}/{len(rows)} patients scored, {len(flagged)} flagged anaemic")
    print(f"Wrote {out}")
    print("\nScreening estimates only — not a diagnosis. Confirm with a laboratory test.")


if __name__ == "__main__":
    main()
