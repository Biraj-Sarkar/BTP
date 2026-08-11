"""Write a synthetic dataset in the Eyes-defy-anemia on-disk layout.

Lets the loaders, splits, QC gate, and training loop be exercised end-to-end
before the real data arrives, and gives a target the pipeline *should* be able
to fit — if a run cannot beat the mean baseline on phantoms where pallor is
literally a linear function of Hb, the bug is in the pipeline, not the data.

    python scripts/make_synthetic_dataset.py --out data/synthetic --patients 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from synthetic import make_eye  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--captures", type=int, default=2)
    parser.add_argument("--subset", type=str, default="India")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    subset_dir = args.out / args.subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for patient in range(1, args.patients + 1):
        hb = float(np.clip(rng.normal(12.0, 2.6), 5.5, 17.5))
        age = float(rng.uniform(18, 65))
        sex = "M" if rng.random() < 0.5 else "F"

        patient_dir = subset_dir / str(patient)
        patient_dir.mkdir(exist_ok=True)

        for capture in range(args.captures):
            # A minority of captures are deliberately unusable, so the QC gate
            # has something real to reject.
            blur = 10.0 if rng.random() < 0.08 else 0.0
            image, mask = make_eye(hb=hb, seed=patient * 100 + capture, blur=blur)
            stem = f"{patient}_{capture}"
            cv2.imwrite(str(patient_dir / f"{stem}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(patient_dir / f"{stem}_forniceal_palpebral.png"), mask)

        rows.append({"Number": patient, "Hgb": round(hb, 1), "Age": round(age), "Sex": sex})

    sheet = subset_dir / f"{args.subset}.xlsx"
    pd.DataFrame(rows).to_excel(sheet, index=False)

    print(f"Wrote {args.patients} patients x {args.captures} captures to {subset_dir}")
    print(f"Metadata: {sheet}")


if __name__ == "__main__":
    main()
