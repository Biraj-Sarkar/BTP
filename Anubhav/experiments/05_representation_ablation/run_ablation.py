"""Which colour representation carries the most haemoglobin signal?

Four candidate representations of the same conjunctiva crop are compared on the
local cohort (26 patients, laboratory Hb, both eyes):

    1. raw RGB              no colour correction
    2. gray-world RGB       illuminant colour corrected      (main pipeline)
    3. raw a*/b*            CLAHE-on-L leaves colour intact  (cielab pipeline)
    4. gray-world a*/b*     both corrections combined        (hybrid)

Method. Segmentation is computed once per image and shared by all four arms, so
the comparison isolates the *representation* rather than the mask. From each
masked crop a small feature vector is taken (channel means inside the mask;
2-3 numbers), and a ridge regression predicts Hb under leave-one-patient-out
cross-validation. With 26 patients a deliberately tiny feature set avoids
fitting noise; the question is not "how accurate can we get" but "how much
Hb-predictive signal survives each representation".

Every arm is reported against the predict-the-mean baseline. A representation
that cannot beat the mean carries no usable signal at this feature scale.

    python experiments/05_representation_ablation/run_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402


def features(image_rgb: np.ndarray, mask: np.ndarray, space: str) -> np.ndarray:
    """Channel means inside the mask, in the requested representation."""

    selected = mask > 0
    if space == "rgb":
        return image_rgb[selected].mean(axis=0).astype(np.float64)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.array([lab[:, :, 1][selected].mean() - 128.0,
                     lab[:, :, 2][selected].mean() - 128.0], dtype=np.float64)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge on standardised features with an intercept."""

    Xc = np.hstack([np.ones((X.shape[0], 1)), X])
    penalty = alpha * np.eye(Xc.shape[1])
    penalty[0, 0] = 0.0                      # never penalise the intercept
    return np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ y)


def leave_one_out_mae(X: np.ndarray, y: np.ndarray) -> float:
    errors = []
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        mu, sd = X[keep].mean(axis=0), X[keep].std(axis=0)
        sd[sd < 1e-9] = 1.0
        beta = ridge_fit((X[keep] - mu) / sd, y[keep])
        prediction = float(np.hstack([[1.0], (X[i] - mu) / sd]) @ beta)
        errors.append(abs(prediction - y[i]))
    return float(np.mean(errors))


def baseline_mae(y: np.ndarray) -> float:
    """Leave-one-out MAE of always predicting the training mean."""

    errors = [abs(np.mean(np.delete(y, i)) - y[i]) for i in range(len(y))]
    return float(np.mean(errors))


def main() -> None:
    samples = load_local_cohort(ROOT)
    arms = {
        "raw RGB": ("raw", "rgb"),
        "gray-world RGB": ("gw", "rgb"),
        "raw a*/b*": ("raw", "lab"),
        "gray-world a*/b* (hybrid)": ("gw", "lab"),
    }

    collected: dict[str, dict[str, list]] = {name: {} for name in arms}
    hb_by_patient: dict[str, float] = {}
    skipped = 0

    for sample in samples:
        rgb = resize(read_rgb(sample.image_path), (512, 512))
        balanced = gray_world_white_balance(rgb)
        mask, _ = heuristic_segmenter(balanced)   # one mask, shared by all arms
        if not (mask > 0).any():
            skipped += 1
            continue
        hb_by_patient[sample.patient_id] = sample.hb
        for name, (source, space) in arms.items():
            image = balanced if source == "gw" else rgb
            collected[name].setdefault(sample.patient_id, []).append(
                features(image, mask, space))

    patients = sorted(hb_by_patient)
    y = np.array([hb_by_patient[p] for p in patients])

    print(f"{len(patients)} patients, {len(samples) - skipped} images "
          f"({skipped} skipped for empty masks)\n")
    print(f"{'representation':<28}{'features':>9}{'LOO MAE':>10}{'vs baseline':>13}")
    print("-" * 60)

    base = baseline_mae(y)
    for name in arms:
        X = np.array([np.mean(collected[name][p], axis=0) for p in patients])
        mae = leave_one_out_mae(X, y)
        gain = base - mae
        flag = "better" if gain > 0 else "NO BETTER"
        print(f"{name:<28}{X.shape[1]:>9}{mae:>10.3f}{gain:>+9.3f} {flag}")

    print("-" * 60)
    print(f"{'predict-the-mean baseline':<28}{'-':>9}{base:>10.3f}")
    print("\nMAE in g/dL, leave-one-patient-out. Positive 'vs baseline' means the")
    print("representation carries signal beyond the cohort average.")


if __name__ == "__main__":
    main()
