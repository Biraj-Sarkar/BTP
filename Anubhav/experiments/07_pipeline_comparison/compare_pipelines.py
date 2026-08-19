"""Both pipelines, end to end, scored on the full metric suite.

Two comparisons are run, because they answer different questions:

  1. END-TO-END — each pipeline uses its *own* segmentation and its *own*
     colour representation, exactly as written. This answers "which pipeline
     would perform better if deployed today", and its result is dominated by
     whichever segmenter finds the tissue.

  2. SHARED-MASK — both representations are measured inside the *same* mask.
     This isolates the colour representation from the segmentation, and
     answers "which colour treatment carries more haemoglobin signal".

Model family is deliberately **linear**: ordinary least squares and ridge, on a
small feature vector, evaluated leave-one-patient-out. No neural network is
involved. The purpose is to establish the floor a CNN must clear — if a linear
model on channel means already explains nothing, that bounds how much of the
signal is in mean colour, and tells the builder what the network has to find
that a linear model cannot.

    python experiments/07_pipeline_comparison/compare_pipelines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from anemia.data import anemia_threshold, is_anemic, load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.segment import heuristic_segmenter  # noqa: E402

import cielab_pipeline as cp  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "06_metrics_evaluation"))
from evaluate import classification_metrics, regression_metrics  # noqa: E402

WORK = (512, 512)


# ---------------------------------------------------------------- models

def fit_predict(X_train, y_train, x_test, alpha):
    """Ridge (alpha>0) or ordinary least squares (alpha=0), standardised."""

    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(y_train), 1)), (X_train - mu) / sd])
    penalty = alpha * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    # lstsq rather than solve so a singular OLS system degrades gracefully
    beta, *_ = np.linalg.lstsq(Z.T @ Z + penalty, Z.T @ y_train, rcond=None)
    return float(np.hstack([[1.0], (x_test - mu) / sd]) @ beta)


def leave_one_out(X, y, alpha):
    return np.array([
        fit_predict(np.delete(X, i, axis=0), np.delete(y, i), X[i], alpha)
        for i in range(len(y))
    ])


# ---------------------------------------------------------------- features

def channel_means(image, mask, space):
    selected = mask > 0
    if space == "rgb":
        return image[selected].mean(axis=0).astype(np.float64)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.array([lab[:, :, 1][selected].mean() - 128.0,
                     lab[:, :, 2][selected].mean() - 128.0], dtype=np.float64)


def rich_features(image, mask):
    """Means, spreads and percentiles — a deliberately over-rich vector."""

    selected = mask > 0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = []
    for plane in (image[:, :, 0], image[:, :, 1], image[:, :, 2],
                  lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0):
        values = plane[selected].astype(np.float64)
        out += [values.mean(), values.std(),
                np.percentile(values, 25), np.percentile(values, 75)]
    return np.array(out)


# ---------------------------------------------------------------- collection

def collect():
    """Per patient: features under every arm, plus label and demographics."""

    data: dict[str, dict] = {}
    seg_backends = {"A": {}, "B": {}}

    for sample in load_local_cohort(ROOT):
        rgb = resize(read_rgb(sample.image_path), WORK)

        # --- Pipeline A: gray-world + refined extractor ---
        balanced = gray_world_white_balance(rgb)
        mask_a, backend_a = heuristic_segmenter(balanced)
        seg_backends["A"][backend_a] = seg_backends["A"].get(backend_a, 0) + 1

        # --- Pipeline B: CLAHE-on-L + its own segmentation ---
        norm_rgb, _ = cp.cielab_normalization(rgb)
        mask_b, backend_b = cp.segment_conjunctiva(norm_rgb)
        seg_backends["B"][backend_b] = seg_backends["B"].get(backend_b, 0) + 1

        if not (mask_a > 0).any() or not (mask_b > 0).any():
            continue

        entry = data.setdefault(sample.patient_id, {
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex,
            "A_e2e": [], "B_e2e": [], "A_shared": [], "B_shared": [], "rich": []})

        # end-to-end: each pipeline's own mask and own representation
        entry["A_e2e"].append(channel_means(balanced, mask_a, "rgb"))
        entry["B_e2e"].append(channel_means(norm_rgb, mask_b, "lab"))
        # shared mask (pipeline A's), representation differs only
        entry["A_shared"].append(channel_means(balanced, mask_a, "rgb"))
        entry["B_shared"].append(channel_means(norm_rgb, mask_a, "lab"))
        entry["rich"].append(rich_features(balanced, mask_a))

    return data, seg_backends


def score(predictions, y, ages, sexes):
    true_an = np.array([is_anemic(h, a, s) for h, a, s in zip(y, ages, sexes)])
    pred_an = np.array([p < anemia_threshold(a, s)
                        for p, a, s in zip(predictions, ages, sexes)])
    return regression_metrics(predictions, y), classification_metrics(pred_an, true_an)


def main() -> None:
    data, seg_backends = collect()
    patients = sorted(data)
    y = np.array([data[p]["hb"] for p in patients])
    ages = [data[p]["age"] for p in patients]
    sexes = [data[p]["sex"] for p in patients]

    print(f"{len(patients)} patients\n")
    print(f"segmentation backends used — pipeline A: {seg_backends['A']}")
    print(f"                             pipeline B: {seg_backends['B']}\n")

    def stack(key):
        return np.array([np.mean(data[p][key], axis=0) for p in patients])

    arms = [
        ("END-TO-END   A: gray-world + refined seg, RGB", "A_e2e", 1.0),
        ("END-TO-END   B: CLAHE-L + own seg, a*/b*", "B_e2e", 1.0),
        ("SHARED-MASK  A: RGB", "A_shared", 1.0),
        ("SHARED-MASK  B: a*/b*", "B_shared", 1.0),
        ("SHARED-MASK  A: RGB, plain OLS", "A_shared", 0.0),
        ("SHARED-MASK  A: 24 rich features, ridge", "rich", 1.0),
        ("SHARED-MASK  A: 24 rich features, plain OLS", "rich", 0.0),
    ]

    baseline = np.array([np.delete(y, i).mean() for i in range(len(y))])
    base_reg, base_cls = score(baseline, y, ages, sexes)

    results = {}
    print(f"{'arm':<46}{'MAE':>7}{'RMSE':>7}{'R2':>8}{'r':>7}{'F1':>7}{'recall':>8}")
    print("-" * 90)
    for label, key, alpha in arms:
        X = stack(key)
        pred = leave_one_out(X, y, alpha)
        reg, cls = score(pred, y, ages, sexes)
        results[label] = {"regression": reg, "classification": cls,
                          "n_features": int(X.shape[1])}
        print(f"{label:<46}{reg['mae']:>7.3f}{reg['rmse']:>7.3f}{reg['r2']:>8.3f}"
              f"{reg.get('pearson_r', float('nan')):>7.3f}{cls['f1']:>7.3f}"
              f"{cls['recall_sensitivity']:>8.3f}")
    print("-" * 90)
    print(f"{'BASELINE: predict the training mean':<46}{base_reg['mae']:>7.3f}"
          f"{base_reg['rmse']:>7.3f}{base_reg['r2']:>8.3f}{'—':>7}"
          f"{base_cls['f1']:>7.3f}{base_cls['recall_sensitivity']:>8.3f}")

    results["BASELINE"] = {"regression": base_reg, "classification": base_cls}
    dest = Path(__file__).resolve().parent / "comparison.json"
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nNo neural network is used here. Linear models only, leave-one-patient-out.")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
