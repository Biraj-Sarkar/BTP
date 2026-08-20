"""Linear haemoglobin model: feature extraction, fitting, and a servable file.

The neural regressor is not the only predictor. A ridge regression over colour
statistics of the extracted conjunctiva is fully specified by a handful of
numbers, runs in microseconds, and is completely interpretable — you can read
its equation and check whether the coefficients make physical sense.

The same self-describing principle as the neural `Checkpoint` applies here: a
saved model carries its coefficients, its feature specification, its
standardisation constants and its preprocessing settings. Coefficients alone
are useless, because they only mean anything relative to the features and the
scaling they were fitted with.

**Feature extraction lives here and is used by both fitting and serving**, so
the two cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

# Representation names must stay stable: they are written into saved models.
REPRESENTATIONS = ("means", "chroma", "erythema", "lab", "a_only")

FEATURE_NAMES: Dict[str, List[str]] = {
    "means": ["mean R", "mean G", "mean B"],
    "chroma": ["r chromaticity", "g chromaticity"],
    "erythema": ["log(R/G)", "log(R/B)"],
    "lab": ["mean a*", "mean b*"],
    "a_only": ["mean a*"],
}


def extract_features(image_rgb: np.ndarray, mask: np.ndarray, representation: str) -> np.ndarray:
    """Colour statistics over the masked conjunctiva, in one representation.

    `chroma` and `erythema` divide out overall intensity, so a brighter capture
    of the same tissue yields the same numbers. That invariance is why they
    outperform raw channel means on this data.
    """

    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}; "
                         f"choose from {list(REPRESENTATIONS)}")

    selected = mask > 0
    if not selected.any():
        raise ValueError("Empty mask — no conjunctiva pixels to measure")

    pixels = image_rgb[selected].astype(np.float64)
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    if representation == "means":
        return np.array([R.mean(), G.mean(), B.mean()])

    if representation == "chroma":
        total = np.maximum(R + G + B, 1e-6)
        return np.array([(R / total).mean(), (G / total).mean()])

    if representation == "erythema":
        return np.array([np.log(np.maximum(R, 1) / np.maximum(G, 1)).mean(),
                         np.log(np.maximum(R, 1) / np.maximum(B, 1)).mean()])

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = (lab[:, :, 1][selected] - 128.0).astype(np.float64)
    if representation == "a_only":
        return np.array([a_star.mean()])
    b_star = (lab[:, :, 2][selected] - 128.0).astype(np.float64)
    return np.array([a_star.mean(), b_star.mean()])


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Closed-form ridge on standardised features. Returns the pieces needed to predict."""

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0

    design = np.hstack([np.ones((len(y), 1)), (X - mean) / scale])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0                      # never penalise the intercept
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(beta[0]), beta[1:], mean, scale


@dataclass
class LinearModel:
    """A servable linear haemoglobin model.

    Everything needed to turn an extracted conjunctiva into g/dL travels in one
    file: which features to compute, how to standardise them, and the fitted
    coefficients.
    """

    representation: str
    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float = 1.0
    work_size: Sequence[int] = (512, 512)
    gray_world: bool = True
    extractor: str = "refined"
    """Which conjunctiva extractor produced the features this model was fitted
    on. Features measured inside a different mask are not comparable, so serving
    with a different extractor is silently wrong — the predictor reads this back
    rather than assuming a default."""

    metrics: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- use

    def predict_features(self, features: np.ndarray) -> float:
        """Apply the equation to an already-extracted feature vector."""

        features = np.asarray(features, dtype=np.float64).reshape(-1)
        if features.shape[0] != self.coefficients.shape[0]:
            raise ValueError(
                f"Expected {self.coefficients.shape[0]} features for "
                f"{self.representation!r}, got {features.shape[0]}")
        standardised = (features - self.feature_mean) / self.feature_scale
        return float(self.intercept + standardised @ self.coefficients)

    def predict_image(self, image_rgb: np.ndarray, mask: np.ndarray) -> float:
        """Extract features from the masked conjunctiva, then apply the equation."""

        return self.predict_features(
            extract_features(image_rgb, mask, self.representation))

    # ------------------------------------------------------------ display

    def raw_equation(self):
        """Coefficients expanded to raw feature units, so the equation is readable.

        Algebraically identical to the standardised form; folding the mean and
        scale into the coefficients makes it directly usable by hand.
        """

        raw = self.coefficients / self.feature_scale
        intercept = self.intercept - float((self.coefficients * self.feature_mean
                                            / self.feature_scale).sum())
        return intercept, raw

    def equation_string(self) -> str:
        intercept, raw = self.raw_equation()
        names = FEATURE_NAMES[self.representation]
        terms = "".join(f" {'+' if c >= 0 else '-'} {abs(c):.6f}*({n})"
                        for c, n in zip(raw, names))
        return f"Hb = {intercept:.4f}{terms}   [g/dL]"

    # --------------------------------------------------------------- i/o

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "anemia-linear-model",
            "format_version": 1,
            "representation": self.representation,
            "feature_names": FEATURE_NAMES[self.representation],
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "alpha": self.alpha,
            "work_size": list(self.work_size),
            "gray_world": self.gray_world,
            "extractor": self.extractor,
            "metrics": self.metrics,
            "equation": self.equation_string(),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearModel":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = {"representation", "intercept", "coefficients",
                   "feature_mean", "feature_scale"} - set(blob)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. Coefficients alone cannot "
                "be served — the feature specification and standardisation "
                "constants are required to interpret them.")
        return cls(
            representation=blob["representation"],
            intercept=float(blob["intercept"]),
            coefficients=np.asarray(blob["coefficients"], dtype=np.float64),
            feature_mean=np.asarray(blob["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(blob["feature_scale"], dtype=np.float64),
            alpha=float(blob.get("alpha", 1.0)),
            work_size=tuple(blob.get("work_size", (512, 512))),
            gray_world=bool(blob.get("gray_world", True)),
            extractor=blob.get("extractor", "refined"),
            metrics=blob.get("metrics", {}),
        )


def fit_linear_model(features: np.ndarray, targets: np.ndarray, representation: str,
                     alpha: float = 1.0, work_size=(512, 512),
                     gray_world: bool = True, extractor: str = "refined",
                     metrics: dict | None = None) -> LinearModel:
    intercept, coefficients, mean, scale = fit_ridge(features, targets, alpha)
    return LinearModel(
        representation=representation,
        intercept=intercept,
        coefficients=coefficients,
        feature_mean=mean,
        feature_scale=scale,
        alpha=alpha,
        work_size=tuple(work_size),
        gray_world=gray_world,
        extractor=extractor,
        metrics=metrics or {},
    )
