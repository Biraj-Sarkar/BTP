"""Synthetic eye phantoms with known ground truth.

Not a substitute for the real datasets — they exist so the pipeline is testable
before the hospital data arrives, and so the segmentation claim (a brightness-
based mask selects sclera, not conjunctiva) can be checked rather than
asserted.

Layout mimics an everted lower lid: skin surround, a bright sclera band, a dark
iris, and the red palpebral conjunctiva strip along the bottom.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

SKIN = (198, 154, 124)      # warm, yellow-leaning
SCLERA = (236, 234, 230)    # bright, near-neutral
IRIS = (78, 62, 48)         # dark
CONJUNCTIVA_HEALTHY = (196, 92, 92)
CONJUNCTIVA_PALE = (216, 168, 162)


def make_eye(
    size: Tuple[int, int] = (512, 512),
    hb: float = 14.0,
    seed: int = 0,
    blur: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb_image, conjunctiva_mask).

    Conjunctiva colour interpolates between pale and healthy with `hb`, so a
    regressor trained on these phantoms should recover the relationship — a
    useful end-to-end check that the plumbing carries signal.
    """

    rng = np.random.default_rng(seed)
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = SKIN

    pallor = np.clip((hb - 7.0) / 8.0, 0.0, 1.0)
    conjunctiva_colour = tuple(
        int(pale + (healthy - pale) * pallor)
        for pale, healthy in zip(CONJUNCTIVA_PALE, CONJUNCTIVA_HEALTHY)
    )

    # Sclera: bright almond in the upper-middle of the frame.
    sclera_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        sclera_mask,
        (width // 2, int(height * 0.42)),
        (int(width * 0.34), int(height * 0.17)),
        0, 0, 360, 255, -1,
    )
    image[sclera_mask > 0] = SCLERA

    # Iris disc.
    cv2.circle(image, (width // 2, int(height * 0.42)), int(height * 0.11), IRIS, -1)

    # Palpebral conjunctiva: red band below the sclera.
    conjunctiva_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        conjunctiva_mask,
        (width // 2, int(height * 0.66)),
        (int(width * 0.30), int(height * 0.075)),
        0, 0, 360, 255, -1,
    )
    image[conjunctiva_mask > 0] = conjunctiva_colour

    noise = rng.normal(0, 4.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if blur > 0:
        kernel = int(blur) * 2 + 1
        image = cv2.GaussianBlur(image, (kernel, kernel), blur)

    return image, conjunctiva_mask


def make_cohort(count: int = 40, seed: int = 0):
    """Phantoms spanning a realistic Hb range, two captures per patient."""

    rng = np.random.default_rng(seed)
    cohort = []
    for patient in range(count):
        hb = float(np.clip(rng.normal(12.0, 2.4), 6.0, 17.0))
        for capture in range(2):
            image, mask = make_eye(hb=hb, seed=patient * 10 + capture)
            cohort.append({"patient": f"p{patient}", "hb": hb, "image": image, "mask": mask})
    return cohort
