"""Does each pipeline's mask land on conjunctiva? Measured, not asserted.

There are no ground-truth masks for the local cohort, so Dice cannot be
computed. What *can* be measured is whether the tissue inside each mask looks
like conjunctiva at all, using the same redness index the extractor is built
on: `a* - 0.5*b*` in CIELAB, high on the red mucosal band, low on sclera and
peri-orbital skin.

This exists because the claim "pipeline B's segmentation lands off-tissue on
most captures" was being quoted as the decisive argument for pipeline A while
resting on visual review alone. A number that can be regenerated is worth more
than a recollection, and if the number disagrees with the recollection the
number wins.

Calibration comes from the extraction work: measured across the cohort, masks
sitting on conjunctiva score a median redness around +9, while a
brightness-based mask that demonstrably selects sclera scores around -4. A
threshold of 0.0 sits between those two populations.

    python experiments/12_mask_placement/measure_placement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import (  # noqa: E402
    gray_world_white_balance,
    read_rgb,
    redness_index,
    resize,
)
from anemia.segment import (  # noqa: E402
    heuristic_segmenter,
    legacy_bright_neutral_mask,
)

import cielab_pipeline as cp  # noqa: E402

WORK = (512, 512)
ON_TISSUE = 0.0
"""Redness index above which a mask is judged to be on conjunctiva."""


def redness_in(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0
    if not selected.any():
        return float("nan")
    return float(redness_index(image_rgb)[selected].mean())


def main() -> None:
    rows = []
    for sample in load_local_cohort(ROOT):
        rgb = resize(read_rgb(sample.image_path), WORK)
        balanced = gray_world_white_balance(rgb)

        mask_a, backend_a = heuristic_segmenter(balanced)
        norm_rgb, _ = cp.cielab_normalization(rgb)
        mask_b, backend_b = cp.segment_conjunctiva(norm_rgb)
        mask_legacy = legacy_bright_neutral_mask(balanced)

        # Every mask is scored on the same image, so only placement differs.
        rows.append({
            "capture": f"{sample.image_path.parent.name}/{sample.image_path.name}",
            "A_redness": redness_in(balanced, mask_a),
            "B_redness": redness_in(balanced, mask_b),
            "legacy_redness": redness_in(balanced, mask_legacy),
            "A_backend": backend_a,
            "B_backend": str(backend_b),
            "overlap": float(
                np.count_nonzero((mask_a > 0) & (mask_b > 0))
                / max(np.count_nonzero(mask_b > 0), 1)
            ),
        })

    def summarise(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        off = int(np.count_nonzero(vals <= ON_TISSUE))
        return vals, off

    n = len(rows)
    print(f"{n} captures, redness index inside each pipeline's own mask\n")
    print(f"{'mask':<28}{'median':>9}{'min':>9}{'max':>9}{'off-tissue':>13}")
    print("-" * 68)
    summary = {}
    for key, label in (("A_redness", "Pipeline A (refined)"),
                       ("B_redness", "Pipeline B (CIELAB)"),
                       ("legacy_redness", "brightness baseline")):
        vals, off = summarise(key)
        summary[label] = {
            "median": float(np.median(vals)), "min": float(vals.min()),
            "max": float(vals.max()), "off_tissue": off, "n": int(vals.size),
        }
        print(f"{label:<28}{np.median(vals):>9.2f}{vals.min():>9.2f}"
              f"{vals.max():>9.2f}{f'{off} of {vals.size}':>13}")
    print("-" * 68)
    print(f"\nOff-tissue means redness <= {ON_TISSUE:+.1f} inside the mask, the value "
          f"that separates\nconjunctiva from sclera and skin in the extraction study.")

    overlaps = np.array([r["overlap"] for r in rows])
    print(f"\nFraction of B's mask that also falls inside A's: "
          f"median {np.median(overlaps):.3f}, mean {overlaps.mean():.3f}")

    out = Path(__file__).resolve().parent
    (out / "placement.json").write_text(
        json.dumps({"threshold": ON_TISSUE, "summary": summary, "per_capture": rows},
                   indent=2),
        encoding="utf-8")
    print(f"\nWrote {out / 'placement.json'}")


if __name__ == "__main__":
    main()
