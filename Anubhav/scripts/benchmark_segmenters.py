"""Score every segmentation backend against the Eyes-defy-anemia ground truth.

Run this the moment the dataset lands. It produces the table that justifies the
extraction design on *real* photographs rather than on phantoms:

    python scripts/benchmark_segmenters.py --data ~/Desktop/data/eyes-defy-anemia

The `legacy` row is a brightness-based baseline, included so the comparison is
measured, not asserted — and so the report can quote a before/after number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anemia.config import Config  # noqa: E402
from anemia.data import load_eyes_defy_anemia  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.metrics import dice_score, iou_score  # noqa: E402
from anemia.segment import (  # noqa: E402
    grabcut_mask,
    heuristic_conjunctiva_mask,
    legacy_bright_neutral_mask,
    refined_conjunctiva_mask,
)
from anemia.train_segmenter import load_mask  # noqa: E402


def build_backends(segmenter_dir: Path | None) -> Dict[str, Callable]:
    backends: Dict[str, Callable] = {
        "brightness baseline": legacy_bright_neutral_mask,
        "redness heuristic": heuristic_conjunctiva_mask,
        "refined (seeded grabcut)": refined_conjunctiva_mask,
        "grabcut": grabcut_mask,
    }
    if segmenter_dir:
        from anemia.segment import Mask2FormerSegmenter

        trained = Mask2FormerSegmenter(segmenter_dir)
        backends["mask2former (trained)"] = lambda rgb: trained(rgb)[0]
    return backends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Eyes-defy-anemia root")
    parser.add_argument("--segmenter", type=Path, default=None, help="trained Mask2Former dir")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = Config()
    size = config.preprocess.work_size

    samples = [s for s in load_eyes_defy_anemia(args.data) if s.has_mask]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No masked samples under {args.data}")

    print(f"Scoring {len(samples)} masked images at {size[0]}x{size[1]}\n")
    backends = build_backends(args.segmenter)
    scores: Dict[str, Dict[str, List[float]]] = {
        name: {"dice": [], "iou": []} for name in backends
    }

    for position, sample in enumerate(samples, start=1):
        truth = load_mask(sample, config.segmentation.mask_kind, size)
        if truth is None:
            continue
        balanced = gray_world_white_balance(resize(read_rgb(sample.image_path), size))

        for name, backend in backends.items():
            predicted = backend(balanced)
            scores[name]["dice"].append(dice_score(predicted, truth))
            scores[name]["iou"].append(iou_score(predicted, truth))

        if position % 25 == 0:
            print(f"  {position}/{len(samples)}")

    print(f"\n{'backend':<24} {'Dice':>16} {'IoU':>16}")
    print("-" * 58)
    summary = {}
    for name, values in sorted(scores.items(), key=lambda kv: -np.mean(kv[1]["dice"] or [0])):
        dice = np.array(values["dice"])
        iou = np.array(values["iou"])
        if dice.size == 0:
            continue
        summary[name] = {
            "dice_mean": float(dice.mean()), "dice_std": float(dice.std()),
            "iou_mean": float(iou.mean()), "iou_std": float(iou.std()),
            "count": int(dice.size),
        }
        print(f"{name:<24} {dice.mean():>8.3f} +/-{dice.std():<5.3f} {iou.mean():>8.3f} +/-{iou.std():<5.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
