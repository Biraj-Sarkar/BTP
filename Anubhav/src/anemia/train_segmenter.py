"""Fine-tune Mask2Former on the Eyes-defy-anemia conjunctiva masks.

This is the only stage with pixel-level ground truth, so it is trained once on
Eyes-defy-anemia and then applied to every other dataset (CP-AnemiC and the
hospital data) to produce crops for the Hb regressor.

Design points that change the outcome:

* The checkpoint downloads on first use. An offline-only load
  (`local_files_only=True`) fails on any clean machine, and if that failure is
  swallowed the run silently degrades to the colour heuristic — reporting
  "trained segmentation" results having trained nothing.
* Validation is patient-grouped and drives model selection.
* Only samples that actually carry masks are used, and the count is asserted.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from .config import Config
from .data import Sample
from .imaging import gray_world_white_balance, read_rgb, resize
from .metrics import dice_score, iou_score
from .splits import holdout


def load_mask(sample: Sample, kind: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Binary ground-truth mask at `size`, or None if this sample has none."""

    path = sample.mask_paths.get(kind) or next(iter(sample.mask_paths.values()), None)
    if path is None:
        return None

    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 3] if mask.shape[2] == 4 else mask.max(axis=2)
    binary = (mask > 0).astype(np.uint8)
    return resize(binary, size, nearest=True)


def _prepare_pair(sample: Sample, kind: str, size: Tuple[int, int]):
    mask = load_mask(sample, kind, size)
    if mask is None:
        return None
    image = gray_world_white_balance(resize(read_rgb(sample.image_path), size))
    return {"image": image, "mask": mask, "sample": sample}


def train_segmenter(
    samples: Sequence[Sample],
    config: Config,
    output_dir: Path,
    device: Optional[torch.device] = None,
    *,
    verbose: bool = True,
) -> Dict[str, float]:
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

    seg = config.segmentation
    size = config.preprocess.work_size
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    masked_samples = [s for s in samples if s.has_mask]
    if not masked_samples:
        raise RuntimeError(
            "No ground-truth masks found. Segmenter training requires "
            "Eyes-defy-anemia, which ships hand-drawn conjunctiva masks."
        )
    if verbose:
        print(f"{len(masked_samples)} masked images from {len(samples)} total")

    split = holdout(masked_samples, config.split.holdout_fraction, config.split.stratify_bins, config.split.seed)
    if verbose:
        print(f"Segmenter split: {split.describe()}")

    train_pairs = [p for p in (_prepare_pair(s, seg.mask_kind, size) for s in split.train) if p]
    val_pairs = [p for p in (_prepare_pair(s, seg.mask_kind, size) for s in split.test) if p]
    if not train_pairs:
        raise RuntimeError("No usable image/mask pairs after loading")

    processor = Mask2FormerImageProcessor.from_pretrained(seg.checkpoint)
    if hasattr(processor, "do_reduce_labels"):
        processor.do_reduce_labels = False

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        seg.checkpoint,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=seg.learning_rate)
    rng = random.Random(config.seed)

    best_dice = -1.0
    best_state = None
    history: List[dict] = []

    for epoch in range(seg.epochs):
        model.train()
        rng.shuffle(train_pairs)
        epoch_loss, steps = 0.0, 0

        for start in range(0, len(train_pairs), seg.batch_size):
            batch = train_pairs[start : start + seg.batch_size]
            encoding = processor(
                images=[item["image"] for item in batch],
                segmentation_maps=[item["mask"] for item in batch],
                return_tensors="pt",
            )
            encoding = {
                key: ([v.to(device) for v in value] if isinstance(value, list) else value.to(device))
                for key, value in encoding.items()
            }

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**encoding)
            outputs.loss.backward()
            optimizer.step()

            epoch_loss += float(outputs.loss.detach().cpu())
            steps += 1

        metrics = evaluate_segmenter(model, processor, val_pairs, device)
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(steps, 1), **metrics})

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{seg.epochs} loss={epoch_loss / max(steps, 1):.4f} "
                f"val_dice={metrics['dice']:.4f} val_iou={metrics['iou']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(output_dir)
    model.save_pretrained(output_dir / "model")
    processor.save_pretrained(output_dir / "processor")
    final = evaluate_segmenter(model, processor, val_pairs, device)
    (output_dir / "segmentation_metrics.json").write_text(
        json.dumps({"best": final, "history": history}, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"\nBest val Dice {final['dice']:.4f} / IoU {final['iou']:.4f} on {final['count']} images")
        print(f"Saved segmenter to {output_dir}")
    return final


def evaluate_segmenter(model, processor, pairs: Sequence[dict], device) -> Dict[str, float]:
    if not pairs:
        return {"dice": 0.0, "iou": 0.0, "count": 0}

    model.eval()
    dice_values, iou_values = [], []
    with torch.inference_mode():
        for item in pairs:
            encoding = processor(images=item["image"], return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            semantic = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[item["image"].shape[:2]]
            )[0]
            predicted = (semantic.detach().cpu().numpy() == 1).astype(np.uint8)
            dice_values.append(dice_score(predicted, item["mask"]))
            iou_values.append(iou_score(predicted, item["mask"]))

    return {
        "dice": float(np.mean(dice_values)),
        "iou": float(np.mean(iou_values)),
        "count": len(dice_values),
    }
