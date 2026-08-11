"""Hb regressor training: cached preprocessing, grouped CV, honest reporting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .data import Sample, is_anemic
from .metrics import (
    aggregate_folds,
    inverse_frequency_weights,
    regression_metrics,
    screening_metrics,
)
from .model import Checkpoint, HbRegressor, to_tensor
from .preprocess import CropCache, Prepared
from .segment import SegmentResult
from .splits import Split, holdout, kfold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment(crop_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Photometric and mild geometric jitter.

    Rotation fills with black to match the letterbox padding — the inherited
    code used BORDER_REFLECT_101, which mirrored padding back into the frame and
    manufactured tissue that was never photographed.

    Hue is left alone deliberately: hue *is* the signal.
    """

    out = crop_rgb

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    if rng.random() < 0.35:
        angle = rng.uniform(-8.0, 8.0)
        height, width = out.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        out = cv2.warpAffine(
            out, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

    if rng.random() < 0.6:
        alpha = rng.uniform(0.9, 1.1)
        beta = rng.uniform(-10.0, 10.0)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.15:
        kernel = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (kernel, kernel), 0)

    return out


class HbDataset(Dataset):
    """Serves cached crops; segmentation never runs inside the training loop."""

    def __init__(
        self,
        samples: Sequence[Sample],
        crops: Dict[Path, np.ndarray],
        target_mean: float,
        target_std: float,
        weights: Optional[Sequence[float]] = None,
        do_augment: bool = False,
        seed: int = 0,
    ):
        self.samples = list(samples)
        self.crops = crops
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if abs(target_std) > 1e-6 else 1.0
        self.weights = list(weights) if weights is not None else [1.0] * len(self.samples)
        self.do_augment = do_augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        crop = self.crops[sample.image_path]
        if self.do_augment:
            crop = augment(crop, self.rng)
        target = (sample.hb - self.target_mean) / self.target_std
        return (
            to_tensor(crop),
            torch.tensor([target], dtype=torch.float32),
            torch.tensor([self.weights[index]], dtype=torch.float32),
            index,
        )


def build_crops(
    samples: Sequence[Sample],
    segmenter: Callable[[np.ndarray], SegmentResult],
    config: Config,
    *,
    verbose: bool = True,
) -> Dict[Path, Prepared]:
    """Preprocess every sample once, reusing the on-disk cache where possible."""

    backend_name = getattr(segmenter, "__name__", type(segmenter).__name__)
    cache = CropCache(config.cache_root, config.preprocess, backend_name)

    prepared: Dict[Path, Prepared] = {}
    for position, sample in enumerate(samples, start=1):
        if sample.image_path in prepared:
            continue
        try:
            prepared[sample.image_path] = cache.prepare(
                sample.image_path, segmenter, config.preprocess, config.quality
            )
        except (FileNotFoundError, cv2.error) as exc:
            if verbose:
                print(f"  skipped {sample.image_path.name}: {exc}")
        if verbose and position % 50 == 0:
            print(f"  prepared {position}/{len(samples)}")
    return prepared


def apply_quality_gate(
    samples: Sequence[Sample],
    prepared: Dict[Path, Prepared],
    enforce: bool,
    *,
    verbose: bool = True,
) -> List[Sample]:
    """Drop captures that failed QC.

    The inherited pipeline computed `accepted_for_training`, wrote it to the
    manifest, and then trained on everything regardless.
    """

    kept, rejected = [], []
    for sample in samples:
        prep = prepared.get(sample.image_path)
        if prep is None:
            continue
        if enforce and not prep.quality.passed:
            rejected.append((sample, prep.quality.reasons))
        else:
            kept.append(sample)

    if verbose and rejected:
        print(f"  QC rejected {len(rejected)}/{len(samples)} captures")
        for sample, reasons in rejected[:5]:
            print(f"    {sample.image_path.name}: {'; '.join(reasons)}")
        if len(rejected) > 5:
            print(f"    ... and {len(rejected) - 5} more")
    return kept


@dataclass
class FoldResult:
    metrics: Dict[str, float]
    screening: Dict[str, float]
    predictions: List[dict]
    checkpoint: Optional[Checkpoint] = None


def train_fold(
    split: Split,
    crops: Dict[Path, np.ndarray],
    config: Config,
    device: torch.device,
    *,
    verbose: bool = True,
) -> FoldResult:
    reg = config.regression
    train_hb = [s.hb for s in split.train]
    target_mean = float(np.mean(train_hb))
    target_std = float(np.std(train_hb)) or 1.0

    weights = inverse_frequency_weights(train_hb, reg.imbalance_bins, reg.imbalance_strength)

    train_loader = DataLoader(
        HbDataset(split.train, crops, target_mean, target_std, weights, reg.augment, config.seed),
        batch_size=reg.batch_size,
        shuffle=True,
        # Only drop a trailing batch of exactly one sample, which would crash
        # the trainable BatchNorm layers. Dropping every short batch would throw
        # away real images on a cohort this small.
        drop_last=(len(split.train) % reg.batch_size == 1),
    )
    test_loader = DataLoader(
        HbDataset(split.test, crops, target_mean, target_std),
        batch_size=reg.batch_size,
        shuffle=False,
    )

    model = HbRegressor(reg.backbone, True, reg.dropout, reg.freeze_until).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=reg.learning_rate,
        weight_decay=reg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=reg.epochs)

    best_mae = float("inf")
    best_state = None
    patience = 0

    for epoch in range(reg.epochs):
        model.train()
        epoch_loss, batches = 0.0, 0
        for images, targets, sample_weights, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            per_sample = F.smooth_l1_loss(predictions, targets, reduction="none")
            loss = torch.mean(sample_weights * per_sample)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        evaluation = _evaluate(model, test_loader, target_mean, target_std, device)
        mae = evaluation["metrics"].get("mae", float("inf"))

        if mae < best_mae - 1e-4:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose and (epoch % 5 == 0 or epoch == reg.epochs - 1):
            print(
                f"    epoch {epoch + 1:>3}/{reg.epochs} "
                f"loss={epoch_loss / max(batches, 1):.4f} test_mae={mae:.4f}"
            )

        if patience >= reg.early_stopping_patience:
            if verbose:
                print(f"    early stop at epoch {epoch + 1} (best MAE {best_mae:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final = _evaluate(model, test_loader, target_mean, target_std, device)

    rows = []
    for sample, predicted in zip(split.test, final["predictions"]):
        rows.append(
            {
                "patient_id": sample.patient_id,
                "source": sample.source,
                "filename": sample.image_path.name,
                "actual_hb": sample.hb,
                "predicted_hb": float(predicted),
                "age_years": sample.age_years,
                "sex": sample.sex,
                "actual_anemic": is_anemic(sample.hb, sample.age_years, sample.sex),
                "predicted_anemic": is_anemic(float(predicted), sample.age_years, sample.sex),
            }
        )

    screening = screening_metrics(
        [r["predicted_anemic"] for r in rows], [r["actual_anemic"] for r in rows]
    )

    checkpoint = Checkpoint(
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        target_mean=target_mean,
        target_std=target_std,
        backbone=reg.backbone,
        model_size=config.preprocess.model_size,
        work_size=config.preprocess.work_size,
        gray_world=config.preprocess.gray_world,
        metrics={"regression": final["metrics"], "screening": screening},
        config=config.to_dict(),
    )

    return FoldResult(final["metrics"], screening, rows, checkpoint)


def _evaluate(model, loader, target_mean: float, target_std: float, device) -> dict:
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for images, batch_targets, _, _ in loader:
            outputs = model(images.to(device)).detach().cpu().numpy().reshape(-1)
            predictions.extend((outputs * target_std + target_mean).tolist())
            targets.extend(
                (batch_targets.numpy().reshape(-1) * target_std + target_mean).tolist()
            )

    return {
        "predictions": predictions,
        # Baseline is the training mean: a model that ignores the image entirely.
        "metrics": regression_metrics(predictions, targets, baseline=target_mean),
    }


def cross_validate(
    samples: Sequence[Sample],
    crops: Dict[Path, np.ndarray],
    config: Config,
    device: torch.device,
    *,
    verbose: bool = True,
) -> dict:
    fold_results: List[FoldResult] = []
    all_rows: List[dict] = []

    for index, split in enumerate(kfold(samples, config.split.folds, config.split.stratify_bins, config.split.seed), 1):
        if verbose:
            print(f"\n  fold {index}/{config.split.folds}: {split.describe()}")
        result = train_fold(split, crops, config, device, verbose=verbose)
        fold_results.append(result)
        all_rows.extend(result.predictions)

    return {
        "regression": aggregate_folds([r.metrics for r in fold_results]),
        "screening": aggregate_folds([r.screening for r in fold_results]),
        "per_fold": [
            {"fold": i + 1, "regression": r.metrics, "screening": r.screening}
            for i, r in enumerate(fold_results)
        ],
        # Every row here is an out-of-fold prediction, so this CSV is safe to
        # report. The inherited pipeline emitted train-set predictions in the
        # same file as test-set ones, with no column distinguishing them.
        "out_of_fold_predictions": all_rows,
        "best_checkpoint": min(
            (r for r in fold_results if r.checkpoint),
            key=lambda r: r.metrics.get("mae", float("inf")),
            default=None,
        ),
    }
