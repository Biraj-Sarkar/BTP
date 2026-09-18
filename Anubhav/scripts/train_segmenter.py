"""Train (or extend) the learned conjunctiva segmenter.

This is the only file that **writes** a parameter bundle. `segment_learned.py`
reads one and cannot produce one, so any bundle in `runs/` is the output of a
recorded training run rather than something a serving process mutated.

Typical use
-----------
First training run on a public dataset::

    python scripts/train_segmenter.py --data datasets/eyes_defy --out runs/segmenter/unet.pt

Later, when a new dataset arrives — **pass every dataset to `--data`**, always::

    python scripts/train_segmenter.py \
        --data datasets/eyes_defy_prepared datasets/hospital \
        --init runs/segmenter/unet.pt \
        --out runs/segmenter/unet.pt

**`--init` alone does not preserve what the model already knew.** This was
measured, not assumed. Fine-tuning the Eyes-Defy model on 547 new images for 40
epochs — warm-started, but with the old data absent — produced a bundle that
returned an **empty mask on 42 of the 52 local captures**, against 2 for the
model it started from, while simultaneously scoring its *best ever* validation
Dice (0.8709) on the new dataset. Mask IoU against its own parent was 0.070.
That is catastrophic forgetting: the new domain overwrote the old one, and no
validation number on the new data gave any warning.

Re-running with **both** datasets in `--data` and the same `--init` restored it
— 2 empty masks, IoU 0.557 against the parent, and a small downstream gain.

So the rule is: `--init` is a starting point for optimisation, **not** a memory.
Anything the model must still handle has to be in `--data` this run. The only
safe use of `--init` without the old data is resuming an interrupted run on the
same dataset.

Abbreviations
-------------
U-Net    encoder-decoder segmentation architecture with skip connections
BCE      binary cross-entropy, the per-pixel classification loss
Dice     overlap score, 2|A∩B| / (|A|+|B|); 1.0 is a perfect mask
LR       learning rate
MPS      Metal Performance Shaders, the GPU backend on Apple silicon
AMP      automatic mixed precision
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anemia.segment_learned import (  # noqa: E402
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    SegmenterConfig,
)

Pair = Tuple[Path, Path]


# ---------------------------------------------------------------------------
# Data


def find_pairs(root: Path) -> List[Pair]:
    """Discover (image, mask) pairs under a dataset root.

    Supported layouts, tried in order:

    1. ``root/images/*`` beside ``root/masks/*``, paired on filename stem.
    2. Any two sibling directories named ``*image*`` and ``*mask*``
       (or ``*segment*``), paired on stem — this is what most published
       conjunctiva sets look like once unzipped.

    A dataset whose "segmented" files are arbitrary rectangular crops rather
    than masks registered to the original frame cannot train a segmenter at
    all, and will simply produce no pairs here rather than training on
    mismatched geometry.
    """

    root = Path(root)
    candidates: List[Tuple[Path, Path]] = []

    explicit = (root / "images", root / "masks")
    if all(d.is_dir() for d in explicit):
        candidates.append(explicit)

    if not candidates:
        directories = [d for d in root.rglob("*") if d.is_dir()]
        images = [d for d in directories if "image" in d.name.lower()]
        masks = [
            d
            for d in directories
            if "mask" in d.name.lower() or "segment" in d.name.lower()
        ]
        for image_dir in images:
            for mask_dir in masks:
                if image_dir.parent == mask_dir.parent:
                    candidates.append((image_dir, mask_dir))

    pairs: List[Pair] = []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for image_dir, mask_dir in candidates:
        by_stem = {
            p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in suffixes
        }
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in suffixes:
                continue
            mask_path = by_stem.get(image_path.stem)
            if mask_path is not None:
                pairs.append((image_path, mask_path))

    return pairs


def subject_of(path: Path) -> str:
    """Group key for splitting. Images of one subject must not straddle the split.

    The same eye photographed twice is not two independent samples: a model
    that memorises one frame scores well on the other, and the validation Dice
    then measures recall rather than generalisation. This is the same
    patient-grouping rule `anemia.splits` already enforces for the regression.
    """

    # Most layouts put a subject id in the parent directory; fall back to the
    # filename stem stripped of a trailing capture index (`sub12_3` -> `sub12`).
    parent = path.parent.name.lower()
    if parent not in {"images", "image", "masks", "mask", "segmented"}:
        return parent
    return path.stem.rsplit("_", 1)[0]


class ConjunctivaDataset(Dataset):
    """Pairs, conditioned for the network.

    Photometric augmentation is deliberately aggressive here, and it is safe in
    a way it would not be elsewhere in this project: the *output* of this model
    is a mask, not a colour. Teaching it that conjunctiva stays conjunctiva
    under a warm bulb or a cold flash is exactly the invariance we want. The
    colour statistics that feed haemoglobin regression are measured later, on
    the original untouched pixels — augmented colour never reaches them.
    """

    def __init__(self, pairs: Sequence[Pair], config: SegmenterConfig, train: bool,
                 augment: str = "standard"):
        self.pairs = list(pairs)
        self.config = config
        self.train = train

        import albumentations as A

        size = config.input_size
        if not train:
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=config.mean, std=config.std),
            ])
            return

        geometry = [
            A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        ]
        photometric = [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.08, p=0.9),
            A.GaussNoise(p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
        ]

        if augment == "rotation":
            # Rotation widened, everything else left exactly as "standard".
            # `strong` changed rotation AND scale together and bought robustness
            # (0 failures, IoU 0.883) at the cost of the accuracy that justified
            # a learned extractor at all (p 0.013 -> 0.165, mask area +112%).
            # Rotation does not change an object's apparent size, so it is not
            # the plausible cause of the over-prediction; scale is. This isolates
            # them.
            geometry = [
                A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT,
                         fill=0, fill_mask=0, p=0.9),
            ]

        if augment == "strong":
            # Calibrated against measured failures, not guessed. The robustness
            # sweep (`experiments/13_learned_extractor/robustness.py`) showed the
            # network losing to a hand-written rule on exactly three axes:
            # rotation (IoU ~0.67 vs 0.93), zooming out (15 outright failures on
            # the older bundle) and over-brightening (10). Each is answered here.
            geometry = [
                # One affine covers scale in BOTH directions — RandomResizedCrop
                # only ever crops in, so it could not teach zoom-out at all,
                # which is where the worst failures were.
                A.Affine(scale=(0.45, 1.6), rotate=(-45, 45),
                         translate_percent=(-0.15, 0.15), shear=(-8, 8),
                         border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.95),
                A.HorizontalFlip(p=0.5),
            ]
            photometric = [
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4, p=0.8),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.1, p=0.9),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.ImageCompression(quality_range=(30, 95), p=0.3),
            ]

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            *geometry,
            *photometric,
            A.Normalize(mean=config.mean, std=config.std),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise RuntimeError(f"Could not read pair {image_path} / {mask_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Datasets encode the region as 255, as 1, or as any non-zero label.
        binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0).astype(np.uint8)

        augmented = self.transform(image=rgb, mask=binary)
        image_tensor = torch.from_numpy(augmented["image"].transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(augmented["mask"]).float().unsqueeze(0)
        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Loss and metric


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice, which optimises overlap directly.

    Conjunctiva occupies a small share of a frame, so plain BCE is minimised
    well by predicting "background everywhere" — a model with excellent loss
    and an empty mask. Dice is scale-free in the positive class and does not
    have that fixed point. The two are summed: BCE gives clean per-pixel
    gradients early, Dice supplies the objective that actually matters.
    """

    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    union = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> float:
    """Mean Dice at the serving threshold, so this number predicts deployment."""

    model.eval()
    scores: List[float] = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predicted = (torch.sigmoid(model(images)) >= threshold).float()
        intersection = (predicted * targets).sum(dim=(1, 2, 3))
        union = predicted.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        # An empty prediction on an empty target is a perfect score, not 0/0.
        scores.extend(((2 * intersection + 1e-6) / (union + 1e-6)).cpu().tolist())
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Bundle writing — the only place parameters are persisted


def save_bundle(path: Path, model, config: SegmenterConfig, provenance: dict, metrics: dict) -> None:
    """Write weights and the context needed to serve and to audit them.

    The provenance block is not decoration. Six months from now the only way to
    answer "which images produced the model currently estimating haemoglobin"
    is for the bundle to carry it.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "provenance": provenance,
            "metrics": metrics,
        },
        path,
    )


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", nargs="+", required=True, type=Path,
                        help="One or more dataset roots. Pass several to train on the union.")
    parser.add_argument("--out", type=Path, default=Path("runs/segmenter/unet.pt"))
    parser.add_argument("--init", type=Path, default=None,
                        help="Warm-start from an existing bundle (use when new data arrives).")
    parser.add_argument("--encoder", default="efficientnet-b0")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", default="standard", choices=["standard", "rotation", "strong"],
                        help="'rotation' widens ONLY rotation; 'strong' widens "
                             "rotation, scale and exposure together. See the class "
                             "docstring for what each cost and bought.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs: List[Pair] = []
    for root in args.data:
        found = find_pairs(root)
        print(f"{root}: {len(found)} image/mask pairs")
        pairs.extend(found)
    if not pairs:
        print("No pairs found. Check that the dataset has masks registered to the "
              "full frame, not pre-cropped conjunctiva patches.", file=sys.stderr)
        return 1

    # Split by subject, never by image.
    subjects = sorted({subject_of(image) for image, _ in pairs})
    random.Random(args.seed).shuffle(subjects)
    cut = max(1, int(len(subjects) * args.val_fraction))
    val_subjects = set(subjects[:cut])
    train_pairs = [p for p in pairs if subject_of(p[0]) not in val_subjects]
    val_pairs = [p for p in pairs if subject_of(p[0]) in val_subjects]
    print(f"{len(subjects)} subjects -> {len(train_pairs)} train / {len(val_pairs)} val images")

    config = SegmenterConfig(
        encoder=args.encoder, input_size=args.input_size, threshold=args.threshold
    )
    device = pick_device(args.device)
    print(f"device: {device}")

    import segmentation_models_pytorch as smp

    # Pretrained encoder unless we are continuing from our own bundle.
    model = smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=None if args.init else "imagenet",
        in_channels=3,
        classes=1,
    )
    if args.init:
        bundle = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(bundle["state_dict"])
        print(f"warm-started from {args.init} "
              f"(val Dice was {bundle.get('metrics', {}).get('val_dice', float('nan')):.4f})")
    model.to(device)

    train_loader = DataLoader(
        ConjunctivaDataset(train_pairs, config, train=True, augment=args.augment),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=len(train_pairs) > args.batch_size,
    )
    val_loader = DataLoader(
        ConjunctivaDataset(val_pairs, config, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_dice, best_epoch = -1.0, -1
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
        scheduler.step()

        val_dice = evaluate(model, val_loader, device, config.threshold)
        marker = ""
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            marker = "  <- best, saved"
            save_bundle(
                args.out, model, config,
                provenance={
                    "trained_at": started,
                    "datasets": [str(d) for d in args.data],
                    "warm_started_from": str(args.init) if args.init else None,
                    "train_images": len(train_pairs),
                    "val_images": len(val_pairs),
                    "val_subjects": sorted(val_subjects),
                    "epochs_requested": args.epochs,
                    "augment": args.augment,
                    "seed": args.seed,
                },
                metrics={"val_dice": best_dice, "best_epoch": best_epoch},
            )
        print(f"epoch {epoch:3d}/{args.epochs}  loss {running / max(len(train_loader), 1):.4f}  "
              f"val Dice {val_dice:.4f}{marker}")

    print(f"\nbest val Dice {best_dice:.4f} at epoch {best_epoch}")
    print(f"bundle written to {args.out}")
    print(f"serve it with:  ANEMIA_SEGMENTER={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
