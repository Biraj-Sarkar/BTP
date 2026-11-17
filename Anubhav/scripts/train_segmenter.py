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

from anemia.architectures import ARCHITECTURES, build  # noqa: E402
from anemia.imaging import gray_world_white_balance  # noqa: E402
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
    #
    # This can only group what the names encode. A set that numbers its files
    # without saying whose eye each one is (the Mendeley set) makes every file
    # its own subject here, repeat shots included. Those repeats are found
    # separately (experiments/23_duplicates) and passed in with `--groups`.
    parent = path.parent.name.lower()
    if parent not in {"images", "image", "masks", "mask", "segmented"}:
        return parent
    return path.stem.rsplit("_", 1)[0]


def read_groups(path: Path) -> dict:
    """subject id -> group label, from a CSV with `group` and `subject` columns.

    Subject ids are the ones `subject_of` returns. They are pooled across every
    `--data` root, so they must already be unique across roots; the trainer has
    always assumed this.
    """

    import csv

    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"group", "subject"} <= set(rows[0]):
        raise ValueError(f"{path}: needs 'group' and 'subject' columns")
    return {r["subject"]: f"group:{r['group']}" for r in rows}


def split_subjects(subject_roots: dict, seed: int, val_fraction: float, test_fraction: float,
                   groups: dict | None = None, buffer: int = 0, block: int = 0) -> dict:
    """Decide which subjects train, validate, test, or sit out.

    `subject_roots` maps each subject id to the dataset root it came from.

    The unit that is shuffled and dealt out is a **group**: a subject named in
    `groups` travels with every other subject of its group, and anyone not named
    is a group of one. With no `groups`, `buffer` or `block` this is exactly the
    split the trainer always made, so earlier bundles keep their test sets.

    `block` is for datasets numbered in capture order, where one person's
    photographs sit together but nothing says where one person ends and the
    next begins (the Mendeley set: a person is often two runs of shots, one per
    eye, that no matcher can link). Instead of dealing out groups one by one, it
    cuts the numbered list into consecutive blocks of about `block` files, never
    cutting through a group, and deals out whole blocks. A person can then be
    split between training and test only at the edge of a block. This is
    "blocked" cross-validation, the usual answer for data ordered in time. The
    first block's length is drawn from the seed, so the edges move between
    seeds. Each dataset root is dealt out separately, so every root gives about
    the same share of test and validation images.

    `buffer` closes the edges. Any group with a numbered member within `buffer`
    files of a test file (same root) is left out of training and validation
    altogether, and it takes its whole group with it: if file 52 is the other
    eye of test file 50, the rest of file 52's run is that person too.
    """

    groups = groups or {}
    unit_of = lambda s: groups.get(s, s)
    members: dict = {}
    for subject in subject_roots:
        members.setdefault(unit_of(subject), []).append(subject)
    rng = random.Random(seed)

    if block <= 0:
        units = sorted(members)
        rng.shuffle(units)
        n_test = int(len(units) * test_fraction)
        n_val = max(1, int(len(units) * val_fraction))
        test_units, rest = units[:n_test], units[n_test:]
        excluded = _buffered(rest, set(test_units), members, subject_roots, buffer)
        rest = [u for u in rest if u not in excluded]
        val_units, train_units = rest[:n_val], rest[n_val:]
    else:
        size = lambda piece: sum(len(members[u]) for u in piece)
        by_root: dict = {}
        for u in sorted(members):
            by_root.setdefault(subject_roots[members[u][0]], []).append(u)

        # Cut each root into pieces: consecutive blocks for numbered files,
        # single units for everything else. Then shuffle the pieces.
        roots = []
        for root in sorted(by_root):
            units = by_root[root]
            numbered = sorted((u for u in units if any(s.isdigit() for s in members[u])),
                              key=lambda u: min(int(s) for s in members[u] if s.isdigit()))
            pieces, current, limit = [], [], rng.randint(1, block)   # first block: random length
            for u in numbered:
                current.append(u)
                if size(current) >= limit:
                    pieces.append(current)
                    current, limit = [], block
            if current:
                pieces.append(current)
            pieces += [[u] for u in units if u not in set(numbered)]
            rng.shuffle(pieces)
            roots.append((pieces, size(units)))

        # Test: whole pieces, in shuffled order, while each brings the root
        # closer to its share (so a block may stop it just short of the share).
        test_units, leftovers = [], []
        for pieces, n_images in roots:
            i = taken = 0
            while i < len(pieces) and taken + size(pieces[i]) / 2 <= test_fraction * n_images:
                test_units += pieces[i]
                taken += size(pieces[i])
                i += 1
            leftovers.append((pieces[i:], n_images))

        excluded = _buffered([u for pieces, _ in leftovers for p in pieces for u in p],
                             set(test_units), members, subject_roots, buffer)

        # Validation the same way, from what the buffer left; the rest trains.
        val_units, train_units = [], []
        for pieces, n_images in leftovers:
            taken = 0
            for piece in pieces:
                piece = [u for u in piece if u not in excluded]
                if taken + size(piece) / 2 <= val_fraction * n_images:
                    val_units += piece
                    taken += size(piece)
                else:
                    train_units += piece

    gather = lambda us: {s for u in us for s in members[u]}
    return {"train": gather(train_units), "val": gather(val_units), "test": gather(test_units),
            "excluded": gather(excluded), "n_units": len(members)}


def _buffered(candidates, test_units, members, subject_roots, buffer: int) -> set:
    """Units among `candidates` with a numbered member within `buffer` of a test file."""

    if buffer <= 0:
        return set()
    near: dict = {}
    for u in test_units:
        for s in members[u]:
            if s.isdigit():
                near.setdefault(subject_roots[s], set()).add(int(s))
    return {u for u in candidates
            if any(s.isdigit() and any(abs(int(s) - n) <= buffer
                                       for n in near.get(subject_roots[s], ()))
                   for s in members[u])}


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
        # Whether to white-balance before the network sees the image. This must
        # match what `preprocess.prepare()` feeds the segmenter at serving time,
        # or the model is served a kind of image it never trained on.
        self.gray_world = config.gray_world

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
        if self.gray_world:
            rgb = gray_world_white_balance(rgb)
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
def evaluate_full(model, loader, device, threshold: float):
    """Mean Dice, mean IoU, and the per-image Dice list.

    The distribution matters as much as the mean: one architecture can match
    another on average while failing outright on a handful of captures, which
    is what an empty mask at serving time looks like.
    """

    model.eval()
    dices, ious = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predicted = (torch.sigmoid(model(images)) >= threshold).float()
        inter = (predicted * targets).sum(dim=(1, 2, 3))
        total = predicted.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        union = total - inter
        dices.extend(((2 * inter + 1e-6) / (total + 1e-6)).cpu().tolist())
        ious.extend(((inter + 1e-6) / (union + 1e-6)).cpu().tolist())
    return (float(np.mean(dices)) if dices else float("nan"),
            float(np.mean(ious)) if ious else float("nan"),
            dices)


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
    parser.add_argument("--gray-world", action="store_true",
                        help="gray-world white-balance images during training. The "
                             "bundle records this, and prepare() shows the segmenter the "
                             "same kind of image at serving time, whatever "
                             "PreprocessConfig.white_balance sets for the crop's colour.")
    parser.add_argument("--arch", default="unet", choices=list(ARCHITECTURES),
                        help="decoder architecture; see anemia.architectures")
    parser.add_argument("--encoder", default="efficientnet-b0")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-select", action="store_true",
                        help="train on every subject for exactly --epochs and save the final "
                             "weights. No validation, so no epoch selection and no held-out "
                             "score: this is the shipping run, after an architecture has been "
                             "chosen on a split. Pass the epoch count that selection found.")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="subjects held out from both training and model selection. "
                             "Non-zero makes the reported Dice an honest held-out figure: "
                             "val Dice is optimistic because the epoch is chosen on it.")
    parser.add_argument("--groups", type=Path, default=None,
                        help="CSV with 'group' and 'subject' columns: subjects that are one "
                             "person (repeat shots) and must fall on one side of the split. "
                             "See experiments/23_duplicates/groups.csv.")
    parser.add_argument("--buffer", type=int, default=0,
                        help="for datasets numbered in capture order: leave out of training "
                             "and validation every group with a file within this many numbers "
                             "of a test file, to cover repeats the grouping missed")
    parser.add_argument("--block", type=int, default=0,
                        help="for datasets numbered in capture order: deal out consecutive "
                             "blocks of about this many files instead of single groups, so a "
                             "person can be split only at a block edge (use with --buffer)")
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
    subject_roots: dict = {}
    for root in args.data:
        found = find_pairs(root)
        print(f"{root}: {len(found)} image/mask pairs")
        pairs.extend(found)
        subject_roots.update({subject_of(image): str(root) for image, _ in found})
    if not pairs:
        print("No pairs found. Check that the dataset has masks registered to the "
              "full frame, not pre-cropped conjunctiva patches.", file=sys.stderr)
        return 1

    # Split by subject (or by group of subjects), never by image.
    groups = read_groups(args.groups) if args.groups else None
    if args.no_select:
        # Everything trains. Nothing is held back, so nothing can be measured
        # here — the estimate for this model is the selection run's test score.
        split = {"train": set(subject_roots), "val": set(), "test": set(), "excluded": set(),
                 "n_units": len(set(groups.get(s, s) for s in subject_roots)) if groups
                 else len(subject_roots)}
    else:
        split = split_subjects(subject_roots, args.seed, args.val_fraction, args.test_fraction,
                               groups=groups, buffer=args.buffer, block=args.block)
    test_subjects, val_subjects = split["test"], split["val"]
    train_subjects, excluded_subjects = split["train"], split["excluded"]
    train_pairs = [p for p in pairs if subject_of(p[0]) in train_subjects]
    val_pairs = [p for p in pairs if subject_of(p[0]) in val_subjects]
    test_pairs = [p for p in pairs if subject_of(p[0]) in test_subjects]
    unit = "groups" if groups else "subjects"
    print(f"{split['n_units']} {unit} -> {len(train_pairs)} train / {len(val_pairs)} val"
          + (f" / {len(test_pairs)} test images" if test_pairs else " images")
          + (f" ({len(excluded_subjects)} left out by the {args.buffer}-number buffer)"
             if excluded_subjects else ""))
    grouping = {}
    if groups or args.buffer or args.block:
        import hashlib
        grouping = {"groups_file": str(args.groups) if groups else None,
                    "groups_sha256": hashlib.sha256(Path(args.groups).read_bytes()).hexdigest()
                    if groups else None,
                    "buffer": args.buffer, "block": args.block, "n_units": split["n_units"],
                    "excluded_subjects": sorted(excluded_subjects)}

    config = SegmenterConfig(
        arch=args.arch, encoder=args.encoder,
        input_size=args.input_size, threshold=args.threshold,
        gray_world=args.gray_world,
    )
    device = pick_device(args.device)
    print(f"device: {device}")

    # Pretrained encoder unless we are continuing from our own bundle.
    model = build(config.arch, config.encoder, pretrained=not args.init)
    print(f"architecture: {config.arch} / {config.encoder} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")
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

        if args.no_select:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {running / max(len(train_loader), 1):.4f}")
            continue

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
                    **grouping,
                },
                metrics={"val_dice": best_dice, "best_epoch": best_epoch},
            )
        print(f"epoch {epoch:3d}/{args.epochs}  loss {running / max(len(train_loader), 1):.4f}  "
              f"val Dice {val_dice:.4f}{marker}")

    if args.no_select:
        # One save, at the end, of a model trained on every available subject.
        save_bundle(
            args.out, model, config,
            provenance={
                "trained_at": started,
                "datasets": [str(d) for d in args.data],
                "warm_started_from": str(args.init) if args.init else None,
                "train_images": len(train_pairs), "val_images": 0,
                "val_subjects": [],
                "epochs_requested": args.epochs,
                "augment": args.augment, "seed": args.seed,
                "selection": "none — trained on all subjects for a fixed epoch count",
                **grouping,
            },
            metrics={"val_dice": float("nan"), "best_epoch": args.epochs},
        )
        print(f"\ntrained on all {len(train_pairs)} images for {args.epochs} epochs")
        print("no held-out score: this run keeps nothing back. Use the selection "
              "run's test score as the estimate for this model.")
        print(f"bundle written to {args.out}")
        return 0

    print(f"\nbest val Dice {best_dice:.4f} at epoch {best_epoch}")

    # Score the *selected* checkpoint on subjects used for neither training nor
    # selection. Validation Dice is optimistic precisely because the epoch was
    # chosen on it; this number is not.
    if test_pairs:
        model.load_state_dict(
            torch.load(args.out, map_location="cpu", weights_only=False)["state_dict"])
        model.to(device)
        test_loader = DataLoader(
            ConjunctivaDataset(test_pairs, config, train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=2,
        )
        test_dice, test_iou, per_image = evaluate_full(model, test_loader, device, config.threshold)
        empty = sum(1 for d in per_image if d == 0.0)
        print(f"held-out test Dice {test_dice:.4f}  IoU {test_iou:.4f}  "
              f"({len(per_image)} images, {empty} empty masks)")
        bundle = torch.load(args.out, map_location="cpu", weights_only=False)
        bundle["metrics"].update({
            "test_dice": test_dice, "test_iou": test_iou,
            "test_images": len(per_image), "test_empty_masks": empty,
            "test_dice_per_image": per_image,
            "test_subjects": sorted(test_subjects),
        })
        torch.save(bundle, args.out)

    print(f"bundle written to {args.out}")
    print(f"serve it with:  ANEMIA_SEGMENTER={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
