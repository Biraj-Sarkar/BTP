"""Look at the candidate pairs, group them, and check experiment 21 for leakage.

Reads pairs.csv from find_duplicates.py; trains nothing.

    python experiments/23_duplicates/review.py --sheet 0 40      # contact sheet of pairs ranked 0-39
    python experiments/23_duplicates/review.py --threshold 50 --near 20 10   # group, check splits

`--sheet` draws each candidate pair side by side with its inlier count, for
judging by eye where "same eye" ends.

`--threshold T --near N D` decides which pairs are the same subject. A pair is
linked if it has at least T RANSAC inliers, or at least N inliers and Mendeley
file numbers at most D apart. The second rule exists because the file numbers
are informative: the set was saved in sessions, consecutive files are usually
the same person, and among pairs with 20-30 inliers those with neighbouring
numbers are the same person while distant ones are mostly chance matches
(checked by eye; see RESULTS.md). Linked pairs are joined into groups (A~B and
B~C make one group), written to `groups.csv`, which the trainer reads with `--groups`. Then, for each experiment-21
split: how many test images had a partner among the training images, and did
those images score higher than the rest?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = Path.home() / "btp-local/datasets"
SOURCES = {"mendeley": DATASETS / "conjunctiva_seg_prepared/images",
           "eyes_defy": DATASETS / "eyes_defy_prepared/images"}
EXP21 = ROOT / "experiments/21_repeated_splits"
EXP20 = ROOT / "experiments/20_architectures"
SEEDS = (0, 1, 2, 3, 42)
ARCHS = ("unet", "unetplusplus", "attention", "linknet", "fpn", "pspnet")


def read_pairs():
    with (HERE / "pairs.csv").open() as h:
        return [dict(r, ransac_inliers=int(r["ransac_inliers"]), ratio_matches=int(r["ratio_matches"]))
                for r in csv.DictReader(h)]


def thumb(key: str, height: int = 220) -> np.ndarray:
    dataset, stem = key.split("/")
    img = cv2.imread(str(SOURCES[dataset] / f"{stem}_0.png"))
    scale = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), height), interpolation=cv2.INTER_AREA)


def sheet(pairs, start: int, stop: int, out: Path) -> None:
    tiles = []
    for rank, p in enumerate(pairs[start:stop], start):
        a, b = thumb(p["a"]), thumb(p["b"])
        gap = np.full((a.shape[0], 6, 3), 255, np.uint8)
        tile = np.hstack([a, gap, b])
        bar = np.full((26, tile.shape[1], 3), 255, np.uint8)
        label = f"#{rank}  {p['a']} | {p['b']}  inliers {p['ransac_inliers']}"
        cv2.putText(bar, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    width = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    rows = []
    for k in range(0, len(tiles), 2):
        row = tiles[k:k + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 255))
        rows.append(np.hstack([row[0], np.full((row[0].shape[0], 20, 3), 255, np.uint8), row[1]]))
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"wrote {out}")


def number(key: str):
    try:
        return int(key.split("/")[1]) if key.startswith("mendeley/") else None
    except ValueError:
        return None


def linked(p, threshold: int, near=None) -> bool:
    if p["ransac_inliers"] >= threshold:
        return True
    if near:
        min_inliers, max_gap = near
        a, b = number(p["a"]), number(p["b"])
        return (a is not None and b is not None and p["ransac_inliers"] >= min_inliers
                and abs(a - b) <= max_gap)
    return False


def groups_from(pairs, threshold: int, near=None):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if linked(p, threshold, near):
            parent[find(p["a"])] = find(p["b"])
    members = defaultdict(list)
    for x in list(parent):
        members[find(x)].append(x)
    return sorted((sorted(v) for v in members.values() if len(v) > 1), key=lambda g: (-len(g), g))


def split_of(seed: int):
    """train / val / test subject sets of one experiment-21 split."""

    import torch
    path = EXP20 / "bundles/attention.pt" if seed == 0 else EXP21 / f"bundles/seed{seed}/attention.pt"
    b = torch.load(path, map_location="cpu", weights_only=False)
    return set(b["provenance"]["val_subjects"]), set(b["metrics"]["test_subjects"])


def test_order(test_subjects):
    """Subjects in the order experiment 21 scored them (per_image_dice.npz)."""

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_segmenter", ROOT / "scripts/train_segmenter.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    roots = [DATASETS / "eyes_defy_prepared", DATASETS / "conjunctiva_seg_prepared"]
    pairs = [p for r in roots for p in trainer.find_pairs(r)]
    return [trainer.subject_of(p[0]) for p in pairs if trainer.subject_of(p[0]) in test_subjects]


def leakage(groups):
    subject = lambda key: key.split("/")[1]
    partners = defaultdict(set)
    for g in groups:
        for k in g:
            partners[subject(k)] |= {subject(x) for x in g if x != k}
    dice = np.load(EXP21 / "per_image_dice.npz")
    out = []
    for seed in SEEDS:
        val, test = split_of(seed)
        order = test_order(test)
        leaked = [s for s in order if any(p not in val and p not in test for p in partners.get(s, ()))]
        row = {"seed": seed, "test_images": len(order), "test_with_partner_in_train": len(leaked),
               "test_with_partner_in_val": sum(1 for s in order if partners.get(s, set()) & val),
               "test_with_partner_in_test": sum(1 for s in order if partners.get(s, set()) & test)}
        flag = np.array([s in set(leaked) for s in order])
        # Every leaked image is a Mendeley image, and Eyes-defy images differ in difficulty,
        # so the fair comparison is within Mendeley: partner in training vs not.
        mendeley = np.array([s.isdigit() for s in order])
        row["mendeley_test_images"] = int(mendeley.sum())
        row["mendeley_test_without_partner"] = int((mendeley & ~flag).sum())
        for arch in ARCHS:
            d = dice[f"seed{seed}_{arch}"]
            assert len(d) == len(order), "test order does not match experiment 21"
            row[arch] = {"dice_all": float(d.mean()),
                         "dice_without_leaked": float(d[~flag].mean()),
                         "mendeley_partner_in_train": float(d[mendeley & flag].mean()) if (mendeley & flag).any() else None,
                         "mendeley_no_partner": float(d[mendeley & ~flag].mean()) if (mendeley & ~flag).any() else None,
                         "eyes_defy": float(d[~mendeley].mean()) if (~mendeley).any() else None}
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", nargs=2, type=int, metavar=("START", "STOP"))
    ap.add_argument("--threshold", type=int)
    ap.add_argument("--near", nargs=2, type=int, metavar=("MIN_INLIERS", "MAX_GAP"))
    args = ap.parse_args()
    pairs = sorted(read_pairs(), key=lambda p: -p["ransac_inliers"])

    if args.sheet:
        start, stop = args.sheet
        sheet(pairs, start, stop, HERE / f"candidates_{start:03d}-{stop - 1:03d}.png")

    if args.threshold is not None:
        groups = groups_from(pairs, args.threshold, args.near)
        with (HERE / "groups.csv").open("w", newline="") as h:
            w = csv.writer(h)
            # `subject` is the id the trainer's `subject_of` gives the image, which is
            # what `train_segmenter.py --groups` reads.
            w.writerow(["group", "image", "subject"])
            for gi, g in enumerate(groups):
                for k in g:
                    w.writerow([gi, k, k.split("/")[1]])
        sizes = defaultdict(int)
        for g in groups:
            sizes[len(g)] += 1
        images_in_groups = sum(len(g) for g in groups)
        print(f"threshold {args.threshold}: {len(groups)} groups, {images_in_groups} images "
              f"(sizes {dict(sorted(sizes.items()))})")
        cross = [g for g in groups if len({k.split('/')[0] for k in g}) > 1]
        print(f"groups spanning both datasets: {len(cross)}")
        result = {"threshold": args.threshold, "near": args.near, "groups": groups, "group_sizes": dict(sizes),
                  "images_in_groups": images_in_groups, "cross_dataset_groups": cross,
                  "leakage": leakage(groups) if groups else []}
        (HERE / "leakage.json").write_text(json.dumps(result, indent=2))
        for r in result["leakage"]:
            print(f"  seed {r['seed']:>2}: {r['test_with_partner_in_train']} of {r['test_images']} test images "
                  f"have a partner in training ({r['mendeley_test_without_partner']} of "
                  f"{r['mendeley_test_images']} Mendeley test images do not)")
        if result["leakage"]:
            print("\n  Mendeley test Dice, partner in training vs none (mean over splits; per-split difference)")
            for arch in ARCHS:
                diffs = [r[arch]["mendeley_partner_in_train"] - r[arch]["mendeley_no_partner"]
                         for r in result["leakage"]]
                yes = np.mean([r[arch]["mendeley_partner_in_train"] for r in result["leakage"]])
                no = np.mean([r[arch]["mendeley_no_partner"] for r in result["leakage"]])
                print(f"    {arch:<13} {yes:.4f} vs {no:.4f}  diff {np.mean(diffs):+.4f}  "
                      f"[{', '.join(f'{d:+.3f}' for d in diffs)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
