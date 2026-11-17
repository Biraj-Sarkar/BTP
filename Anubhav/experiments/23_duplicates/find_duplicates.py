"""Are some Mendeley photographs of the same eye?

The Mendeley set (547 images) carries no subject identifiers: files are just
numbered. The trainer therefore treats every image as its own subject, so two
photographs of one eye can land on opposite sides of a train / test split. A
model that has trained on one of them partly recognises the other, and the test
score overstates performance on a new patient. This script looks for such pairs
and, if any exist, measures whether they actually straddled the splits of
experiment 21 and inflated its scores.

Three tests, from the cheapest and narrowest to the most general
----------------------------------------------------------------
1. **Exact copies.** A hash of the decoded pixels. Catches byte- or
   pixel-identical files.
2. **Near copies.** A 64-bit perceptual hash (DCT of a 32x32 greyscale
   thumbnail, as in pHash) compared under all eight flips and quarter-turns.
   Catches a photograph that was resized, recompressed, mirrored or rotated —
   the pre-augmented copies some public sets ship with.
3. **Same eye, different shot.** SIFT keypoints on every image at 1024 px, matched
   between **every** pair (no shortlist, so nothing is missed by a coarse first
   pass), then a RANSAC fit of a homography. Two photographs of one eye share
   vessels, lashes and lid-margin texture in a geometrically consistent
   arrangement, so many matches survive the fit; photographs of different eyes
   leave only chance matches. Reported per pair as the inlier count.

The threshold between "same eye" and "different eyes" is read off the
distribution of inlier counts and confirmed by looking at the candidate pairs
(`candidates.png`), not fixed in advance.

What it cannot see: the left and right eye of one person are different eyes
and will not match geometrically, though they share skin tone and lashes.

The Eyes-defy images (101) are compared with the Mendeley images too, in case
one set reused photographs from the other.

Run
---
    python experiments/23_duplicates/find_duplicates.py     # features + all pairs, a few minutes
    python experiments/23_duplicates/review.py --sheet 0 40   # then look at the top pairs (see review.py)

Abbreviations
-------------
SIFT    scale-invariant feature transform (keypoint detector and descriptor)
RANSAC  random sample consensus, a robust fit that ignores outlying matches
pHash   perceptual hash; similar images give hashes a few bits apart
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = Path.home() / "btp-local/datasets"
MENDELEY = DATASETS / "conjunctiva_seg/Eye Conjunctiva Segmentation Dataset/Images"
MENDELEY_PREPARED = DATASETS / "conjunctiva_seg_prepared/images"
EYES_DEFY_PREPARED = DATASETS / "eyes_defy_prepared/images"
CACHE = Path.home() / "btp-local/cache/duplicates"   # features; outside the repo

WORK_SIDE = 1024        # longest side for keypoints
N_FEATURES = 1500
RATIO = 0.75            # Lowe's ratio test
RANSAC_PX = 8.0         # reprojection tolerance at 1024 px
MIN_GOOD = 12           # fewer ratio-test matches than this cannot support a fit


# ----------------------------------------------------------------- items

def items():
    out = []
    for p in sorted(MENDELEY_PREPARED.iterdir()):
        if p.suffix == ".png":
            stem = p.stem.rsplit("_", 1)[0]
            out.append({"key": f"mendeley/{stem}", "path": str(MENDELEY / f"{stem}.png")})
    for p in sorted(EYES_DEFY_PREPARED.iterdir()):
        if p.suffix == ".png":
            out.append({"key": f"eyes_defy/{p.stem.rsplit('_', 1)[0]}", "path": str(p)})
    return out


# ----------------------------------------------------------------- per image

def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8].flatten()[1:]          # drop the DC term
    bits = block > np.median(block)
    return int("".join("1" if b else "0" for b in bits), 2)


def dihedral(gray: np.ndarray):
    """The eight flips and quarter-turns of an image."""

    for k in range(4):
        r = np.rot90(gray, k)
        yield r
        yield np.fliplr(r)


def describe(item) -> dict:
    bgr = cv2.imread(item["path"], cv2.IMREAD_COLOR)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = WORK_SIDE / max(gray_full.shape)
    gray = cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray_full
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((0, 128), np.float32)
    return {
        "key": item["key"],
        "shape": list(gray_full.shape),
        "pixel_sha": hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest(),
        "phash": [phash(g) for g in dihedral(gray)],
        "points": np.array([k.pt for k in keypoints], np.float32),
        # SIFT descriptors are small non-negative values; uint8 keeps the cache small
        "desc": np.clip(descriptors, 0, 255).astype(np.uint8),
    }


# ----------------------------------------------------------------- per pair

_FEATS = None


def _init(path):
    global _FEATS
    data = np.load(path, allow_pickle=True)
    _FEATS = {"points": list(data["points"]), "desc": list(data["desc"])}


def match_row(args):
    """Match image i against every j in `others`; returns (i, j, good, inliers)."""

    i, others = args
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    di = _FEATS["desc"][i].astype(np.float32)
    pi = _FEATS["points"][i]
    rows = []
    for j in others:
        dj = _FEATS["desc"][j].astype(np.float32)
        if len(di) < 2 or len(dj) < 2:
            rows.append((i, j, 0, 0))
            continue
        pairs = matcher.knnMatch(di, dj, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
        inliers = 0
        if len(good) >= MIN_GOOD:
            src = pi[[m.queryIdx for m in good]]
            dst = _FEATS["points"][j][[m.trainIdx for m in good]]
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=2000, confidence=0.999)
            inliers = int(mask.sum()) if mask is not None else 0
        rows.append((i, j, len(good), inliers))
    return rows


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------- main steps

def build_features(all_items, workers):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "features.npz"
    meta_path = CACHE / "meta.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["keys"] == [it["key"] for it in all_items] and meta["n_features"] == N_FEATURES:
            print(f"reusing features from {path}")
            return path, meta
    started = time.time()
    with Pool(workers) as pool:
        feats = pool.map(describe, all_items, chunksize=4)
    print(f"features for {len(feats)} images in {time.time() - started:.0f}s")
    points = np.empty(len(feats), dtype=object)
    desc = np.empty(len(feats), dtype=object)
    for k, f in enumerate(feats):
        points[k], desc[k] = f["points"], f["desc"]
    np.savez(path, points=points, desc=desc)
    meta = {"keys": [f["key"] for f in feats], "shapes": [f["shape"] for f in feats],
            "pixel_sha": [f["pixel_sha"] for f in feats], "phash": [f["phash"] for f in feats],
            "n_keypoints": [int(len(f["points"])) for f in feats], "n_features": N_FEATURES}
    meta_path.write_text(json.dumps(meta))
    return path, meta


def all_pairs(meta, feature_path, workers):
    keys = meta["keys"]
    n = len(keys)
    mendeley = [k for k in range(n) if keys[k].startswith("mendeley/")]
    defy = [k for k in range(n) if keys[k].startswith("eyes_defy/")]
    # Mendeley x Mendeley, and Eyes-defy x Mendeley. Eyes-defy x Eyes-defy is one image per
    # subject by construction and is skipped.
    jobs = [(i, [j for j in mendeley if j > i]) for i in mendeley]
    jobs += [(i, mendeley) for i in defy]
    jobs = [j for j in jobs if j[1]]
    total = sum(len(o) for _, o in jobs)
    print(f"matching {total} pairs on {workers} workers", flush=True)
    started, done, rows = time.time(), 0, []
    with Pool(workers, initializer=_init, initargs=(str(feature_path),)) as pool:
        for chunk in pool.imap_unordered(match_row, jobs, chunksize=1):
            rows.extend(chunk)
            done += len(chunk)
            if len(rows) and done % 20000 < len(chunk):
                print(f"  {done}/{total}  {time.time() - started:.0f}s", flush=True)
    print(f"matched in {time.time() - started:.0f}s")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    all_items = items()
    feature_path, meta = build_features(all_items, args.workers)
    keys = meta["keys"]

    # 1. exact copies
    by_sha = {}
    for k, sha in zip(keys, meta["pixel_sha"]):
        by_sha.setdefault(sha, []).append(k)
    exact = [v for v in by_sha.values() if len(v) > 1]

    # 2. near copies: smallest Hamming distance over the eight orientations of one image
    #    against the upright hash of the other
    n = len(keys)
    near = []
    for i, j in combinations(range(n), 2):
        if keys[i].startswith("eyes_defy/") and keys[j].startswith("eyes_defy/"):
            continue
        d = min(hamming(meta["phash"][i][0], h) for h in meta["phash"][j])
        near.append((i, j, d))

    # 3. every pair, keypoints + RANSAC
    rows = all_pairs(meta, feature_path, args.workers)
    phash_d = {(i, j): d for i, j, d in near}

    with (HERE / "pairs.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["a", "b", "ratio_matches", "ransac_inliers", "phash_distance"])
        for i, j, good, inl in sorted(rows, key=lambda r: -r[3]):
            if inl > 0 or phash_d.get((min(i, j), max(i, j)), 64) <= 16:
                w.writerow([keys[i], keys[j], good, inl, phash_d.get((min(i, j), max(i, j)))])

    inliers = np.array([r[3] for r in rows])
    hist = {f"{lo}-{hi}": int(((inliers >= lo) & (inliers < hi)).sum())
            for lo, hi in ((0, 1), (1, 10), (10, 15), (15, 20), (20, 30), (30, 50),
                           (50, 100), (100, 200), (200, 10 ** 6))}
    pd = np.array([d for _, _, d in near])
    summary = {
        "images": {"mendeley": sum(k.startswith("mendeley/") for k in keys),
                   "eyes_defy": sum(k.startswith("eyes_defy/") for k in keys)},
        "pairs_matched": len(rows),
        "exact_duplicate_groups": exact,
        "phash_distance_hist": {f"<= {t}": int((pd <= t).sum()) for t in (0, 4, 8, 12, 16)},
        "ransac_inlier_hist": hist,
        "top_pairs": [{"a": keys[i], "b": keys[j], "ratio_matches": g, "inliers": inl,
                       "phash_distance": phash_d.get((min(i, j), max(i, j)))}
                      for i, j, g, inl in sorted(rows, key=lambda r: -r[3])[:60]],
        "keypoints_median": float(np.median(meta["n_keypoints"])),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("images", "pairs_matched", "exact_duplicate_groups",
                                              "phash_distance_hist", "ransac_inlier_hist")}, indent=1))
    print("top 15 pairs by inliers:")
    for t in summary["top_pairs"][:15]:
        print("  ", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
