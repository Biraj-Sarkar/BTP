"""How much do resizing and squashing cost? Three measurements, no training.

Every image in this project is resized at least twice before anything reads it,
and every resize to a square also changes its shape, because 97% of the photos
are 3:4 portrait. This script measures what that costs, on the annotators' own
masks, so that no model's errors are mixed in.

What the app does to a photograph (see `preprocess.prepare`)
-----------------------------------------------------------
1. Squash the whole frame to 512x512 (area interpolation). A 3000x4000 photo
   loses about 98% of its pixels and is stretched 1.33x sideways.
2. The segmenter squashes that again to 320x320, predicts a probability per
   pixel, and the probability map is enlarged back to 512x512 and thresholded.
3. Colour is read inside the mask from the 512x512 frame.
4. The neural model's input is the mask's bounding box cut from the 512x512
   frame, padded to a square and resized to 224x224.

The three measurements
----------------------
1. **Mask ceiling.** The annotator's mask is shrunk to a grid, enlarged back and
   scored against itself at native resolution. This is the best Dice that any
   model working on that grid could reach, since a perfect model can do no
   better than the true mask sampled on its grid. "Soft" keeps the fraction of
   each grid cell that is conjunctiva, which is what an ideal probability map
   is, enlarges it smoothly and thresholds it at 0.5, as the app does. "Hard"
   thresholds on the grid first and enlarges the blocks. Also scored: the
   training labels themselves (the 576x768 copies the models learn from)
   against the native annotation.
2. **Colour.** The colour features the project uses, computed inside the
   annotator's mask on the native photograph, and again after each resize.
   Differences are reported against the spread of that feature between people,
   because a shift that is small next to the difference between two people
   cannot move a haemoglobin estimate much.
3. **Structure.** The green channel (where vessels have the most contrast) is
   shrunk and enlarged back, and compared with the original inside the
   conjunctiva: SSIM, and the share of detail energy kept in each band of
   feature size. Losing a band means features of that size are no longer in the
   image the network sees.

Measurements 2 and 3 use the 547 Mendeley images only: their masks were drawn
on the full 12-megapixel photograph. The 101 Eyes-defy annotations exist only
at 800x1067, so for those the "native" mask is already a reduced one; they are
included in measurement 1 and reported separately.

Run
---
    python experiments/22_resolution/measure.py            # all images, about 15 minutes
    python experiments/22_resolution/measure.py --limit 5  # quick check

Abbreviations
-------------
Dice   overlap score, 2|A∩B| / (|A|+|B|)
SSIM   structural similarity index (1 = identical)
SD     standard deviation
px     pixels
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anemia.imaging import pad_to_square, tight_bbox  # noqa: E402
from anemia.linear_model import extract_features  # noqa: E402

DATASETS = Path.home() / "btp-local/datasets"
MENDELEY = DATASETS / "conjunctiva_seg/Eye Conjunctiva Segmentation Dataset"
MENDELEY_PREPARED = DATASETS / "conjunctiva_seg_prepared"
EYES_DEFY_PREPARED = DATASETS / "eyes_defy_prepared"

SQUARE_GRIDS = (160, 224, 320, 448, 512, 640, 768)
PORTRAIT_GRIDS = ((240, 320), (288, 384), (384, 512), (576, 768))   # (width, height), 3:4
COLOUR_FEATURES = [("chroma", ["r chroma", "g chroma"]),
                   ("erythema", ["log(R/G)", "log(R/B)"]),
                   ("lab", ["a*", "b*"]),
                   ("means", ["mean R", "mean G", "mean B"])]
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32))   # Gaussian sigma pairs, native px
ERODE_PX = 8          # keep structure metrics off the blurred mask edge
CNN_SIZE, WORK_SIZE, SEG_SIZE = 224, 512, 320


# ----------------------------------------------------------------- loading

def mendeley_items():
    for p in sorted((MENDELEY_PREPARED / "images").iterdir()):
        if p.suffix == ".png":
            stem = p.stem.rsplit("_", 1)[0]
            yield {"dataset": "mendeley", "stem": stem}


def eyes_defy_items():
    for p in sorted((EYES_DEFY_PREPARED / "images").iterdir()):
        if p.suffix == ".png":
            yield {"dataset": "eyes_defy", "stem": p.stem.rsplit("_", 1)[0]}


def native_mask(item) -> np.ndarray:
    """The annotation at the resolution it was drawn, as booleans.

    Mendeley: intersection of the two annotators, the same policy as the
    prepared copies, but at 12 MP instead of 576x768.
    """

    if item["dataset"] == "mendeley":
        masks = []
        for folder in ("Masks Annotator 1", "Masks Annotator 2"):
            m = cv2.imread(str(MENDELEY / folder / f"{item['stem']}.png"), cv2.IMREAD_UNCHANGED)
            masks.append((m.max(axis=2) if m.ndim == 3 else m) > 0)
        return masks[0] & masks[1]
    m = cv2.imread(str(EYES_DEFY_PREPARED / "masks" / f"{item['stem']}_0.png"), cv2.IMREAD_GRAYSCALE)
    return m > 127


def native_image(item) -> np.ndarray:
    bgr = cv2.imread(str(MENDELEY / "Images" / f"{item['stem']}.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepared_mask(item) -> np.ndarray:
    root = MENDELEY_PREPARED if item["dataset"] == "mendeley" else EYES_DEFY_PREPARED
    return cv2.imread(str(root / "masks" / f"{item['stem']}_0.png"), cv2.IMREAD_GRAYSCALE) > 127


# ----------------------------------------------------------------- helpers

def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return 1.0 if total == 0 else 2.0 * inter / total


def oriented(grid, shape):
    """A (width, height) portrait grid, turned to match a landscape frame."""

    w, h = grid
    return (w, h) if shape[0] >= shape[1] else (h, w)


def soft_ceiling(mask: np.ndarray, size) -> tuple[float, float]:
    """Best Dice on a grid: (soft, hard). `size` is (width, height)."""

    height, width = mask.shape
    coverage = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    soft = cv2.resize(coverage, (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
    hard = cv2.resize((coverage >= 0.5).astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST) > 0
    return dice(soft, mask), dice(hard, mask)


def app_chain_ceiling(mask: np.ndarray) -> float:
    """Ideal model inside the app: 320 probability -> 512 -> threshold -> native."""

    height, width = mask.shape
    coverage = cv2.resize(mask.astype(np.float32), (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
    at_work = cv2.resize(coverage, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_LINEAR) >= 0.5
    back = cv2.resize(at_work.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return dice(back, mask)


def thickness_px(mask: np.ndarray) -> float:
    """Mean thickness of the region, 2 x area / perimeter, in native px."""

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * float(mask.sum()) / perimeter if perimeter else float("nan")


def features(image: np.ndarray, mask: np.ndarray) -> dict:
    out = {}
    for rep, names in COLOUR_FEATURES:
        for name, value in zip(names, extract_features(image, mask.astype(np.uint8), rep)):
            out[name] = float(value)
    return out


def cnn_crop(image: np.ndarray, mask: np.ndarray):
    """The neural model's input: bounding box (8% pad) -> pad to square -> 224."""

    left, top, right, bottom = tight_bbox(mask.astype(np.uint8), 0.08)
    crop_i, crop_m = image[top:bottom, left:right], mask[top:bottom, left:right].astype(np.uint8)
    small_i = cv2.resize(pad_to_square(crop_i), (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(pad_to_square(crop_m), (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_NEAREST)
    return small_i, small_m > 0, (left, top, right, bottom)


def unpad_resize(small: np.ndarray, crop_h: int, crop_w: int, out_w: int, out_h: int) -> np.ndarray:
    """Undo pad_to_square + resize, then scale the crop to (out_w, out_h)."""

    side = max(crop_h, crop_w)
    padded = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
    top, left = (side - crop_h) // 2, (side - crop_w) // 2
    crop = padded[top:top + crop_h, left:left + crop_w]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def ssim_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SSIM per pixel, Gaussian window sigma 1.5 (Wang et al., 2004)."""

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def band_energy(g: np.ndarray, inside: np.ndarray) -> list:
    """Detail energy inside the mask in each band of feature size."""

    blurred = {s: cv2.GaussianBlur(g, (0, 0), s) for s in {s for pair in BANDS for s in pair}}
    return [float(np.mean((blurred[a] - blurred[b])[inside] ** 2)) for a, b in BANDS]


# ----------------------------------------------------------------- one image

def measure(item) -> dict:
    mask = native_mask(item)
    height, width = mask.shape
    row = {"dataset": item["dataset"], "stem": item["stem"], "height": height, "width": width,
           "coverage": float(mask.mean()), "thickness_px": thickness_px(mask)}

    # 1. Mask ceiling
    ceiling = {}
    for s in SQUARE_GRIDS:
        ceiling[f"sq{s}_soft"], ceiling[f"sq{s}_hard"] = soft_ceiling(mask, (s, s))
    for grid in PORTRAIT_GRIDS:
        w, h = grid
        ceiling[f"p{w}x{h}_soft"], ceiling[f"p{w}x{h}_hard"] = soft_ceiling(mask, oriented(grid, mask.shape))
    ceiling["app_chain"] = app_chain_ceiling(mask)
    labels = cv2.resize(prepared_mask(item).astype(np.uint8), (width, height),
                        interpolation=cv2.INTER_NEAREST) > 0
    ceiling["training_labels"] = dice(labels, mask)
    row["ceiling"] = ceiling
    # Above 1 means the labels the models learn from are fatter than the annotation.
    row["label_area_ratio"] = float(labels.sum() / max(mask.sum(), 1))

    if item["dataset"] != "mendeley":
        return row

    image = native_image(item)
    if image.shape[:2] != mask.shape:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # 2. Colour
    colour = {"native": features(image, mask)}
    scale = 768 / max(height, width)
    prep_size = (int(round(width * scale)), int(round(height * scale)))
    for name, size, interp in (("prepared_768", prep_size, cv2.INTER_AREA),
                               ("app_512", (WORK_SIZE, WORK_SIZE), cv2.INTER_AREA),
                               ("seg_320_area", (SEG_SIZE, SEG_SIZE), cv2.INTER_AREA),
                               ("seg_320_linear", (SEG_SIZE, SEG_SIZE), cv2.INTER_LINEAR)):
        small = cv2.resize(image, size, interpolation=interp)
        small_m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        colour[name] = features(small, small_m)
    work = cv2.resize(image, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    work_m = cv2.resize(mask.astype(np.uint8), (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_NEAREST) > 0
    app_crop, app_crop_m, app_box = cnn_crop(work, work_m)
    colour["cnn_224_app"] = features(app_crop, app_crop_m)
    nat_crop, nat_crop_m, nat_box = cnn_crop(image, mask)
    colour["cnn_224_native"] = features(nat_crop, nat_crop_m)
    coverage = cv2.resize(mask.astype(np.float32), (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
    ideal_mask = cv2.resize(coverage, (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
    colour["native_pixels_ideal320_mask"] = features(image, ideal_mask)
    row["colour"] = colour

    # 3. Structure, inside the native bounding box only (the rest is not read)
    left, top, right, bottom = nat_box
    region = (slice(top, bottom), slice(left, right))
    green = image[:, :, 1].astype(np.float32)
    ref = green[region]
    inside = cv2.erode(mask[region].astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)) > 0
    if inside.sum() < 50:
        inside = mask[region]
    ref_energy = band_energy(ref, inside)

    rebuilt, full_frame = {}, {}
    for name, size in (("prepared_768", prep_size), ("app_512", (WORK_SIZE, WORK_SIZE)),
                       ("seg_320", (SEG_SIZE, SEG_SIZE))):
        small = cv2.resize(green, size, interpolation=cv2.INTER_AREA)
        full_frame[name] = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        rebuilt[name] = full_frame[name][region]
    # Neural model input, cut from the native frame
    nat_small = cv2.resize(pad_to_square(ref), (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_AREA)
    rebuilt["cnn_224_native"] = unpad_resize(nat_small, bottom - top, right - left, right - left, bottom - top)
    # Neural model input as the app builds it, cut from the 512 frame, mapped back to native
    l5, t5, r5, b5 = app_box
    work_g = work[:, :, 1].astype(np.float32)
    app_small = cv2.resize(pad_to_square(work_g[t5:b5, l5:r5]), (CNN_SIZE, CNN_SIZE),
                           interpolation=cv2.INTER_AREA)
    sx, sy = width / WORK_SIZE, height / WORK_SIZE
    x0, y0 = int(round(l5 * sx)), int(round(t5 * sy))
    x1, y1 = int(round(r5 * sx)), int(round(b5 * sy))
    # Outside its box the app has only the 512 frame, so fill with that rather
    # than zeros, which would add a false edge to the coarse detail bands.
    app_native = full_frame["app_512"].copy()
    app_native[y0:y1, x0:x1] = unpad_resize(app_small, b5 - t5, r5 - l5, x1 - x0, y1 - y0)
    rebuilt["cnn_224_app"] = app_native[region]

    structure = {}
    for name, rec in rebuilt.items():
        energy = band_energy(rec, inside)
        structure[name] = {"ssim": float(ssim_map(ref, rec)[inside].mean()),
                           "band_kept": [e / r if r > 0 else float("nan")
                                         for e, r in zip(energy, ref_energy)]}
    row["structure"] = structure
    row["crop_px"] = {"native_box_w": right - left, "native_box_h": bottom - top,
                      "app_box_w_at_512": r5 - l5, "app_box_h_at_512": b5 - t5}
    return row


def safe_measure(item):
    try:
        return measure(item)
    except Exception as exc:  # report and continue; one bad file should not stop the run
        return {"dataset": item["dataset"], "stem": item["stem"], "error": repr(exc)}


# ----------------------------------------------------------------- summary

def quantiles(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "min": float(v.min())}


def summarise(rows: list) -> dict:
    out = {"ceiling": {}, "colour": {}, "structure": {}}
    for dataset in ("mendeley", "eyes_defy", "all"):
        group = [r for r in rows if "ceiling" in r and (dataset == "all" or r["dataset"] == dataset)]
        if not group:
            continue
        keys = group[0]["ceiling"].keys()
        out["ceiling"][dataset] = {k: quantiles([r["ceiling"][k] for r in group]) for k in keys}
        out["ceiling"][dataset]["thickness_px"] = quantiles([r["thickness_px"] for r in group])
        out["ceiling"][dataset]["coverage"] = quantiles([r["coverage"] for r in group])
        out["ceiling"][dataset]["label_area_ratio"] = quantiles([r["label_area_ratio"] for r in group])

    # Thin masks lose more: the 320 soft ceiling by thickness tercile (Mendeley)
    men = [r for r in rows if r.get("dataset") == "mendeley" and "ceiling" in r]
    t = np.array([r["thickness_px"] for r in men])
    cuts = np.percentile(t, [100 / 3, 200 / 3])
    out["ceiling"]["mendeley_by_thickness"] = {
        label: {"thickness_range_px": [float(t[sel].min()), float(t[sel].max())],
                "sq320_soft": quantiles([r["ceiling"]["sq320_soft"] for r, s in zip(men, sel) if s]),
                "app_chain": quantiles([r["ceiling"]["app_chain"] for r, s in zip(men, sel) if s])}
        for label, sel in (("thinnest third", t <= cuts[0]),
                           ("middle third", (t > cuts[0]) & (t <= cuts[1])),
                           ("thickest third", t > cuts[1]))}

    col = [r for r in rows if "colour" in r]
    if col:
        names = list(col[0]["colour"]["native"].keys())
        variants = [k for k in col[0]["colour"] if k != "native"]
        for name in names:
            native = np.array([r["colour"]["native"][name] for r in col])
            spread = float(native.std(ddof=1))
            entry = {"between_person_sd": spread}
            for v in variants:
                d = np.array([r["colour"][v][name] for r in col]) - native
                other = native + d
                entry[v] = {"mean_shift": float(d.mean()),
                            "mean_abs_shift": float(np.abs(d).mean()),
                            "max_abs_shift": float(np.abs(d).max()),
                            "mean_abs_shift_in_sd": float(np.abs(d).mean() / spread),
                            "max_abs_shift_in_sd": float(np.abs(d).max() / spread),
                            "pearson_with_native": float(np.corrcoef(native, other)[0, 1])}
            out["colour"][name] = entry

    st = [r for r in rows if "structure" in r]
    if st:
        for v in st[0]["structure"]:
            kept = np.array([r["structure"][v]["band_kept"] for r in st], dtype=float)
            out["structure"][v] = {
                "ssim": quantiles([r["structure"][v]["ssim"] for r in st]),
                "band_kept_median": {f"{a}-{b}px": float(np.nanmedian(kept[:, i]))
                                     for i, (a, b) in enumerate(BANDS)}}
        out["structure"]["crop_px_median"] = {
            k: float(np.median([r["crop_px"][k] for r in st])) for k in st[0]["crop_px"]}
    return out


def write(rows: list, summary: dict) -> None:
    good = [r for r in rows if "ceiling" in r]
    with (HERE / "ceiling_per_image.csv").open("w", newline="") as h:
        keys = list(good[0]["ceiling"].keys())
        w = csv.writer(h)
        w.writerow(["dataset", "stem", "width", "height", "coverage", "thickness_px", *keys])
        for r in good:
            w.writerow([r["dataset"], r["stem"], r["width"], r["height"], round(r["coverage"], 6),
                        round(r["thickness_px"], 2), *(round(r["ceiling"][k], 5) for k in keys)])
    col = [r for r in rows if "colour" in r]
    if col:
        with (HERE / "colour_per_image.csv").open("w", newline="") as h:
            names = list(col[0]["colour"]["native"].keys())
            variants = list(col[0]["colour"].keys())
            w = csv.writer(h)
            w.writerow(["stem", "variant", *names])
            for r in col:
                for v in variants:
                    w.writerow([r["stem"], v, *(round(r["colour"][v][n], 6) for n in names)])
    st = [r for r in rows if "structure" in r]
    if st:
        with (HERE / "structure_per_image.csv").open("w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["stem", "variant", "ssim", *(f"kept_{a}-{b}px" for a, b in BANDS)])
            for r in st:
                for v, d in r["structure"].items():
                    w.writerow([r["stem"], v, round(d["ssim"], 5), *(round(x, 5) for x in d["band_kept"])])
    errors = [r for r in rows if "error" in r]
    summary["errors"] = errors
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="first N images of each dataset (quick check)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    items = list(mendeley_items()), list(eyes_defy_items())
    if args.limit:
        items = items[0][:args.limit], items[1][:args.limit]
    work = items[0] + items[1]
    print(f"{len(items[0])} Mendeley + {len(items[1])} Eyes-defy images, {args.workers} workers", flush=True)

    started = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(safe_measure, work, chunksize=2), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(work):
                print(f"  {i}/{len(work)}  {time.time() - started:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["dataset"], r["stem"]))
    summary = summarise(rows)
    write(rows, summary)
    errors = summary["errors"]
    print(f"done in {time.time() - started:.0f}s; {len(errors)} errors"
          + (f": {errors[:3]}" if errors else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
