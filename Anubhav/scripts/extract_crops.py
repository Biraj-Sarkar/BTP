"""Extract conjunctiva crops with one or more extractors, ready for a regressor.

Built for one comparison in particular: does gray-world white balance help or
hurt haemoglobin prediction? Two Attention U-Net extractors exist, identical
except for white balance:

    raw   runs/segmenter/attention_raw_full.pt   trained and served on raw pixels
    gw    runs/segmenter/attention_gw_full.pt    trained and served white-balanced

This script runs any set of extractors over a folder of photographs and writes
one folder of crops per extractor, plus a manifest saying which images every
extractor accepted. A regressor trained on each folder in turn, with everything
else held fixed, answers the question.

    python scripts/extract_crops.py --images /path/to/photos --out crops/
    -> crops/raw/<same relative path>.png
       crops/gw/<same relative path>.png
       crops/manifest.csv       one row per photograph, both extractors
       crops/extractors.json    exactly which bundle produced which folder

What a crop is
--------------
The same thing the serving path produces (`preprocess.prepare()`): the frame is
segmented, everything outside the mask is set to black, the result is cut to
the mask's bounding box, padded to a square without stretching, and resized to
224x224. Saved as PNG, because the regressor reads colour and JPEG compression
would alter it.

**Colour follows the extractor.** With `--crop-colour match`, the default, the
crop's pixels are white-balanced exactly when the extractor was trained on
white-balanced images. Each extractor then defines one complete, consistent
pipeline, and "with versus without white balance" is a single switch rather
than two settings that can be combined by mistake. `--crop-colour raw` or `gw`
fixes the crop colour for every extractor instead, which separates the effect
of white balance on the mask from its effect on the colour the regressor reads.

Comparing fairly
----------------
Quality control can accept a photograph under one extractor and reject it under
the other, because the two produce different masks. Scoring each pipeline on
whatever it accepted would compare them on different patients. The manifest's
`usable_in_all` column marks photographs every extractor accepted; restrict both
training and evaluation to those rows.

Abbreviations
-------------
gw     gray-world white balance
QC     quality control
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.imaging import read_rgb  # noqa: E402
from anemia.preprocess import prepare  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

ALIASES = {"raw": "attention_raw_full.pt", "gw": "attention_gw_full.pt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve(spec: str, bundle_dir: Path) -> tuple[str, Path]:
    """`raw`, `gw`, or a path to any bundle. Returns (folder name, bundle path)."""

    if spec in ALIASES:
        return spec, bundle_dir / ALIASES[spec]
    path = Path(spec).expanduser()
    return path.stem, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and not p.name.startswith("."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", type=Path, required=True,
                        help="folder of photographs; searched recursively")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extractors", nargs="+", default=["raw", "gw"],
                        help="'raw', 'gw', or paths to bundles (default: raw gw)")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "runs/segmenter",
                        help="where the 'raw' and 'gw' aliases are looked up")
    parser.add_argument("--crop-colour", choices=["match", "raw", "gw"], default="match",
                        help="match: crop colour follows each extractor's training condition. "
                             "raw/gw: fixed for every extractor, to separate the mask effect "
                             "from the colour effect")
    parser.add_argument("--limit", type=int, default=None, help="first N photographs only")
    args = parser.parse_args()

    images = find_images(args.images)
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"No images found under {args.images}", file=sys.stderr)
        return 1

    extractors = []
    for spec in args.extractors:
        name, path = resolve(spec, args.bundle_dir)
        if not path.exists():
            print(f"Bundle not found for '{spec}': {path}\n"
                  f"Put the .pt files in {args.bundle_dir} or pass --bundle-dir.", file=sys.stderr)
            return 1
        segmenter = LearnedSegmenter(path)
        if args.crop_colour == "match":
            crop_gw = segmenter.wants_gray_world
        else:
            crop_gw = args.crop_colour == "gw"
        extractors.append({
            "name": name, "path": path, "segmenter": segmenter,
            "preprocess": replace(PreprocessConfig(), gray_world=crop_gw),
        })
    names = [e["name"] for e in extractors]
    if len(set(names)) != len(names):
        print(f"Two extractors share an output folder name: {names}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for e in extractors:
        seg = e["segmenter"]
        provenance[e["name"]] = {
            "bundle": str(e["path"]),
            "sha256": sha256(e["path"]),
            "cache_key": seg.cache_key,
            "arch": seg.config.arch,
            "segmenter_trained_on": "gray-world" if seg.wants_gray_world else "raw pixels",
            "crop_colour": "gray-world" if e["preprocess"].gray_world else "raw pixels",
            "trained_at": seg.provenance.get("trained_at"),
            "train_images": seg.provenance.get("train_images"),
        }
        print(f"{e['name']:<6} {e['path'].name}   segmenter on "
              f"{provenance[e['name']]['segmenter_trained_on']}, crop colour "
              f"{provenance[e['name']]['crop_colour']}")
    (args.out / "extractors.json").write_text(json.dumps(
        {"crop_colour_mode": args.crop_colour, "extractors": provenance}, indent=2))

    quality = QualityConfig()
    rows = []
    print(f"\n{len(images)} photographs\n")
    for i, image_path in enumerate(images, 1):
        relative = image_path.relative_to(args.images)
        row = {"image": relative.as_posix()}
        try:
            rgb = read_rgb(image_path)
        except Exception as error:                      # unreadable file
            for e in extractors:
                row.update({f"{e['name']}_passed": False,
                            f"{e['name']}_reasons": f"unreadable: {error}",
                            f"{e['name']}_crop": "", f"{e['name']}_mask_fraction": ""})
            rows.append(row)
            continue

        for e in extractors:
            prepared = prepare(rgb, e["segmenter"], e["preprocess"], quality)
            passed = bool(prepared.quality.passed)
            crop_path = ""
            if passed:
                target = args.out / e["name"] / relative.with_suffix(".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), cv2.cvtColor(prepared.crop, cv2.COLOR_RGB2BGR))
                crop_path = target.relative_to(args.out).as_posix()
            row.update({
                f"{e['name']}_passed": passed,
                f"{e['name']}_reasons": "; ".join(prepared.quality.reasons),
                f"{e['name']}_crop": crop_path,
                f"{e['name']}_mask_fraction": round(float((prepared.mask > 0).mean()), 5),
            })
        row["usable_in_all"] = all(row[f"{n}_passed"] for n in names)
        rows.append(row)
        if i % 25 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    for row in rows:
        row.setdefault("usable_in_all", False)
    fields = ["image", "usable_in_all"] + [f"{n}_{k}" for n in names
                                          for k in ("passed", "crop", "mask_fraction", "reasons")]
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\naccepted by quality control:")
    for n in names:
        print(f"  {n:<6} {sum(1 for r in rows if r[f'{n}_passed'])} of {len(rows)}")
    both = sum(1 for r in rows if r["usable_in_all"])
    print(f"  usable in all: {both} of {len(rows)}   <- train and evaluate on these rows only")
    print(f"\nWrote {args.out / 'manifest.csv'} and {args.out / 'extractors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
