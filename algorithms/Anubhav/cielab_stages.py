"""Stage-by-stage trace of the CIELAB pipeline (`cielab_pipeline.py`).

Runs the exact preprocessing flow from cielab_pipeline.py on one image or a
folder, saving every intermediate so the progression can be inspected file by
file. Nothing in cielab_pipeline.py is modified — this script only imports it.

    python cielab_stages.py --image left_eye/2.jpeg --out stages_cielab
    python cielab_stages.py --dir   left_eye        --out stages_cielab/left_eye

Each image produces ONE contact sheet, `<name>_stages_cielab.jpg`, with every
stage side by side so each can be compared against the one before it:

    00 original -> 01 resized -> 02 CLAHE on L* only -> 03 mask ->
    04 overlay  -> 05 masked  -> 06 crop + letterbox ->
    07 a* channel -> 08 b* channel

The last two panels are the point of this pipeline: the regression network
receives ONLY those two channels. L* is used for lighting normalization and
then discarded, so lighting cannot influence the prediction.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cielab_pipeline as cp  # noqa: E402


def save_rgb(image_rgb: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def heatmap(channel_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.applyColorMap(channel_u8, cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


def panel(image: np.ndarray, label: str, box: int = 300, bar: int = 30) -> np.ndarray:
    """Captioned panel, aspect ratio preserved by padding rather than stretching."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    height, width = image.shape[:2]
    scale = min(box / width, box / height)
    resized = cv2.resize(image, (max(int(width * scale), 1), max(int(height * scale), 1)),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((box + bar, box, 3), 28, dtype=np.uint8)
    y0, x0 = (box - resized.shape[0]) // 2, (box - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    cv2.putText(canvas, label, (6, box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (box - 1, box + bar - 1), (70, 70, 70), 1)
    return canvas


def contact_sheet(panels, header: str, per_row: int = 5) -> np.ndarray:
    rows = []
    for start in range(0, len(panels), per_row):
        chunk = list(panels[start:start + per_row])
        while len(chunk) < per_row:
            chunk.append(np.full_like(chunk[0], 28))
        rows.append(np.concatenate(chunk, axis=1))
    grid = np.concatenate(rows, axis=0)
    strip = np.full((34, grid.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(strip, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([strip, grid], axis=0)


def trace(image_path: Path, out_root: Path, config: cp.PipelineConfig,
          separate: bool = False) -> str:
    stem = image_path.stem
    rgb = cp.read_rgb(image_path)

    # Stage 1: resize — mirrors preprocess_hb_image in cielab_pipeline.
    resized = cp.resize_image(rgb, config.target_size)

    # Stage 2: CIELAB normalization — CLAHE on L* only, colour untouched.
    normalized_rgb, normalized_lab = cp.cielab_normalization(resized)

    # Stage 3-4: segmentation on the normalized RGB.
    mask, backend = cp.segment_conjunctiva(normalized_rgb)
    overlay = cp.build_overlay(normalized_rgb, mask)

    # Stage 5: mask applied to the LAB image — the branch the regressor uses.
    masked_lab = cp.apply_mask(normalized_lab, mask)

    # Stage 6: tight crop + square padding, exactly as crop_to_mask does it.
    cropped_lab, _ = cp.crop_to_mask(masked_lab, mask)
    source_lab = cropped_lab if cropped_lab.size else masked_lab

    # Stage 7-8: the 224x224 a* and b* channels — the network's actual input.
    lab224 = cv2.resize(source_lab, config.regression_image_size,
                        interpolation=cv2.INTER_AREA)

    stages = [
        (rgb, "00  original capture"),
        (resized, "01  resized"),
        (normalized_rgb, "02  CLAHE on L* only"),
        (mask, "03  segmentation mask"),
        (overlay, "04  mask over photo"),
        (cv2.cvtColor(masked_lab, cv2.COLOR_LAB2RGB), "05  outside mask removed"),
        (cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB), "06  crop + letterbox"),
        (heatmap(lab224[:, :, 1]), "07  a* channel -> network"),
        (heatmap(lab224[:, :, 2]), "08  b* channel -> network"),
    ]

    out_root.mkdir(parents=True, exist_ok=True)
    save_rgb(contact_sheet([panel(i, l) for i, l in stages],
                           f"{image_path.name}   backend={backend}   CIELAB pipeline"),
             out_root / f"{stem}_stages_cielab.jpg")

    if separate:
        folder = out_root / stem
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, folder / image_path.name)
        for index, (image, label) in enumerate(stages[1:], start=1):
            name = label.split("  ", 1)[1].replace(" ", "_").replace("*", "").replace("->", "to")
            if image.ndim == 2:
                cv2.imwrite(str(folder / f"{stem}_{index:02d}_{name}.png"), image)
            else:
                save_rgb(image, folder / f"{stem}_{index:02d}_{name}.jpg")

    return backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="single photo")
    source.add_argument("--dir", type=Path, help="folder of photos")
    parser.add_argument("--out", type=Path, default=Path("stages_cielab"))
    parser.add_argument("--separate", action="store_true",
                        help="also write each stage as its own file")
    args = parser.parse_args()

    if args.image:
        files = [args.image]
    else:
        files = sorted(
            (p for p in args.dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
        )
    if not files:
        raise SystemExit("No images found.")

    # dataset_root/output_root are unused by the preprocessing path but the
    # config dataclass requires them.
    config = cp.PipelineConfig(dataset_root=Path("."), output_root=args.out)

    print(f"Tracing {len(files)} image(s) -> {args.out}\n")
    for file in files:
        backend = trace(file, args.out, config, args.separate)
        print(f"  {file.stem}: backend={backend}")

    print("\nDone. One sheet per capture: <name>_stages_cielab.jpg")
    print("The last two panels (a* and b* heatmaps) are the only channels the")
    print("regressor sees. Pass --separate to also write each stage as a file.")


if __name__ == "__main__":
    main()
