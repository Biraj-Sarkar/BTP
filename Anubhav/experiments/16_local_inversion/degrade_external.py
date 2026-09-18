"""Could image processing alone have erased the local cohort's colour signal?

The local captures are hand-cropped exports: 52 different sizes, 29-97 KB, JPEG
quality ~75 with 4:2:0 chroma subsampling, no EXIF or colour profile. The
external photographs are untouched camera files and carry a strong
physiological signal (red chromaticity vs haemoglobin r = +0.67).

This applies the local cohort's processing to the external photographs and
checks whether that signal survives. Scale is set so the learned extractor's
crescents end up as thick, in pixels, as they are in the local images; then the
images are JPEG-compressed at the local quality, once and twice.

If the correlation collapses or inverts, processing is a sufficient cause. If it
survives, processing is ruled out and the explanation lies in the photographs
themselves.

Run: python experiments/16_local_inversion/degrade_external.py

Abbreviations
-------------
JPEG   lossy image compression; q = quality setting (0-100)
Hb     haemoglobin (g/dL)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_dataset  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure_local import chroma, thickness  # noqa: E402


def jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main() -> None:
    unet = get_extractor("unet")
    local = pd.read_csv(HERE / "local_measurements.csv")
    local_thick = float(local["unet_thick_px"].median())

    samples = load_dataset(ROOT / "datasets/eyes_defy/dataset anemia")
    natives, thick = [], []
    for s in samples:
        rgb = read_rgb(s.image_path)
        mask, _ = unet(rgb)
        t = thickness(mask)
        natives.append((s, rgb))
        thick.append(t)
    ext_thick = float(np.nanmedian(thick))
    scale = local_thick / ext_thick
    print(f"median crescent thickness: external {ext_thick:.1f}px, local {local_thick:.1f}px "
          f"-> scale external by {scale:.3f}")

    variants = {
        "original": lambda im: im,
        "downscaled": lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
        "downscaled + JPEG q75": lambda im: jpeg(cv2.resize(im, None, fx=scale, fy=scale,
                                                            interpolation=cv2.INTER_AREA), 75),
        "downscaled + JPEG q75 twice": lambda im: jpeg(jpeg(cv2.resize(
            im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 75), 75),
        "downscaled + JPEG q40 twice": lambda im: jpeg(jpeg(cv2.resize(
            im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 40), 40),
    }

    results = {}
    print(f"\n{'variant':<30}{'n':>5}{'r vs Hb (raw)':>16}{'b vs Hb (raw)':>16}"
          f"{'r vs Hb (gw)':>15}{'b vs Hb (gw)':>15}")
    for name, fn in variants.items():
        rows = []
        for s, native in natives:
            im = fn(native)
            mask, _ = unet(im)
            if np.count_nonzero(mask) < 50:
                continue
            rr, _, rb = chroma(im, mask)
            gr, _, gb = chroma(gray_world_white_balance(im), mask)
            rows.append((s.hb, rr, rb, gr, gb))
        a = np.array(rows)
        cors = [stats.pearsonr(a[:, k], a[:, 0])[0] for k in (1, 2, 3, 4)]
        results[name] = {"n": len(a), "raw_r": cors[0], "raw_b": cors[1],
                         "gw_r": cors[2], "gw_b": cors[3]}
        print(f"{name:<30}{len(a):>5}{cors[0]:>+16.3f}{cors[1]:>+16.3f}{cors[2]:>+15.3f}{cors[3]:>+15.3f}")

    print("\nlocal cohort, for reference (26 patients, both eyes averaged):  "
          "r raw -0.243, b raw +0.318, r gw -0.357, b gw +0.506")
    (HERE / "degrade_external.json").write_text(json.dumps(
        {"scale": scale, "external_thickness_px": ext_thick, "local_thickness_px": local_thick,
         "results": results}, indent=2))


if __name__ == "__main__":
    main()
