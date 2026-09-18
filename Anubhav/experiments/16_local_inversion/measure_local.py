"""Why does the local cohort lose the colour-haemoglobin signal? Measure the suspects.

Under the learned extractor, the 217-subject external cohort shows a strong
physiological relationship (red chromaticity vs haemoglobin r = +0.67), while the
26 local patients show none on red (r = -0.26, p = 0.20) and a *positive* blue
correlation under gray-world (r = +0.52, p = 0.006). The two cohorts' red
correlations differ at p < 1e-6, so the discrepancy is real even though the local
"inversion" on red is not significant on its own.

The local images carry no EXIF and no colour profile, and all 52 have different
dimensions (695x414 to 1556x500) at 29-97 KB — hand-cropped, recompressed
exports rather than camera files. That suggests specific mechanisms, each
measured here per capture:

* **Thin masks in small JPEGs.** JPEG stores colour at half resolution in 2x2
  blocks (4:2:0) inside 16x16 coding units, so a crescent a few pixels thick has
  its colour averaged with the lid skin and sclera beside it. Measured as mask
  thickness in native pixels, and the chroma-subsampling mode of each file.
* **Crop composition.** White balance — the phone's, before export, or
  gray-world here — rescales colour by frame content, and every crop frames a
  different mix of tissue. Measured as whole-frame chromaticity and mask share.
* **Illumination / session.** Sclera is roughly neutral and its colour should not
  track haemoglobin, so sclera chromaticity correlating with Hb would indicate a
  capture confound. Capture dates span nine months (`Created On`).
* **Demographics.** Age, sex, socio-economic status.

Writes one row per capture to `local_measurements.csv`.

Abbreviations
-------------
Hb     haemoglobin (g/dL)
JPEG   the image compression format; 4:2:0 means colour at half resolution
MCU    minimum coded unit, the JPEG block (16x16 px under 4:2:0)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import gray_world_white_balance, read_rgb  # noqa: E402
from anemia.segment import get_extractor, legacy_bright_neutral_mask  # noqa: E402

OUT = Path(__file__).resolve().parent / "local_measurements.csv"
SAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "unknown"}

# IJG standard luminance quantisation table at quality 50, for estimating quality.
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def jpeg_quality(image: Image.Image) -> float:
    """Approximate IJG quality from the luminance quantisation table."""

    tables = getattr(image, "quantization", None)
    if not tables:
        return float("nan")
    q = np.array(tables[0], dtype=float)
    scale = float(np.median(q / IJG_LUMA)) * 100.0
    return 100.0 - scale / 2.0 if scale <= 100 else 5000.0 / scale


def chroma(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    px = rgb[mask > 0].astype(np.float64)
    if not len(px):
        return (np.nan, np.nan, np.nan)
    total = np.maximum(px.sum(axis=1), 1e-6)
    r, g = (px[:, 0] / total).mean(), (px[:, 1] / total).mean()
    return (float(r), float(g), float(1.0 - r - g))


def thickness(mask: np.ndarray) -> float:
    """Typical crescent thickness in pixels: twice the median inner distance
    along the mask's ridge (its local maxima of the distance transform)."""

    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return float("nan")
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist > 0)
    return float(2.0 * np.median(dist[ridge]))


def main() -> None:
    unet = get_extractor("unet")
    classical = get_extractor("refined")
    table = pd.read_csv(ROOT / "DATASAMPLE.csv")
    table["Image ID"] = table["Image ID"].astype(str)

    rows = []
    for sample in load_local_cohort(ROOT):
        image_id = sample.image_path.stem
        meta = table[table["Image ID"] == image_id].iloc[0]
        pil = Image.open(sample.image_path)

        raw = read_rgb(sample.image_path)           # native resolution, no balancing
        balanced = gray_world_white_balance(raw)
        full = np.full(raw.shape[:2], 255, np.uint8)

        mask_u, _ = unet(raw)
        mask_c, _ = classical(balanced)
        sclera = cv2.bitwise_and(legacy_bright_neutral_mask(balanced),
                                 cv2.bitwise_not(cv2.bitwise_or(mask_u, mask_c)))

        row = {
            "capture": f"{sample.image_path.parent.name}/{sample.image_path.name}",
            "patient": sample.patient_id, "eye": sample.image_path.parent.name,
            "hb": sample.hb, "age": sample.age_years, "sex": sample.sex,
            "ses": meta.get("Socio-economic status"),
            "created": pd.to_datetime(str(meta.get("Created On"))[:24], errors="coerce"),
            "width": raw.shape[1], "height": raw.shape[0],
            "kb": sample.image_path.stat().st_size / 1024,
            "jpeg_q": jpeg_quality(pil),
            "subsampling": SAMPLING.get(JpegImagePlugin.get_sampling(pil), "unknown"),
            "unet_area": float((mask_u > 0).mean()),
            "unet_thick_px": thickness(mask_u),
            "classical_area": float((mask_c > 0).mean()),
            "classical_thick_px": thickness(mask_c),
            "sclera_area": float((sclera > 0).mean()),
        }
        for tag, image in (("raw", raw), ("gw", balanced)):
            for region, m in (("frame", full), ("unet", mask_u), ("classical", mask_c),
                              ("sclera", sclera)):
                r, g, b = chroma(image, m)
                row[f"{region}_{tag}_r"], row[f"{region}_{tag}_g"], row[f"{region}_{tag}_b"] = r, g, b
        rows.append(row)
        print(f"{row['capture']:<18} hb {row['hb']:>5}  {row['width']}x{row['height']}  "
              f"q~{row['jpeg_q']:.0f} {row['subsampling']}  unet thick {row['unet_thick_px']:.1f}px  "
              f"classical thick {row['classical_thick_px']:.1f}px")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
