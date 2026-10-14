"""Side-by-side masks from the three architectures on the 52-capture cohort.

The three stage-1 bundles — trained on the same 389 images, selected on the same
130, scored on the same 129 — are run over the 52 captures the project started
with, and their masks are drawn side by side.

**These images carry no segmentation annotations.** Only haemoglobin labels, so
there is no reference mask and nothing here is an accuracy measurement. Two
things it *can* do, both useful:

1. **Show what the Dice difference looks like.** A gap of 0.01 Dice on the
   held-out set is abstract. Seeing where two masks actually diverge on a real
   capture is not.
2. **Measure agreement between the models.** With no ground truth, pairwise IoU
   between the three predictions still says where the architectures disagree.
   Images are ordered by that, worst agreement first, so the first sheet holds
   the cases worth looking at rather than fifty near-identical rows.

Disagreement is not error — all three can be wrong together, and on a cohort
whose captures differ from the training data that is a real possibility. What
unanimous agreement does rule out is one architecture failing alone.

Run: python experiments/20_architectures/overlays.py
Outputs: `overlays/sheet_NN.jpg`, `agreement.csv`.

Abbreviations
-------------
IoU    intersection over union, |A∩B| / |A∪B|
px     pixels
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anemia.data import load_local_cohort  # noqa: E402
from anemia.imaging import read_rgb, resize  # noqa: E402
from anemia.segment_learned import LearnedSegmenter  # noqa: E402

ARCHS = ("unet", "unetplusplus", "attention")
LABEL = {"unet": "U-Net", "unetplusplus": "UNet++", "attention": "Attention U-Net"}
# Distinct, colour-blind-safe outline colours, BGR for cv2.
COLOUR = {"unet": (74, 136, 212), "unetplusplus": (99, 184, 108), "attention": (160, 84, 198)}
WORK = (512, 512)
ROWS_PER_SHEET = 7
PANEL = 300          # px per panel in the contact sheet
PAD = 0.18           # fraction of the union box added as margin


def overlay(rgb: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    """Tint the mask interior and draw its outline, leaving the rest untouched."""

    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if mask.any():
        tint = out.copy()
        tint[mask > 0] = colour
        out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def crop_box(masks: list, shape) -> tuple:
    """Bounding box of every mask together, padded, so all panels show the same region."""

    union = np.zeros(shape[:2], dtype=bool)
    for m in masks:
        union |= m > 0
    if not union.any():
        return 0, 0, shape[1], shape[0]
    ys, xs = np.where(union)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1 - x0) * PAD) + 10, int((y1 - y0) * PAD) + 10
    return (max(0, x0 - px), max(0, y0 - py),
            min(shape[1], x1 + px), min(shape[0], y1 + py))


def fit(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox into a square panel so nothing is stretched."""

    h, w = img.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 245, dtype=np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def label(panel: np.ndarray, text: str, colour=(40, 30, 34)) -> np.ndarray:
    strip = np.full((26, panel.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def main() -> None:
    bundles = {a: HERE / "bundles" / f"{a}.pt" for a in ARCHS}
    missing = [a for a, p in bundles.items() if not p.exists()]
    if missing:
        print(f"Missing bundles: {missing}", file=sys.stderr)
        raise SystemExit(1)
    segmenters = {a: LearnedSegmenter(p) for a, p in bundles.items()}

    samples = load_local_cohort(ROOT)
    print(f"{len(samples)} captures from {len({s.patient_id for s in samples})} patients\n")

    records, panels = [], {}
    for sample in samples:
        rgb = resize(read_rgb(sample.image_path), WORK)     # no colour correction
        masks = {}
        for arch, seg in segmenters.items():
            mask, _ = seg(rgb)
            masks[arch] = mask

        row = {"image": sample.image_path.name, "patient": sample.patient_id,
               "eye": sample.image_path.parent.name, "hb": sample.hb}
        ious = []
        for i, a in enumerate(ARCHS):
            row[f"{a}_coverage"] = float((masks[a] > 0).mean())
            for b in ARCHS[i + 1:]:
                inter = np.logical_and(masks[a] > 0, masks[b] > 0).sum()
                union = np.logical_or(masks[a] > 0, masks[b] > 0).sum()
                iou = float(inter / union) if union else 1.0
                row[f"iou_{a}_{b}"] = iou
                ious.append(iou)
        row["mean_iou"] = float(np.mean(ious))
        row["empty"] = int(sum(1 for a in ARCHS if not (masks[a] > 0).any()))
        records.append(row)

        x0, y0, x1, y1 = crop_box(list(masks.values()), rgb.shape)
        original = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        cells = [label(fit(original, PANEL),
                       f"{sample.image_path.parent.name}/{sample.image_path.name}  Hb {sample.hb}")]
        for arch in ARCHS:
            drawn = overlay(rgb, masks[arch], COLOUR[arch])[y0:y1, x0:x1]
            cover = 100 * (masks[arch] > 0).mean()
            tag = f"{LABEL[arch]}  {cover:.1f}%" + ("  EMPTY" if cover == 0 else "")
            cells.append(label(fit(drawn, PANEL), tag, COLOUR[arch][::-1]))
        panels[sample.image_path.as_posix()] = np.hstack(cells)

    df = pd.DataFrame(records).sort_values("mean_iou")
    df.to_csv(HERE / "agreement.csv", index=False)

    print("Agreement between the three models (no ground truth available):")
    print(f"  mean pairwise IoU   {df.mean_iou.mean():.3f}")
    print(f"  worst capture       {df.mean_iou.min():.3f}  ({df.iloc[0]['image']}, "
          f"{df.iloc[0]['eye']})")
    print(f"  captures below 0.80 {(df.mean_iou < 0.80).sum()} of {len(df)}")
    print(f"  any empty mask      {int((df['empty'] > 0).sum())} captures")
    for i, a in enumerate(ARCHS):
        for b in ARCHS[i + 1:]:
            print(f"  {LABEL[a]:<16} vs {LABEL[b]:<16} mean IoU {df[f'iou_{a}_{b}'].mean():.3f}")
    print("\n  median mask coverage of frame:")
    for a in ARCHS:
        print(f"    {LABEL[a]:<17}{100 * df[f'{a}_coverage'].median():.2f}%")

    out_dir = HERE / "overlays"
    out_dir.mkdir(exist_ok=True)
    for old in list(out_dir.glob("sheet_*.png")) + list(out_dir.glob("sheet_*.jpg")):
        old.unlink()
    # Worst agreement first: the first sheet is the one worth looking at.
    order = [ROOT / r["eye"] / r["image"] for _, r in df.iterrows()]
    for n, start in enumerate(range(0, len(order), ROWS_PER_SHEET), 1):
        chunk = [panels[p.as_posix()] for p in order[start:start + ROWS_PER_SHEET]]
        sheet = np.vstack([np.vstack([c, np.full((8, c.shape[1], 3), 245, np.uint8)])
                           for c in chunk])
        # JPEG: these sheets are photographs, and PNG costs ~8x the space for no gain.
        cv2.imwrite(str(out_dir / f"sheet_{n:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nWrote {len(list(out_dir.glob('sheet_*.jpg')))} sheets to {out_dir}")
    print(f"Wrote {HERE / 'agreement.csv'}")


if __name__ == "__main__":
    main()
