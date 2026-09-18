"""The extractor comparison, restricted to photographs no segmenter was trained on.

`compare_extractors_full.py` reported the learned extractor at R² 0.336 against
the classical chain's 0.116 on the 217-subject external cohort. That cohort *is*
Eyes-Defy-Anemia, and every learned bundle was trained on 101 Eyes-Defy images
with their masks. The segmenter never saw a haemoglobin value, so this is not
label leakage — but on those 101 photographs it had already learned exactly
where the conjunctiva is, which flatters its masks, and therefore its colour
features, relative to an extractor that was never trained at all.

The clean test uses only the external subjects whose photographs were never in
any segmenter's training data: the ones `prepare_eyes_defy.py` rejected (their
annotations were rectangles or missing). Recorded in
`datasets_manifests/eyes_defy_prepared.json`.

It also reports the raw correlation of each chromaticity channel with laboratory
haemoglobin, because the direction is itself a finding: physiology predicts
redder conjunctiva with higher haemoglobin, and the local cohort under the
learned extractor shows the reverse.

Run: python experiments/13_learned_extractor/unseen_external.py

Abbreviations
-------------
LOO    leave-one-out cross-validation, here leaving out one patient
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "11_head_to_head"))
sys.path.insert(0, str(ROOT / "experiments" / "14_pooled_regression"))

from anemia.imaging import gray_world_white_balance, read_rgb, resize  # noqa: E402
from anemia.linear_model import extract_features  # noqa: E402
from anemia.segment import get_extractor  # noqa: E402

from head_to_head import leave_one_out, score  # noqa: E402
from pooled import EXTERNAL, european_float  # noqa: E402

PERMUTATIONS = 2000
EXTRACTORS = ("unet", "refined")


def r2_of(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def seen_by_segmenter() -> set[str]:
    manifest = json.loads((ROOT / "datasets_manifests/eyes_defy_prepared.json").read_text())
    # stems look like "India-11_0" -> subject key "India:11"
    return {entry["stem"].rsplit("_", 1)[0].replace("-", ":", 1) for entry in manifest["kept"]}


def main() -> None:
    seen = seen_by_segmenter()
    models = {name: get_extractor(name) for name in EXTRACTORS}
    rows = {name: {"seen": [], "unseen": []} for name in EXTRACTORS}

    for country in ("India", "Italy"):
        df = pd.read_excel(EXTERNAL / country / f"{country}.xlsx", engine="openpyxl")
        labels = {int(r["Number"]): r for _, r in df.iterrows()
                  if pd.notna(r.get("Number")) and pd.notna(r.get("Hgb"))}
        for folder in sorted((EXTERNAL / country).iterdir()):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            row = labels.get(int(folder.name))
            photos = sorted(folder.glob("*.jpg"))
            if row is None or not photos:
                continue
            hb = european_float(row["Hgb"])
            if hb is None or not 3.0 <= hb <= 22.0:
                continue
            key = f"{country}:{folder.name}"
            split = "seen" if key in seen else "unseen"
            sex = str(row.get("Gender", "")).strip().upper()[:1] or None
            age = european_float(row.get("Age"))
            rgb = gray_world_white_balance(resize(read_rgb(photos[0]), (512, 512)))
            for name, seg in models.items():
                mask, _ = seg(rgb)
                if np.count_nonzero(mask) < 500:
                    continue
                rows[name][split].append((key, extract_features(rgb, mask, "chroma"), hb, age, sex))

    results = {}
    for split in ("unseen", "seen"):
        keys = set.intersection(*[{r[0] for r in rows[n][split]} for n in EXTRACTORS])
        print(f"\n=== external subjects {split.upper()} by the segmenter — "
              f"{len(keys)} usable by both extractors ===")
        print(f"{'extractor':<10}{'MAE':>8}{'R²':>8}{'p':>8}   corr with Hb:  r      g      b")
        for name in EXTRACTORS:
            data = sorted([r for r in rows[name][split] if r[0] in keys])
            X = np.array([r[1] for r in data])
            y = np.array([r[2] for r in data])
            meta = [(r[3], r[4]) for r in data]
            m = score(leave_one_out(X, y), y, meta)
            rng = np.random.default_rng(0)
            p = float(np.mean([r2_of(leave_one_out(X, s), s) >= m["r2"]
                               for s in (rng.permutation(y) for _ in range(PERMUTATIONS))]))
            rch, gch = X[:, 0], X[:, 1]
            c = np.corrcoef(np.c_[rch, gch, 1 - rch - gch, y], rowvar=False)[3, :3]
            results.setdefault(split, {})[name] = {**m, "p": p, "n": len(y),
                                                   "corr_r": c[0], "corr_g": c[1], "corr_b": c[2]}
            print(f"{name:<10}{m['mae']:>8.3f}{m['r2']:>8.3f}{p:>8.3f}"
                  f"               {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}")
        base_y = np.array([r[2] for r in rows[EXTRACTORS[0]][split] if r[0] in keys])
        print(f"{'baseline':<10}{np.abs(base_y - base_y.mean()).mean():>8.3f}")

    out = Path(__file__).resolve().parent / "unseen_external.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
