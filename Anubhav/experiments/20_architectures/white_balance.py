"""Is gray-world white balance helping the extractor, or hurting it?

The question has two halves, and they had been conflated.

**Half one: a train/serve mismatch.** Segmenter training reads raw image/mask
pairs, but `preprocess.prepare()` white-balanced the frame before handing it to
the segmenter. So every bundle in this project was trained on raw pixels and
served balanced ones. A segmentation network is as sensitive to a global colour
shift as to anything else in its input, so that is not a cosmetic difference:
measured on the 52-capture cohort, the raw mask and the balanced mask of the
*same model on the same photograph* agree at only IoU 0.57–0.67, and coverage
falls by about a third under balancing.

**Half two: which condition is actually better.** Answering that needs a model
trained under each condition, so `attention_gw.pt` is Attention U-Net trained
with white balance applied, identical to `attention.pt` in every other respect —
same seed, split, augmentation, encoder and epochs.

The 2×2 below is the whole answer. Rows are what the model trained on, columns
what it is served:

                     served raw      served gray-world
    trained raw      matched         ← what the pipeline actually did
    trained gw       mismatch        matched

Comparing the two matched cells says whether white balance helps. Comparing a
matched cell with the mismatch beside it says what the bug cost.

Scored against annotator masks on the same 129 held-out subjects used throughout
experiment 20, so these numbers sit directly beside the architecture comparison.

Run: python experiments/20_architectures/white_balance.py

Abbreviations
-------------
gw     gray-world white balance
Dice   overlap score, 2|A∩B| / (|A|+|B|)
IoU    intersection over union
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from anemia.architectures import build  # noqa: E402
from anemia.imaging import gray_world_white_balance  # noqa: E402

DATA_ROOTS = [Path.home() / "btp-local/datasets/eyes_defy_prepared",
              Path.home() / "btp-local/datasets/conjunctiva_seg_prepared"]
BUNDLES = {"trained raw": HERE / "bundles" / "attention.pt",
           "trained gw": HERE / "bundles" / "attention_gw.pt"}


def _trainer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ts", ROOT / "scripts/train_segmenter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ts"] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def score(path: Path, pairs, serve_gw: bool, device) -> dict:
    """Dice, IoU, precision, recall per image, for one bundle under one serving condition."""

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    model = build(config.get("arch", "unet"), config["encoder"], pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()

    size, threshold = config["input_size"], config["threshold"]
    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)

    dice, iou, prec, rec = [], [], [], []
    for image_path, mask_path in pairs:
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if serve_gw:
            rgb = gray_world_white_balance(rgb)
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        truth = cv2.resize(truth, (size, size), interpolation=cv2.INTER_NEAREST) > 127

        x = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        predicted = (torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0).to(device)))
                     [0, 0].cpu().numpy() >= threshold)

        inter = float(np.logical_and(predicted, truth).sum())
        p_sum, t_sum = float(predicted.sum()), float(truth.sum())
        dice.append((2 * inter + 1e-6) / (p_sum + t_sum + 1e-6))
        iou.append((inter + 1e-6) / (p_sum + t_sum - inter + 1e-6))
        prec.append((inter + 1e-6) / (p_sum + 1e-6))
        rec.append((inter + 1e-6) / (t_sum + 1e-6))
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec}


def main() -> None:
    missing = [k for k, v in BUNDLES.items() if not v.exists()]
    if missing:
        print(f"Missing bundles for: {missing}\n"
              f"Train the white-balanced variant with:\n"
              f"  python scripts/train_segmenter.py --arch attention --gray-world "
              f"--out experiments/20_architectures/bundles/attention_gw.pt "
              f"--epochs 40 --seed 0 --augment rotation --val-fraction 0.2 --test-fraction 0.2",
              file=sys.stderr)
        raise SystemExit(1)

    trainer = _trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    first = torch.load(BUNDLES["trained raw"], map_location="cpu", weights_only=False)
    test_subjects = set(first["metrics"]["test_subjects"])
    other = torch.load(BUNDLES["trained gw"], map_location="cpu", weights_only=False)
    assert set(other["metrics"]["test_subjects"]) == test_subjects, \
        "the two bundles were tested on different subjects — not comparable"

    pairs = []
    for root in DATA_ROOTS:
        pairs.extend(trainer.find_pairs(root))
    test_pairs = [p for p in pairs if trainer.subject_of(p[0]) in test_subjects]
    print(f"{len(test_pairs)} held-out images from {len(test_subjects)} subjects\n")

    cells = {}
    for trained, path in BUNDLES.items():
        for served, gw in (("served raw", False), ("served gw", True)):
            cells[(trained, served)] = score(path, test_pairs, gw, device)
            print(f"  scored {trained:<12} / {served}")

    print("\n2x2 — mean Dice (bold cells are the matched conditions)\n")
    print(f"   {'':<14}{'served raw':>14}{'served gray-world':>20}")
    for trained in BUNDLES:
        row = ""
        for served in ("served raw", "served gw"):
            v = np.mean(cells[(trained, served)]["dice"])
            matched = (trained == "trained raw") == (served == "served raw")
            row += f"{('*' if matched else ' ') + f'{v:.4f}':>{14 if served == 'served raw' else 20}}"
        print(f"   {trained:<14}{row}")
    print("\n   * = train and serve conditions agree")

    print("\nFull metrics for each cell\n")
    print(f"   {'configuration':<30}{'Dice':>9}{'IoU':>9}{'precision':>11}{'recall':>9}")
    for (trained, served), v in cells.items():
        tag = f"{trained} / {served}"
        print(f"   {tag:<30}{np.mean(v['dice']):>9.4f}{np.mean(v['iou']):>9.4f}"
              f"{np.mean(v['precision']):>11.4f}{np.mean(v['recall']):>9.4f}")

    print("\nThe two comparisons that matter, paired per image\n")
    questions = [
        ("Does white balance help? (matched vs matched)",
         ("trained raw", "served raw"), ("trained gw", "served gw")),
        ("What did the mismatch cost? (matched vs what the pipeline did)",
         ("trained raw", "served raw"), ("trained raw", "served gw")),
    ]
    out = {}
    for name, a, b in questions:
        x = np.array(cells[a]["dice"])
        y = np.array(cells[b]["dice"])
        diff = x - y
        p = float(stats.wilcoxon(x, y).pvalue) if np.any(diff) else 1.0
        rng = np.random.default_rng(0)
        boot = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        print(f"   {name}")
        print(f"     {a[0]}/{a[1]} minus {b[0]}/{b[1]}")
        print(f"     mean difference {diff.mean():+.4f}   95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]   "
              f"p {p:.4g}\n")
        out[name] = {"a": list(a), "b": list(b), "mean_diff": float(diff.mean()),
                     "ci": list(ci), "p": p}

    (HERE / "white_balance.json").write_text(json.dumps(
        {"cells": {f"{t} / {s}": {k: float(np.mean(v)) for k, v in d.items()}
                   for (t, s), d in cells.items()},
         "comparisons": out}, indent=2))
    print(f"Wrote {HERE / 'white_balance.json'}")


if __name__ == "__main__":
    main()
