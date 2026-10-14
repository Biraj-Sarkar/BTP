"""Compare three encoder-decoder architectures for conjunctiva segmentation.

`run.sh` trains U-Net, UNet++ and Attention U-Net under identical conditions —
same seed, same subject-grouped split, same augmentation, same ImageNet
EfficientNet-B0 encoder, same 40 epochs — so what differs is the decoder. This
script scores the three bundles it produced, using only segmentation metrics.

How an encoder-decoder segmentation model is evaluated
======================================================
The model emits one probability per pixel. Threshold it at 0.5 and you have a
predicted mask; the annotator's mask is the reference. Everything below is a way
of describing how those two sets of pixels differ.

**Region overlap — did it find the right pixels?**

    Dice = 2|A∩B| / (|A|+|B|)          IoU = |A∩B| / |A∪B|

Both measure overlap; IoU is the stricter one. They rank models identically, so
reporting both is a consistency check rather than two separate findings. Dice is
the convention in medical segmentation, which is why it is the headline here.

**Precision and recall — which way did it go wrong?** Dice is a single number
that hides the direction of the error, and the direction matters for this task:

    precision = |A∩B| / |A|    of the pixels the model called conjunctiva,
                               how many really were — low precision means the
                               mask has spilled onto lashes, lid or sclera, and
                               colour read inside it is contaminated
    recall    = |A∩B| / |B|    of the true conjunctiva pixels, how many were
                               found — low recall means the mask is too small
                               and colour is measured on a biased sample

For this project precision is the more expensive failure. The whole reason for
replacing the rule-based extractor was that its masks bled onto surrounding
tissue, which Dice alone would have under-stated.

**Boundary accuracy — is the outline in the right place?** Overlap is dominated
by the interior of a large region, so a mask can score well while its edge
wanders. HD95 — the 95th percentile of the distance from each predicted boundary
pixel to the nearest true boundary pixel — measures that directly, in pixels,
with the 95th percentile rather than the maximum so one stray pixel does not
dominate. Lower is better. It matters here because the conjunctiva is a thin
crescent whose boundary is most of it.

**The distribution, not just the mean.** An architecture that ties on average
while failing outright on a few captures is worse in deployment, because a
failure is an empty mask and a refused reading. So the 10th percentile and the
worst case are reported beside the mean, along with the count of empty masks.

Three things make the comparison trustworthy, and all three are enforced:

1. **Split by subject, never by image.** Both eyes of one person are nearly the
   same photograph; splitting by image measures memorisation.
2. **Score on subjects used for neither training nor model selection.** The epoch
   is chosen on validation Dice, so validation Dice is optimistic. A third split
   of 20% of subjects, untouched by both, is what gets reported.
3. **Compare per image, paired.** All three are scored on the same test images,
   so differences are tested per image (Wilcoxon signed-rank, plus a paired
   bootstrap interval) rather than by comparing two means from unrelated samples.

Run: python experiments/20_architectures/compare.py

Abbreviations
-------------
Dice   overlap score, 2|A∩B| / (|A|+|B|); 1.0 is a perfect mask
IoU    intersection over union, |A∩B| / |A∪B|
HD95   95th-percentile Hausdorff distance between boundaries, in pixels
IQR    interquartile range
px     pixels
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anemia.architectures import build  # noqa: E402

ARCHS = ("unet", "unetplusplus", "attention")
LABEL = {"unet": "U-Net", "unetplusplus": "UNet++", "attention": "Attention U-Net"}
DATA_ROOTS = [Path.home() / "btp-local/datasets/eyes_defy_prepared",
              Path.home() / "btp-local/datasets/conjunctiva_seg_prepared"]


def _trainer():
    """Import the training script for its dataset helpers, without running it."""

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_segmenter", ROOT / "scripts/train_segmenter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_segmenter"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- metrics

def hd95(predicted: np.ndarray, truth: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance between the two boundaries.

    Returns NaN when either mask is empty: a distance to an absent boundary is
    undefined, and scoring it as 0 would flatter a model that predicted nothing.
    """

    if not predicted.any() or not truth.any():
        return float("nan")
    pe = predicted ^ cv2.erode(predicted.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    te = truth ^ cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not pe.any() or not te.any():
        return float("nan")
    # Distance from every pixel to the nearest boundary pixel of the other mask.
    dist_to_truth = distance_transform_edt(~te)
    dist_to_pred = distance_transform_edt(~pe)
    both = np.concatenate([dist_to_truth[pe], dist_to_pred[te]])
    return float(np.percentile(both, 95))


def score_pair(predicted: np.ndarray, truth: np.ndarray) -> dict:
    inter = float(np.logical_and(predicted, truth).sum())
    p_sum, t_sum = float(predicted.sum()), float(truth.sum())
    union = p_sum + t_sum - inter
    return {
        "dice": (2 * inter + 1e-6) / (p_sum + t_sum + 1e-6),
        "iou": (inter + 1e-6) / (union + 1e-6),
        "precision": (inter + 1e-6) / (p_sum + 1e-6),
        "recall": (inter + 1e-6) / (t_sum + 1e-6),
        "hd95": hd95(predicted, truth),
        "empty": p_sum == 0,
    }


@torch.no_grad()
def evaluate_bundle(path: Path, pairs, size: int, threshold: float, device) -> list:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    model = build(config.get("arch", "unet"), config["encoder"], pretrained=False)
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()

    mean = np.array(config["mean"], dtype=np.float32)
    std = np.array(config["std"], dtype=np.float32)
    rows = []
    for image_path, mask_path in pairs:
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        truth = cv2.resize(truth, (size, size), interpolation=cv2.INTER_NEAREST) > 127

        x = ((image.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
        predicted = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold)

        row = score_pair(predicted, truth)
        row["image"] = image_path.name
        rows.append(row)
    return rows


# ---------------------------------------------------------------- reporting

def summarise(rows: list) -> dict:
    d = np.array([r["dice"] for r in rows])
    hd = np.array([r["hd95"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "dice": float(d.mean()), "dice_sd": float(d.std()),
        "dice_median": float(np.median(d)),
        "dice_p10": float(np.percentile(d, 10)), "dice_worst": float(d.min()),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "hd95": float(np.nanmedian(hd)),
        "empty": int(sum(r["empty"] for r in rows)),
        "per_image_dice": d.tolist(),
    }


def paired_tests(summaries: dict) -> list:
    rows = []
    for i, a in enumerate(ARCHS):
        for b in ARCHS[i + 1:]:
            if a not in summaries or b not in summaries:
                continue
            x = np.array(summaries[a]["per_image_dice"])
            y = np.array(summaries[b]["per_image_dice"])
            diff = x - y
            p = float(stats.wilcoxon(x, y).pvalue) if np.any(diff) else 1.0
            rng = np.random.default_rng(0)
            boot = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
            rows.append({"a": a, "b": b, "mean_diff": float(diff.mean()),
                         "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         "p": p, "a_wins": int((diff > 0).sum()), "b_wins": int((diff < 0).sum())})
    return rows


def main() -> None:
    trainer = _trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    bundles = {a: HERE / "bundles" / f"{a}.pt" for a in ARCHS}
    missing = [a for a, p in bundles.items() if not p.exists()]
    if missing:
        print(f"Missing bundles: {missing}. Run experiments/20_architectures/run.sh first.",
              file=sys.stderr)
        raise SystemExit(1)

    first = torch.load(bundles["unet"], map_location="cpu", weights_only=False)
    test_subjects = set(first["metrics"]["test_subjects"])
    size = first["config"]["input_size"]
    threshold = first["config"]["threshold"]

    pairs = []
    for root in DATA_ROOTS:
        pairs.extend(trainer.find_pairs(root))
    test_pairs = [p for p in pairs if trainer.subject_of(p[0]) in test_subjects]
    print(f"{len(test_pairs)} held-out images from {len(test_subjects)} subjects, "
          f"at {size}px, threshold {threshold}\n")

    # Every architecture must be scored on exactly the same images.
    for arch in ARCHS:
        b = torch.load(bundles[arch], map_location="cpu", weights_only=False)
        assert set(b["metrics"]["test_subjects"]) == test_subjects, \
            f"{arch} was tested on different subjects — the comparison would be invalid"

    summaries = {}
    for arch in ARCHS:
        rows = evaluate_bundle(bundles[arch], test_pairs, size, threshold, device)
        summaries[arch] = summarise(rows)
        summaries[arch]["params_m"] = sum(
            v.numel() for v in torch.load(bundles[arch], map_location="cpu",
                                          weights_only=False)["state_dict"].values()) / 1e6
        print(f"  scored {LABEL[arch]}")

    print("\n1. OVERLAP — held-out subjects, used for neither training nor selection\n")
    print(f"   {'architecture':<20}{'params':>9}{'Dice':>9}{'IoU':>9}"
          f"{'precision':>11}{'recall':>9}{'HD95 px':>10}{'empty':>7}")
    for arch in ARCHS:
        s = summaries[arch]
        print(f"   {LABEL[arch]:<20}{s['params_m']:>8.2f}M{s['dice']:>9.4f}{s['iou']:>9.4f}"
              f"{s['precision']:>11.4f}{s['recall']:>9.4f}{s['hd95']:>10.1f}{s['empty']:>7d}")

    print("\n2. DISTRIBUTION — the mean hides the failures\n")
    print(f"   {'architecture':<20}{'mean':>9}{'sd':>8}{'median':>9}{'10th pct':>11}{'worst':>9}")
    for arch in ARCHS:
        s = summaries[arch]
        print(f"   {LABEL[arch]:<20}{s['dice']:>9.4f}{s['dice_sd']:>8.4f}"
              f"{s['dice_median']:>9.4f}{s['dice_p10']:>11.4f}{s['dice_worst']:>9.4f}")

    print("\n3. PAIRED — per image, same images, Wilcoxon signed-rank\n")
    tests = paired_tests(summaries)
    print(f"   {'pair':<36}{'mean diff':>11}{'95% CI':>20}{'p':>9}{'wins':>10}")
    for t in tests:
        ci = f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
        name = f"{LABEL[t['a']]} − {LABEL[t['b']]}"
        print(f"   {name:<36}{t['mean_diff']:>+11.4f}{ci:>20}{t['p']:>9.4f}"
              f"{str(t['a_wins']) + '/' + str(t['b_wins']):>10}")
    print("\n   'wins' = images where the first scores higher / the second does.")
    print("   A difference matters only if the interval excludes zero.")

    out = {a: {k: v for k, v in s.items() if k != "per_image_dice"} for a, s in summaries.items()}
    (HERE / "results.json").write_text(json.dumps(
        {"test_images": len(test_pairs), "test_subjects": len(test_subjects),
         "overlap": out, "paired": tests}, indent=2, default=float))
    np.save(HERE / "per_image_dice.npy",
            np.array([summaries[a]["per_image_dice"] for a in ARCHS]))
    print(f"\nWrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
