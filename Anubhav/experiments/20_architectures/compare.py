"""Compare encoder-decoder architectures for conjunctiva segmentation.

`scripts/select_architecture.py` trains each architecture under identical
conditions — same seed, same subject-grouped split, same augmentation, same
ImageNet EfficientNet-B0 encoder, same 40 epochs — so what differs is the
decoder. This script scores every bundle it produced, using only segmentation
metrics: U-Net, UNet++ and Attention U-Net first, then LinkNet, FPN and PSPNet,
added after a published comparison reported LinkNet ahead of the other four.

With six architectures there are fifteen possible pairs. Testing all of them at
0.05, or testing everything against whichever wins on this test set, would find
"significant" differences by chance. So two families of comparisons are fixed in
advance — the published claim, and whether anything beats the current choice —
and each is Holm-corrected. All fifteen pairs are still written to
`results.json`, unadjusted, for the record.

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

ARCHS = ("unet", "unetplusplus", "attention", "linknet", "fpn", "pspnet")
LABEL = {"unet": "U-Net", "unetplusplus": "UNet++", "attention": "Attention U-Net",
         "linknet": "LinkNet", "fpn": "FPN", "pspnet": "PSPNet"}

# Comparisons fixed before seeing results, so the test set is not used both to
# pick a winner and to test it. Each family gets a Holm correction.
FAMILIES = {
    "The paper's claim: LinkNet against the four it reportedly beat":
        [("linknet", "unet"), ("linknet", "unetplusplus"), ("linknet", "fpn"), ("linknet", "pspnet")],
    "Does anything beat the current choice, Attention U-Net?":
        [(a, "attention") for a in ("unet", "unetplusplus", "linknet", "fpn", "pspnet")],
}
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
        # Pixel accuracy, and what predicting "background everywhere" would
        # score on the same image. Conjunctiva is a few percent of the frame,
        # so the second is already ~0.97: accuracy is reported because it is
        # commonly asked for, not because it separates models.
        "accuracy": float((predicted == truth).mean()),
        "trivial_accuracy": float((~truth).mean()),
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


def paired(summaries: dict, a: str, b: str) -> dict:
    """Per-image Dice of a minus b: mean, paired bootstrap 95% CI, Wilcoxon p."""

    x = np.array(summaries[a]["per_image_dice"])
    y = np.array(summaries[b]["per_image_dice"])
    diff = x - y
    p = float(stats.wilcoxon(x, y).pvalue) if np.any(diff) else 1.0
    rng = np.random.default_rng(0)
    boot = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
    return {"a": a, "b": b, "mean_diff": float(diff.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": p, "a_wins": int((diff > 0).sum()), "b_wins": int((diff < 0).sum())}


def holm(rows: list) -> list:
    """Holm step-down adjustment within one family of comparisons.

    Testing several comparisons at 0.05 each makes at least one false positive
    likely. Holm controls that family-wide while costing less power than
    Bonferroni: sort by p, multiply the k-th smallest by (m - k + 1), and keep
    the adjusted values non-decreasing.
    """

    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows


def paired_tests(summaries: dict) -> list:
    """Every pair, unadjusted — stored for the record, not used to declare winners."""

    present = [a for a in ARCHS if a in summaries]
    return [paired(summaries, a, b) for i, a in enumerate(present) for b in present[i + 1:]]


def main() -> None:
    trainer = _trainer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    bundles = {a: HERE / "bundles" / f"{a}.pt" for a in ARCHS}
    missing = [a for a, p in bundles.items() if not p.exists()]
    if missing:
        print(f"Missing bundles, skipped: {missing}. Train them with "
              f"scripts/select_architecture.py --archs {' '.join(missing)}", file=sys.stderr)
    bundles = {a: p for a, p in bundles.items() if p.exists()}
    if "unet" not in bundles:
        raise SystemExit("The U-Net bundle is required: it defines the shared test split.")

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
    for arch in bundles:
        b = torch.load(bundles[arch], map_location="cpu", weights_only=False)
        assert set(b["metrics"]["test_subjects"]) == test_subjects, \
            f"{arch} was tested on different subjects — the comparison would be invalid"

    summaries = {}
    for arch in bundles:
        rows = evaluate_bundle(bundles[arch], test_pairs, size, threshold, device)
        summaries[arch] = summarise(rows)
        summaries[arch]["params_m"] = sum(
            v.numel() for v in torch.load(bundles[arch], map_location="cpu",
                                          weights_only=False)["state_dict"].values()) / 1e6
        print(f"  scored {LABEL[arch]}")

    ranked = sorted(summaries, key=lambda a: summaries[a]["dice"], reverse=True)
    print("\n1. OVERLAP — held-out subjects, used for neither training nor selection\n")
    print(f"   {'architecture':<20}{'params':>9}{'Dice':>9}{'IoU':>9}"
          f"{'precision':>11}{'recall':>9}{'HD95 px':>10}{'empty':>7}")
    for arch in ranked:
        s = summaries[arch]
        print(f"   {LABEL[arch]:<20}{s['params_m']:>8.2f}M{s['dice']:>9.4f}{s['iou']:>9.4f}"
              f"{s['precision']:>11.4f}{s['recall']:>9.4f}{s['hd95']:>10.1f}{s['empty']:>7d}")

    print("\n2. DISTRIBUTION — the mean hides the failures\n")
    print(f"   {'architecture':<20}{'mean':>9}{'sd':>8}{'median':>9}{'10th pct':>11}{'worst':>9}")
    for arch in ranked:
        s = summaries[arch]
        print(f"   {LABEL[arch]:<20}{s['dice']:>9.4f}{s['dice_sd']:>8.4f}"
              f"{s['dice_median']:>9.4f}{s['dice_p10']:>11.4f}{s['dice_worst']:>9.4f}")

    print("\n3. PAIRED — per image, same images, Wilcoxon signed-rank, Holm-adjusted per family\n")
    families = {}
    for title, pairs_ in FAMILIES.items():
        rows = holm([paired(summaries, a, b) for a, b in pairs_
                     if a in summaries and b in summaries])
        families[title] = rows
        if not rows:
            continue
        print(f"   {title}")
        print(f"   {'pair':<34}{'mean diff':>11}{'95% CI':>20}{'p':>8}{'p Holm':>9}{'wins':>10}")
        for t in rows:
            ci = f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"
            name = f"{LABEL[t['a']]} \u2212 {LABEL[t['b']]}"
            verdict = "  resolved" if t["p_holm"] < 0.05 else ""
            print(f"   {name:<34}{t['mean_diff']:>+11.4f}{ci:>20}{t['p']:>8.4f}{t['p_holm']:>9.4f}"
                  f"{str(t['a_wins']) + '/' + str(t['b_wins']):>10}{verdict}")
        print()
    print("   'wins' = images where the first scores higher / the second does.")
    print("   Positive mean diff = the first is better. 'resolved' = Holm-adjusted p < 0.05.")
    tests = paired_tests(summaries)

    out = {a: {k: v for k, v in s.items() if k != "per_image_dice"} for a, s in summaries.items()}
    (HERE / "results.json").write_text(json.dumps(
        {"test_images": len(test_pairs), "test_subjects": len(test_subjects),
         "overlap": out, "families": families, "paired_all_unadjusted": tests},
        indent=2, default=float))
    np.save(HERE / "per_image_dice.npy",
            np.array([summaries[a]["per_image_dice"] for a in summaries]))
    print(f"\nWrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
