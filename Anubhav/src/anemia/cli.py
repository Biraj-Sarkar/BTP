"""Command line entry points.

    python -m anemia inspect  --data <root>
    python -m anemia train-hb --data <root> [--data <root2>] --out runs/exp1
    python -m anemia predict  --checkpoint runs/exp1/best.pt --image photo.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import List

from .config import Config, REPO_ROOT
from .data import Sample, load_dataset, summarise


def _collect(data_roots: List[Path]) -> List[Sample]:
    samples: List[Sample] = []
    for root in data_roots:
        loaded = load_dataset(root)
        print(f"  {root}: {len(loaded)} images")
        samples.extend(loaded)
    if not samples:
        raise SystemExit("No samples loaded — check --data paths.")
    return samples


def cmd_inspect(args: argparse.Namespace) -> None:
    """Describe the cohort before training anything.

    Worth running the moment a dataset lands: it is the cheapest way to catch a
    loader that silently matched 40 of 710 images.
    """

    samples = _collect(args.data)
    stats = summarise(samples)
    print("\nCohort summary")
    print(json.dumps(stats, indent=2))

    if stats["images"] < 100:
        print("\nWARNING: fewer than 100 images parsed. Verify the loader matched the layout.")
    if stats.get("with_masks", 0) == 0:
        print("\nNote: no ground-truth masks found — segmenter training needs Eyes-defy-anemia.")


def cmd_train_hb(args: argparse.Namespace) -> None:
    import torch

    from .segment import heuristic_segmenter, load_segmenter
    from .train import apply_quality_gate, build_crops, cross_validate, pick_device, set_seed

    config = Config(
        data_roots=tuple(args.data),
        output_root=args.out,
        cache_root=args.cache,
    )
    if args.epochs:
        config = replace(config, regression=replace(config.regression, epochs=args.epochs))
    if args.folds:
        config = replace(config, split=replace(config.split, folds=args.folds))

    set_seed(config.seed)
    device = pick_device()
    print(f"Device: {device}")

    samples = _collect(list(args.data))
    print(json.dumps(summarise(samples), indent=2))

    segmenter = load_segmenter(args.segmenter) if args.segmenter else heuristic_segmenter
    print(f"\nPreprocessing {len(samples)} images (cached at {config.cache_root})")
    prepared = build_crops(samples, segmenter, config)

    samples = apply_quality_gate(samples, prepared, config.quality.enforce)
    print(f"  {len(samples)} images passed QC")
    crops = {path: prep.crop for path, prep in prepared.items()}

    print(f"\nCross-validating ({config.split.folds} folds, patient-grouped)")
    results = cross_validate(samples, crops, config, device)

    args.out.mkdir(parents=True, exist_ok=True)
    config.save(args.out / "config.json")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {"regression": results["regression"], "screening": results["screening"],
             "per_fold": results["per_fold"]},
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = results["out_of_fold_predictions"]
    if rows:
        with (args.out / "out_of_fold_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    best = results["best_checkpoint"]
    if best and best.checkpoint:
        best.checkpoint.save(args.out / "best.pt")
        print(f"\nSaved checkpoint to {args.out / 'best.pt'}")

    reg = results["regression"]
    print("\nCross-validated results")
    print(f"  MAE  {reg.get('mae_mean', float('nan')):.3f} +/- {reg.get('mae_std', 0):.3f} g/dL")
    print(f"  RMSE {reg.get('rmse_mean', float('nan')):.3f} g/dL")
    print(f"  R^2  {reg.get('r2_mean', float('nan')):.3f}")
    gain = reg.get("mae_vs_baseline_mean")
    if gain is not None:
        verdict = "better than" if gain > 0 else "NO BETTER THAN"
        print(f"  {abs(gain):.3f} g/dL {verdict} predicting the training mean")
    scr = results["screening"]
    print(f"  sensitivity {scr.get('sensitivity_mean', float('nan')):.3f}, "
          f"specificity {scr.get('specificity_mean', float('nan')):.3f}")


def cmd_train_segmenter(args: argparse.Namespace) -> None:
    from .train import pick_device, set_seed
    from .train_segmenter import train_segmenter

    config = Config(data_roots=tuple(args.data), output_root=args.out)
    if args.epochs:
        config = replace(config, segmentation=replace(config.segmentation, epochs=args.epochs))

    set_seed(config.seed)
    samples = _collect(list(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    train_segmenter(samples, config, args.out, pick_device())


def cmd_debug(args: argparse.Namespace) -> None:
    """Dump what the pipeline sees for one photo. No trained model required.

    This is the tool for checking a capture on a real eye: it shows whether the
    segmenter found the conjunctiva or wandered onto skin, and why QC passed or
    failed. Deliberately reports no haemoglobin value — segmentation quality is
    judged by eye, and a number from an untrained or unrelated model would only
    distract from that.
    """

    import numpy as np

    from .config import PreprocessConfig, QualityConfig
    from .imaging import build_overlay, read_rgb, resize, write_gray, write_rgb
    from .preprocess import prepare
    from .segment import heuristic_segmenter, load_segmenter

    segmenter = load_segmenter(args.segmenter) if args.segmenter else heuristic_segmenter

    if args.dir:
        _debug_folder(args, segmenter)
        return

    raw = read_rgb(args.image)
    prepared = prepare(raw, segmenter, PreprocessConfig(), QualityConfig())

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_rgb(prepared.balanced, out / "1_balanced.png")
    write_gray(prepared.mask, out / "2_mask.png")
    write_rgb(build_overlay(prepared.balanced, prepared.mask), out / "3_overlay.png")
    write_rgb(prepared.crop, out / "4_crop.png")

    # Side-by-side sheet: original | detected region | what the model would see.
    panels = [
        resize(prepared.balanced, (320, 320)),
        resize(build_overlay(prepared.balanced, prepared.mask), (320, 320)),
        resize(prepared.crop, (320, 320)),
    ]
    write_rgb(np.concatenate(panels, axis=1), out / "0_contact_sheet.png")

    report = prepared.quality
    print(f"\nInput      {args.image}  {raw.shape[1]}x{raw.shape[0]}")
    print(f"Backend    {prepared.backend}")
    print(f"Mask       {100 * report.mask_ratio:.2f}% of frame")
    print("\nQuality")
    print(f"  focus    {report.focus:8.1f}   (need >= 35)")
    print(f"  clipped  {report.clipped:8.4f}   (blown-out/black pixel fraction)")
    print(f"  exposure {report.exposure:8.1f}")
    print(f"  passed   {report.passed}")
    for reason in report.reasons:
        print(f"    - {reason}")

    print(f"\nWrote {out}/")
    print("  0_contact_sheet.png   original | detected region | model input")
    print("  3_overlay.png         green tint = what was segmented  <- check this one")
    print("\nIf the green tint is not on the red inner eyelid, segmentation failed")
    print("on this capture — that is the thing worth reporting, not any number.")


def _debug_folder(args: argparse.Namespace, segmenter) -> None:
    """Screen a whole folder: per-image QC stats, a CSV, and an overlay sheet.

    Built for unlabelled real photographs. Without ground-truth masks there is
    no Dice to compute, so the overlay sheet is the instrument — segmentation
    quality on real tissue is judged by eye until masks are available.
    """

    import numpy as np

    from .config import PreprocessConfig, QualityConfig
    from .data import load_unlabelled
    from .imaging import build_overlay, read_rgb, write_rgb
    from .preprocess import prepare

    samples = load_unlabelled(args.dir)
    if not samples:
        raise SystemExit(f"No images found under {args.dir}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Screening {len(samples)} images from {args.dir}\n")

    rows, thumbs = [], []
    for sample in samples:
        prepared = prepare(
            read_rgb(sample.image_path), segmenter, PreprocessConfig(), QualityConfig()
        )
        report = prepared.quality
        rows.append({
            "folder": sample.image_path.parent.name,
            "filename": sample.image_path.name,
            "focus": round(report.focus, 2),
            "exposure": round(report.exposure, 2),
            "clipped": round(report.clipped, 5),
            "mask_ratio": round(report.mask_ratio, 4),
            "backend": prepared.backend,
            "passed": report.passed,
            "reasons": "; ".join(report.reasons),
        })
        if len(thumbs) < args.sheet_size:
            thumbs.append(
                cv2_resize(build_overlay(prepared.balanced, prepared.mask), 260)
            )

    with (out / "quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if thumbs:
        per_row = 4
        grid = [
            np.concatenate(thumbs[i : i + per_row], axis=1)
            for i in range(0, len(thumbs) - len(thumbs) % per_row, per_row)
        ]
        if grid:
            write_rgb(np.concatenate(grid, axis=0), out / "overlay_sheet.png")

    passed = [r for r in rows if r["passed"]]
    focus = np.array([r["focus"] for r in rows])
    mask = np.array([r["mask_ratio"] for r in rows])

    print(f"QC passed      {len(passed)}/{len(rows)} ({100 * len(passed) / len(rows):.0f}%)")
    print(f"focus          min {focus.min():.1f}  median {np.median(focus):.1f}  max {focus.max():.1f}")
    print(f"mask ratio     min {mask.min():.3f}  median {np.median(mask):.3f}  max {mask.max():.3f}")

    failures = [r for r in rows if not r["passed"]]
    if failures:
        print("\nrejected:")
        for row in failures:
            print(f"  {row['folder']}/{row['filename']:<12} {row['reasons']}")

    print(f"\nWrote {out}/quality_report.csv and overlay_sheet.png")
    print("Inspect the overlay sheet: green must sit on the red inner lid, not on")
    print("lashes or cheek skin. Mask ratio alone cannot tell you this.")


def cv2_resize(image, size: int):
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _letterbox_panel(image, label: str, box: int = 300, bar: int = 30):
    """One captioned panel: image fitted into `box`x`box` without distortion.

    Aspect ratio is preserved by padding rather than stretching, because the
    stages being compared differ in shape (full frame, square working image,
    crescent crop) and stretching them would misrepresent exactly the geometry
    this pipeline takes care to protect.
    """

    import cv2
    import numpy as np

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    height, width = image.shape[:2]
    scale = min(box / width, box / height)
    resized = cv2.resize(image, (max(int(width * scale), 1), max(int(height * scale), 1)),
                         interpolation=cv2.INTER_AREA)

    canvas = np.full((box + bar, box, 3), 28, dtype=np.uint8)
    y0 = (box - resized.shape[0]) // 2
    x0 = (box - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

    cv2.putText(canvas, label, (6, box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (box - 1, box + bar - 1), (70, 70, 70), 1)
    return canvas


def _contact_sheet(panels, header: str, per_row: int = 5):
    """Grid of captioned panels with a header strip."""

    import cv2
    import numpy as np

    rows = []
    for start in range(0, len(panels), per_row):
        chunk = list(panels[start:start + per_row])
        while len(chunk) < per_row:                      # pad the last row
            chunk.append(np.full_like(chunk[0], 28))
        rows.append(np.concatenate(chunk, axis=1))
    grid = np.concatenate(rows, axis=0)

    strip = np.full((34, grid.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(strip, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([strip, grid], axis=0)


def cmd_stages(args: argparse.Namespace) -> None:
    """Trace every pipeline stage for each image onto one contact sheet.

    Built as a learning tool: one image per capture showing the whole
    progression side by side, so each stage can be compared against the one
    before it without opening separate files.

    The redness map is included even though it is an *internal* detail of the
    extractor, because it is the single most instructive panel in the set: it
    shows what the segmentation actually "sees".
    """

    import shutil

    import cv2
    import numpy as np

    from .config import PreprocessConfig, QualityConfig
    from .imaging import (
        apply_mask,
        build_overlay,
        gray_world_white_balance,
        pad_to_square,
        read_rgb,
        redness_index,
        resize,
        tight_bbox,
    )
    from .preprocess import assess_quality
    from .segment import heuristic_segmenter, load_segmenter

    segmenter = load_segmenter(args.segmenter) if args.segmenter else heuristic_segmenter
    preprocess = PreprocessConfig()
    quality_config = QualityConfig()

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

    def save_rgb(image_rgb, path: Path) -> None:
        cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Tracing {len(files)} image(s) -> {args.out}\n")

    for file in files:
        stem = file.stem
        rgb = read_rgb(file)

        # Stage 1: resize to the working resolution the pipeline operates at.
        resized = resize(rgb, preprocess.work_size)

        # Stage 2: gray-world white balance neutralises the illuminant.
        balanced = gray_world_white_balance(resized)

        # Stage 3: the redness map — what the extractor actually scores.
        redness = redness_index(balanced)
        spread = float(np.ptp(redness)) or 1.0
        red_u8 = np.clip((redness - redness.min()) / spread * 255, 0, 255).astype(np.uint8)
        redness_view = cv2.cvtColor(
            cv2.applyColorMap(red_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

        # Stage 4-5: segmentation, as raw mask and as overlay.
        mask, backend = segmenter(balanced)
        overlay = build_overlay(balanced, mask)

        # Stage 6: everything outside the mask blacked out.
        masked = apply_mask(balanced, mask)

        # Stage 7: tight crop around the mask (still the crescent shape).
        left, top, right, bottom = tight_bbox(mask, preprocess.bbox_pad_ratio)
        cropped = masked[top:bottom, left:right]
        if cropped.size == 0:
            cropped = masked

        # Stage 8: letterbox to a square WITHOUT stretching the crescent.
        letterboxed = pad_to_square(cropped)

        # Stage 9: the exact input the regression network would receive.
        model_input = resize(letterboxed, preprocess.model_size)

        report = assess_quality(balanced, mask, quality_config)

        stages = [
            (rgb, "00  original capture"),
            (resized, "01  resized to 512x512"),
            (balanced, "02  gray-world white balance"),
            (redness_view, "03  redness map (a*-0.5b*)"),
            (mask, "04  segmentation mask"),
            (overlay, "05  mask over photo"),
            (masked, "06  outside mask removed"),
            (cropped, "07  tight crop"),
            (letterboxed, "08  letterboxed to square"),
            (model_input, "09  model input 224x224"),
        ]
        header = (f"{file.name}   backend={backend}   "
                  f"mask={report.mask_ratio:.3f}   focus={report.focus:.0f}   "
                  f"QC={'pass' if report.passed else 'FAIL: ' + '; '.join(report.reasons)}")

        sheet = _contact_sheet(
            [_letterbox_panel(image, label, box=args.panel) for image, label in stages],
            header, per_row=args.per_row)
        save_rgb(sheet, args.out / f"{stem}_stages.jpg")

        if args.separate:
            folder = args.out / stem
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, folder / file.name)
            for index, (image, label) in enumerate(stages[1:], start=1):
                name = label.split("  ", 1)[1].replace(" ", "_").replace("*", "")
                if image.ndim == 2:
                    cv2.imwrite(str(folder / f"{stem}_{index:02d}_{name}.png"), image)
                else:
                    save_rgb(image, folder / f"{stem}_{index:02d}_{name}.jpg")
            (folder / f"{stem}_quality.json").write_text(
                json.dumps({"backend": backend, **report.to_dict()}, indent=2),
                encoding="utf-8")

        flag = "" if report.passed else f"  QC FAIL: {'; '.join(report.reasons)}"
        print(f"  {stem}: backend={backend} mask={report.mask_ratio:.3f} "
              f"focus={report.focus:.0f}{flag}")

    print(f"\nDone. One sheet per capture: <name>_stages.jpg")
    print("Panels read left to right, top to bottom:")
    print("  original -> resized -> white balance -> redness map -> mask ->")
    print("  overlay -> masked -> crop -> letterbox -> model input")
    print("Pass --separate to also write each stage as its own file.")


def cmd_fit_linear(args: argparse.Namespace) -> None:
    """Fit the linear model and save it as a servable file.

    Reports leave-one-patient-out performance against a predict-the-mean
    baseline before saving, so a model that has learned nothing cannot be
    written out silently.
    """

    import numpy as np

    from .config import PreprocessConfig, QualityConfig
    from .imaging import read_rgb
    from .linear_model import extract_features, fit_linear_model
    from .metrics import regression_metrics
    from .preprocess import prepare
    from .segment import get_extractor

    segmenter = get_extractor(args.extractor, args.segmenter)
    preprocess, quality = PreprocessConfig(), QualityConfig()
    print(f"Extractor: {args.extractor if not args.segmenter else args.segmenter}")

    samples = _collect(list(args.data))
    by_patient: dict = {}
    skipped = 0
    for sample in samples:
        prepared = prepare(read_rgb(sample.image_path), segmenter, preprocess, quality)
        if not (prepared.mask > 0).any() or not prepared.quality.passed:
            skipped += 1
            continue
        by_patient.setdefault(sample.patient_id, {"hb": sample.hb, "f": []})["f"].append(
            extract_features(prepared.balanced, prepared.mask, args.representation))

    if len(by_patient) < 5:
        raise SystemExit(f"Only {len(by_patient)} usable patients — too few to fit.")

    patients = sorted(by_patient)
    X = np.array([np.mean(by_patient[p]["f"], axis=0) for p in patients])
    y = np.array([by_patient[p]["hb"] for p in patients])
    print(f"\n{len(patients)} patients usable ({skipped} images skipped)")

    # The DEPLOYED model is fitted on every patient — standard practice, since
    # the shipped model should use all available data.
    model = fit_linear_model(X, y, args.representation, args.alpha,
                             preprocess.work_size, preprocess.gray_world,
                             args.extractor)
    in_sample = np.array([model.predict_features(row) for row in X])
    fitted = regression_metrics(in_sample, y, baseline=float(y.mean()))

    # Held-out estimate, kept alongside. In-sample metrics describe how well the
    # model fits data it has already seen; only held-out numbers estimate how it
    # will behave on a new patient, so both are stored and clearly labelled.
    predictions = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.ones(len(y), bool)
        keep[i] = False
        fold = fit_linear_model(X[keep], y[keep], args.representation, args.alpha)
        predictions[i] = fold.predict_features(X[i])

    baseline = np.array([np.delete(y, i).mean() for i in range(len(y))])
    held_out = regression_metrics(predictions, y, baseline=float(y.mean()))
    base = regression_metrics(baseline, y)

    print(f"\nFitted on all {len(patients)} patients "
          f"({args.representation}, alpha={args.alpha})")
    print(f"\n  {'metric':<8}{'in-sample':>12}{'held-out':>12}{'baseline':>12}")
    print("  " + "-" * 44)
    for key, label in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")):
        print(f"  {label:<8}{fitted[key]:>12.3f}{held_out[key]:>12.3f}{base[key]:>12.3f}")
    print("  " + "-" * 44)
    print("  in-sample = fit and scored on the same patients (describes the fit)")
    print("  held-out  = leave-one-patient-out (estimates a new patient)")
    if held_out["mae"] >= base["mae"]:
        print("\n  NOTE: held-out performance does not beat predicting the mean.")

    model.metrics = {"in_sample": fitted, "held_out": held_out,
                     "baseline": base, "patients": len(patients),
                     "extractor": args.extractor}
    model.save(args.out)
    print(f"\n{model.equation_string()}")
    print(f"\nSaved {args.out}")


def cmd_predict(args: argparse.Namespace) -> None:
    if args.linear:
        from .predict import LinearPredictor

        predictor = LinearPredictor(args.linear, segmenter_dir=args.segmenter)
    else:
        from .predict import AnemiaPredictor

        if not args.checkpoint:
            raise SystemExit("Pass --checkpoint for the neural model or --linear for the linear one.")
        predictor = AnemiaPredictor(args.checkpoint, segmenter_dir=args.segmenter)

    result = predictor.predict_path(args.image, age_years=args.age, sex=args.sex)
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anemia", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a dataset without training")
    inspect.add_argument("--data", type=Path, action="append", required=True)
    inspect.set_defaults(func=cmd_inspect)

    train = sub.add_parser("train-hb", help="cross-validate the Hb regressor")
    train.add_argument("--data", type=Path, action="append", required=True)
    train.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "hb")
    train.add_argument("--cache", type=Path, default=REPO_ROOT / ".cache" / "crops")
    train.add_argument("--segmenter", type=Path, default=None,
                       help="trained Mask2Former dir; omit to use the colour heuristic")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--folds", type=int, default=None)
    train.set_defaults(func=cmd_train_hb)

    segment = sub.add_parser("train-segmenter", help="fine-tune Mask2Former on ground-truth masks")
    segment.add_argument("--data", type=Path, action="append", required=True,
                         help="Eyes-defy-anemia root (the only dataset with masks)")
    segment.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "segmenter")
    segment.add_argument("--epochs", type=int, default=None)
    segment.set_defaults(func=cmd_train_segmenter)

    debug = sub.add_parser("debug", help="dump masks/overlays; no trained model needed")
    source = debug.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="single photo")
    source.add_argument("--dir", type=Path, help="folder of photos, screened in batch")
    debug.add_argument("--out", type=Path, default=REPO_ROOT / "debug")
    debug.add_argument("--segmenter", type=Path, default=None)
    debug.add_argument("--sheet-size", type=int, default=12,
                       help="how many overlays to place on the contact sheet")
    debug.set_defaults(func=cmd_debug)

    stages = sub.add_parser("stages", help="dump every pipeline stage per image, one folder each")
    stage_source = stages.add_mutually_exclusive_group(required=True)
    stage_source.add_argument("--image", type=Path, help="single photo")
    stage_source.add_argument("--dir", type=Path, help="folder of photos")
    stages.add_argument("--out", type=Path, default=REPO_ROOT / "stages")
    stages.add_argument("--segmenter", type=Path, default=None)
    stages.add_argument("--panel", type=int, default=300, help="panel size in pixels")
    stages.add_argument("--per-row", type=int, default=5, help="panels per row")
    stages.add_argument("--separate", action="store_true",
                        help="also write each stage as its own file")
    stages.set_defaults(func=cmd_stages)

    fit_linear = sub.add_parser("fit-linear", help="fit and save the linear Hb model")
    fit_linear.add_argument("--data", type=Path, action="append", required=True)
    fit_linear.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "linear_model.json")
    fit_linear.add_argument("--representation", default="erythema",
                            choices=["means", "chroma", "erythema", "lab", "a_only"])
    fit_linear.add_argument("--alpha", type=float, default=1.0)
    fit_linear.add_argument("--extractor", default="refined",
                            choices=["refined", "redness", "brightness", "grabcut", "cielab"],
                            help="which conjunctiva extractor to use")
    fit_linear.add_argument("--segmenter", type=Path, default=None,
                            help="trained segmenter directory (overrides --extractor)")
    fit_linear.set_defaults(func=cmd_fit_linear)

    predict = sub.add_parser("predict", help="score a single photo")
    predict.add_argument("--checkpoint", type=Path, default=None,
                         help="neural checkpoint (.pt)")
    predict.add_argument("--linear", type=Path, default=None,
                         help="linear model (.json) — use instead of --checkpoint")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--segmenter", type=Path, default=None)
    predict.add_argument("--age", type=float, default=None, help="patient age in years")
    predict.add_argument("--sex", type=str, default=None, choices=["M", "F"])
    predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
