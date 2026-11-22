"""Non-destructive candidate-versus-production extractor assessment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from extractor import ROIExtractor
from predictor.predict import predict_from_crop
from predictor.training import load_checkpoint


def _labels(path: Path) -> dict[str, float]:
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                out[str(row["RegistrationNo"]).strip()] = float(row["Haemoglobin"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _mean(values):
    return float(np.mean(values)) if values else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("extractor/experiments/final_pretrained320/split_manifest.csv"))
    p.add_argument("--labels", type=Path, default=Path("datasets/captured_data.csv"))
    p.add_argument("--predictor", type=Path, default=Path("predictor/runs/resnet/best.pt"))
    p.add_argument("--candidate", type=Path, default=Path("extractor/artifacts/attention_unet_320/attention_unet_candidate.pt"))
    p.add_argument("--production", type=Path, default=Path("Models/UNet_extractor/unet.pt"))
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--output", type=Path, default=Path("extractor/artifacts/attention_unet_320/preproduction_assessment"))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    labels = _labels(a.labels)
    records = []
    with a.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != a.split:
                continue
            stem = Path(row["image"]).stem
            if stem in labels:
                records.append((Path(row["image"]), labels[stem]))
    if not records:
        raise RuntimeError(f"No labelled {a.split} records matched the manifest")

    candidate = ROIExtractor(checkpoint_path=a.candidate)
    production = ROIExtractor(checkpoint_path=a.production)
    predictor, bundle = load_checkpoint(a.predictor, device="cpu")
    rows = []
    for rel, target in records:
        image = Path(rel)
        t = time.perf_counter(); rc = candidate.extract_from_path(image); candidate_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); rp = production.extract_from_path(image); production_ms = (time.perf_counter() - t) * 1000
        for name, result in (("candidate", rc), ("production", rp)):
            if result.crop.shape != (224, 224, 3) or result.crop.dtype != np.uint8:
                raise AssertionError(f"{name} crop contract failed for {image}")
            if result.crop.ndim != 3:
                raise AssertionError(f"{name} crop is not RGB for {image}")
        pred_c = None if not rc.mask.any() else predict_from_crop(predictor, rc.crop)["hb_g_dl"]
        pred_p = None if not rp.mask.any() else predict_from_crop(predictor, rp.crop)["hb_g_dl"]
        rows.append({"image": str(image), "target_hb": target,
                     "candidate_hb": pred_c, "production_hb": pred_p,
                     "candidate_abs_error": None if pred_c is None else abs(pred_c-target),
                     "production_abs_error": None if pred_p is None else abs(pred_p-target),
                     "candidate_empty": not bool(rc.mask.any()), "production_empty": not bool(rp.mask.any()),
                     "candidate_ms": candidate_ms, "production_ms": production_ms,
                     "candidate_bbox": list(rc.bbox), "production_bbox": list(rp.bbox),
                     "candidate_quality": rc.quality.to_dict(), "production_quality": rp.quality.to_dict()})
    with (a.output / "rows.json").open("w", encoding="utf-8") as f: json.dump(rows, f, indent=2)
    def stats(prefix):
        valid = [r for r in rows if r[prefix+"_hb"] is not None]
        errors = [r[prefix+"_hb"]-r["target_hb"] for r in valid]
        abs_errors = [abs(x) for x in errors]
        return {"count": len(valid), "empty_masks": sum(r[prefix+"_empty"] for r in rows),
                "mae_g_dl": _mean(abs_errors), "rmse_g_dl": math.sqrt(_mean([x*x for x in errors])) if errors else None,
                "bias_g_dl": _mean(errors), "mean_runtime_ms": _mean([r[prefix+"_ms"] for r in rows])}
    summary = {"split": a.split, "records": len(rows), "predictor_checkpoint": str(a.predictor),
               "predictor_architecture": bundle.get("architecture"), "candidate": stats("candidate"),
               "production": stats("production"), "test_split_used": False}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
