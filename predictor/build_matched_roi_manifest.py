"""Build an auditable ROI/label/split manifest for matched predictor experiments."""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from pathlib import Path
import cv2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("datasets"))
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--extraction-manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("datasets/captured_data.csv"))
    p.add_argument("--splits", type=Path, default=Path("extractor/experiments/final_pretrained320/split_manifest.csv"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)

    labels = {}
    with a.labels.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = str(row["RegistrationNo"]).strip()
            if key in labels: raise RuntimeError(f"Duplicate label ID: {key}")
            raw = (row.get("Haemoglobin") or "").strip()
            try: hb = float(raw); valid_hb = math.isfinite(hb) and 3.0 <= hb <= 25.0
            except ValueError: valid_hb = False
            labels[key] = (row, valid_hb)
    splits = {}
    with a.splits.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = Path(row["image"]).stem
            if key in splits: raise RuntimeError(f"Duplicate split ID: {key}")
            splits[key] = row
    with a.extraction_manifest.open(encoding="utf-8", newline="") as f:
        extraction = {Path(row["image"]).stem: row for row in csv.DictReader(f)}
    if set(extraction) != set(splits):
        raise RuntimeError(f"Extraction/split ID mismatch: extraction={len(extraction)}, splits={len(splits)}")
    checkpoint_hash = sha256(a.checkpoint)
    rows = []
    fields = ["image_id","source_image","roi_path","hemoglobin","label_valid","age","sex","split","extractor_backend","architecture","checkpoint","checkpoint_sha256","passed","empty_mask","training_eligible","roi_exists","roi_shape","roi_dtype","mask_ratio","focus","quality_reasons"]
    for image_id, split in sorted(splits.items()):
        if image_id not in labels: raise RuntimeError(f"No label row for image ID {image_id}")
        ext = extraction[image_id]; label, label_valid = labels[image_id]
        split_path = Path(split["image"])
        candidates = [a.images / split_path, a.images / split_path.name]
        source = next((x for x in candidates if x.is_file()), candidates[-1])
        if not source.is_file():
            raise RuntimeError(f"Source image missing for {image_id}: tried {candidates}")
        roi = a.rois / ext["roi_file"]
        img = cv2.imread(str(roi), cv2.IMREAD_UNCHANGED) if roi.is_file() else None
        shape = "x".join(map(str, img.shape)) if img is not None else ""
        dtype = str(img.dtype) if img is not None else ""
        passed = str(ext.get("passed", "")).lower() == "true"
        empty = str(ext.get("backend", "")).lower().endswith("empty") or float(ext.get("mask_ratio", 0) or 0) <= 0
        if passed and (img is None or img.shape != (224,224,3) or str(img.dtype) != "uint8"):
            raise RuntimeError(f"Invalid successful ROI format: {roi} ({shape}, {dtype})")
        training_eligible = passed and not empty and label_valid and roi.is_file() and shape == "224x224x3" and dtype == "uint8"
        rows.append({"image_id":image_id,"source_image":str(source),"roi_path":str(roi),"hemoglobin":label["Haemoglobin"],"label_valid":label_valid,"age":label.get("AgeInYears",""),"sex":label.get("Gender",""),"split":split["split"],"extractor_backend":a.backend,"architecture":a.architecture,"checkpoint":str(a.checkpoint),"checkpoint_sha256":checkpoint_hash,"passed":passed,"empty_mask":empty,"training_eligible":training_eligible,"roi_exists":roi.is_file(),"roi_shape":shape,"roi_dtype":dtype,"mask_ratio":ext.get("mask_ratio",""),"focus":ext.get("focus",""),"quality_reasons":ext.get("reasons","")})
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    audit={"records":len(rows),"splits":{s:sum(r["split"]==s for r in rows) for s in ("train","val","test")},"passed":sum(r["passed"] for r in rows),"empty":sum(r["empty_mask"] for r in rows),"failed":sum(not r["passed"] for r in rows),"valid_labels":sum(r["label_valid"] for r in rows),"training_eligible":sum(r["training_eligible"] for r in rows),"roi_format_valid":sum((not r["passed"]) or (r["roi_shape"]=="224x224x3" and r["roi_dtype"]=="uint8") for r in rows),"backend":a.backend,"architecture":a.architecture,"checkpoint_sha256":checkpoint_hash}
    a.output.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__": main()
