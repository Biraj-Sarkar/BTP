# Annotation workspace

Manual annotation is the only active annotation workflow in this directory.
The verified masks are the ground-truth labels. Old model-assisted predictions
and pilot reports are preserved under `D:\BTP\archive\annotation\` and are not
used as labels or as the current audit.

The Attention U-Net belongs to the extraction/segmentation phase. Its old
annotation-assistance outputs and model-selection artifacts are archived; no
model is loaded by the active GUI.

## Manual annotation GUI

From the project root:

```powershell
python experiments/annotation/scripts/manual_annotation_gui.py `
  --images "datasets/raw/hospital/images" `
  --output "experiments/annotation" `
  --reviewer "your-name"
```

Every new image starts with a completely blank mask. The GUI never loads model
predictions. `S` saves a draft, `V` saves a manually reviewed verified mask, `C`
clears the mask, and `Q` exits. Verification of an empty mask is blocked.

Drafts are stored in `masks/drafts/`, verified ground truth is stored in
`masks/verified/`, and the active manifest is
`masks/reports/annotation_manifest.csv`.
Masks are PNG files named exactly after the source image stem, for example
`0115H-569.jpg` maps to `0115H-569.png`.

## Current audit

`reports/final_annotation_dataset_audit.json` and
`reports/final_annotation_dataset_audit.md` are the current filesystem audit.
They are distinct from the archived 30-image pilot audit. The current audit
found 288 verified manifest entries and 89 image-mask dimension mismatches;
therefore, it does not claim that all masks are structurally valid.
