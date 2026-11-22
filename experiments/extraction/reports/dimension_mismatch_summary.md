# Image-Mask Dimension Mismatch Investigation

Generated from the active manifest: `D:\BTP\BTP-Anemia-Final\experiments\annotation\masks\reports\annotation_manifest.csv`. This is diagnostic only; no source image, mask, manifest entry, or status was modified.

## Scope and counts
- Manifest rows inspected: **288**
- Verified rows inspected: **288**
- Dimension mismatches: **89**
- Matching image/mask dimensions: **199**
- Source images available: **288**
- Verified masks available: **288**

## Category counts
- Width/height swapped: **89**
- Uniform resizing: **0**
- Cropping or aspect-ratio changes: **0**
- Different source image or incorrect pairing: **0**
- Other / unknown: **0**

## Findings
- Exact width/height swaps: **89** of 89.
- Same aspect ratio but different resolution: **0** of 89.
- All mismatches are `JPG` source images paired with PNG masks.

### Dimension evidence
- Image 4096x3072 -> mask 3072x4096: **89**

## File metadata and historical records
- Image modification dates represented: `{'2026-10-19': 182, '2026-11-09': 106}`
- Mask modification dates represented: `{'2026-11-10': 288}`
- File sizes and formats for every mismatch are recorded in `dimension_mismatch_report.csv`.
- Git history is unavailable: the project directory is not a Git repository.
- Historical archive files found:
  - D:\BTP\archive\annotation\reports\pilot_30_image\annotation_manifest.csv: present (19003 bytes)
  - D:\BTP\archive\annotation\reports\pilot_30_image\annotation_validation.txt: present (84 bytes)
  - D:\BTP\archive\annotation\reports\pilot_30_image\final_annotation_dataset_audit.json: present (34648 bytes)
  - D:\BTP\archive\annotation\reports\pilot_30_image\final_annotation_dataset_audit.md: present (1205 bytes)
- No metadata was found that proves original image dimensions or filenames changed during cleanup/preprocessing. The current evidence only establishes the dimensions presently on disk and the historical reports preserved in the archive.

## Correction decision
- Clearly identifiable safe corrections: **0 approved automatically**. 89 cases are exact axis swaps and are safe candidates for a future, explicitly approved coordinate/orientation correction, but must not be changed without confirming the annotation/display convention.
- Require further investigation: **0** cases, plus provenance confirmation for the 89 swapped cases.
- No automatic correction was attempted.
