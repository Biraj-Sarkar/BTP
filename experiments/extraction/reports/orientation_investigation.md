# Orientation Investigation: 89 Transposed Masks

Generated from the active annotation manifest. This diagnostic did not modify any original image, verified mask, manifest entry, or status.

## EXIF findings
- Mismatched pairs inspected: **89**
- EXIF Orientation 6: **89**
- Other EXIF orientations: **0**
- Raw encoded dimensions: **4096 x 3072** for all 89 images.
- Dimensions after EXIF orientation: **3072 x 4096** for all 89 images.
- Mask dimensions: **3072 x 4096** for all 89 masks.
- OpenCV default color loading applies EXIF orientation; `IMREAD_IGNORE_ORIENTATION` and `IMREAD_UNCHANGED` expose the raw 4096x3072 pixels.

## Candidate transformation results
- Unchanged mask + EXIF-oriented image: compatible for **89/89** cases.
- Mask rotated 90 degrees counterclockwise + raw image: compatible for **89/89** cases; equivalent geometry to honoring EXIF Orientation 6 on the image.
- Mask rotated 90 degrees clockwise + raw image: shape-compatible for **89/89**, but contradicted by EXIF Orientation 6.
- Mask transposed + raw image: shape-compatible for **89/89**, but is not the EXIF Orientation 6 transform.
- Mask rotated 180 degrees + raw image: incompatible for all 89 cases.

## Visual alignment comparison
The contact sheet compares the unchanged mask over the EXIF-oriented image with raw-image overlays using 90-degree clockwise, 90-degree counterclockwise, and transpose candidates. The unchanged mask aligns with the EXIF-oriented image geometry consistently; the equivalent raw-pixel operation is the mask rotated 90 degrees counterclockwise.

## All-89 conclusion
- EXIF-consistent orientation: **89/89**.
- Clearly supported processing interpretation: load images with EXIF orientation applied, leaving stored masks unchanged.
- Equivalent derived-view transform: rotate a mask copy 90 degrees counterclockwise when comparing against raw OpenCV pixels.
- No case-specific exceptions were found in metadata or dimensions.
- No original or verified files were changed.

## Historical evidence
- The archived pilot audit reported zero dimension errors for its 30 verified records, but it did not record EXIF orientation or raw-versus-oriented dimensions.
- No Git history is available in the project directory.
- The current evidence explains the mismatch as an orientation metadata/display-coordinate issue, not a cleanup rename or resize.

## Decision
Do not modify the verified masks. Before extraction, make the image-loading path consistently honor EXIF orientation, or use the equivalent in-memory mask transformation in a derived processing view. Await approval before any correction or pipeline change.
