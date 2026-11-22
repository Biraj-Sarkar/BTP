# Phase 6.1 Hb dataset readiness report

## Final status

**READY FOR Hb MODEL DEVELOPMENT**

## Dataset finalization

- Source CSV SHA-256 matched: `1FD1519AB27E09B338242887BCFC86CFBDADE315ACA19D2D3611EE67E7AF7A2B`
- Total records: 288
- Unique registration numbers: 288
- Eligible Hb targets: 279
- Excluded missing/non-numeric Hb records: 9
- Numeric targets labeled unit: `gm/dL`
- Conversion applied: none
- Imputation applied: none
- ROI crops readable: 288

The 279 eligible values equal their original numeric CSV values exactly.
`gm/dL` is interpreted as grams per deciliter, equivalent to `g/dL`.

## Provenance

The exact website heading supplied by the project owner is
`HAEMOGLOBIN (GM/DL)`. The project owner confirms that the current CSV was
downloaded from that same hospital webpage.

No hospital screenshot file was found in the project workspace or session
attachments. The screenshot was not recreated or fabricated. The final
provenance report distinguishes the screenshot heading supplied by the owner
from the owner's confirmation of the CSV download origin.

## Nine excluded records

The nine retained audit records remain in
`hb_dataset_manifest.csv` and are identified by
`excluded_missing_or_invalid_hb`. They are excluded solely because the original
`Haemoglobin` field is missing or non-numeric. No replacement values were
invented or imputed.

## Split validation

Existing split memberships were retained:

- Train eligible: 195
- Validation eligible: 43
- Test eligible: 41
- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0
- Eligible records assigned exactly once: 279
- Splits regenerated: no
- Test records used for training or model selection: no

## Preservation

Before-and-after scope digests show no change to original images, verified or
draft masks, annotation manifest, extraction predictions, overlays, ROI crops,
existing split manifests, or protected extraction artifacts.

Protected checkpoint SHA-256 values and Phase 6 preservation status are recorded
in `artifact_integrity.json`.

## Recommendation

The dataset is finalized for supervised Hb model development under the
confirmed gm/dL unit, but the screenshot file itself should be copied into the
project as an evidence attachment if available to the project owner. No
training or inference was performed in this phase.
