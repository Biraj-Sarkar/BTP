# Phase 6.1 Hb unit provenance

## Verified source

- Source file: `datasets/raw/hospital/captured_data.csv`
- SHA-256: `1FD1519AB27E09B338242887BCFC86CFBDADE315ACA19D2D3611EE67E7AF7A2B`
- Source records: 288
- Unique `RegistrationNo`: 288
- Downloaded column: `Haemoglobin`

## Unit evidence

### 1. Hospital website evidence supplied by the project owner

The hospital website heading supplied for this dataset is:

`HAEMOGLOBIN (GM/DL)`

The project owner confirms that the current `captured_data.csv` was downloaded
from that same hospital webpage. This confirmation establishes the origin of
the actual CSV used in this project.

### 2. Screenshot evidence

The stated heading is supported here only by the project owner's supplied
account of the screenshot. A separate screenshot file was not found in the
active project workspace or session files. Therefore, no screenshot is
claimed as an on-disk provenance artifact. The screenshot must be copied into
the project manually before it can be included as a file-backed evidence
attachment. It was not recreated or fabricated.

### 3. Unit interpretation

`GM/DL` means grams per deciliter and is equivalent to `g/dL`.
Therefore, the 279 valid original numeric values remain numerically unchanged
and are labeled `gm/dL`. No division by 10 or other conversion was applied.

## Distinction between evidence types

- Screenshot heading evidence: `HAEMOGLOBIN (GM/DL)` as supplied by the project
  owner; no screenshot file was available in the workspace.
- Project-owner confirmation: confirms the current CSV was downloaded from the
  hospital webpage that displays the heading.
- CSV content evidence: confirms 279 numeric values and 9 missing/non-numeric
  values under the `Haemoglobin` column.

These sources jointly support unit provenance for the current CSV; the
website/screenshot image itself is not available as a project file.
