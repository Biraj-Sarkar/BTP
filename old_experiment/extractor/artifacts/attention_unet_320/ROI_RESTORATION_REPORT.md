# ROI restoration report

Date: 2026-10-22

## Interfaces and source audit

The extractor CLI is `python -m extractor.extract` and requires `--input`, `--output`, and optional `--checkpoint`. The source directory is `datasets/`, containing 183 eye images directly at its root. The split manifest stores paths as `segmentation_dataset/images/<name>.jpg`; the manifest builder resolves these to the actual `datasets/<name>.jpg` files and verifies existence. Labels are joined by `RegistrationNo`/image stem from `datasets/captured_data.csv`, never by row order. The fixed split is `extractor/experiments/final_pretrained320/split_manifest.csv` (129 train, 27 validation, 27 test).

## Commands run

```powershell
python -m extractor.extract --input datasets --output ROI --checkpoint Models\UNet_extractor\unet.pt
python -m extractor.extract --input datasets --output ROI_attention_verified --checkpoint extractor\artifacts\attention_unet_320\attention_unet_candidate.pt

python -m predictor.build_matched_roi_manifest --rois ROI --extraction-manifest ROI\extraction_manifest.csv --output predictor\experiments\roi_provenance\production_manifest.csv --backend production_unet --architecture unet --checkpoint Models\UNet_extractor\unet.pt
python -m predictor.build_matched_roi_manifest --rois ROI_attention_verified --extraction-manifest ROI_attention_verified\extraction_manifest.csv --output predictor\experiments\roi_provenance\attention_manifest.csv --backend attention_unet_candidate --architecture attention_unet --checkpoint extractor\artifacts\attention_unet_320\attention_unet_candidate.pt
```

The pre-existing `ROI_attention/` directory was not overwritten or used.

## Results

| Dataset | Records | Successful | Empty/failed | Splits |
|---|---:|---:|---:|---|
| `ROI/` production U-Net | 183 | 182 | 1 | 129 / 27 / 27 |
| `ROI_attention_verified/` candidate | 183 | 182 | 1 | 129 / 27 / 27 |

The production empty extraction is `112-3643.jpg` (train, Hb 5.60); the candidate empty extraction is `1951.jpg` (train, Hb 11.40). The old uploaded manifest's `1951.jpg` production-empty entry does not match regeneration with the current unchanged production checkpoint and is not used as authority. Both failed rows remain in the manifests with quality reasons; no zero crop is silently treated as usable.

Authoritative manifests and audits:

- `predictor/experiments/roi_provenance/production_manifest.csv` and `.audit.json`
- `predictor/experiments/roi_provenance/attention_manifest.csv` and `.audit.json`

All 183 source IDs, labels, and split assignments match between manifests. Every successful ROI exists and is exactly `224x224x3` `uint8`; all source images exist. The two checkpoint hashes are unchanged and match the required baselines:

- Production: `35b07d3c8bf7f36603d3b38f96ec95bfee8aac367089111046cc163d6983865c`
- Candidate: `c7c8eea36c2f162297e7dc1d5683051b8090faac881a5cd98ec62588593f1498`

## Training readiness and stop point

The restored datasets are provenance-valid. For matched predictor training, the explicit usable-sample policy should use the intersection of successful rows (181 common images), preserving their existing train/validation/test split labels; the two non-overlapping failed images must remain recorded and excluded. This policy has not been applied to training because predictor training was explicitly deferred. No predictor checkpoint, extractor checkpoint, threshold, default, or split was changed.
