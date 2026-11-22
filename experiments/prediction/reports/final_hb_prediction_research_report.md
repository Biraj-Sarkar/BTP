# Final Hb Prediction Research Report

## Final organization

The active prediction area contains:

```text
experiments/prediction/
├── data_audit/
│   ├── artifact_integrity.json
│   ├── eligibility_summary.json
│   ├── hb_dataset_manifest.csv
│   ├── readiness_report.md
│   ├── split_validation.json
│   └── unit_provenance.md
├── benchmarks/
│   ├── v1/
│   ├── v2/
│   └── v2_stability/
├── final_evaluation/
├── reports/
│   └── final_hb_prediction_research_report.md
├── scripts/
│   ├── hb_benchmark_v2.py
│   ├── hb_final_test_evaluation.py
│   └── inference.py
└── README.md
```

The selected checkpoint is
`models/predictors/densenet121_seed_20261118_best.pt`. Other Hb checkpoints
remain under `D:\BTP\archive\prediction_models\`, preserving their architecture
and version structure. The Attention U-Net files remain in `models/extraction/`
unchanged.

Historical scripts are under `D:\BTP\archive\prediction_scripts\`:
`hb_benchmark.py`, `hb_benchmark_finalize_test.py`,
`hb_benchmark_v2_stability.py`, `phase6_finalize_hb_dataset.py`, and
`phase6_1_finalize_hb_dataset.py`. The active reproducibility and evaluation
scripts remain under `scripts/`.

## Dataset and model

The finalized dataset contains 288 records, 279 eligible Hb targets, and 9
excluded missing/non-numeric targets. Existing patient-wise splits are train
195, validation 43, and test 41. Hb values are from the hospital
`HAEMOGLOBIN (GM/DL)` field and are represented as `g/dL`.

The selected architecture is DenseNet-121. It uses RGB ROI crops resized to
224 x 224 with ImageNet normalization. Training-target standardization is
fitted on the training split only; inference applies the checkpoint's stored
mean and standard deviation to return `g/dL`.

## Existing final evaluation

Selected checkpoint validation metrics:

- MAE: 0.7736 g/dL
- RMSE: 1.1696 g/dL
- R²: 0.4510
- Best epoch: 9

Held-out evaluation metrics for 41 test records are retained in
`final_evaluation/test_metrics.json`:

- MAE: 1.3452 g/dL
- RMSE: 1.7005 g/dL
- R²: 0.2097
- Mean bias: 0.4078 g/dL

The training-mean baseline remains documented beside these results. The test
set was not used for model selection.

## Inference verification

`scripts/inference.py` was syntax-checked and exercised on an existing ROI
crop using CPU inference. The output was finite and returned in `g/dL`:

```json
{
  "image": "experiments/extraction/roi_crops/0115H-569.jpg",
  "predicted_hb_g_dl": 12.395803451538086
}
```

The script accepts an already extracted ROI. It does not call the
Attention U-Net or perform full-image segmentation.

## Integrity and preservation

Final SHA-256 verification was recorded after the reorganization:

| Artifact | SHA-256 | Status |
|---|---|---|
| Selected DenseNet checkpoint | `57EA961A29524563059F5293147E962945AE9BDB1042E357570057F4C533288B` | unchanged |
| Attention U-Net metadata | `EE23E2FBB1BCD683056090F804B95282DF2EC026D7473D569EDBE7D881C0A2BB` | unchanged |
| Attention U-Net best checkpoint | `1F53832F8DC2A8FFBFCBEEBD426B8A53F8C9EC69B0A9229C21A4D7F66AD86AF3` | unchanged |
| Attention U-Net final checkpoint | `959CCF4EF94E93EFB5CBC83926CB537A764D55E21C071541B62881A70AD9E` | unchanged |

All 21 tracked model/protected-artifact hashes matched their pre-cleanup
values. Historical benchmark reports, predictions, metrics, plots, logs, and
audits were preserved. No training script was executed.

The superseded cleanup plans, directory-tree notes, and checkpoint hash CSVs
were removed only after this verification. Detailed verification is recorded
in `reports/final_hash_verification.json`.

## Limitations

This is an academic Hb estimation/screening research model. It must not be
described as a diagnostic device or as clinically validated. Full-image
end-to-end integration and genuinely independent external evaluation remain
future work.
