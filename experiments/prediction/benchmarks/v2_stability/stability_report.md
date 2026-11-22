# Phase 7.2 Multi-Seed Hb Regression Stability Evaluation

This evaluation reuses the corrected V2 implementation for DenseNet-121,
VGG16, and MobileNetV3-Large. Seed 20261116 is the completed V2 result;
only seeds 20261117 and 20261118 were trained here.

## Results

| Architecture | MAE mean +/- SD | RMSE mean +/- SD | R2 mean +/- SD | Mean runtime (s) |
|---|---:|---:|---:|---:|
| densenet121 | 0.8760 +/- 0.0895 | 1.2530 +/- 0.0838 | 0.3682 +/- 0.0843 | 357.8 |
| vgg16 | 1.0019 +/- 0.0588 | 1.3035 +/- 0.0835 | 0.3164 +/- 0.0887 | 542.8 |
| mobilenet_v3_large | 0.9878 +/- 0.0301 | 1.3545 +/- 0.0364 | 0.2635 +/- 0.0398 | 61.2 |

## Interpretation

The ranking should be assessed across all three metrics and seed variability,
not from one validation score. The complete per-seed values are in
`stability_results.csv`. No test targets, images, predictions, or metrics
were accessed. This remains an internal validation stability study and does
not establish clinical diagnostic validity.

DenseNet-121 was the most stable leader by validation MAE: it ranked first
for MAE at all three seeds and had the best mean MAE, mean RMSE, and mean R2.
VGG16 had the best RMSE and R2 at seed 20261116, but was behind DenseNet-121
on those metrics at the two new seeds. MobileNetV3-Large had the smallest MAE
standard deviation and the shortest runtime, but its mean RMSE and R2 were
weaker. Thus the MAE ranking was stable, while supporting-metric rankings were
not completely stable; this is validation evidence, not a final clinical
model-selection claim.

Protected extraction artifacts were hash-verified before and after the run.
