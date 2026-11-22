# Held-Out Test Evaluation

The selected DenseNet-121 checkpoint was evaluated once on the 41-record
patient-wise test split using inference mode only.

## Model

- MAE: 1.3452 g/dL
- RMSE: 1.7005 g/dL
- R2: 0.2097
- Bias: 0.4078 g/dL
- Median absolute error: 1.0662 g/dL
- Maximum absolute error: 4.5003 g/dL
- Bland-Altman 95% LoA: -2.8682 to 3.6837 g/dL

## Training-mean baseline

- Training mean: 11.3764 g/dL
- MAE: 1.3713 g/dL
- RMSE: 1.9149 g/dL
- R2: -0.0021

The V1 Phase 7 VGG16 test result exists as historical evidence. This V2
evaluation is not independent external validation, clinical validation, or
evidence that the model can diagnose anemia. It is an internal held-out
research evaluation on a small single-source cohort.
