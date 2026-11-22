# Final Hb Model Selection

DenseNet-121 seed 20261118 was selected using validation MAE only from the
three-seed Phase 7.2 stability results. Its validation MAE was
0.7736 g/dL, RMSE
1.1696 g/dL, and R2
0.4510, at epoch 9.

Checkpoint: `D:\BTP\BTP-Anemia-Final\models\predictors\densenet121_seed_20261118_best.pt`

The test set was not used for this selection. No checkpoint averaging,
retraining, or test-driven tuning was performed.
