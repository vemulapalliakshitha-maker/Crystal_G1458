# Experiment history

Original project (PS1): historical original scripts archived.

C001: corrected feature-label alignment, group-separated Li-only and mixed train/H sets, train-only scaler/PCA, sodium transformation.

C002: automatic validation-selected scaling, PCA, hyperparameters and DNN architecture experiments. Chosen frozen checkpoint is the **model-tuning stage**, not the subsequent DNN architecture stage.

The preserved reference values are in `results/reference/recorded_c002_model_tuning_checkpoint.csv`. No newly measured scores are asserted in this package.
