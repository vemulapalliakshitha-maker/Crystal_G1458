# AI-Based Battery Electrode Voltage Prediction

**Selected checkpoint:** C002 model-tuning stage (before DNN architecture tuning).

This project contains source Materials Project data, the original reference implementation, corrected feature extraction, battery-group-aware preprocessing, six standalone ML model programs, archived original experiments, and recorded performance.

## Quick start (VS Code / PowerShell)

Open this folder in VS Code. In your activated Python environment:

```powershell
pip install -r requirements.txt
python run_pipeline.py prepare
python run_pipeline.py train
```

Or run `python run_pipeline.py all` for the complete sequence. Feature extraction reads thousands of CIF structures and can be computationally expensive. Train only one model, e.g. `python models/mixed_dnn.py`.

## Folder map

- `data/raw/dataset/` — ORIGINAL downloadable research dataset (CSV, HDF5 crystal structures and element tables).
- `data/interim/` — generated `remaining_instances.csv` and `na_instances.csv` with Na/K absent from training.
- `data/processed/` — generated, aligned 239-feature matrices and processed train/holdout/Na tables.
- `scripts/01_prepare_records.py` — clean/filter data.
- `scripts/02_extract_training_features.py` — corrected 239-feature extraction with row tracking.
- `scripts/03_extract_sodium_features.py` — separate sodium feature extraction.
- `scripts/04_prepare_group_safe.py` — prepare C001 group-safe Li/mixed pipelines.
- `scripts/06_auto_improve_six_models.py` — optional experimental optimization (separate from frozen models).
- `models/li_dnn.py`, `models/mixed_dnn.py` — complete standalone neural network training/evaluation for backend deployment.
- `models/li_svr.py`, `models/li_krr.py`, `models/mixed_svr.py`, `models/mixed_krr.py` — complete classical ML baseline scripts.
- `models/run_all_six.py` — sequentially trains all six.
- `artifacts/frozen_models/` — created on training: `.keras` / `.joblib`, predictions, measured metrics.
- `results/reference/` — **recorded numbers from the user's earlier local run, not measurements made by this package**.
- `results/measured/` — future runs' feature-extraction audits and optional optimization outputs.
- `archive/original_scripts/` — original user-uploaded experimental scripts (not main reproducible pipeline).
- `reference/original_repository/` — original authors' documentation, preprocessing and regression reference code.

## Frozen checkpoint models

| Model | Scaling | PCA | Config |
|---|---|---|---|
| Li DNN | StandardScaler | 80 | 60→30 ReLU; dropout .25/.10; L2 .0001; RMSprop lr .001; 120 epochs; batch 32 |
| Li SVR | MinMaxScaler(-1,1) | 80 | RBF, C=100, gamma=.02, epsilon=.1 |
| Li KRR | MinMaxScaler(-1,1) | 80 | RBF, alpha=.01, gamma=.02 |
| Mixed DNN | StandardScaler | 80 | Same architecture, 100 epochs |
| Mixed SVR | MinMaxScaler(-1,1) | 120 | RBF, C=100, gamma=.01, epsilon=.1 |
| Mixed KRR | MinMaxScaler(-1,1) | 80 | RBF, alpha=.01, gamma=.02 |

Training ions: Li-only model uses Li; mixed model uses Li/Ca/Mg/Zn/Al/Y. Na remains external evaluation only; K excluded. All reported metrics are regression MAE (volts) and R², **not classification accuracy**.

## Measurement integrity

`results/reference/recorded_c002_model_tuning_checkpoint.csv` contains the provided C002 printed scores. It is not re-created by hardcoding predictions. Running `models/*.py` computes new scores from trained models and writes measured files to `artifacts/frozen_models/`. DNN results can differ across environments and hardware. Models use a deterministic battery-ID group split and training-only fitted scalers/PCA. Cross-validation is a separate concern: holdout/Na results are not used for hyperparameter selection. No saved trained `.keras` or `.joblib` weights were provided with the uploaded files, so **trained weights are not included in the download**.

## Reviewer demo

1. Show source CSV/HDF5 and preprocessing code.
2. Show `results/reference/recorded_c002_model_tuning_checkpoint.csv` as the prior local experiment.
3. Execute `python models/mixed_dnn.py` or load the `artifacts/frozen_models/mixed_dnn.keras` created by a completed run.
4. Compare `artifacts/frozen_models/frozen_six_model_metrics.csv` (new measured predictions) with the reference CSV.
5. For deployment, generate the same 239 input descriptors; load the saved DNN and its own fitted preprocessor.

## Data provenance

The raw materials dataset and original authors' code originate from the research repository ZIP uploaded by the user. Preserve its copyright/license terms where applicable. `reference/original_repository/` retains reference files, while the original raw dataset is placed under `data/raw/dataset/`.
