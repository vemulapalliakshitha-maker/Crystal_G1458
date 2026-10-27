
"""Train DNN, SVR and KRR on mixed training data; evaluate on H and Na sets."""
import os, random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.regularizers import l2
except ImportError as exc:
    raise SystemExit("TensorFlow is required for DNN. Activate your project .venv.") from exc

tf.keras.utils.set_random_seed(SEED)

TRAIN_FILE = "mixed_T_set.csv"
H_FILE = "mixed_H_set.csv"
NA_FILE = "na_pca.csv"
METRICS_FILE = "16_mixed_models_metrics.csv"
PREDICTIONS_FILE = "16_mixed_models_predictions.csv"

# Previously tuned hyperparameters
DNN_EPOCHS, DNN_BATCH_SIZE = 100, 32
DNN_LEARNING_RATE, DNN_L2 = 0.001, 0.0001
SVR_C, SVR_GAMMA = 100, 0.01
KRR_ALPHA, KRR_GAMMA = 0.01, 0.02


def get_target(df):
    for col in ("average_voltage", "voltage"):
        if col in df.columns:
            return col
    raise ValueError("Expected target column 'average_voltage' or 'voltage'.")


def get_features(df):
    pcs = [c for c in df.columns if c.startswith("PC_")]
    if pcs:
        return sorted(pcs, key=lambda c: int(c.split("_")[1]))
    feats = [c for c in df.columns if c.startswith("feat_")]
    if feats:
        return sorted(feats, key=lambda c: int(c.split("_")[1]))
    raise ValueError("No PC_ or feat_ feature columns found.")


def prepare(df, features, target):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns, e.g. {missing[:5]}")
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid], df.loc[valid].copy().reset_index(drop=True)


def make_dnn(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(60, activation="relu", kernel_regularizer=l2(DNN_L2)),
        Dropout(0.25),
        Dense(30, activation="relu", kernel_regularizer=l2(DNN_L2)),
        Dropout(0.10),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=DNN_LEARNING_RATE),
        loss="mae"
    )
    return model


def main():
    train = pd.read_csv(TRAIN_FILE)
    h_set = pd.read_csv(H_FILE)
    na_set = pd.read_csv(NA_FILE)

    target = get_target(train)
    features = get_features(train)
    if target not in h_set.columns or target not in na_set.columns:
        raise ValueError(f"Both test sets must contain '{target}'.")

    X_train, y_train, _ = prepare(train, features, target)
    X_h, y_h, h_valid = prepare(h_set, features, target)
    X_na, y_na, na_valid = prepare(na_set, features, target)

    print(f"Training rows: {len(X_train)} | Features: {len(features)}")
    print(f"H-set rows: {len(X_h)} | Na-set rows: {len(X_na)}")

    
    print("\nTraining DNN...")
    dnn = make_dnn(X_train.shape[1])
    dnn.fit(X_train, y_train, epochs=DNN_EPOCHS,
            batch_size=DNN_BATCH_SIZE, verbose=0)

    print("Training SVR...")
    svr = SVR(kernel="rbf", C=SVR_C, gamma=SVR_GAMMA)
    svr.fit(X_train, y_train)

    print("Training KRR...")
    krr = KernelRidge(kernel="rbf", alpha=KRR_ALPHA, gamma=KRR_GAMMA)
    krr.fit(X_train, y_train)

    results = []
    for name, model in [
        ("DNN", dnn), ("SVR", svr), ("KRR", krr)
    ]:
        for set_name, X, y in [
            ("H_set", X_h, y_h), ("Na_set", X_na, y_na)
        ]:
            pred = (
                model.predict(X, verbose=0).ravel()
                if name == "DNN" else model.predict(X)
            )
            results.append({
                "Model": name,
                "Test_Set": set_name,
                "N": len(y),
                "MAE": mean_absolute_error(y, pred),
                "R2": r2_score(y, pred)
            })

    results = pd.DataFrame(results)
    results.to_csv(METRICS_FILE, index=False)
    print("\n=== MAE and R2 ===")
    print(results.to_string(index=False))
    print(f"\nSaved results to {METRICS_FILE}")


if __name__ == "__main__":
    main()
