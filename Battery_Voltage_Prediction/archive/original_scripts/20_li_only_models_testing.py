
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

FEATURES = [f"PC_{i}" for i in range(1, 81)]
TARGET = "average_voltage"


def load_data(file):
    df = pd.read_csv(file)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid]


X, y = load_data("li_T_set.csv")
Xh, yh = load_data("li_H_set.csv")
Xna, yna = load_data("na_pca.csv")

print("Li training:", len(y), "| Li H-set:", len(yh), "| Na-set:", len(yna))

print("\nTraining DNN...")
dnn = Sequential([
    Input(shape=(80,)),
    Dense(60, activation="relu", kernel_regularizer=l2(0.0001)),
    Dropout(0.25),
    Dense(30, activation="relu", kernel_regularizer=l2(0.0001)),
    Dropout(0.10),
    Dense(1)
])
dnn.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss="mae"
)
dnn.fit(X, y, epochs=100, batch_size=32, verbose=0)

print("Training SVR...")
svr = SVR(kernel="rbf", C=100, gamma=0.01).fit(X, y)

print("Training KRR...")
krr = KernelRidge(kernel="rbf", alpha=0.01, gamma=0.02).fit(X, y)

results, predictions = [], []

for name, model in [("DNN", dnn), ("SVR", svr), ("KRR", krr)]:
    for set_name, Xt, yt in [
        ("Li_H_set", Xh, yh),
        ("Na_set", Xna, yna)
    ]:
        pred = (
            model.predict(Xt, verbose=0).ravel()
            if name == "DNN" else model.predict(Xt)
        )
        results.append({
            "Model": name,
            "Test_Set": set_name,
            "N": len(yt),
            "MAE": mean_absolute_error(yt, pred),
            "R2": r2_score(yt, pred)
        })

        for actual, estimate in zip(yt, pred):
            predictions.append({
                "Model": name,
                "Test_Set": set_name,
                "Actual_Voltage": actual,
                "Predicted_Voltage": estimate,
                "Absolute_Error_V": abs(actual - estimate)
            })

pd.DataFrame(results).to_csv("li_only_models_metrics.csv", index=False)
pd.DataFrame(predictions).to_csv(
    "li_only_models_predictions.csv", index=False
)

print("\n=== Li-only model results ===")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved li_only_models_metrics.csv")
print("Saved li_only_models_predictions.csv")
