
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


def load_data(filename):
    df = pd.read_csv(filename)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


X, y = load_data("li_train_pca_clean.csv")
Xh, yh = load_data("li_holdout_pca_clean.csv")
Xna, yna = load_data("na_pca_li_clean.csv")

print("Li training:", len(y))
print("Li holdout:", len(yh))
print("Na test set:", len(yna))


# Train DNN
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


# Train SVR
print("Training SVR...")
svr = SVR(kernel="rbf", C=100, gamma=0.01)
svr.fit(X, y)


# Train KRR
print("Training KRR...")
krr = KernelRidge(kernel="rbf", alpha=0.01, gamma=0.02)
krr.fit(X, y)


# Evaluate models
results = []
predictions = []

for name, model in [("DNN", dnn), ("SVR", svr), ("KRR", krr)]:

    # Training predictions and MAE
    train_pred = (
        model.predict(X, verbose=0).ravel()
        if name == "DNN"
        else model.predict(X)
    )
    train_mae = mean_absolute_error(y, train_pred)

    print(f"\n===== {name} =====")
    print(f"Training MAE: {train_mae:.4f} V")

    for set_name, Xt, yt in [
        ("Li_H_set", Xh, yh),
        ("Na_set", Xna, yna)
    ]:
        pred = (
            model.predict(Xt, verbose=0).ravel()
            if name == "DNN"
            else model.predict(Xt)
        )

        test_mae = mean_absolute_error(yt, pred)
        test_r2 = r2_score(yt, pred)

        print(
            f"{set_name}: "
            f"Testing MAE = {test_mae:.4f} V, "
            f"Testing R2 = {test_r2:.4f}"
        )

        results.append({
            "Model": name,
            "Training_MAE_V": train_mae,
            "Test_Set": set_name,
            "Testing_MAE_V": test_mae,
            "Testing_R2": test_r2,
            "N": len(yt)
        })

        for actual, estimate in zip(yt, pred):
            predictions.append({
                "Model": name,
                "Test_Set": set_name,
                "Actual_Voltage": actual,
                "Predicted_Voltage": estimate,
                "Absolute_Error_V": abs(actual - estimate)
            })


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("li_only_models_metrics_clean.csv", index=False)

pd.DataFrame(predictions).to_csv(
    "li_only_models_predictions_clean.csv", index=False
)

print("\n===== FINAL RESULTS =====")
print(results_df.to_string(index=False))

print("\nSaved:")
print("li_only_models_metrics_clean.csv")
print("li_only_models_predictions_clean.csv")
