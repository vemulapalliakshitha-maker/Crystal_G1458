
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

df = pd.read_csv("mixed_T_set.csv")

X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

learning_rate = 0.001
l2_values = [0.00001, 0.00005, 0.0001, 0.0005]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

for l2_value in l2_values:
    fold_maes = []
    print(f"\nL2: {l2_value}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Sequential([
            Dense(60, activation="relu", input_shape=(80,),
                  kernel_regularizer=l2(l2_value)),
            Dropout(0.25),
            Dense(30, activation="relu",
                  kernel_regularizer=l2(l2_value)),
            Dropout(0.10),
            Dense(1)
        ])

        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"]
        )

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            verbose=0
        )

        pred = model.predict(X_val, verbose=0).ravel()
        mae = mean_absolute_error(y_val, pred)
        fold_maes.append(mae)

        print(f"Fold {fold}: MAE = {mae:.4f} V")

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes, ddof=1)

    results.append({
        "learning_rate": learning_rate,
        "l2": l2_value,
        "mean_MAE": mean_mae,
        "std_MAE": std_mae
    })

    print(f"Mean MAE: {mean_mae:.4f} V")
    print(f"Std MAE: {std_mae:.4f} V")

result_df = pd.DataFrame(results).sort_values("mean_MAE")
result_df.to_csv("dnn_l2_tuning_results.csv", index=False)

print("\nFINAL COMPARISON")
print(result_df.to_string(index=False))
print("\nSaved to dnn_l2_tuning_results.csv")
