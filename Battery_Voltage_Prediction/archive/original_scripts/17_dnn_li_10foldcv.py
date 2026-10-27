
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

np.random.seed(42)
tf.random.set_seed(42)

# Load Li-only training set
df = pd.read_csv("li_T_set.csv")

X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_maes = []

for fold, (tr, va) in enumerate(kf.split(X), start=1):
    tf.keras.backend.clear_session()
    tf.random.set_seed(42 + fold)

    # Fresh DNN for each fold
    model = Sequential([
        Input(shape=(80,)),
        Dense(60, activation="relu",
              kernel_regularizer=l2(0.0001)),
        Dropout(0.25),
        Dense(30, activation="relu",
              kernel_regularizer=l2(0.0001)),
        Dropout(0.10),
        Dense(1)
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="mae"
    )

    # Train on 9 folds
    model.fit(
        X[tr], y[tr],
        epochs=100,
        batch_size=32,
        verbose=0
    )

    # Validate on the remaining fold
    pred = model.predict(X[va], verbose=0).flatten()
    mae = mean_absolute_error(y[va], pred)
    fold_maes.append(mae)

    print(f"Fold {fold:02d} MAE: {mae:.4f} V")

# Summary
mean_mae = np.mean(fold_maes)
std_mae = np.std(fold_maes, ddof=1)

print("\n===== Li-only DNN 10-Fold CV =====")
print(f"Mean MAE: {mean_mae:.4f} V")
print(f"Std MAE:  {std_mae:.4f} V")
print(f"Result: {mean_mae:.4f} ± {std_mae:.4f} V")

# Save fold results
pd.DataFrame({
    "Fold": range(1, 11),
    "MAE_V": fold_maes
}).to_csv("dnn_li_10fold_results.csv", index=False)

print("\nSaved: dnn_li_10fold_results.csv")
