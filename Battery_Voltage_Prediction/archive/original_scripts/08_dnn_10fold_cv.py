
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

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load mixed-ion T-set
df = pd.read_csv("mixed_T_set.csv")

# Input: 80 PCA components
X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_maes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create a fresh DNN for every fold
    tf.keras.backend.clear_session()
    tf.random.set_seed(42 + fold)

    model = Sequential([
        Input(shape=(80,)),
        Dense(60, activation="relu",
              kernel_regularizer=l2(0.001)),
        Dropout(0.25),
        Dense(30, activation="relu",
              kernel_regularizer=l2(0.001)),
        Dropout(0.10),
        Dense(1)
    ])

    model.compile(
        optimizer=RMSprop(),
        loss="mse",
        metrics=["mae"]
    )

    # Train on 9 folds
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=0
    )

    # Predict the remaining fold
    predictions = model.predict(X_val, verbose=0).flatten()

    mae = mean_absolute_error(y_val, predictions)
    fold_maes.append(mae)

    print(f"Fold {fold:02d} MAE: {mae:.4f} V")

# Calculate mean and standard deviation
mean_mae = np.mean(fold_maes)
std_mae = np.std(fold_maes, ddof=1)

print("\n===== DNN 10-Fold CV Results =====")
print(f"Mean MAE: {mean_mae:.4f} V")
print(f"Std MAE:  {std_mae:.4f} V")
print(f"Result:   {mean_mae:.4f} ± {std_mae:.4f} V")

# Save fold results
results = pd.DataFrame({
    "Fold": range(1, 11),
    "MAE_V": fold_maes
})

results.to_csv("dnn_10fold_results.csv", index=False)
print("\nSaved results to dnn_10fold_results.csv")