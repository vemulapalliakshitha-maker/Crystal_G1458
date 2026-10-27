
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

target = "average_voltage"
features = [f"PC_{i}" for i in range(1, 81)]

train = pd.read_csv("mixed_train_pca_clean.csv")
holdout = pd.read_csv("mixed_holdout_pca_clean.csv")
na = pd.read_csv("na_pca_mixed_clean.csv")

X = train[features].to_numpy(dtype=np.float32)
y = train[target].to_numpy(dtype=np.float32)

X_holdout = holdout[features].to_numpy(dtype=np.float32)
y_holdout = holdout[target].to_numpy(dtype=np.float32)

X_na = na[features].to_numpy(dtype=np.float32)
y_na = na[target].to_numpy(dtype=np.float32)

# Validation split from training data only
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Mixed holdout rows:", len(X_holdout))
print("Sodium rows:", len(X_na))

# --------------------------------------------------
# 2. DNN CONFIGURATIONS
# --------------------------------------------------

configs = [
    {
        "name": "DNN_1",
        "layers": [60, 30],
        "dropout": [0.25, 0.10],
        "l2": 0.0001,
        "optimizer": "RMSprop",
        "lr": 0.001,
        "batch": 32
    },
    {
        "name": "DNN_2",
        "layers": [60, 30],
        "dropout": [0.10, 0.05],
        "l2": 0.0001,
        "optimizer": "Adam",
        "lr": 0.001,
        "batch": 32
    },
    {
        "name": "DNN_3",
        "layers": [60, 30],
        "dropout": [0.25, 0.10],
        "l2": 0.0001,
        "optimizer": "Adam",
        "lr": 0.0005,
        "batch": 32
    },
    {
        "name": "DNN_4",
        "layers": [80, 40],
        "dropout": [0.20, 0.10],
        "l2": 0.0001,
        "optimizer": "Adam",
        "lr": 0.0005,
        "batch": 32
    },
    {
        "name": "DNN_5",
        "layers": [128, 64],
        "dropout": [0.25, 0.10],
        "l2": 0.0001,
        "optimizer": "Adam",
        "lr": 0.0005,
        "batch": 32
    },
    {
        "name": "DNN_6",
        "layers": [128, 64],
        "dropout": [0.30, 0.20],
        "l2": 0.001,
        "optimizer": "Adam",
        "lr": 0.0005,
        "batch": 64
    }
]

# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

def build_model(config):

    model = Sequential()
    model.add(Input(shape=(80,)))

    for neurons, dropout_rate in zip(
        config["layers"], config["dropout"]
    ):
        model.add(
            Dense(
                neurons,
                activation="relu",
                kernel_regularizer=l2(config["l2"])
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    if config["optimizer"] == "Adam":
        optimizer = Adam(learning_rate=config["lr"])
    else:
        optimizer = RMSprop(learning_rate=config["lr"])

    model.compile(
        optimizer=optimizer,
        loss="mae",
        metrics=["mae"]
    )

    return model

# --------------------------------------------------
# 4. TRAIN AND TUNE
# --------------------------------------------------

results = []
best_model = None
best_config = None
best_val_mae = float("inf")

for config in configs:

    print("\n" + "=" * 50)
    print("Testing:", config["name"])
    print("=" * 50)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    model = build_model(config)

    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=config["batch"],
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate using validation data
    val_pred = model.predict(X_val, verbose=0).ravel()

    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)

    train_pred = model.predict(X_train, verbose=0).ravel()
    train_mae = mean_absolute_error(y_train, train_pred)

    best_epoch = len(history.history["loss"])

    print("Training MAE:", round(train_mae, 4))
    print("Validation MAE:", round(val_mae, 4))
    print("Validation R2:", round(val_r2, 4))
    print("Epochs trained:", best_epoch)

    results.append({
        "Configuration": config["name"],
        "Training_MAE": train_mae,
        "Validation_MAE": val_mae,
        "Validation_R2": val_r2,
        "Epochs": best_epoch
    })

    # Select best model using validation MAE
    if val_mae < best_val_mae:

        best_val_mae = val_mae
        best_model = model
        best_config = config

# --------------------------------------------------
# 5. SAVE TUNING RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Validation_MAE")

results_df.to_csv(
    "dnn_mixed_tuning_results.csv",
    index=False
)

print("\n" + "=" * 50)
print("BEST DNN CONFIGURATION")
print("=" * 50)

print("Configuration:", best_config["name"])
print("Validation MAE:", round(best_val_mae, 4))

print("\nAll tuning results:")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 6. FINAL EVALUATION
# --------------------------------------------------

print("\n" + "=" * 50)
print("FINAL BEST DNN EVALUATION")
print("=" * 50)

# Training performance
train_pred = best_model.predict(X, verbose=0).ravel()

train_mae = mean_absolute_error(y, train_pred)

print("\nTraining MAE:", round(train_mae, 4), "V")

# Mixed holdout
holdout_pred = best_model.predict(
    X_holdout, verbose=0
).ravel()

holdout_mae = mean_absolute_error(
    y_holdout, holdout_pred
)

holdout_r2 = r2_score(
    y_holdout, holdout_pred
)

print("\nMixed Holdout:")
print("MAE:", round(holdout_mae, 4), "V")
print("R2:", round(holdout_r2, 4))

# Sodium evaluation
na_pred = best_model.predict(X_na, verbose=0).ravel()

na_mae = mean_absolute_error(y_na, na_pred)
na_r2 = r2_score(y_na, na_pred)

print("\nSodium Set:")
print("MAE:", round(na_mae, 4), "V")
print("R2:", round(na_r2, 4))

# --------------------------------------------------
# 7. SAVE FINAL MODEL AND METRICS
# --------------------------------------------------

best_model.save("best_mixed_dnn.keras")

metrics = pd.DataFrame([
    {
        "Model": "Best_DNN",
        "Dataset": "Training",
        "N": len(y),
        "MAE": train_mae,
        "R2": r2_score(y, train_pred)
    },
    {
        "Model": "Best_DNN",
        "Dataset": "Mixed_Holdout",
        "N": len(y_holdout),
        "MAE": holdout_mae,
        "R2": holdout_r2
    },
    {
        "Model": "Best_DNN",
        "Dataset": "Na_Set",
        "N": len(y_na),
        "MAE": na_mae,
        "R2": na_r2
    }
])

metrics.to_csv(
    "best_mixed_dnn_metrics.csv",
    index=False
)

# Save predictions
predictions = pd.concat([
    pd.DataFrame({
        "Dataset": "Mixed_Holdout",
        "Actual_V": y_holdout,
        "Predicted_V": holdout_pred
    }),
    pd.DataFrame({
        "Dataset": "Na_Set",
        "Actual_V": y_na,
        "Predicted_V": na_pred
    })
], ignore_index=True)

predictions.to_csv(
    "best_mixed_dnn_predictions.csv",
    index=False
)

print("\nSaved files:")
print("dnn_mixed_tuning_results.csv")
print("best_mixed_dnn.keras")
print("best_mixed_dnn_metrics.csv")
print("best_mixed_dnn_predictions.csv")

print("\nDNN tuning completed.")
