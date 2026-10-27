
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

features = [f"PC_{i}" for i in range(1, 81)]
target = "average_voltage"

train = pd.read_csv("mixed_train_pca_clean.csv")
holdout = pd.read_csv("mixed_holdout_pca_clean.csv")
na = pd.read_csv("na_pca_mixed_clean.csv")

X = train[features].to_numpy(dtype=np.float32)
y = train[target].to_numpy(dtype=np.float32)

X_holdout = holdout[features].to_numpy(dtype=np.float32)
y_holdout = holdout[target].to_numpy(dtype=np.float32)

X_na = na[features].to_numpy(dtype=np.float32)
y_na = na[target].to_numpy(dtype=np.float32)

# Validation split from mixed training data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

# --------------------------------------------------
# 2. BUILD DNN_1
# --------------------------------------------------

def build_model(loss):

    model = Sequential([
        Input(shape=(80,)),

        Dense(
            60,
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        Dropout(0.25),

        Dense(
            30,
            activation="relu",
            kernel_regularizer=l2(0.0001)
        ),
        Dropout(0.10),

        Dense(1)
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss=loss,
        metrics=["mae"]
    )

    return model

# --------------------------------------------------
# 3. LOSS FUNCTIONS
# --------------------------------------------------

losses = {
    "MAE": "mae",
    "MSE": "mse",
    "Huber": tf.keras.losses.Huber(delta=1.0)
}

results = []

best_model = None
best_loss = None
best_val_mae = float("inf")

# --------------------------------------------------
# 4. TRAIN AND COMPARE
# --------------------------------------------------

for name, loss in losses.items():

    print("\n" + "=" * 45)
    print("Testing loss:", name)
    print("=" * 45)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    model = build_model(loss)

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
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Training predictions
    train_pred = model.predict(
        X_train, verbose=0
    ).ravel()

    train_mae = mean_absolute_error(
        y_train, train_pred
    )

    # Validation predictions
    val_pred = model.predict(
        X_val, verbose=0
    ).ravel()

    val_mae = mean_absolute_error(
        y_val, val_pred
    )

    val_r2 = r2_score(
        y_val, val_pred
    )

    print("Training MAE:", round(train_mae, 4))
    print("Validation MAE:", round(val_mae, 4))
    print("Validation R2:", round(val_r2, 4))
    print("Epochs:", len(history.history["loss"]))

    results.append({
        "Loss": name,
        "Training_MAE": train_mae,
        "Validation_MAE": val_mae,
        "Validation_R2": val_r2,
        "Epochs": len(history.history["loss"])
    })

    # Select using validation MAE only
    if val_mae < best_val_mae:

        best_val_mae = val_mae
        best_model = model
        best_loss = name

# --------------------------------------------------
# 5. SAVE TUNING RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Validation_MAE"
)

results_df.to_csv(
    "dnn_loss_tuning_results.csv",
    index=False
)

print("\n" + "=" * 45)
print("BEST LOSS")
print("=" * 45)

print("Best loss:", best_loss)
print("Validation MAE:", round(best_val_mae, 4))

print("\nAll results:")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 6. FINAL EVALUATION
# --------------------------------------------------

print("\n" + "=" * 45)
print("FINAL DNN EVALUATION")
print("=" * 45)

# Training data
train_pred = best_model.predict(
    X, verbose=0
).ravel()

train_mae = mean_absolute_error(
    y, train_pred
)

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

# Sodium
na_pred = best_model.predict(
    X_na, verbose=0
).ravel()

na_mae = mean_absolute_error(
    y_na, na_pred
)

na_r2 = r2_score(
    y_na, na_pred
)

print("\nSodium Set:")
print("MAE:", round(na_mae, 4), "V")
print("R2:", round(na_r2, 4))

# --------------------------------------------------
# 7. SAVE MODEL, METRICS AND PREDICTIONS
# --------------------------------------------------

best_model.save("best_mixed_dnn_loss.keras")

metrics = pd.DataFrame([
    {
        "Model": "DNN",
        "Loss": best_loss,
        "Dataset": "Training",
        "N": len(y),
        "MAE": train_mae,
        "R2": r2_score(y, train_pred)
    },
    {
        "Model": "DNN",
        "Loss": best_loss,
        "Dataset": "Mixed_Holdout",
        "N": len(y_holdout),
        "MAE": holdout_mae,
        "R2": holdout_r2
    },
    {
        "Model": "DNN",
        "Loss": best_loss,
        "Dataset": "Na_Set",
        "N": len(y_na),
        "MAE": na_mae,
        "R2": na_r2
    }
])

metrics.to_csv(
    "best_mixed_dnn_loss_metrics.csv",
    index=False
)

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
    "best_mixed_dnn_loss_predictions.csv",
    index=False
)

print("\nSaved:")
print("dnn_loss_tuning_results.csv")
print("best_mixed_dnn_loss.keras")
print("best_mixed_dnn_loss_metrics.csv")
print("best_mixed_dnn_loss_predictions.csv")

print("\nLoss tuning completed.")
