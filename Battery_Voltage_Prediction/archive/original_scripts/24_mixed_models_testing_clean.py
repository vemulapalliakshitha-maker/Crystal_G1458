
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

target = "average_voltage"
features = [f"PC_{i}" for i in range(1, 81)]

# Load datasets
train = pd.read_csv("mixed_train_pca_clean.csv")
holdout = pd.read_csv("mixed_holdout_pca_clean.csv")
sodium = pd.read_csv("na_pca_mixed_clean.csv")

X_train = train[features].to_numpy(dtype=float)
y_train = train[target].to_numpy(dtype=float)

X_holdout = holdout[features].to_numpy(dtype=float)
y_holdout = holdout[target].to_numpy(dtype=float)

X_na = sodium[features].to_numpy(dtype=float)
y_na = sodium[target].to_numpy(dtype=float)

# Model definitions
def build_dnn():
    model = Sequential([
        Input(shape=(80,)),
        Dense(60, activation="relu", kernel_regularizer=l2(0.0001)),
        Dropout(0.25),
        Dense(30, activation="relu", kernel_regularizer=l2(0.0001)),
        Dropout(0.10),
        Dense(1)
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="mae"
    )
    return model

models = {
    "DNN": build_dnn(),
    "SVR": SVR(kernel="rbf", C=100, gamma=0.01),
    "KRR": KernelRidge(kernel="rbf", alpha=0.01, gamma=0.02)
}

results = []
predictions = []

def evaluate(model_name, dataset_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(
        f"{dataset_name}: MAE = {mae:.4f} V, "
        f"R2 = {r2:.4f}"
    )

    results.append({
        "Model": model_name,
        "Dataset": dataset_name,
        "N": len(y_true),
        "MAE_V": mae,
        "R2": r2
    })

    for actual, predicted in zip(y_true, y_pred):
        predictions.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Actual_V": actual,
            "Predicted_V": predicted
        })

# Train and evaluate each model
for name, model in models.items():
    print(f"\n===== {name} =====")

    if name == "DNN":
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            verbose=0
        )

        train_pred = model.predict(X_train, verbose=0).ravel()
        holdout_pred = model.predict(X_holdout, verbose=0).ravel()
        na_pred = model.predict(X_na, verbose=0).ravel()

    else:
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        holdout_pred = model.predict(X_holdout)
        na_pred = model.predict(X_na)

    train_mae = mean_absolute_error(y_train, train_pred)
    print(f"Training MAE = {train_mae:.4f} V")

    results.append({
        "Model": name,
        "Dataset": "Training",
        "N": len(y_train),
        "MAE_V": train_mae,
        "R2": r2_score(y_train, train_pred)
    })

    evaluate(name, "Mixed_Holdout", y_holdout, holdout_pred)
    evaluate(name, "Na_Set", y_na, na_pred)

    # Save actual vs predicted values
    for dataset_name, actual, predicted in [
        ("Training", y_train, train_pred),
        ("Mixed_Holdout", y_holdout, holdout_pred),
        ("Na_Set", y_na, na_pred)
    ]:
        for a, p in zip(actual, predicted):
            predictions.append({
                "Model": name,
                "Dataset": dataset_name,
                "Actual_V": a,
                "Predicted_V": p
            })

# Save results
pd.DataFrame(results).to_csv(
    "mixed_models_metrics_clean.csv", index=False
)
pd.DataFrame(predictions).to_csv(
    "mixed_models_predictions_clean.csv", index=False
)

print("\nResults saved:")
print("mixed_models_metrics_clean.csv")
print("mixed_models_predictions_clean.csv")
