
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge

np.random.seed(42)

# 1. Load the mixed-ion training dataset
df = pd.read_csv("mixed_T_set.csv")

X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

# 2. Create 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_maes = []

# 3. Train and validate KRR on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = KernelRidge(
        kernel="rbf",
        alpha=1.0,
        gamma=None
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    fold_maes.append(mae)

    print(f"Fold {fold:02d} MAE: {mae:.4f} V")

# 4. Calculate mean and standard deviation
mean_mae = np.mean(fold_maes)
std_mae = np.std(fold_maes, ddof=1)

print("\n===== KRR 10-Fold CV Results =====")
print(f"Mean MAE: {mean_mae:.4f} V")
print(f"Std MAE:  {std_mae:.4f} V")
print(f"Result:   {mean_mae:.4f} ± {std_mae:.4f} V")

# 5. Save the fold results
pd.DataFrame({
    "Fold": range(1, 11),
    "MAE_V": fold_maes
}).to_csv("krr_10fold_results.csv", index=False)

print("\nSaved results to krr_10fold_results.csv")
