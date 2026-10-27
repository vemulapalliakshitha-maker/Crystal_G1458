
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("mixed_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].values
y = df["average_voltage"].values

cv = KFold(n_splits=10, shuffle=True, random_state=42)

params = [
    (1.0, None),
    (0.01, 0.1),   # Paper settings
    (0.1, 0.1),
    (0.001, 0.1),
    (0.01, 0.01),
    (0.01, 0.5),
    (0.1, 0.01),
    (0.1, 0.5),
]

results = []

for alpha, gamma in params:
    maes = []

    for train_idx, val_idx in cv.split(X):
        model = KernelRidge(
            kernel="rbf", alpha=alpha, gamma=gamma
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], pred))

    results.append({
        "alpha": alpha,
        "gamma": gamma,
        "mean_MAE": np.mean(maes),
        "std_MAE": np.std(maes, ddof=1)
    })

    print(
        f"alpha={alpha}, gamma={gamma}: "
        f"MAE={np.mean(maes):.4f} ± "
        f"{np.std(maes, ddof=1):.4f} V"
    )

out = pd.DataFrame(results).sort_values("mean_MAE")
out.to_csv("krr_tuning_results.csv", index=False)

print("\nFINAL COMPARISON")
print(out.to_string(index=False))
print("\nSaved: krr_tuning_results.csv")
