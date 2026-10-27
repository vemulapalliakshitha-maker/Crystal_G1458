
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("mixed_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].values
y = df["average_voltage"].values

cv = KFold(n_splits=10, shuffle=True, random_state=42)

params = [
    (1, "scale"),
    (10, 0.1),
    (10, 0.01),
    (10, 0.5),
    (100, 0.1),
    (100, 0.01),
    (100, 0.5),
]

results = []

for C, gamma in params:
    maes = []

    for train_idx, val_idx in cv.split(X):
        model = SVR(kernel="rbf", C=C, gamma=gamma)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], pred))

    results.append({
        "C": C,
        "gamma": gamma,
        "mean_MAE": np.mean(maes),
        "std_MAE": np.std(maes, ddof=1)
    })

    print(f"C={C}, gamma={gamma}: "
          f"MAE={np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f} V")

out = pd.DataFrame(results).sort_values("mean_MAE")
out.to_csv("svr_tuning_results.csv", index=False)

print("\nFINAL COMPARISON")
print(out.to_string(index=False))
print("\nSaved: svr_tuning_results.csv")
