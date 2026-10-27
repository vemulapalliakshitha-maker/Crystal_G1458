
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("mixed_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].values
y = df["average_voltage"].values

cv = KFold(n_splits=10, shuffle=True, random_state=42)
params = [(100, 0.001), (100, 0.005), (100, 0.01)]

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

out = pd.DataFrame(results).sort_values("mean_MAE")
out.to_csv("svr_fine_tuning_results.csv", index=False)

print("\nFINE-TUNING RESULTS")
print(out.to_string(index=False))
print("\nSaved: svr_fine_tuning_results.csv")
