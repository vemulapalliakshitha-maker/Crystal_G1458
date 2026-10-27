
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("mixed_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].values
y = df["average_voltage"].values

cv = KFold(n_splits=10, shuffle=True, random_state=42)

alphas = [0.001, 0.003, 0.01, 0.03]
gammas = [0.003, 0.005, 0.01, 0.02]

results = []

for alpha in alphas:
    for gamma in gammas:
        maes = []

        for tr, va in cv.split(X):
            model = KernelRidge(
                kernel="rbf", alpha=alpha, gamma=gamma
            )
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            maes.append(mean_absolute_error(y[va], pred))

        results.append({
            "alpha": alpha,
            "gamma": gamma,
            "mean_MAE": np.mean(maes),
            "std_MAE": np.std(maes, ddof=1)
        })

out = pd.DataFrame(results).sort_values("mean_MAE")
out.to_csv("krr_finetuning_results.csv", index=False)

print(out.to_string(index=False))
print("\nSaved: krr_finetuning_results.csv")
