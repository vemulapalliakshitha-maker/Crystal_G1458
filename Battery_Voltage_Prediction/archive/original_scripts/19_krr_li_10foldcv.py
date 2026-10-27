
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge

df = pd.read_csv("li_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
maes = []

for fold, (tr, va) in enumerate(kf.split(X), 1):
    model = KernelRidge(kernel="rbf", alpha=0.01, gamma=0.02)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    mae = mean_absolute_error(y[va], pred)
    maes.append(mae)
    print(f"Fold {fold:02d} MAE: {mae:.4f} V")

print(f"\nMean MAE: {np.mean(maes):.4f} V")
print(f"Std MAE:  {np.std(maes, ddof=1):.4f} V")

pd.DataFrame({"Fold": range(1, 11), "MAE_V": maes}).to_csv(
    "krr_li_10fold_results.csv", index=False
)
