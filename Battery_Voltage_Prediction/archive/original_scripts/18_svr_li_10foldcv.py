
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

df = pd.read_csv("li_T_set.csv")
X = df[[f"PC_{i}" for i in range(1, 81)]].to_numpy()
y = df["average_voltage"].to_numpy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
maes = []

for fold, (tr, va) in enumerate(kf.split(X), 1):
    model = SVR(kernel="rbf", C=100, gamma=0.01)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    mae = mean_absolute_error(y[va], pred)
    maes.append(mae)
    print(f"Fold {fold:02d} MAE: {mae:.4f} V")

print(f"\nMean MAE: {np.mean(maes):.4f} V")
print(f"Std MAE:  {np.std(maes, ddof=1):.4f} V")

pd.DataFrame({"Fold": range(1, 11), "MAE_V": maes}).to_csv(
    "svr_li_10fold_results.csv", index=False
)
