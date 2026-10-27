
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Files
mixed_file = "remaining_features.csv"
na_file = "na_features.csv"

target = "average_voltage"
features = [f"feat_{i}" for i in range(1, 240)]
pc_cols = [f"PC_{i}" for i in range(1, 81)]

# Load datasets
mixed = pd.read_csv(mixed_file)
na = pd.read_csv(na_file)

# Check required columns
for name, df in [("Mixed", mixed), ("Na", na)]:
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {missing}")

# Keep the same feature order in both datasets
mixed = mixed.dropna(subset=features + [target]).copy()
na = na.dropna(subset=features + [target]).copy()

X = mixed[features].astype(float)
y = mixed[target].astype(float)

X_na = na[features].astype(float)
y_na = na[target].astype(float)

# Split mixed data: 90% train, 10% holdout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Fit scaler ONLY on mixed training data
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_na_scaled = scaler.transform(X_na)

# Fit PCA ONLY on mixed training data
pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_na_pca = pca.transform(X_na_scaled)

# Save transformed datasets
def save_data(filename, X_pca, y_values, original=None):
    result = pd.DataFrame(X_pca, columns=pc_cols)
    result[target] = np.asarray(y_values)

    if original is not None:
        if "working_ion" in original.columns:
            result["working_ion"] = original["working_ion"].to_numpy()
        if "source_row" in original.columns:
            result["source_row"] = original["source_row"].to_numpy()

    result.to_csv(filename, index=False)

save_data("mixed_train_pca_clean.csv", X_train_pca, y_train)
save_data("mixed_holdout_pca_clean.csv", X_test_pca, y_test)
save_data("na_pca_mixed_clean.csv", X_na_pca, y_na, na)

# Save fitted preprocessing objects
joblib.dump(scaler, "mixed_scaler_clean.joblib")
joblib.dump(pca, "mixed_pca_model_clean.joblib")

print("Preprocessing completed successfully.")
print("Mixed training rows:", len(X_train))
print("Mixed holdout rows:", len(X_test))
print("Sodium rows:", len(X_na))
print("PCA components:", pca.n_components_)
print(
    "Variance retained:",
    round(pca.explained_variance_ratio_.sum(), 6)
)
print("Saved all three transformed datasets.")
