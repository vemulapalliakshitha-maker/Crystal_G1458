import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Files
feature_cols = [f"feat_{i}" for i in range(1, 240)]

train_data = pd.read_csv("remaining_features.csv")
na_data = pd.read_csv("na_features.csv")

# Recreate the original mixed-data scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data[feature_cols])

# Normalize Na features using the mixed-data scaler
na_scaled = scaler.transform(na_data[feature_cols])

# Recreate the original PCA from mixed normalized data
mixed_data = pd.read_csv("mixed_normalized.csv")
pca = PCA(n_components=80)
pca.fit(mixed_data[feature_cols])

# Transform Na features
na_pca = pca.transform(na_scaled)

# Save reusable preprocessing objects
joblib.dump(scaler, "mixed_scaler.joblib")
joblib.dump(pca, "mixed_pca_model.joblib")

# Save transformed Na dataset
output = pd.DataFrame(
    na_pca,
    columns=[f"PC_{i}" for i in range(1, 81)]
)
output["average_voltage"] = na_data["average_voltage"].values
output["working_ion"] = na_data["working_ion"].values
output["source_row"] = na_data["source_row"].values

output.to_csv("na_pca.csv", index=False)

print("Na preprocessing complete!")
print("Output shape:", na_pca.shape)
print("Saved: na_pca.csv")
print("Explained variance:", pca.explained_variance_ratio_.sum())