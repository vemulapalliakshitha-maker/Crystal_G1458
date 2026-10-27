#this is to perform pca after normalization

import pandas as pd
from sklearn.decomposition import PCA

# Input files
mixed_file = "mixed_normalized.csv"
li_file = "li_normalized.csv"

# Output files
mixed_output = "mixed_pca.csv"
li_output = "li_pca.csv"

# 239 feature columns
feature_cols = [f"feat_{i}" for i in range(1, 240)]

# Number of principal components
n_components = 80


# ---------- MIXED DATASET ----------

mixed_df = pd.read_csv(mixed_file)

mixed_X = mixed_df[feature_cols]

pca_mixed = PCA(n_components=n_components)
mixed_pca = pca_mixed.fit_transform(mixed_X)

mixed_pca_df = pd.DataFrame(
    mixed_pca,
    columns=[f"PC_{i}" for i in range(1, 81)]
)

# Keep target and ion information
mixed_pca_df["average_voltage"] = mixed_df["average_voltage"].values
mixed_pca_df["working_ion"] = mixed_df["working_ion"].values

mixed_pca_df.to_csv(mixed_output, index=False)


# ---------- LI-ONLY DATASET ----------

li_df = pd.read_csv(li_file)

li_X = li_df[feature_cols]

pca_li = PCA(n_components=n_components)
li_pca = pca_li.fit_transform(li_X)

li_pca_df = pd.DataFrame(
    li_pca,
    columns=[f"PC_{i}" for i in range(1, 81)]
)

# Keep target and ion information
li_pca_df["average_voltage"] = li_df["average_voltage"].values
li_pca_df["working_ion"] = li_df["working_ion"].values

li_pca_df.to_csv(li_output, index=False)


# ---------- RESULTS ----------

print("PCA complete!")

print("\nMixed dataset:")
print("Original features: 239")
print("PCA features:", mixed_pca.shape[1])
print("Shape:", mixed_pca_df.shape)
print("Saved to:", mixed_output)

print("\nLi-only dataset:")
print("Original features: 239")
print("PCA features:", li_pca.shape[1])
print("Shape:", li_pca_df.shape)
print("Saved to:", li_output)

print("\nExplained variance:")
print("Mixed:", pca_mixed.explained_variance_ratio_.sum())
print("Li-only:", pca_li.explained_variance_ratio_.sum())