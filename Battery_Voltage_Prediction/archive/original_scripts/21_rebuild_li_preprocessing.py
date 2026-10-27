
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib

# Load raw feature data
mixed = pd.read_csv("remaining_features.csv")
na = pd.read_csv("na_features.csv")

features = [f"feat_{i}" for i in range(1, 240)]
target = "average_voltage"

# Select lithium rows from the raw mixed dataset
ion = mixed["working_ion"].astype(str).str.strip().str.lower()
li = mixed[ion.isin(["li", "lithium"])].copy()

print("Li rows found:", len(li))
print("Na rows found:", len(na))

if len(li) == 0:
    print("Working-ion values found:", mixed["working_ion"].unique())
    raise ValueError("Could not identify Li rows. Check the values printed above.")

# Keep valid rows and preserve original feature columns
li = li.dropna(subset=features + [target]).reset_index(drop=True)
na = na.dropna(subset=features + [target]).reset_index(drop=True)

# Split raw Li data BEFORE fitting the scaler or PCA
li_train, li_holdout = train_test_split(
    li, test_size=0.10, random_state=42
)

# Fit preprocessing ONLY on Li training features
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(li_train[features])

X_holdout_scaled = scaler.transform(li_holdout[features])
X_na_scaled = scaler.transform(na[features])

pca = PCA(n_components=80)
X_train_pca = pca.fit_transform(X_train_scaled)

X_holdout_pca = pca.transform(X_holdout_scaled)
X_na_pca = pca.transform(X_na_scaled)

pc_cols = [f"PC_{i}" for i in range(1, 81)]

def save_data(X, df, filename):
    out = pd.DataFrame(X, columns=pc_cols)
    out[target] = df[target].to_numpy()
    out.to_csv(filename, index=False)
    print("Saved:", filename, "| Rows:", len(out))

save_data(X_train_pca, li_train, "li_train_pca_clean.csv")
save_data(X_holdout_pca, li_holdout, "li_holdout_pca_clean.csv")
save_data(X_na_pca, na, "na_pca_li_clean.csv")

joblib.dump(scaler, "li_scaler_clean.joblib")
joblib.dump(pca, "li_pca_model_clean.joblib")

print("Retained PCA variance:",
      round(pca.explained_variance_ratio_.sum(), 6))
print("Preprocessing complete.")
