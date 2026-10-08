# here we do the normalization:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Input files
mixed_file = "remaining_features.csv"
li_file = "li_features.csv"

# Output files
mixed_output = "mixed_normalized.csv"
li_output = "li_normalized.csv"

# Read datasets
mixed_df = pd.read_csv(mixed_file)
li_df = pd.read_csv(li_file)

# 239 input feature columns
feature_cols = [f"feat_{i}" for i in range(1, 240)]

# Create scaler for -1 to +1
scaler = MinMaxScaler(feature_range=(-1, 1))

# Normalize mixed dataset
mixed_df[feature_cols] = scaler.fit_transform(mixed_df[feature_cols])

# Normalize Li-only dataset separately
li_scaler = MinMaxScaler(feature_range=(-1, 1))
li_df[feature_cols] = li_scaler.fit_transform(li_df[feature_cols])

# Save as NEW files
mixed_df.to_csv(mixed_output, index=False)
li_df.to_csv(li_output, index=False)

print("Normalization complete!")

print("\nMixed dataset:")
print("Shape:", mixed_df.shape)
print("Saved to:", mixed_output)

print("\nLi-only dataset:")
print("Shape:", li_df.shape)
print("Saved to:", li_output)

print("\nVoltage columns were NOT normalized.")