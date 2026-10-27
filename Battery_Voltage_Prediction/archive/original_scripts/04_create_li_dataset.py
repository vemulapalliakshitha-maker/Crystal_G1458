#this file will create a lionly data from mixed dataset after the feature extraction

import pandas as pd

# Input: existing mixed dataset
input_file = "remaining_features.csv"

# Output: new Li-only dataset
output_file = "li_features.csv"

# Read the mixed dataset
df = pd.read_csv(input_file)

# Create a NEW dataframe containing only Li instances
li_df = df[df["working_ion"] == "Li"].copy()

# Save as a separate file
li_df.to_csv(output_file, index=False)

print("Li-only dataset created!")
print("Shape:", li_df.shape)
print("Working ions:")
print(li_df["working_ion"].value_counts())
print("Saved to:", output_file)