import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# 1. Na instances
na_df = df[df["working_ion"] == "Na"]
na_df.to_csv("na_instances.csv", index=False)

# 2. K instances
k_df = df[df["working_ion"] == "K"]
k_df.to_csv("k_instances.csv", index=False)

# 3. Remaining ions
remaining_ions = ["Li", "Ca", "Mg", "Zn", "Al", "Y", "Rb", "Cs"]

remaining_df = df[df["working_ion"].isin(remaining_ions)]
remaining_df.to_csv("remaining_instances.csv", index=False)

# Display results
print("Na instances:", len(na_df))
print("K instances:", len(k_df))
print("Remaining instances:", len(remaining_df))

print("\nRemaining ion counts:")
print(remaining_df["working_ion"].value_counts())

print("\nTotal:")
print(len(na_df) + len(k_df) + len(remaining_df))
print("Original dataset:", len(df))