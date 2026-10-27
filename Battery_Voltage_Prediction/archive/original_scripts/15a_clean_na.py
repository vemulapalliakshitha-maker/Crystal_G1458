
import pandas as pd

df = pd.read_excel("na_external_test.xlsx")

# Remove spaces from column names and text
df.columns = df.columns.str.strip()

for col in ["high_formula", "low_formula", "working_ion"]:
    df[col] = df[col].astype(str).str.strip()

# Convert voltage columns to numeric
voltage_cols = [
    "dft_voltage_V",
    "paper_DNN_voltage_V",
    "paper_KRR_voltage_V",
    "paper_SVR_voltage_V"
]

for col in voltage_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid or incomplete rows
df = df.dropna(subset=[
    "high_formula", "low_formula", "working_ion",
    *voltage_cols
])

df = df[df["working_ion"].str.lower() == "na"]
df = df.drop_duplicates().reset_index(drop=True)

df.to_csv("na_cleaned.csv", index=False)

print("Cleaning complete!")
print("Shape:", df.shape)
print("Saved as na_cleaned.csv")
