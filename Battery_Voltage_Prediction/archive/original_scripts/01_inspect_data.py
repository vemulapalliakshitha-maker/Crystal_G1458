import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Neha chandrika\OneDrive\Desktop\PS1\data_2019_12_03.csv")

print("========== BASIC INFORMATION ==========")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


print("\n========== DUPLICATE ROWS ==========")
print("Duplicate rows:", df.duplicated().sum())


print("\n========== WORKING IONS ==========")
print(df["working_ion"].value_counts())


print("\n========== VOLTAGE STATISTICS ==========")
print(df["average_voltage"].describe())


print("\n========== LOWEST VOLTAGE RECORDS ==========")
print(
    df.nsmallest(10, "average_voltage")[
        ["battery_id", "formula_charge", "formula_discharge",
         "working_ion", "average_voltage"]
    ]
)


print("\n========== HIGHEST VOLTAGE RECORDS ==========")
print(
    df.nlargest(10, "average_voltage")[
        ["battery_id", "formula_charge", "formula_discharge",
         "working_ion", "average_voltage"]
    ]
)


print("\n========== UNIQUE BATTERY IDs ==========")
print("Unique battery IDs:", df["battery_id"].nunique())


print("\n========== UNIQUE CHARGE/DISCHARGE PAIRS ==========")
print(
    df[["formula_charge", "formula_discharge"]]
    .drop_duplicates()
    .shape[0]
)

# ============================================================
# REPEATED RECORD CHECK
# ============================================================

print("\n========== REPEATED BATTERY IDs ==========")

battery_counts = df["battery_id"].value_counts()

print("Battery IDs appearing more than once:")
print(battery_counts[battery_counts > 1].head(20))

print("\nNumber of repeated battery IDs:",
      (battery_counts > 1).sum())


print("\n========== REPEATED MATERIAL PAIRS ==========")

pair_counts = df.groupby(
    ["id_charge", "id_discharge"]
).size()

print("Repeated material pairs:")
print(pair_counts[pair_counts > 1].head(20))

print("\nNumber of repeated material pairs:",
      (pair_counts > 1).sum())


print("\n========== INDEX COLUMN ==========")

print("First 5 index values:")
print(df["Unnamed: 0"].head().tolist())

print("Last 5 index values:")
print(df["Unnamed: 0"].tail().tolist())


# ============================================================
# DATA CLEANING
# ============================================================

print("\n========== CLEANING ==========")

# Remove the unnecessary CSV index column
df_clean = df.drop(columns=["Unnamed: 0"])

print("Removed 'Unnamed: 0' column.")

# Remove completely duplicated rows
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
after = len(df_clean)

print("Duplicate rows removed:", before - after)

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned dataset shape:", df_clean.shape)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

output_file = r"C:\Users\Neha chandrika\OneDrive\Desktop\PS1\cleaned_dataset.csv"

df_clean.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:")
print(output_file)

print("\n========== PAPER ION GROUP ==========")

paper_ions = ["Li", "Ca", "Mg", "Zn", "Y", "Al"]

paper_df = df_clean[df_clean["working_ion"].isin(paper_ions)]

print("Records for paper's main ion group:")
print(paper_df["working_ion"].value_counts())

print("\nTotal records:", len(paper_df))

print("\nOther ions:")
print(
    df_clean[
        ~df_clean["working_ion"].isin(paper_ions)
    ]["working_ion"].value_counts()
)