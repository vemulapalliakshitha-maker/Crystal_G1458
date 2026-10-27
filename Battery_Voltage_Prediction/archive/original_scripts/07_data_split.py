#after pca we are going to split the data into 90 and 10 percent

import pandas as pd
from sklearn.model_selection import train_test_split

# Input files
mixed_file = "mixed_pca.csv"
li_file = "li_pca.csv"

# Output files
mixed_t = "mixed_T_set.csv"
mixed_h = "mixed_H_set.csv"

li_t = "li_T_set.csv"
li_h = "li_H_set.csv"

# Reproducible split
RANDOM_STATE = 42


# ---------- MIXED DATASET ----------

mixed_df = pd.read_csv(mixed_file)

mixed_T, mixed_H = train_test_split(
    mixed_df,
    test_size=0.10,
    random_state=RANDOM_STATE
)

mixed_T.to_csv(mixed_t, index=False)
mixed_H.to_csv(mixed_h, index=False)


# ---------- LI-ONLY DATASET ----------

li_df = pd.read_csv(li_file)

li_T, li_H = train_test_split(
    li_df,
    test_size=0.10,
    random_state=RANDOM_STATE
)

li_T.to_csv(li_t, index=False)
li_H.to_csv(li_h, index=False)


# ---------- RESULTS ----------

print("Data splitting complete!")

print("\nMixed dataset:")
print("Original:", mixed_df.shape)
print("T-set:", mixed_T.shape)
print("H-set:", mixed_H.shape)

print("\nLi-only dataset:")
print("Original:", li_df.shape)
print("T-set:", li_T.shape)
print("H-set:", li_H.shape)

print("\nFiles created:")
print(mixed_t)
print(mixed_h)
print(li_t)
print(li_h)
