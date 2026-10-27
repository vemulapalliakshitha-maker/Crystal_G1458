
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
matches = pd.read_csv(BASE / "na_structure_matches.csv")
na = pd.read_csv(BASE / "na_cleaned.csv")

# Keep sodium-containing high-formula matches
high = matches[
    matches["column"].eq("high_formula")
    & matches["requested_formula"].str.contains("Na", case=False, na=False)
].copy()

# Count structures found for each test material
summary = (
    high.groupby("na_row")
    .agg(
        formula=("requested_formula", "first"),
        structure_count=("mp_id", "nunique"),
        mp_ids=("mp_id", lambda x: "; ".join(sorted(set(x))))
    )
    .reset_index()
)

# Include all 32 sodium test rows
all_rows = pd.DataFrame({"na_row": range(len(na))})
summary = all_rows.merge(summary, on="na_row", how="left")
summary["formula"] = summary["formula"].fillna(
    na["high_formula"].astype(str)
)
summary["structure_count"] = (
    summary["structure_count"].fillna(0).astype(int)
)
summary["mp_ids"] = summary["mp_ids"].fillna("NO MATCH")

summary.to_csv(BASE / "na_structure_summary.csv", index=False)

print("Saved: na_structure_summary.csv")
print(summary.to_string(index=False))
