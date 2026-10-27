import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "na_instances.csv"
ELEMENT_DATA_DIR = BASE_DIR / "dataset" / "element_data"
H5_FILE = BASE_DIR / "dataset" / "mp_data" / "data_2019_12_03.h5"
OUTPUT_FILE = BASE_DIR / "na_features.csv"
FAILED_FILE = BASE_DIR / "na_failed_rows.csv"

COLUMNS = [
    "AtimicNumber", "AtomicVolume", "AtomicWeight", "BCCefflatcnt",
    "BCCenergydiff", "BCCfermi", "BCCmagmom", "BCCvolume_pa",
    "BCCvolume_padiff", "BoilingT", "ColumnNumber", "CovalentRadius",
    "Density", "FirstIonizationEnergy", "GSbandgap", "GSefflatcnt",
    "GSestBCClatcnt", "GSestFCClatcnt", "GSmagmom", "GSvolume_pa",
    "HeatCapacityMass", "HeatCapacityMolar", "ICSDVolume", "IsAlkali",
    "IsDBlock", "IsFBlock", "IsMetal", "IsMetalloid", "IsNonmetal",
    "MendeleevNumber", "NUnfilled", "NValance", "NdUnfilled",
    "NdValence", "NfUnfilled", "NfValence", "NpUnfilled", "NpValence",
    "NsUnfilled", "NsValence", "OxidationStates", "Polarizability",
    "RowNumber", "SpaceGroupNumber"
]

# Load elemental property tables
element_data = {}
for col in COLUMNS:
    if col == "OxidationStates":
        element_data[col] = None
        continue
    with open(ELEMENT_DATA_DIR / f"{col}.table") as f:
        element_data[col] = [
            0 if x.strip() in ("", "Missing") else float(x.strip())
            for x in f
        ]

def create_formula_features(formula):
    comp = Composition(formula)
    result = np.zeros(44)
    if len(comp) == 1:
        el = comp.elements[0]
        for i, col in enumerate(COLUMNS):
            if col != "OxidationStates":
                result[i] = element_data[col][el.number - 1]
        return result

    mean = np.zeros(44)
    dev = np.zeros(44)
    amounts = comp.get_el_amt_dict()
    total = sum(amounts.values())

    for i, col in enumerate(COLUMNS):
        if col == "OxidationStates":
            continue
        vals = {
            symbol: element_data[col][Element(symbol).number - 1]
            for symbol in amounts
        }
        mean[i] = sum(vals[s] * n for s, n in amounts.items()) / total
        dev[i] = sum(
            abs(vals[s] - mean[i]) * n for s, n in amounts.items()
        ) / total

    return np.concatenate([mean, dev])

def create_ion_features(ion):
    ions = ["Li", "Ca", "Cs", "Rb", "K", "Y", "Na", "Al", "Zn", "Mg"]
    out = np.zeros(10)
    if ion in ions:
        out[ions.index(ion)] = 1
    return out

def create_crystal_features(system):
    systems = [
        "triclinic", "monoclinic", "orthorhombic",
        "tetragonal", "trigonal", "hexagonal", "cubic"
    ]
    out = np.zeros(7)
    if system in systems:
        out[systems.index(system)] = 1
    return out

df = pd.read_csv(INPUT_FILE)
print("Input rows:", len(df))

features = []
successful_rows = []
failed = []

with h5py.File(H5_FILE, "r") as cif_data:
    for index, row in df.iterrows():
        try:
            ion = str(row["working_ion"]).strip()

            charge_id = str(row["id_charge"]).strip()
            discharge_id = str(row["id_discharge"]).strip()

            charge_crystal = Structure.from_str(
                cif_data[charge_id][()].decode(), fmt="cif"
            )
            discharge_crystal = Structure.from_str(
                cif_data[discharge_id][()].decode(), fmt="cif"
            )

            finder = SpacegroupAnalyzer(charge_crystal)
            space_group = np.array([finder.get_space_group_number()])
            crystal = create_crystal_features(finder.get_crystal_system())
            ion_features = create_ion_features(ion)

            composition = Composition(discharge_crystal.formula)
            concentration = np.array([
                composition.get_atomic_fraction(Element(ion))
            ])

            charge_features = create_formula_features(
                charge_crystal.composition.reduced_formula
            )
            discharge_features = create_formula_features(
                discharge_crystal.composition.reduced_formula
            )
            ion_features_element = create_formula_features(ion)

            # Ensure charge features always have 88 values
            if len(charge_features) == 44:
                charge_features = np.concatenate(
                    [charge_features, np.zeros(44)]
                )

            combined = np.concatenate([
                space_group, crystal, ion_features, concentration,
                charge_features, discharge_features, ion_features_element
            ])

            if len(combined) != 239:
                raise ValueError(f"Expected 239 features, got {len(combined)}")

            features.append(combined)
            successful_rows.append(index)

        except Exception as e:
            print(f"Failed row {index}: {e}")
            failed.append({
                "row_index": index,
                "battery_id": row.get("battery_id", ""),
                "id_charge": row.get("id_charge", ""),
                "id_discharge": row.get("id_discharge", ""),
                "error": str(e)
            })

# Save only successful rows, keeping their labels aligned
feature_cols = [f"feat_{i}" for i in range(1, 240)]
result = pd.DataFrame(features, columns=feature_cols)
result["average_voltage"] = df.loc[successful_rows, "average_voltage"].to_numpy()
result["working_ion"] = df.loc[successful_rows, "working_ion"].to_numpy()
result["source_row"] = successful_rows
result.to_csv(OUTPUT_FILE, index=False)

pd.DataFrame(failed).to_csv(FAILED_FILE, index=False)

print("\nFeature extraction finished.")
print("Successful rows:", len(result))
print("Failed rows:", len(failed))
print("Features shape:", result[feature_cols].shape)
print("Saved:", OUTPUT_FILE)
print("Failures saved:", FAILED_FILE)