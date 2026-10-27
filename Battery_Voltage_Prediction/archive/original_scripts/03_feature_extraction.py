import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "remaining_instances.csv"

ELEMENT_DATA_DIR = BASE_DIR / "dataset" / "element_data"
H5_FILE = BASE_DIR / "dataset" / "mp_data" / "data_2019_12_03.h5"

OUTPUT_FILE = BASE_DIR / "remaining_features.csv"


# -----------------------------
# 44 ELEMENTAL PROPERTIES
# -----------------------------

COLUMNS = [
    "AtimicNumber",
    "AtomicVolume",
    "AtomicWeight",
    "BCCefflatcnt",
    "BCCenergydiff",
    "BCCfermi",
    "BCCmagmom",
    "BCCvolume_pa",
    "BCCvolume_padiff",
    "BoilingT",
    "ColumnNumber",
    "CovalentRadius",
    "Density",
    "FirstIonizationEnergy",
    "GSbandgap",
    "GSefflatcnt",
    "GSestBCClatcnt",
    "GSestFCClatcnt",
    "GSmagmom",
    "GSvolume_pa",
    "HeatCapacityMass",
    "HeatCapacityMolar",
    "ICSDVolume",
    "IsAlkali",
    "IsDBlock",
    "IsFBlock",
    "IsMetal",
    "IsMetalloid",
    "IsNonmetal",
    "MendeleevNumber",
    "NUnfilled",
    "NValance",
    "NdUnfilled",
    "NdValence",
    "NfUnfilled",
    "NfValence",
    "NpUnfilled",
    "NpValence",
    "NsUnfilled",
    "NsValence",
    "OxidationStates",
    "Polarizability",
    "RowNumber",
    "SpaceGroupNumber"
]


# -----------------------------
# LOAD ELEMENT DATA
# -----------------------------

element_data = {}

for column in COLUMNS:

    # The original repository does not use
    # OxidationStates as a numerical feature.
    if column == "OxidationStates":
        element_data[column] = None
        continue

    values = []

    with open(ELEMENT_DATA_DIR / (column + ".table"), "r") as f:

        for line in f:

            line = line.strip()

            if line == "Missing" or line == "":
                values.append(0)

            else:
                values.append(float(line))

    element_data[column] = values

# -----------------------------
# CREATE FEATURES FROM FORMULA
# -----------------------------

def create_formula_features(formula):

    comp = Composition(formula)

    # 44 elemental properties
    num_features = 44

    # Single-element material
    if len(comp) == 1:

        features = np.zeros(num_features)

        element = comp.elements[0]

        for i, column in enumerate(COLUMNS):

            if column != "OxidationStates":
                features[i] = element_data[column][element.number - 1]

            else:
                features[i] = 0

        return features

    # Multi-element material
    weighted_mean = np.zeros(num_features)
    mean_deviation = np.zeros(num_features)

    element_dict = comp.to_reduced_dict

    total_ratio = sum(element_dict.values())

    for i, column in enumerate(COLUMNS):

        if column != "OxidationStates":

            total = 0

            for element_symbol, ratio in element_dict.items():

                element = Element(element_symbol)

                total += (
                    element_data[column][element.number - 1]
                    * ratio
                )

            mean = total / total_ratio

            weighted_mean[i] = mean

            deviation = 0

            for element_symbol, ratio in element_dict.items():

                element = Element(element_symbol)

                deviation += (
                    abs(
                        element_data[column][element.number - 1]
                        - mean
                    )
                    * ratio
                )

            mean_deviation[i] = deviation / total_ratio

        else:

            weighted_mean[i] = 0
            mean_deviation[i] = 0

    return np.concatenate(
        [weighted_mean, mean_deviation]
    )


# -----------------------------
# WORKING-ION FEATURES
# -----------------------------

def create_ion_features(ion):

    feature = np.zeros(10)

    if ion == "Li":
        feature[0] = 1

    elif ion == "Ca":
        feature[1] = 1

    elif ion == "Cs":
        feature[2] = 1

    elif ion == "Rb":
        feature[3] = 1

    elif ion == "K":
        feature[4] = 1

    elif ion == "Y":
        feature[5] = 1

    elif ion == "Na":
        feature[6] = 1

    elif ion == "Al":
        feature[7] = 1

    elif ion == "Zn":
        feature[8] = 1

    elif ion == "Mg":
        feature[9] = 1

    return feature


# -----------------------------
# CRYSTAL SYSTEM FEATURES
# -----------------------------

def create_crystal_features(crystal_system):

    feature = np.zeros(7)

    if crystal_system == "triclinic":
        feature[0] = 1

    elif crystal_system == "monoclinic":
        feature[1] = 1

    elif crystal_system == "orthorhombic":
        feature[2] = 1

    elif crystal_system == "tetragonal":
        feature[3] = 1

    elif crystal_system == "trigonal":
        feature[4] = 1

    elif crystal_system == "hexagonal":
        feature[5] = 1

    elif crystal_system == "cubic":
        feature[6] = 1

    return feature


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print("Input dataset:", len(df))
print("Starting feature extraction...")


# -----------------------------
# OPEN CRYSTAL STRUCTURE DATA
# -----------------------------

cif_data = h5py.File(H5_FILE, "r")


# -----------------------------
# CREATE FEATURES
# -----------------------------

features = []

failed = []


for index, row in df.iterrows():

    try:

        id_charge = row["id_charge"]
        id_discharge = row["id_discharge"]
        ion = row["working_ion"]


        # Get crystal structures
        charge_crystal = Structure.from_str(
            cif_data[id_charge][()].decode(),
            fmt="cif"
        )

        discharge_crystal = Structure.from_str(
            cif_data[id_discharge][()].decode(),
            fmt="cif"
        )


        # -------------------------
        # Space group
        # -------------------------

        finder = SpacegroupAnalyzer(charge_crystal)

        space_group = np.array([
            finder.get_space_group_number()
        ])


        # -------------------------
        # Crystal system
        # -------------------------

        crystal_system = create_crystal_features(
            finder.get_crystal_system()
        )


        # -------------------------
        # Working ion
        # -------------------------

        ion_features = create_ion_features(ion)


        # -------------------------
        # Ion concentration
        # -------------------------

        discharge_composition = Composition(
            discharge_crystal.formula
        )

        ion_concentration = np.array([
            discharge_composition.get_atomic_fraction(
                Element(ion)
            )
        ])


        # -------------------------
        # Elemental features
        # -------------------------

        charge_formula = charge_crystal.formula

        discharge_formula = discharge_crystal.formula


        charge_features = create_formula_features(
            charge_formula
        )

        discharge_features = create_formula_features(
            discharge_formula
        )

        ion_element_features = create_formula_features(
            ion
        )


        # -------------------------
        # Match repository logic
        # -------------------------

        # If charge material is a single element,
        # add 44 zeros so it becomes 88 features.

        if charge_features.shape[0] == 44:

            charge_features = np.concatenate([
                charge_features,
                np.zeros(44)
            ])


        # -------------------------
        # Combine everything
        # -------------------------

        final_features = np.concatenate([
            space_group,
            crystal_system,
            ion_features,
            ion_concentration,
            charge_features,
            discharge_features,
            ion_element_features
        ])


        features.append(final_features)


    except Exception as e:

        print(
            f"Failed at row {index}: {e}"
        )

        failed.append(index)


# -----------------------------
# CLOSE HDF5 FILE
# -----------------------------

cif_data.close()


# -----------------------------
# SAVE FEATURES
# -----------------------------

features = np.array(features)

print("\nFeature matrix shape:", features.shape)

feature_columns = [
    f"feat_{i + 1}"
    for i in range(features.shape[1])
]

feature_df = pd.DataFrame(
    features,
    columns=feature_columns
)


# Add original information needed later
feature_df["average_voltage"] = df["average_voltage"].values[:len(feature_df)]
feature_df["working_ion"] = df["working_ion"].values[:len(feature_df)]


feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFeature extraction complete!")
print("Saved to:", OUTPUT_FILE)
print("Number of failed rows:", len(failed))