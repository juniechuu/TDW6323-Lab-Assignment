"""
Central configuration — all paths, column lists, and encoding maps live here.
No other module hard-codes dataset-specific strings or magic values.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT                 = Path(__file__).resolve().parent.parent
DATASET_PATH         = ROOT / "dataset" / "diabetic_data.csv"
OUTPUT_DIR           = ROOT / "output" / "part1"
CLEANED_DATASET_PATH = ROOT / "output" / "cleaned_data.csv"

# ── Columns to drop ────────────────────────────────────────────────────────────
# encounter_id / patient_nbr : administrative IDs, zero predictive value
# weight                     : ~97 % missing — imputation would fabricate data
# payer_code                 : ~40 % missing, purely administrative
# medical_specialty          : ~49 % missing
NON_IMPACTFUL_COLS = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
]

# ── Medication (drug) columns ──────────────────────────────────────────────────
# 23 columns present in this dataset version (assignment spec lists 21;
# examide and citoglipton are constant-'No' columns but kept for completeness)
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Numerical columns appropriate for outlier detection (excludes binary/ID-like cols)
NUMERICAL_OUTLIER_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# ── Encoding maps ──────────────────────────────────────────────────────────────

# Any drug activity (Steady / Up / Down) = 1; no use = 0
DRUG_ENCODING: dict[str, int] = {"No": 0, "Steady": 1, "Up": 1, "Down": 1}

# Abnormal/normal test result → 1/0; no test taken → -99 (sentinel, not NaN,
# so the column stays integer and downstream models can optionally mask it)
A1C_ENCODING: dict[str, int] = {">7": 1, ">8": 1, "Norm": 0, "None": -99}
GLU_ENCODING: dict[str, int] = {">200": 1, ">300": 1, "Norm": 0, "None": -99}

# Any readmission event → 1; no readmission → 0
READMITTED_ENCODING: dict[str, int] = {"NO": 0, "<30": 1, ">30": 1}

# Ordered insulin categories for plots (no use → decreasing → steady → increasing)
INSULIN_CATEGORIES: list[str] = ["No", "Down", "Steady", "Up"]
