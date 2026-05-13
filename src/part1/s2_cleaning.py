"""
Part 1 · Section 2 — Data Cleaning and Transformation  (3 marks)
=================================================================
Pipeline (order matters — insulin plots must run before drug encoding):
  1. Encode 'readmitted' to binary int, plot class distribution
  2. Plot impact of gender / age / race on readmission rate
  3. Insulin analysis for readmitted patients (by category, age, gender, race)
  4. Categorise diag_1 / diag_2 / diag_3 using ICD-9 ranges, plot distribution
  5. Encode 23 drug columns to binary integer (any use = 1, no use = 0)
  6. Encode A1Cresult and max_glu_serum to integer
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    A1C_ENCODING,
    DRUG_COLS,
    DRUG_ENCODING,
    GLU_ENCODING,
    INSULIN_CATEGORIES,
    READMITTED_ENCODING,
)
from src.utils.plotting import save_fig


# ── 1.  Readmitted encoding and distribution ───────────────────────────────────

def encode_readmitted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'readmitted' string values to binary.
    '<30' and '>30' both become 1 — the assignment treats any readmission as the
    positive class regardless of timing.
    """
    df = df.copy()
    df["readmitted"] = df["readmitted"].map(READMITTED_ENCODING)

    # Drop rows where the value wasn't in the expected set (edge-case guard)
    before = len(df)
    df.dropna(subset=["readmitted"], inplace=True)
    df["readmitted"] = df["readmitted"].astype(int)

    dropped = before - len(df)
    if dropped:
        print(f"[s2] Dropped {dropped} rows with unrecognised 'readmitted' values.")
    return df


def plot_readmitted_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart showing count of readmitted vs not-readmitted patients."""
    label_map = {0: "Not Readmitted", 1: "Readmitted"}
    counts = df["readmitted"].map(label_map).value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Class Distribution: Readmitted vs Not Readmitted")
    ax.set_ylabel("Count")

    # Annotate bar heights for quick reading
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=10,
        )
    save_fig(fig, output_dir, "readmitted_distribution.png")


# ── 2.  Demographic impact on readmission ─────────────────────────────────────

def plot_demographic_impact(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Grouped bar charts: readmission *rate* (%) per gender / age / race.
    Using rate instead of raw count corrects for unequal group sizes.
    """
    demo_cols = ["gender", "age", "race"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col in zip(axes, demo_cols):
        rate = (
            df.groupby(col)["readmitted"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        rate.plot(kind="bar", ax=ax, color=sns.color_palette("muted"), edgecolor="black")
        ax.set_title(f"Readmission Rate by {col.capitalize()}", fontsize=11)
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_xlabel(col.capitalize())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)

    fig.suptitle("Impact of Demographics on Hospital Readmission", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "demographics_readmitted.png")


# ── 3.  Insulin analysis for readmitted patients ──────────────────────────────

def plot_insulin_impact(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Focuses only on readmitted patients to surface the relationship between
    insulin management and length of stay.

    Chart 1: average time_in_hospital per insulin category (summary).
    Chart 2: breakdown of the above by age group, gender, and race (3 subplots).
    """
    readmitted = df[df["readmitted"] == 1].copy()

    # Categorical ordering makes the x-axis progression intuitive
    readmitted["insulin"] = pd.Categorical(
        readmitted["insulin"], categories=INSULIN_CATEGORIES, ordered=True
    )

    # --- Chart 1: summary average per insulin category ---
    avg_by_insulin = (
        readmitted.groupby("insulin", observed=True)["time_in_hospital"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        avg_by_insulin["insulin"].astype(str),
        avg_by_insulin["time_in_hospital"],
        color=sns.color_palette("muted"), edgecolor="black",
    )
    ax.set_title("Avg. Time in Hospital by Insulin Category\n(Readmitted Patients Only)")
    ax.set_xlabel("Insulin Category")
    ax.set_ylabel("Avg. Days in Hospital")
    save_fig(fig, output_dir, "insulin_avg_time.png")

    # --- Chart 2: breakdown by age / gender / race ---
    breakdown_cols = ["age", "gender", "race"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, col in zip(axes, breakdown_cols):
        pivot = (
            readmitted.groupby([col, "insulin"], observed=True)["time_in_hospital"]
            .mean()
            .unstack("insulin")
        )
        pivot.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title(f"Insulin Impact by {col.capitalize()}", fontsize=11)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Avg. Days in Hospital")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.legend(title="Insulin", fontsize=8)

    fig.suptitle("Insulin Intake Impact on Readmitted Patients", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, "insulin_breakdown.png")


# ── 4.  Diagnosis column categorisation ───────────────────────────────────────

def _map_icd9_to_category(raw_value: object) -> str:
    """
    Convert a single ICD-9 code to a disease category string.

    int() is applied to the float so that sub-codes like 250.83 correctly
    resolve to 'Diabetes' (matches intent of the assignment spec 'value==250').
    Non-numeric strings (V-codes, E-codes) fall through to 'Other'.
    """
    try:
        v = int(float(str(raw_value)))
    except (ValueError, TypeError):
        return "Other"

    if v == -1:
        return "NAN"
    if (390 <= v <= 459) or v == 785:
        return "Circulatory"
    if (460 <= v <= 519) or v == 786:
        return "Respiratory"
    if (520 <= v <= 579) or v == 787:
        return "Digestive"
    if v == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if (580 <= v <= 629) or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def clean_diagnosis_columns(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Apply ICD-9 category mapping to diag_1 / diag_2 / diag_3, then plot
    readmission counts by primary diagnosis category (diag_1).
    """
    df = df.copy()

    for col in ["diag_1", "diag_2", "diag_3"]:
        # Guard: ensure NaNs are filled before mapping (s1 already did this, but safe to repeat)
        df[col] = df[col].fillna("-1").apply(_map_icd9_to_category)

    # Readmission count per primary diagnosis category
    diag_summary = (
        df.groupby("diag_1")["readmitted"]
        .agg(total="count", readmitted_count="sum")
        .sort_values("readmitted_count", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    diag_summary["readmitted_count"].plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="black"
    )
    ax.set_title("Readmission Count by Primary Diagnosis Category (diag_1)")
    ax.set_xlabel("Diagnosis Category")
    ax.set_ylabel("Readmitted Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, output_dir, "diagnosis_readmission.png")

    top_cat = diag_summary["readmitted_count"].idxmax()
    print(f"[s2] Most readmitted primary diagnosis: '{top_cat}' "
          f"({int(diag_summary.loc[top_cat, 'readmitted_count']):,} cases)")

    return df


# ── 5.  Drug column encoding ───────────────────────────────────────────────────

def encode_drug_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace drug activity strings with binary integers.
    Any active dosage (Steady / Up / Down) → 1;  no drug usage → 0.
    """
    df = df.copy()
    encoded = 0
    for col in DRUG_COLS:
        if col in df.columns:
            df[col] = df[col].map(DRUG_ENCODING).astype(int)
            encoded += 1
    print(f"[s2] Encoded {encoded} drug columns to binary int.")
    return df


# ── 6.  Lab result encoding ────────────────────────────────────────────────────

def encode_lab_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform A1Cresult and max_glu_serum from category strings to integers.

    Mapping:
      A1Cresult   : '>7'/''>8'' → 1 (above threshold), 'Norm' → 0, missing → -99
      max_glu_serum: '>200'/'>300' → 1, 'Norm' → 0, missing → -99

    -99 is used (not NaN) to keep the column dtype as int and allow downstream
    code to optionally filter or flag untested patients.
    """
    df = df.copy()

    df["A1Cresult"]     = df["A1Cresult"].map(A1C_ENCODING).fillna(-99).astype(int)
    df["max_glu_serum"] = df["max_glu_serum"].map(GLU_ENCODING).fillna(-99).astype(int)

    print("[s2] Encoded 'A1Cresult' and 'max_glu_serum' to integer.")
    return df


# ── Entry point ────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Execute Section 2 end-to-end.

    NOTE: insulin plots (step 3) intentionally run before drug encoding (step 5)
    because encoding converts the insulin column to 0/1, destroying the string
    categories needed for the grouped visualisation.

    Parameters
    ----------
    df         : cleaned DataFrame from Section 1
    output_dir : directory where plots are saved

    Returns
    -------
    df : fully transformed DataFrame ready for Section 3
    """
    print("\n" + "=" * 60)
    print("  SECTION 2 - Data Cleaning and Transformation")
    print("=" * 60)

    df = encode_readmitted(df)
    print(f"\n[s2] Readmitted distribution:\n{df['readmitted'].value_counts().to_string()}")
    plot_readmitted_distribution(df, output_dir)

    plot_demographic_impact(df, output_dir)

    # Must run BEFORE encode_drug_columns — insulin still holds string values here
    plot_insulin_impact(df, output_dir)

    df = clean_diagnosis_columns(df, output_dir)

    df = encode_drug_columns(df)

    df = encode_lab_results(df)

    print(f"\n[s2] Transformation complete. Final shape: {df.shape}")
    return df
