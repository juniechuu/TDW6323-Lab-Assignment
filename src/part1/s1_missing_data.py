"""
Part 1 · Section 1 — Missing Data Handling  (4 marks)
======================================================
Pipeline:
  1. Display and describe the raw dataset
  2. Report shape (rows × cols)
  3. Audit missing values and visualise their pattern
  4. Apply imputation strategy (mode for low-missing cols, drop for high-missing)
  5. Drop non-impactful columns
  6. Split numerical / categorical attributes
  7. Verify all missing values are resolved
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import NON_IMPACTFUL_COLS
from src.utils.plotting import save_fig


# ── 1 & 2.  Dataset overview ───────────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame) -> None:
    """Print shape, column dtypes, and numeric summary statistics."""
    print(f"\n  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\n--- dtypes ---")
    print(df.dtypes.to_string())
    print("\n--- Statistical summary (numerical columns) ---")
    print(df.describe().to_string())
    print("\n--- First 5 rows ---")
    print(df.head().to_string())


# ── 3.  Missing-value audit ────────────────────────────────────────────────────

def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table: column → (missing count, missing %)."""
    total   = df.isnull().sum()
    pct     = (total / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": total, "missing_%": pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_%", ascending=False)


def plot_missing_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Boolean heatmap — each cell is True (yellow) if the value is NaN.
    Only columns with at least one NaN are included to keep the chart readable.
    """
    missing_cols = df.columns[df.isnull().any()].tolist()

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        df[missing_cols].isnull(),
        yticklabels=False,
        cbar=False,
        cmap="viridis",
        ax=ax,
    )
    ax.set_title("Missing Value Pattern  (yellow = NaN)", fontsize=13)
    ax.set_xlabel("Columns with missing values")
    save_fig(fig, output_dir, "missing_heatmap.png")


def plot_missing_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart of missing-value percentages per column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    summary["missing_%"].plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Percentage by Column")
    ax.invert_yaxis()
    save_fig(fig, output_dir, "missing_percentage_bar.png")


# ── 4–6.  Imputation, column drop, and split ──────────────────────────────────

def handle_missing(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute and clean the dataset.

    Strategy decisions:
      • weight / payer_code / medical_specialty — dropped via NON_IMPACTFUL_COLS
        because their missing rates (97 %, 40 %, 49 %) are too high for reliable
        imputation; they also add minimal predictive signal for readmission.
      • encounter_id / patient_nbr — dropped as pure administrative identifiers.
      • race (~2.2 % missing) — mode-imputed; small fraction, categorical column.
      • diag_1/2/3 — filled with '-1' string so the downstream ICD-9 categoriser
        maps them to 'NAN'. Filling preserves row count.

    Returns
    -------
    df_cleaned    : imputed + non-impactful columns removed
    df_numerical  : numerical-only view of df_cleaned
    df_categorical: categorical-only view of df_cleaned
    """
    df = df.copy()

    # --- Drop non-impactful columns ---
    df.drop(columns=NON_IMPACTFUL_COLS, inplace=True, errors="ignore")
    print(f"\n[s1] Dropped {len(NON_IMPACTFUL_COLS)} non-impactful columns: {NON_IMPACTFUL_COLS}")

    # --- Impute 'race' with the most frequent category ---
    race_mode = df["race"].mode()[0]
    n_race    = df["race"].isnull().sum()
    df["race"] = df["race"].fillna(race_mode)
    print(f"[s1] Imputed {n_race:,} missing 'race' values with mode '{race_mode}'")

    # --- Fill diagnosis NaNs with '-1' so the ICD-9 mapper returns 'NAN' ---
    for col in ["diag_1", "diag_2", "diag_3"]:
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna("-1")
        print(f"[s1] Filled {n_missing:,} missing '{col}' values with '-1'")

    # --- Split numerical and categorical for inspection ---
    df_numerical   = df.select_dtypes(include="number")
    df_categorical = df.select_dtypes(exclude="number")

    print(f"\n[s1] Numerical columns   ({len(df_numerical.columns)}): "
          f"{df_numerical.columns.tolist()}")
    print(f"[s1] Categorical columns ({len(df_categorical.columns)}): "
          f"{df_categorical.columns.tolist()}")

    return df, df_numerical, df_categorical


# ── 7.  Verification ───────────────────────────────────────────────────────────

def verify_no_missing(df: pd.DataFrame) -> None:
    """
    Confirm that zero NaN values remain after Section 1 imputation.

    A1Cresult and max_glu_serum are intentionally excluded from this check:
    their NaN values represent 'test not conducted' and are encoded to -99
    in Section 2's encode_lab_results step.
    """
    # These two columns are deferred to Section 2 — exclude from s1 check
    DEFERRED_COLS = {"A1Cresult", "max_glu_serum"}

    remaining = df.drop(columns=list(DEFERRED_COLS & set(df.columns))).isnull().sum()
    still_missing = remaining[remaining > 0]

    if still_missing.empty:
        print("\n[s1] OK  All imputable missing values resolved.")
        print("[s1]     (A1Cresult, max_glu_serum NaN deferred to Section 2 encoding)")
    else:
        print("\n[s1] WARNING  Remaining missing values detected:")
        print(still_missing.to_string())


# ── Entry point ────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Execute Section 1 end-to-end.

    Parameters
    ----------
    df         : raw DataFrame (loaded with '?' already replaced by NaN)
    output_dir : directory where plots are saved

    Returns
    -------
    df_cleaned : DataFrame ready for Section 2
    """
    print("\n" + "=" * 60)
    print("  SECTION 1 - Missing Data Handling")
    print("=" * 60)

    describe_dataset(df)

    missing_summary = audit_missing(df)
    print("\n--- Missing value audit ---")
    print(missing_summary.to_string())

    plot_missing_heatmap(df, output_dir)
    plot_missing_bar(missing_summary, output_dir)

    df_cleaned, _num, _cat = handle_missing(df)

    verify_no_missing(df_cleaned)

    return df_cleaned
