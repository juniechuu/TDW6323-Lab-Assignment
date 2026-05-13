"""
Part 1 · Section 3 — Outlier Detection and Treatment  (3 marks)
================================================================
Method chosen : Interquartile Range (IQR)
Treatment     : Winsorisation — clip values to [Q1 − 1.5·IQR, Q3 + 1.5·IQR]

Why IQR over DBSCAN / Z-Score:
  • IQR is non-parametric — the numeric columns here are heavily right-skewed
    (e.g. number_outpatient has a long tail), so Z-Score would under-detect.
  • DBSCAN requires distance tuning and is computationally heavier for 100k rows.

Why Winsorisation over row removal:
  • Removing rows reduces dataset size and can introduce selection bias.
  • Clipping retains all records while neutralising the distorting effect of
    extreme values on model training and visualisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import NUMERICAL_OUTLIER_COLS
from src.utils.plotting import save_fig


# ── Visual exploration ─────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """
    Box plots for every numerical outlier column.
    *suffix* ('before' or 'after') differentiates the two calls in the pipeline.
    """
    cols = [c for c in NUMERICAL_OUTLIER_COLS if c in df.columns]
    ncols = 4
    nrows = -(-len(cols) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        ax.boxplot(
            df[col].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.6),
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3, color="tomato"),
        )
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Value")

    # Hide unused subplot slots
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    title_suffix = suffix.replace("_", " ").title()
    fig.suptitle(f"Box Plots — {title_suffix} Outlier Treatment", fontsize=13)
    fig.tight_layout()
    save_fig(fig, output_dir, f"boxplots_{suffix}.png")


def plot_scatter_exploration(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter plot of time_in_hospital vs num_medications — one of the clearest
    combinations where joint outliers (long stay + many drugs) are visible.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        df["time_in_hospital"],
        df["num_medications"],
        alpha=0.15, s=6, color="steelblue",
    )
    ax.set_xlabel("Time in Hospital (days)")
    ax.set_ylabel("Number of Medications")
    ax.set_title("Scatter: Time in Hospital vs Number of Medications\n"
                 "(dense clusters = typical; lone points = potential outliers)")
    save_fig(fig, output_dir, "scatter_exploration.png")


# ── IQR detection ──────────────────────────────────────────────────────────────

def compute_iqr_bounds(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute Q1, Q3, IQR, and the standard 1.5·IQR fence bounds for each column.
    Returns a DataFrame indexed by column name.
    """
    records = {}
    for col in cols:
        if col not in df.columns:
            continue
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        records[col] = {
            "Q1":    q1,
            "Q3":    q3,
            "IQR":   iqr,
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
        }
    return pd.DataFrame(records).T


def count_outliers(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.Series:
    """Return the number of outlier values per column (outside IQR fences)."""
    counts = {}
    for col, row in bounds.iterrows():
        if col in df.columns:
            mask = (df[col] < row["lower"]) | (df[col] > row["upper"])
            counts[str(col)] = int(mask.sum())
    return pd.Series(counts, name="outlier_count")


def plot_outlier_counts(counts: pd.Series, output_dir: Path) -> None:
    """Bar chart of outlier count per column, sorted descending."""
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.sort_values(ascending=False).plot(
        kind="bar", ax=ax, color="tomato", edgecolor="black"
    )
    ax.set_title("Outlier Count per Column (IQR Method, 1.5 × IQR fence)")
    ax.set_ylabel("Number of Outlier Values")
    ax.set_xlabel("Column")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, output_dir, "outlier_counts.png")


# ── Treatment — Winsorisation ──────────────────────────────────────────────────

def winsorise(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Clip each column's values to its computed IQR fence boundaries."""
    df = df.copy()
    for col, row in bounds.iterrows():
        if col in df.columns:
            df[col] = df[col].clip(lower=row["lower"], upper=row["upper"])
    return df


# ── Before & After comparison ─────────────────────────────────────────────────

def plot_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Side-by-side box plots for three representative columns so the effect of
    Winsorisation is immediately visible without scrolling through all columns.
    """
    # Choose columns that tend to have the largest outlier counts
    spotlight = ["number_outpatient", "num_medications", "time_in_hospital"]
    spotlight = [c for c in spotlight if c in df_before.columns]

    fig, axes = plt.subplots(len(spotlight), 2, figsize=(12, 4 * len(spotlight)))

    for i, col in enumerate(spotlight):
        for j, (label, data, color) in enumerate([
            ("Before", df_before, "tomato"),
            ("After",  df_after,  "steelblue"),
        ]):
            ax = axes[i][j]
            ax.boxplot(
                data[col].dropna(),
                vert=True, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.6),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker="o", markersize=2, alpha=0.3),
            )
            ax.set_title(f"{col}  ({label})", fontsize=10)
            ax.set_ylabel("Value")

    fig.suptitle("Outlier Treatment — Before vs After Winsorisation", fontsize=13)
    fig.tight_layout()
    save_fig(fig, output_dir, "before_after_outliers.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Execute Section 3 end-to-end.

    Parameters
    ----------
    df         : transformed DataFrame from Section 2
    output_dir : directory where plots are saved

    Returns
    -------
    df_treated : DataFrame with outliers Winsorised, ready for Parts 2 & 3
    """
    print("\n" + "=" * 60)
    print("  SECTION 3 - Outlier Detection and Treatment (IQR)")
    print("=" * 60)

    cols = [c for c in NUMERICAL_OUTLIER_COLS if c in df.columns]

    # --- Explore ---
    plot_boxplots(df, output_dir, suffix="before")
    plot_scatter_exploration(df, output_dir)

    # --- Detect ---
    bounds         = compute_iqr_bounds(df, cols)
    outlier_counts = count_outliers(df, bounds)

    print("\n[s3] IQR bounds per column:")
    print(bounds.to_string())
    print(f"\n[s3] Outlier counts (values outside 1.5 × IQR fences):")
    print(outlier_counts.to_string())

    plot_outlier_counts(outlier_counts, output_dir)

    # --- Treat ---
    df_treated = winsorise(df, bounds)
    total_winsorised = sum(
        int(count_outliers(df, bounds)[col])
        for col in outlier_counts.index
    )
    print(f"\n[s3] Winsorisation applied to {len(cols)} columns "
          f"({total_winsorised:,} values clipped). Shape unchanged: {df_treated.shape}")

    # --- Compare ---
    plot_before_after(df, df_treated, output_dir)
    plot_boxplots(df_treated, output_dir, suffix="after")

    return df_treated
