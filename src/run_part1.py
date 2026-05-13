"""
Part 1 — Pipeline Orchestrator
===============================
Chains Sections 1 → 2 → 3, then saves the cleaned dataset to
output/cleaned_data.csv for use in Parts 2 and 3.

Usage (from project root):
    python -m src.run_part1
"""

import matplotlib
# Use the non-interactive Agg backend when running as a script so the process
# doesn't hang waiting for a GUI display.  Notebooks override this with
# %matplotlib inline before importing any src module.
matplotlib.use("Agg")

from src.config import CLEANED_DATASET_PATH, DATASET_PATH, OUTPUT_DIR
from src.part1 import s1_missing_data, s2_cleaning, s3_outliers
from src.utils.io import load_dataset, save_dataset
from src.utils.plotting import setup_style


def main() -> None:
    setup_style()

    print("\n" + "=" * 60)
    print("  TDW6323 - Lab Assignment Part 1")
    print("=" * 60)
    print(f"\n[run] Loading dataset from: {DATASET_PATH}")

    df_raw = load_dataset(DATASET_PATH)
    print(f"[run] Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns.")

    # ── Section 1: Missing Data Handling ──────────────────────────────────────
    df_s1 = s1_missing_data.run(df_raw, OUTPUT_DIR / "s1_missing")

    # ── Section 2: Data Cleaning & Transformation ──────────────────────────────
    df_s2 = s2_cleaning.run(df_s1, OUTPUT_DIR / "s2_cleaning")

    # ── Section 3: Outlier Detection & Treatment ───────────────────────────────
    df_s3 = s3_outliers.run(df_s2, OUTPUT_DIR / "s3_outliers")

    # ── Persist cleaned dataset for Parts 2 and 3 ─────────────────────────────
    save_dataset(df_s3, CLEANED_DATASET_PATH)

    print("\n" + "=" * 60)
    print("  Part 1 complete.")
    print(f"  Cleaned dataset : {CLEANED_DATASET_PATH}")
    print(f"  Plots saved to  : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
