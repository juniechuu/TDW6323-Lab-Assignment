"""Dataset I/O helpers — load and persist DataFrames."""

from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load CSV and immediately replace the '?' sentinel with NaN.
    low_memory=False avoids mixed-type DtypeWarnings on columns like payer_code.
    """
    return pd.read_csv(path, na_values="?", low_memory=False)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Persist *df* to CSV without the row index. Creates parent dirs if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"[io] Saved {len(df):,} rows x {df.shape[1]} cols  ->  {dest}")
