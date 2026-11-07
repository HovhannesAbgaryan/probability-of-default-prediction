"""Loading of the raw extract and the processed train/test split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Load the supplied pickle as-is.

    Security: unpickling can execute code, so only load files from a trusted source (here, the file
    received through the official task channel)."""
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} does not contain a DataFrame (got {type(df).__name__})")
    return df


def load_split(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(processed_dir / "train.parquet"), pd.read_parquet(processed_dir / "test.parquet")
