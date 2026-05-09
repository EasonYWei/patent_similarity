"""I/O helpers for preprocessing raw patent data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl


def pandas_to_polars(pdf) -> pl.DataFrame:
    """Convert a pandas frame without relying on optional pyarrow conversion."""
    return pl.DataFrame({str(col): pdf[col].to_list() for col in pdf.columns})


def read_stata(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for Stata .dta compatibility I/O") from exc
    pdf = pd.read_stata(path, columns=columns, convert_categoricals=False)
    return pandas_to_polars(pdf)


def write_stata(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for Stata .dta compatibility I/O") from exc
    pd.DataFrame(df.to_dict(as_series=False)).to_stata(path, write_index=False)
