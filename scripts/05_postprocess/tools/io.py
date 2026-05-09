"""I/O helpers for postprocess outputs."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def read_csv(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pl.read_csv(path, infer_schema_length=10_000)


def write_csv(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
