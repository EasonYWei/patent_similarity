"""I/O helpers for similarity and panel-building scripts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl


def ensure_columns(df: pl.DataFrame, required: Iterable[str], source: str | Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}. Found columns: {df.columns}")


def embedding_columns(df: pl.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("emb_")]
    if not cols:
        raise ValueError("No embedding columns found. Expected columns named emb_0, emb_1, ...")
    return cols


def read_csv(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pl.read_csv(path, infer_schema_length=10_000)


def read_parquet(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pl.read_parquet(path)


def read_frame(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return read_parquet(path)
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported input format for {path}. Expected .parquet or .csv")


def write_csv(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_excel(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pl.read_excel(path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading Excel with Polars requires the optional fastexcel package. Install dependencies from requirements.txt."
        ) from exc


def pandas_to_polars(pdf) -> pl.DataFrame:
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


def select_existing(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    return df.select([col for col in columns if col in df.columns])


def cast_common_keys(df: pl.DataFrame, *, has_firm: bool = False, has_city: bool = False) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if has_firm and "stkcd" in df.columns:
        exprs.append(pl.col("stkcd").cast(pl.Utf8, strict=False).str.strip_chars().alias("stkcd"))
    if has_city and "city_code" in df.columns:
        exprs.append(pl.col("city_code").cast(pl.Utf8, strict=False).str.strip_chars().alias("city_code"))
    if "p_year" in df.columns:
        exprs.append(pl.col("p_year").cast(pl.Int64, strict=False).alias("p_year"))
    return df.with_columns(exprs) if exprs else df
