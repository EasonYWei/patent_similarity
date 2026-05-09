"""Build city-enriched cleaned patent files from raw Stata input."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from .config import COLUMN_MAPPING, PATENT_TYPES, RAW_COLUMNS, STOCK_PREFIXES
from .io import pandas_to_polars, read_stata, write_stata


def read_raw_patent_chunks(path: Path, chunk_size: int | None) -> Iterator[pl.DataFrame]:
    """Read raw Stata data through the pandas compatibility boundary."""
    if chunk_size is None or chunk_size <= 0:
        yield read_stata(path, columns=RAW_COLUMNS)
        return
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for Stata .dta compatibility I/O") from exc
    for chunk in pd.read_stata(
        path, columns=RAW_COLUMNS, convert_categoricals=False, chunksize=chunk_size
    ):
        yield pandas_to_polars(chunk)


def clean_city_patents(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw patent columns and keep records usable for city workflows."""
    df = df.rename({source: target for source, target in COLUMN_MAPPING.items() if source in df.columns})
    return (
        df.with_columns(
            pl.col("stkcd").cast(pl.Utf8, strict=False).str.strip_chars().alias("stkcd"),
            pl.col("p_year").cast(pl.Int64, strict=False).alias("p_year"),
            pl.col("p_cite").cast(pl.Float64, strict=False).fill_null(0.0).alias("p_cite"),
            pl.col("city_code").cast(pl.Utf8, strict=False).str.strip_chars().alias("city_code"),
        )
        .filter(pl.col("p_type").is_in(PATENT_TYPES))
        .filter(pl.col("stkcd").str.slice(0, 1).is_in(STOCK_PREFIXES))
        .filter(pl.col("city_code").is_not_null() & pl.col("p_year").is_not_null())
    )


def build_city_enriched_patents(
    *,
    input_path: Path,
    output_path: Path,
    chunk_size: int | None = None,
    target_rows: int | None = None,
    max_chunks: int | None = None,
) -> None:
    parts: list[pl.DataFrame] = []
    for idx, chunk in enumerate(read_raw_patent_chunks(input_path, chunk_size), start=1):
        if max_chunks is not None and idx > max_chunks:
            break
        cleaned = clean_city_patents(chunk)
        if not cleaned.is_empty():
            parts.append(cleaned)
        if target_rows is not None and sum(part.height for part in parts) >= target_rows:
            break

    df = pl.concat(parts, how="vertical") if parts else pl.DataFrame()
    if target_rows is not None:
        df = df.head(target_rows)
    write_stata(df, output_path)
    logging.getLogger(__name__).info("Saved city-enriched patents: %s rows=%d", output_path, df.height)
