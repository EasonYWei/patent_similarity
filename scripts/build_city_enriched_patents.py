#!/usr/bin/env python3
"""Build a city-enriched cleaned patent Stata file from raw patent data."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from patent_similarity.config import DEFAULT_DATA_DIR
from patent_similarity.io import pandas_to_polars, read_stata, write_stata
from patent_similarity.runtime import setup_logging

RAW_COLUMNS = [
    "股票代码",
    "newipzlid",
    "年份",
    "标题",
    "摘要",
    "申请日",
    "专利类型",
    "IPC",
    "被引证次数",
    "市",
    "市代码",
    "省",
    "省代码",
]

COLUMN_MAPPING = {
    "股票代码": "stkcd",
    "newipzlid": "p_id",
    "年份": "p_year",
    "标题": "p_tt",
    "摘要": "p_abs",
    "申请日": "p_date",
    "专利类型": "p_type",
    "IPC": "p_ipc",
    "被引证次数": "p_cite",
    "市": "city",
    "市代码": "city_code",
    "省": "province",
    "省代码": "province_code",
}

PATENT_TYPES = ("发明申请", "发明授权", "实用新型")
STOCK_PREFIXES = ("0", "3", "6")


def read_raw_patent_chunks(path: Path, chunk_size: int | None) -> Iterator[pl.DataFrame]:
    """Read raw Stata data through the pandas compatibility boundary."""
    if chunk_size is None or chunk_size <= 0:
        yield read_stata(path, columns=RAW_COLUMNS)
        return
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required only for Stata .dta compatibility I/O") from exc
    for chunk in pd.read_stata(path, columns=RAW_COLUMNS, convert_categoricals=False, chunksize=chunk_size):
        yield pandas_to_polars(chunk)


def clean_city_patents(df: pl.DataFrame) -> pl.DataFrame:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patents_cleaned_with_city.dta from raw patents.dta")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "patents.dta")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR / "patents_cleaned_with_city.dta")
    parser.add_argument("--chunk-size", type=int, default=None, help="Read raw Stata data in row chunks")
    parser.add_argument("--target-rows", type=int, default=None, help="Optional debug limit after filtering")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional debug limit on raw chunks read")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    parts: list[pl.DataFrame] = []
    for idx, chunk in enumerate(read_raw_patent_chunks(args.input, args.chunk_size), start=1):
        if args.max_chunks is not None and idx > args.max_chunks:
            break
        cleaned = clean_city_patents(chunk)
        if not cleaned.is_empty():
            parts.append(cleaned)
        if args.target_rows is not None and sum(part.height for part in parts) >= args.target_rows:
            break
    df = pl.concat(parts, how="vertical") if parts else pl.DataFrame()
    if args.target_rows is not None:
        df = df.head(args.target_rows)
    write_stata(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
