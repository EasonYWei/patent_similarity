"""Patent-level input and output helpers for the embedding stage."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from .config import (
    CITATION_COLUMN,
    CITY_CODE_COLUMN,
    PATENT_LEVEL_COLUMNS,
    PATENT_ROW_ID,
    RAW_COLUMN_RENAMES,
    STKCD_COLUMN,
    TEXT_IS_EMPTY_FIELD,
    YEAR_COLUMN,
)
from .text import with_combined_text


def pandas_to_polars(pdf) -> pl.DataFrame:
    """Convert a pandas frame without relying on optional pyarrow conversion."""
    return pl.DataFrame({str(col): pdf[col].to_list() for col in pdf.columns})


def ensure_columns(df: pl.DataFrame, required: Iterable[str], source: str | Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}. Found columns: {df.columns}")


def _available_stata_columns(path: Path) -> list[str] | None:
    try:
        import pandas as pd

        with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
            varlist = getattr(reader, "varlist", None)
            if varlist:
                return list(varlist)
            labels = reader.variable_labels()
            return list(labels.keys()) if labels else None
    except Exception:
        logging.getLogger(__name__).debug("Could not inspect Stata columns for %s", path, exc_info=True)
        return None


def _source_columns(columns: Sequence[str] | None, available: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    requested = list(dict.fromkeys(columns))
    available_set = set(available) if available is not None else None
    selected: list[str] = []
    for column in requested:
        candidates = [column]
        candidates.extend(raw for raw, normalized in RAW_COLUMN_RENAMES.items() if normalized == column)
        for candidate in candidates:
            if (available_set is None or candidate in available_set) and candidate not in selected:
                selected.append(candidate)
                break
    return selected


def read_stata(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for Stata .dta compatibility I/O") from exc
    selected = _source_columns(columns, _available_stata_columns(path))
    pdf = pd.read_stata(path, columns=selected, convert_categoricals=False)
    return pandas_to_polars(pdf)


def _available_parquet_columns(path: Path) -> list[str] | None:
    try:
        return pl.scan_parquet(path).collect_schema().names()
    except Exception:
        logging.getLogger(__name__).debug("Could not inspect Parquet columns for %s", path, exc_info=True)
        return None


def read_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    selected = _source_columns(columns, _available_parquet_columns(path))
    return pl.read_parquet(path, columns=selected)


def read_patent_table(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return read_stata(path, columns=columns)
    if suffix == ".parquet":
        return read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported patent input format for {path}. Expected .dta or .parquet")


def normalize_source_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {
        raw: normalized
        for raw, normalized in RAW_COLUMN_RENAMES.items()
        if raw in df.columns and normalized not in df.columns
    }
    return df.rename(rename_map) if rename_map else df


def _read_columns() -> list[str]:
    base = ["p_id", "stkcd", "p_year", "p_tt", "p_abs", "p_date", "p_type", "p_ipc", "p_cite", "city", "city_code", "province", "province_code"]
    return list(dict.fromkeys([*base, *RAW_COLUMN_RENAMES.keys()]))


def _normalize_optional_identifiers(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if STKCD_COLUMN in df.columns:
        exprs.append(pl.col(STKCD_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars().alias(STKCD_COLUMN))
    if CITY_CODE_COLUMN in df.columns:
        exprs.append(pl.col(CITY_CODE_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars().alias(CITY_CODE_COLUMN))
    if YEAR_COLUMN in df.columns:
        exprs.append(pl.col(YEAR_COLUMN).cast(pl.Int64, strict=False).alias(YEAR_COLUMN))
    if CITATION_COLUMN in df.columns:
        citation = pl.col(CITATION_COLUMN).cast(pl.Float64, strict=False).fill_nan(0.0).fill_null(0.0)
        exprs.append(pl.when(citation < 0).then(0.0).otherwise(citation).alias(CITATION_COLUMN))
    if "p_date" in df.columns:
        exprs.append(pl.col("p_date").cast(pl.Datetime, strict=False).alias("p_date"))
    return df.with_columns(exprs) if exprs else df


def _add_entity_keys(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if STKCD_COLUMN in df.columns and YEAR_COLUMN in df.columns:
        exprs.append(
            pl.concat_str([pl.col(STKCD_COLUMN), pl.lit("_"), pl.col(YEAR_COLUMN).cast(pl.Utf8)], separator="").alias("stkcd_year")
        )
    if CITY_CODE_COLUMN in df.columns and YEAR_COLUMN in df.columns:
        exprs.append(
            pl.concat_str([pl.col(CITY_CODE_COLUMN), pl.lit("_"), pl.col(YEAR_COLUMN).cast(pl.Utf8)], separator="").alias("city_year")
        )
    return df.with_columns(exprs) if exprs else df


def prepare_patent_records(path: str | Path) -> pl.DataFrame:
    """Load cleaned patent records and add text plus patent-level metadata columns."""
    df = normalize_source_columns(read_patent_table(path, columns=_read_columns()))
    ensure_columns(df, [YEAR_COLUMN, "p_tt", "p_abs"], path)
    df = _normalize_optional_identifiers(df).filter(pl.col(YEAR_COLUMN).is_not_null())
    sort_cols = [col for col in (STKCD_COLUMN, "p_date", YEAR_COLUMN) if col in df.columns]
    if sort_cols:
        df = df.sort(sort_cols, maintain_order=True)
    df = with_combined_text(df)
    df = _add_entity_keys(df)
    if PATENT_ROW_ID not in df.columns:
        df = df.with_row_count(PATENT_ROW_ID)
    logging.getLogger(__name__).info(
        "Loaded patent records rows=%d empty_text=%d from %s",
        df.height,
        df.select(pl.col(TEXT_IS_EMPTY_FIELD).sum()).item(),
        path,
    )
    return df


def patent_level_meta(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([col for col in PATENT_LEVEL_COLUMNS if col in df.columns])


def patent_level_paths(output_dir: Path, model_short: str) -> tuple[Path, Path]:
    suffix = f"_{model_short}" if model_short else ""
    return output_dir / f"patent_level{suffix}_meta.csv", output_dir / f"patent_level{suffix}_embeddings.npy"


def save_empty_patent_level(output_dir: Path, model_short: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path, emb_path = patent_level_paths(output_dir, model_short)
    pl.DataFrame({PATENT_ROW_ID: []}).write_csv(meta_path)
    np.save(emb_path, np.empty((0, 0), dtype=np.float32))
