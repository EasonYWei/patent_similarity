"""I/O and schema helpers for Polars-based scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from .config import CITATION_COLUMN, STKCD_COLUMN, TEXT_IS_EMPTY_FIELD
from .entities import EntitySpec
from .text import with_combined_text

RAW_COLUMN_RENAMES = {
    "股票代码": STKCD_COLUMN,
    "年份": "p_year",
    "标题": "p_tt",
    "摘要": "p_abs",
    "申请日": "p_date",
    "专利类型": "p_type",
    "IPC": "p_ipc",
    "被引证次数": CITATION_COLUMN,
    "市": "city",
    "市代码": "city_code",
    "省": "province",
    "省代码": "province_code",
}
RAW_FILTER_COLUMNS = ("p_type",)
RAW_ALLOWED_PATENT_TYPES = ("发明申请", "发明授权", "实用新型")
RAW_ALLOWED_STOCK_PREFIXES = ("0", "3", "6")


def pandas_to_polars(pdf) -> pl.DataFrame:
    """Convert a pandas frame without relying on optional pyarrow conversion."""
    return pl.DataFrame({str(col): pdf[col].to_list() for col in pdf.columns})


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


def write_csv(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def read_excel(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pl.read_excel(path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading Excel with Polars requires the optional fastexcel package. "
            "Install dependencies from requirements.txt."
        ) from exc


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


def read_stata(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read Stata data through pandas, then immediately return a Polars frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required only for Stata .dta compatibility I/O") from exc

    selected = None
    if columns is not None:
        available = _available_stata_columns(path)
        selected = [col for col in columns if available is None or col in available]
    pdf = pd.read_stata(path, columns=selected, convert_categoricals=False)
    return pandas_to_polars(pdf)


def _available_parquet_columns(path: Path) -> list[str] | None:
    try:
        return pl.scan_parquet(path).collect_schema().names()
    except Exception:
        logging.getLogger(__name__).debug("Could not inspect Parquet columns for %s", path, exc_info=True)
        return None


def _parquet_source_columns(columns: Sequence[str] | None, available: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None

    requested = list(dict.fromkeys([*columns, *RAW_FILTER_COLUMNS]))
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


def read_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read Parquet data, selecting only existing requested columns when possible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    selected = _parquet_source_columns(columns, _available_parquet_columns(path))
    return pl.read_parquet(path, columns=selected)


def read_patent_table(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read a cleaned patent table in the supported on-disk formats."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return read_stata(path, columns=columns)
    if suffix == ".parquet":
        return read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported patent input format for {path}. Expected .dta or .parquet")


def has_raw_columns(df: pl.DataFrame) -> bool:
    return any(column in df.columns for column in RAW_COLUMN_RENAMES)


def normalize_source_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {
        raw: normalized
        for raw, normalized in RAW_COLUMN_RENAMES.items()
        if raw in df.columns and normalized not in df.columns
    }
    return df.rename(rename_map) if rename_map else df


def apply_raw_preprocessing_filters(df: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    if "p_type" in df.columns:
        df = df.filter(pl.col("p_type").cast(pl.Utf8, strict=False).is_in(RAW_ALLOWED_PATENT_TYPES))
    if spec.id_col == STKCD_COLUMN and STKCD_COLUMN in df.columns:
        stock_code = pl.col(STKCD_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars()
        df = df.filter(stock_code.str.slice(0, 1).is_in(RAW_ALLOWED_STOCK_PREFIXES))
    return df


def write_stata(df: pl.DataFrame, path: str | Path) -> None:
    """Write Stata data through pandas. This is the only supported .dta write boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required only for Stata .dta compatibility I/O") from exc
    pd.DataFrame(df.to_dict(as_series=False)).to_stata(path, write_index=False)


def normalize_identifier_columns(df: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    exprs: list[pl.Expr] = [
        pl.col(spec.id_col).cast(pl.Utf8, strict=False).str.strip_chars().alias(spec.id_col),
        pl.col(spec.year_col).cast(pl.Int64, strict=False).alias(spec.year_col),
    ]
    return df.with_columns(exprs).filter(
        pl.col(spec.id_col).is_not_null() & pl.col(spec.year_col).is_not_null()
    )


def prepare_patent_data(path: str | Path, spec: EntitySpec) -> pl.DataFrame:
    """Load cleaned patent data, validate it, and add text/key columns."""
    df = read_patent_table(path, columns=spec.read_cols)
    raw_source = has_raw_columns(df)
    df = normalize_source_columns(df)
    ensure_columns(df, spec.required_cols, path)
    if raw_source:
        df = apply_raw_preprocessing_filters(df, spec)
    df = normalize_identifier_columns(df, spec)

    if "p_date" in df.columns:
        df = df.with_columns(pl.col("p_date").cast(pl.Datetime, strict=False)).sort(
            [spec.id_col, "p_date"], maintain_order=True
        )
    else:
        df = df.sort([spec.id_col, spec.year_col], maintain_order=True)

    df = with_combined_text(df)
    df = df.with_columns(
        pl.concat_str(
            [pl.col(spec.id_col), pl.lit("_"), pl.col(spec.year_col).cast(pl.Utf8)],
            separator="",
        ).alias(spec.key_col)
    )

    if CITATION_COLUMN in df.columns:
        citation = pl.col(CITATION_COLUMN).cast(pl.Float64, strict=False).fill_nan(0.0).fill_null(0.0)
        df = df.with_columns(pl.when(citation < 0).then(0.0).otherwise(citation).alias(CITATION_COLUMN))

    logging.getLogger(__name__).info(
        "Loaded %s rows=%d entity_years=%d empty_text=%d",
        Path(path).name,
        df.height,
        df.select(pl.col(spec.key_col).n_unique()).item(),
        df.select(pl.col(TEXT_IS_EMPTY_FIELD).sum()).item(),
    )
    return df


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


def save_embedding_bundle(
    output_dir: Path,
    prefix: str,
    meta: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    model_short: str,
    save_npy: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{model_short}" if model_short else ""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape((embeddings.shape[0], 1))
    emb_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    emb_df = pl.DataFrame(emb_cols) if emb_cols else pl.DataFrame()
    full = pl.concat([meta, emb_df], how="horizontal") if emb_cols else meta
    csv_path = output_dir / f"{prefix}{suffix}_embeddings.csv"
    full.write_csv(csv_path)
    logging.getLogger(__name__).info("Saved embeddings CSV: %s", csv_path)

    if save_npy:
        meta_path = output_dir / f"{prefix}{suffix}_meta.csv"
        emb_path = output_dir / f"{prefix}{suffix}_embeddings.npy"
        meta.write_csv(meta_path)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))
        logging.getLogger(__name__).info("Saved metadata CSV: %s", meta_path)
        logging.getLogger(__name__).info("Saved embeddings NPY: %s", emb_path)


def save_patent_level_embeddings(
    output_dir: Path,
    meta: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    model_short: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{model_short}" if model_short else ""
    meta.write_csv(output_dir / f"patent_level{suffix}_meta.csv")
    np.save(output_dir / f"patent_level{suffix}_embeddings.npy", embeddings.astype(np.float32, copy=False))
