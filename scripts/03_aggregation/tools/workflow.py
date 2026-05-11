"""Aggregate patent-level embeddings into entity-year embeddings."""

from __future__ import annotations

import logging
from itertools import islice
from pathlib import Path

import numpy as np
import polars as pl

from .aggregation import AggregateParts, build_aggregate_parts, finalize_aggregates
from .config import (
    CITATION_COLUMN,
    CITY_CODE_COLUMN,
    CITY_COLUMN,
    COUNTY_ID_COLUMN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATENT_LEVEL_DIR,
    INDUSTRY_COLUMN,
    PATENT_ROW_ID,
    PROVINCE_COLUMN,
    STKCD_COLUMN,
    YEAR_COLUMN,
    model_short_name,
)
from .entities import EntitySpec
from .io import ensure_columns, load_patent_level_bundle, read_csv, save_embedding_bundle


def iter_slices(df: pl.DataFrame, chunk_size: int | None):
    if chunk_size is None or chunk_size <= 0 or df.height <= chunk_size:
        yield df
        return
    for offset in range(0, df.height, chunk_size):
        yield df.slice(offset, chunk_size)


def _normalized_stock_expr(column: str = STKCD_COLUMN) -> pl.Expr:
    stock = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    stock = pl.when(stock.str.contains(r"^\d+\.0+$")).then(
        stock.str.replace(r"\.0+$", "")
    ).otherwise(stock)
    return stock.str.zfill(6).alias(STKCD_COLUMN)


def _city_code_from_county_expr() -> pl.Expr:
    county = pl.col(COUNTY_ID_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars()
    county = pl.when(county.str.contains(r"^\d+\.0+$")).then(
        county.str.replace(r"\.0+$", "")
    ).otherwise(county)
    return county.str.zfill(6).str.slice(0, 4).alias(CITY_CODE_COLUMN)


def _with_city_code(df: pl.DataFrame) -> pl.DataFrame:
    ensure_columns(df, [COUNTY_ID_COLUMN], "city metadata")
    return df.with_columns(_city_code_from_county_expr())


def _with_entity_key(df: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    return df.with_columns(
        pl.concat_str(
            [pl.col(spec.id_col), pl.lit("_"), pl.col(spec.year_col).cast(pl.Utf8)],
            separator="",
        ).alias(spec.key_col)
    )


def _prepare_meta_for_entity(meta: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    if spec.name == "city":
        meta = _with_city_code(meta)
    ensure_columns(meta, spec.required_cols, f"patent-level metadata for {spec.name}")
    exprs: list[pl.Expr] = [
        pl.col(spec.id_col).cast(pl.Utf8, strict=False).str.strip_chars().alias(spec.id_col),
        pl.col(spec.year_col).cast(pl.Int64, strict=False).alias(spec.year_col),
    ]
    meta = meta.with_columns(exprs).filter(
        pl.col(spec.id_col).is_not_null() & pl.col(spec.year_col).is_not_null()
    )
    if spec.key_col not in meta.columns:
        meta = _with_entity_key(meta, spec)
    return meta.sort(list(spec.sort_cols), maintain_order=True)


def _read_balance_source(path: Path) -> pl.DataFrame:
    df = read_csv(path)
    rename_map = {
        col: col.strip().lstrip("\ufeff")
        for col in df.columns
        if col != col.strip().lstrip("\ufeff")
    }
    if rename_map:
        df = df.rename(rename_map)
    required = {STKCD_COLUMN, "year", CITY_COLUMN, COUNTY_ID_COLUMN, INDUSTRY_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns for balanced panels: {missing}")
    return (
        df.with_columns(
            _normalized_stock_expr(),
            pl.col("year").cast(pl.Int64, strict=False).alias(YEAR_COLUMN),
            pl.col(CITY_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars().alias(CITY_COLUMN),
            pl.col(PROVINCE_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars().alias(PROVINCE_COLUMN)
            if PROVINCE_COLUMN in df.columns
            else pl.lit(None, dtype=pl.Utf8).alias(PROVINCE_COLUMN),
            pl.col(INDUSTRY_COLUMN).cast(pl.Utf8, strict=False).str.strip_chars().alias(INDUSTRY_COLUMN),
            _city_code_from_county_expr(),
        )
        .filter(pl.col(YEAR_COLUMN).is_not_null())
    )


def _balanced_years(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(YEAR_COLUMN).unique().sort(YEAR_COLUMN)


def _city_balance_frame(source: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    valid = source.filter(pl.col(CITY_CODE_COLUMN).is_not_null())
    labels = (
        valid.select(CITY_CODE_COLUMN, PROVINCE_COLUMN, CITY_COLUMN, YEAR_COLUMN)
        .sort(
            [CITY_CODE_COLUMN, YEAR_COLUMN, PROVINCE_COLUMN, CITY_COLUMN],
            descending=[False, True, False, False],
            maintain_order=True,
        )
        .unique(CITY_CODE_COLUMN, keep="first", maintain_order=True)
        .select(CITY_COLUMN, CITY_CODE_COLUMN, PROVINCE_COLUMN)
    )
    panel = labels.join(_balanced_years(source), how="cross")
    return _with_entity_key(panel, spec).select(list(spec.embedding_metadata_cols)).sort(
        list(spec.sort_cols),
        maintain_order=True,
    )


def _industry_balance_frame(source: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    labels = source.select(INDUSTRY_COLUMN).drop_nulls().unique().sort(INDUSTRY_COLUMN)
    panel = labels.join(_balanced_years(source), how="cross")
    return _with_entity_key(panel, spec).select(list(spec.embedding_metadata_cols)).sort(
        list(spec.sort_cols),
        maintain_order=True,
    )


def build_balance_frame(spec: EntitySpec, balance_path: Path) -> pl.DataFrame:
    source = _read_balance_source(balance_path)
    if spec.name == "city":
        return _city_balance_frame(source, spec)
    if spec.name == "industry":
        return _industry_balance_frame(source, spec)
    raise ValueError(f"Balanced panel is not configured for {spec.name}")


def _apply_balance_panel(
    *,
    spec: EntitySpec,
    balance_frame: pl.DataFrame,
    meta: pl.DataFrame,
    simple_embeddings: np.ndarray,
    citation_embeddings: np.ndarray,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    if meta.height:
        observed = meta.with_row_index("_observed_idx").select(
            [
                spec.key_col,
                "_observed_idx",
                "n_patents",
                "n_texts_used",
                "total_citations",
                "mean_citations",
            ]
        )
    else:
        observed = pl.DataFrame(
            {
                spec.key_col: [],
                "_observed_idx": [],
                "n_patents": [],
                "n_texts_used": [],
                "total_citations": [],
                "mean_citations": [],
            }
        )
    balanced = balance_frame.join(observed, on=spec.key_col, how="left").with_columns(
        [
            pl.col("n_patents").fill_null(0).cast(pl.Int64).alias("n_patents"),
            pl.col("n_texts_used").fill_null(0).cast(pl.Int64).alias("n_texts_used"),
            pl.col("total_citations").fill_null(0.0).cast(pl.Float64).alias("total_citations"),
        ]
    )
    balanced = balanced.with_columns(
        pl.when(pl.col("n_patents") > 0)
        .then(pl.col("mean_citations").cast(pl.Float64))
        .otherwise(None)
        .alias("mean_citations")
    )

    observed_idx = (
        balanced.get_column("_observed_idx")
        .cast(pl.Int64)
        .fill_null(-1)
        .to_numpy()
        .astype(np.int64, copy=False)
    )
    valid = observed_idx >= 0
    dim = simple_embeddings.shape[1] if simple_embeddings.ndim == 2 else 0
    balanced_simple = np.full((balanced.height, dim), np.nan, dtype=np.float32)
    balanced_citation = np.full((balanced.height, dim), np.nan, dtype=np.float32)
    if np.any(valid):
        balanced_simple[valid] = simple_embeddings[observed_idx[valid]]
        balanced_citation[valid] = citation_embeddings[observed_idx[valid]]

    return (
        balanced.select(list(spec.embedding_output_cols)),
        balanced_simple,
        balanced_citation,
    )


def _limited_slices(df: pl.DataFrame, row_chunk_size: int | None, max_chunks: int | None):
    chunks = iter_slices(df, row_chunk_size)
    if max_chunks is not None:
        chunks = islice(chunks, max_chunks)
    yield from chunks


def run_entity_aggregation_pipeline(
    *,
    spec: EntitySpec,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    patent_input_dir: Path = DEFAULT_PATENT_LEVEL_DIR,
    model: str,
    patent_meta_path: Path | None = None,
    patent_embeddings_path: Path | None = None,
    row_chunk_size: int | None = None,
    max_chunks: int | None = None,
    include_empty_in_agg: bool = False,
    save_npy: bool = False,
    balance_path: Path | None = None,
) -> None:
    short = model_short_name(model)
    meta, embeddings = load_patent_level_bundle(
        patent_input_dir,
        short,
        meta_path=patent_meta_path,
        embeddings_path=patent_embeddings_path,
    )
    meta = _prepare_meta_for_entity(meta, spec)
    if meta.is_empty():
        raise ValueError(f"No patent-level rows are usable for {spec.name} aggregation")

    aggregate_parts: list[AggregateParts] = []
    for chunk in _limited_slices(meta, row_chunk_size, max_chunks):
        row_indices = chunk.get_column(PATENT_ROW_ID).to_numpy().astype(np.int64, copy=False)
        chunk_embeddings = np.asarray(embeddings[row_indices], dtype=np.float32)
        aggregate_parts.append(
            build_aggregate_parts(
                chunk,
                chunk_embeddings,
                spec,
                include_empty_text=include_empty_in_agg,
                citation_col=CITATION_COLUMN,
            )
        )

    final_meta, simple_embeddings, citation_embeddings = finalize_aggregates(aggregate_parts, spec)
    if balance_path is not None:
        balance_frame = build_balance_frame(spec, balance_path)
        final_meta, simple_embeddings, citation_embeddings = _apply_balance_panel(
            spec=spec,
            balance_frame=balance_frame,
            meta=final_meta,
            simple_embeddings=simple_embeddings,
            citation_embeddings=citation_embeddings,
        )
        logging.getLogger(__name__).info(
            "Balanced %s panel from %s to %s entity-years",
            spec.name,
            balance_path,
            final_meta.height,
        )
    save_embedding_bundle(output_dir, spec.output_prefix, final_meta, simple_embeddings, model_short=short, save_npy=save_npy)
    save_embedding_bundle(
        output_dir,
        f"{spec.output_prefix}_citweighted",
        final_meta,
        citation_embeddings,
        model_short=short,
        save_npy=save_npy,
    )
    logging.getLogger(__name__).info(
        "Aggregated %s embeddings for %s entity-years", short, final_meta.height
    )
