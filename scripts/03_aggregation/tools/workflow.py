"""Aggregate patent-level embeddings into entity-year embeddings."""

from __future__ import annotations

import logging
from itertools import islice
from pathlib import Path

import numpy as np
import polars as pl

from .aggregation import AggregateParts, build_aggregate_parts, finalize_aggregates
from .config import CITATION_COLUMN, DEFAULT_OUTPUT_DIR, PATENT_ROW_ID, model_short_name
from .entities import EntitySpec
from .io import ensure_columns, load_patent_level_bundle, save_embedding_bundle


def iter_slices(df: pl.DataFrame, chunk_size: int | None):
    if chunk_size is None or chunk_size <= 0 or df.height <= chunk_size:
        yield df
        return
    for offset in range(0, df.height, chunk_size):
        yield df.slice(offset, chunk_size)


def _prepare_meta_for_entity(meta: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    ensure_columns(meta, spec.required_cols, f"patent-level metadata for {spec.name}")
    exprs: list[pl.Expr] = [
        pl.col(spec.id_col).cast(pl.Utf8, strict=False).str.strip_chars().alias(spec.id_col),
        pl.col(spec.year_col).cast(pl.Int64, strict=False).alias(spec.year_col),
    ]
    meta = meta.with_columns(exprs).filter(
        pl.col(spec.id_col).is_not_null() & pl.col(spec.year_col).is_not_null()
    )
    if spec.key_col not in meta.columns:
        meta = meta.with_columns(
            pl.concat_str([pl.col(spec.id_col), pl.lit("_"), pl.col(spec.year_col).cast(pl.Utf8)], separator="").alias(spec.key_col)
        )
    return meta.sort(list(spec.sort_cols), maintain_order=True)


def _limited_slices(df: pl.DataFrame, row_chunk_size: int | None, max_chunks: int | None):
    chunks = iter_slices(df, row_chunk_size)
    if max_chunks is not None:
        chunks = islice(chunks, max_chunks)
    yield from chunks


def run_entity_aggregation_pipeline(
    *,
    spec: EntitySpec,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str,
    patent_meta_path: Path | None = None,
    patent_embeddings_path: Path | None = None,
    row_chunk_size: int | None = None,
    max_chunks: int | None = None,
    include_empty_in_agg: bool = False,
    save_npy: bool = False,
) -> None:
    short = model_short_name(model)
    meta, embeddings = load_patent_level_bundle(
        output_dir,
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
