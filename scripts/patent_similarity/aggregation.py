"""Entity-year embedding aggregation with Polars metadata and NumPy vectors."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import polars as pl

from .config import CITATION_COLUMN, TEXT_IS_EMPTY_FIELD, ZERO_DIVISION_EPSILON
from .entities import EntitySpec


@dataclass
class AggregateParts:
    meta: pl.DataFrame
    embedding_sums: np.ndarray
    text_counts: np.ndarray
    citation_weighted_sums: np.ndarray


def divide_rows(sums: np.ndarray, counts: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if sums.size == 0:
        return np.empty_like(sums, dtype=dtype)
    out = np.empty_like(sums, dtype=dtype)
    out[:] = np.nan
    mask = counts > ZERO_DIVISION_EPSILON
    if np.any(mask):
        out[mask] = (sums[mask] / counts[mask, None]).astype(dtype)
    return out


def _coerce_citations(df: pl.DataFrame, citation_col: str | None) -> tuple[np.ndarray, int]:
    if citation_col is None or citation_col not in df.columns:
        return np.zeros(df.height, dtype=np.float64), 0
    casted = df.select(pl.col(citation_col).cast(pl.Float64, strict=False).alias(citation_col))
    invalid = int(casted.select(pl.col(citation_col).is_null().sum()).item())
    values = casted.get_column(citation_col).fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    values = np.where(values < 0, 0.0, values)
    return values, invalid


def build_aggregate_parts(
    df: pl.DataFrame,
    embeddings: np.ndarray,
    spec: EntitySpec,
    *,
    include_empty_text: bool = False,
    citation_col: str | None = CITATION_COLUMN,
) -> AggregateParts:
    if df.height != embeddings.shape[0]:
        raise ValueError(f"Row mismatch: df has {df.height} rows but embeddings has {embeddings.shape[0]} rows")
    dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    if df.height == 0:
        return AggregateParts(
            meta=pl.DataFrame({col: [] for col in spec.embedding_output_cols}),
            embedding_sums=np.empty((0, dim), dtype=np.float32),
            text_counts=np.empty((0,), dtype=np.float64),
            citation_weighted_sums=np.empty((0, dim), dtype=np.float32),
        )

    keys = df.get_column(spec.key_col).cast(pl.Utf8).to_numpy()
    uniq_keys, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_groups = len(uniq_keys)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if include_empty_text or TEXT_IS_EMPTY_FIELD not in df.columns:
        text_weights = np.ones(df.height, dtype=np.float32)
    else:
        text_weights = (~df.get_column(TEXT_IS_EMPTY_FIELD).fill_null(True).to_numpy()).astype(np.float32)

    embedding_sums = np.zeros((n_groups, emb.shape[1]), dtype=np.float32)
    np.add.at(embedding_sums, inverse, emb * text_weights[:, None])
    text_counts = np.bincount(inverse, weights=text_weights).astype(np.float64)
    all_counts = np.bincount(inverse).astype(np.float64)

    fallback = text_counts == 0
    if np.any(fallback):
        all_sums = np.zeros((n_groups, emb.shape[1]), dtype=np.float32)
        np.add.at(all_sums, inverse, emb)
        embedding_sums[fallback] = all_sums[fallback]
        text_counts[fallback] = all_counts[fallback]
        logging.getLogger(__name__).warning("Some groups had only empty text; using all rows for those groups.")

    citations, invalid = _coerce_citations(df, citation_col)
    if invalid:
        logging.getLogger(__name__).warning("%d invalid citation values were treated as 0", invalid)
    citation_weighted_sums = np.zeros((n_groups, emb.shape[1]), dtype=np.float32)
    np.add.at(citation_weighted_sums, inverse, emb * citations[:, None].astype(np.float32))
    total_citations = np.bincount(inverse, weights=citations).astype(np.float64)

    first_idx = np.full(n_groups, df.height, dtype=np.int64)
    np.minimum.at(first_idx, inverse, np.arange(df.height, dtype=np.int64))
    meta_data: dict[str, list[object]] = {}
    for col in spec.embedding_metadata_cols:
        if col in df.columns:
            meta_data[col] = df.get_column(col).gather(first_idx).to_list()
        else:
            meta_data[col] = [None] * n_groups
    meta_data["n_patents"] = all_counts.astype(np.int64).tolist()
    meta_data["n_texts_used"] = text_counts.astype(np.int64).tolist()
    meta_data["total_citations"] = total_citations.tolist()
    meta_data["mean_citations"] = np.divide(
        total_citations,
        all_counts,
        out=np.full(n_groups, np.nan),
        where=all_counts > 0,
    ).tolist()
    return AggregateParts(pl.DataFrame(meta_data), embedding_sums, text_counts, citation_weighted_sums)


def finalize_aggregates(parts: list[AggregateParts], spec: EntitySpec) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    if not parts:
        return (
            pl.DataFrame({col: [] for col in spec.embedding_output_cols}),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    nonempty = [part for part in parts if part.meta.height > 0]
    if not nonempty:
        dim = parts[0].embedding_sums.shape[1] if parts else 0
        return (
            pl.DataFrame({col: [] for col in spec.embedding_output_cols}),
            np.empty((0, dim), dtype=np.float32),
            np.empty((0, dim), dtype=np.float32),
        )

    meta_all = pl.concat([part.meta for part in nonempty], how="vertical")
    first_cols = [
        pl.col(col).first().alias(col)
        for col in spec.embedding_metadata_cols
        if col != spec.key_col
    ]
    global_meta = (
        meta_all.group_by(spec.key_col)
        .agg(
            first_cols
            + [
                pl.col("n_patents").sum().alias("n_patents"),
                pl.col("n_texts_used").sum().alias("n_texts_used"),
                pl.col("total_citations").sum().alias("total_citations"),
            ]
        )
        .with_columns(
            (pl.col("total_citations") / pl.col("n_patents")).alias("mean_citations")
        )
    )
    ordered_cols = [col for col in spec.embedding_output_cols if col in global_meta.columns]
    global_meta = global_meta.select(ordered_cols).sort(list(spec.sort_cols), maintain_order=True)

    keys = global_meta.get_column(spec.key_col).cast(pl.Utf8).to_list()
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    dim = nonempty[0].embedding_sums.shape[1]
    n_groups = len(keys)
    embedding_sums = np.zeros((n_groups, dim), dtype=np.float64)
    text_counts = np.zeros(n_groups, dtype=np.float64)
    citation_sums = np.zeros((n_groups, dim), dtype=np.float64)

    for part in nonempty:
        part_keys = part.meta.get_column(spec.key_col).cast(pl.Utf8).to_list()
        idx = np.array([key_to_idx[key] for key in part_keys], dtype=np.int64)
        np.add.at(embedding_sums, idx, part.embedding_sums)
        np.add.at(text_counts, idx, part.text_counts)
        np.add.at(citation_sums, idx, part.citation_weighted_sums)

    global_meta = global_meta.with_columns(pl.Series("n_texts_used", text_counts.astype(np.int64)))
    simple = divide_rows(embedding_sums, text_counts)
    citation_denominator = global_meta.get_column("total_citations").to_numpy().astype(np.float64, copy=False)
    weighted = divide_rows(citation_sums, citation_denominator)
    fallback = citation_denominator <= ZERO_DIVISION_EPSILON
    if np.any(fallback):
        weighted[fallback] = simple[fallback]
    return global_meta.select(list(spec.embedding_output_cols)), simple.astype(np.float32), weighted.astype(np.float32)
