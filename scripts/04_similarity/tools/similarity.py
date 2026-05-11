"""Lagged cosine similarity calculations for entity-year embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import (
    SAFE_COSINE_TOLERANCE,
    embedding_parquet_name,
    model_short_name,
    similarity_parquet_name,
)
from .entities import EntitySpec
from .io import embedding_columns, ensure_columns, read_frame, write_parquet

SIMILARITY_LAGS = (1, 2, 3, 4, 5)
SIMILARITY_COLUMNS = tuple(f"cos_sim_lag{lag}" for lag in SIMILARITY_LAGS) + (
    "cos_sim_cumulative",
)
CITW_SIMILARITY_COLUMNS = tuple(f"{col}_citw" for col in SIMILARITY_COLUMNS)


def safe_cosine_similarity(
    v1: np.ndarray, v2: np.ndarray, tolerance: float = SAFE_COSINE_TOLERANCE
) -> float:
    """Return cosine similarity, or NaN for undefined inputs."""
    if v1.size == 0 or v2.size == 0 or v1.shape != v2.shape:
        return np.nan
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        return np.nan
    n1 = float(np.sqrt(np.sum(v1 * v1)))
    n2 = float(np.sqrt(np.sum(v2 * v2)))
    if n1 <= tolerance or n2 <= tolerance:
        return np.nan
    return float(np.sum(v1 * v2) / (n1 * n2))


def _rowwise_cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = SAFE_COSINE_TOLERANCE,
) -> np.ndarray:
    """Return row-wise cosine similarities, with NaN for undefined rows."""
    if left.shape != right.shape:
        raise ValueError(f"Cosine input shape mismatch: {left.shape} != {right.shape}")
    out = np.full(left.shape[0], np.nan, dtype=np.float64)
    if left.size == 0:
        return out

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left_norms = np.sqrt(np.sum(left * left, axis=1))
    right_norms = np.sqrt(np.sum(right * right, axis=1))
    valid = finite & (left_norms > tolerance) & (right_norms > tolerance)
    if np.any(valid):
        out[valid] = np.sum(left[valid] * right[valid], axis=1) / (
            left_norms[valid] * right_norms[valid]
        )
    return out


def _replace_nan_with_nulls(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    exprs = [
        pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        for col in columns
        if col in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def calculate_entity_similarities(
    df: pl.DataFrame, spec: EntitySpec, emb_cols: list[str] | None = None
) -> pl.DataFrame:
    """Calculate lagged and cumulative similarities by entity."""
    ensure_columns(df, [*spec.similarity_metadata_cols], spec.name)
    emb_cols = emb_cols or embedding_columns(df)
    df = df.with_columns(
        [
            pl.col(spec.id_col)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .alias(spec.id_col),
            pl.col(spec.year_col).cast(pl.Int64, strict=False).alias(spec.year_col),
        ]
    ).sort([spec.id_col, spec.year_col], maintain_order=True)

    embeddings = df.select(emb_cols).to_numpy().astype(np.float64, copy=False)
    ids = df.get_column(spec.id_col).to_list()
    result = df.select(
        [col for col in spec.similarity_metadata_cols if col in df.columns]
    )
    n = df.height
    lag_values = {
        f"cos_sim_lag{lag}": np.full(n, np.nan, dtype=np.float64)
        for lag in SIMILARITY_LAGS
    }
    cumulative = np.full(n, np.nan, dtype=np.float64)

    start = 0
    while start < n:
        entity = ids[start]
        end = start + 1
        while end < n and ids[end] == entity:
            end += 1
        group = embeddings[start:end]
        group_size = group.shape[0]
        if group_size >= 2:
            prefix = np.vstack(
                [
                    np.zeros((1, group.shape[1]), dtype=np.float64),
                    np.cumsum(group, axis=0),
                ]
            )
            previous_counts = np.arange(1, group_size, dtype=np.float64)[:, None]
            previous_means = prefix[1:group_size] / previous_counts
            cumulative[start + 1 : end] = _rowwise_cosine_similarity(
                group[1:group_size], previous_means
            )

            for lag in SIMILARITY_LAGS:
                if group_size <= lag:
                    continue
                rolling_means = (
                    prefix[lag:group_size] - prefix[: group_size - lag]
                ) / lag
                lag_values[f"cos_sim_lag{lag}"][start + lag : end] = (
                    _rowwise_cosine_similarity(
                        group[lag:group_size],
                        rolling_means,
                    )
                )
        start = end

    result = result.with_columns(
        [pl.Series(col, values) for col, values in lag_values.items()]
        + [pl.Series("cos_sim_cumulative", cumulative)]
    )
    return _replace_nan_with_nulls(result, SIMILARITY_COLUMNS)


def _summarize(result: pl.DataFrame, label: str) -> None:
    log = logging.getLogger(__name__)
    for col in SIMILARITY_COLUMNS:
        if col not in result.columns:
            continue
        clean = pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None)
        stats = result.select(
            clean.mean().alias("mean"),
            clean.std().alias("std"),
            clean.count().alias("n"),
        ).row(0, named=True)
        mean = stats["mean"] if stats["mean"] is not None else np.nan
        std = stats["std"] if stats["std"] is not None else np.nan
        log.info("%s %s: mean=%.4f sd=%.4f n=%d", label, col, mean, std, stats["n"])


def run_similarity_for_model(
    spec: EntitySpec,
    model: str,
    input_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Load embedding Parquet files, calculate similarities, and write Parquet outputs."""
    short = model_short_name(model)
    output_dir = output_dir or input_dir
    simple_input = input_dir / embedding_parquet_name(spec.output_prefix, short)
    cit_input = input_dir / embedding_parquet_name(
        f"{spec.output_prefix}_citweighted", short
    )
    simple_output = output_dir / similarity_parquet_name(spec.output_prefix, short)
    cit_output = output_dir / similarity_parquet_name(
        spec.output_prefix, short, weighted=True
    )
    merged_output = output_dir / similarity_parquet_name(
        spec.output_prefix, short, merged=True
    )

    simple_df = read_frame(simple_input)
    simple_emb_cols = embedding_columns(simple_df)
    result_simple = calculate_entity_similarities(simple_df, spec, simple_emb_cols)
    write_parquet(result_simple, simple_output)
    logging.getLogger(__name__).info("Saved simple similarity: %s", simple_output)
    _summarize(result_simple, "Simple")

    if not cit_input.exists():
        logging.getLogger(__name__).warning(
            "Citation-weighted input not found, skipping: %s", cit_input
        )
        return

    cit_df = read_frame(cit_input)
    cit_emb_cols = embedding_columns(cit_df)
    result_cit = calculate_entity_similarities(cit_df, spec, cit_emb_cols).rename(
        dict(zip(SIMILARITY_COLUMNS, CITW_SIMILARITY_COLUMNS))
    )
    write_parquet(result_cit, cit_output)
    logging.getLogger(__name__).info(
        "Saved citation-weighted similarity: %s", cit_output
    )

    join_keys = [spec.id_col, spec.year_col]
    merged = result_simple.join(
        result_cit.select(join_keys + list(CITW_SIMILARITY_COLUMNS)),
        on=join_keys,
        how="left",
    ).sort(join_keys, maintain_order=True)
    write_parquet(merged, merged_output)
    logging.getLogger(__name__).info("Saved merged similarity: %s", merged_output)
