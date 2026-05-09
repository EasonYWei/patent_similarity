"""Lagged cosine similarity calculations for entity-year embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import SAFE_COSINE_TOLERANCE, embedding_csv_name, model_short_name, similarity_csv_name
from .entities import EntitySpec
from .io import embedding_columns, ensure_columns, read_csv, write_csv

SIMILARITY_COLUMNS = ("cos_sim_lag1", "cos_sim_lag3", "cos_sim_cumulative")
CITW_SIMILARITY_COLUMNS = ("cos_sim_lag1_citw", "cos_sim_lag3_citw", "cos_sim_cumulative_citw")


def safe_cosine_similarity(v1: np.ndarray, v2: np.ndarray, tolerance: float = SAFE_COSINE_TOLERANCE) -> float:
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


def calculate_entity_similarities(df: pl.DataFrame, spec: EntitySpec, emb_cols: list[str] | None = None) -> pl.DataFrame:
    """Calculate lag-1, lag-3, and cumulative similarities by entity."""
    ensure_columns(df, [*spec.similarity_metadata_cols], spec.name)
    emb_cols = emb_cols or embedding_columns(df)
    df = df.with_columns(
        [
            pl.col(spec.id_col).cast(pl.Utf8, strict=False).str.strip_chars().alias(spec.id_col),
            pl.col(spec.year_col).cast(pl.Int64, strict=False).alias(spec.year_col),
        ]
    ).sort([spec.id_col, spec.year_col], maintain_order=True)

    embeddings = df.select(emb_cols).to_numpy().astype(np.float64, copy=False)
    ids = df.get_column(spec.id_col).to_list()
    result = df.select([col for col in spec.similarity_metadata_cols if col in df.columns]).to_dict(as_series=False)
    n = df.height
    lag1 = np.full(n, np.nan, dtype=np.float64)
    lag3 = np.full(n, np.nan, dtype=np.float64)
    cumulative = np.full(n, np.nan, dtype=np.float64)

    start = 0
    while start < n:
        entity = ids[start]
        end = start + 1
        while end < n and ids[end] == entity:
            end += 1
        group = embeddings[start:end]
        running_sum = np.zeros(group.shape[1], dtype=np.float64)
        for offset in range(group.shape[0]):
            idx = start + offset
            current = group[offset]
            if offset >= 1:
                lag1[idx] = safe_cosine_similarity(current, group[offset - 1])
                cumulative[idx] = safe_cosine_similarity(current, running_sum / offset)
            if offset >= 3:
                lag3[idx] = safe_cosine_similarity(current, group[offset - 3 : offset].mean(axis=0))
            running_sum += current
        start = end

    result["cos_sim_lag1"] = lag1
    result["cos_sim_lag3"] = lag3
    result["cos_sim_cumulative"] = cumulative
    return pl.DataFrame(result)


def _summarize(result: pl.DataFrame, label: str) -> None:
    log = logging.getLogger(__name__)
    for col in SIMILARITY_COLUMNS:
        if col not in result.columns:
            continue
        stats = result.select(
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).count().alias("n"),
        ).row(0, named=True)
        log.info("%s %s: mean=%.4f sd=%.4f n=%d", label, col, stats["mean"] or np.nan, stats["std"] or np.nan, stats["n"])


def run_similarity_for_model(spec: EntitySpec, model: str, output_dir: Path) -> None:
    """Load embedding CSVs, calculate similarities, and write compatible outputs."""
    short = model_short_name(model)
    simple_input = output_dir / embedding_csv_name(spec.output_prefix, short)
    cit_input = output_dir / embedding_csv_name(f"{spec.output_prefix}_citweighted", short)
    simple_output = output_dir / similarity_csv_name(spec.output_prefix, short)
    cit_output = output_dir / similarity_csv_name(spec.output_prefix, short, weighted=True)
    merged_output = output_dir / similarity_csv_name(spec.output_prefix, short, merged=True)

    simple_df = read_csv(simple_input)
    simple_emb_cols = embedding_columns(simple_df)
    result_simple = calculate_entity_similarities(simple_df, spec, simple_emb_cols)
    write_csv(result_simple, simple_output)
    logging.getLogger(__name__).info("Saved simple similarity: %s", simple_output)
    _summarize(result_simple, "Simple")

    if not cit_input.exists():
        logging.getLogger(__name__).warning("Citation-weighted input not found, skipping: %s", cit_input)
        return

    cit_df = read_csv(cit_input)
    cit_emb_cols = embedding_columns(cit_df)
    result_cit = calculate_entity_similarities(cit_df, spec, cit_emb_cols).rename(
        dict(zip(SIMILARITY_COLUMNS, CITW_SIMILARITY_COLUMNS))
    )
    write_csv(result_cit, cit_output)
    logging.getLogger(__name__).info("Saved citation-weighted similarity: %s", cit_output)

    join_keys = [spec.id_col, spec.year_col]
    merged = result_simple.join(
        result_cit.select(join_keys + list(CITW_SIMILARITY_COLUMNS)),
        on=join_keys,
        how="outer",
    ).sort(join_keys, maintain_order=True)
    write_csv(merged, merged_output)
    logging.getLogger(__name__).info("Saved merged similarity: %s", merged_output)
