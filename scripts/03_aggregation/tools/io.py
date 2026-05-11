"""Patent-level bundle and aggregate embedding I/O."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from .config import PATENT_ROW_ID, model_suffix, patent_level_paths


def ensure_columns(df: pl.DataFrame, required: Iterable[str], source: str | Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}. Found columns: {df.columns}")


def read_csv(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pl.read_csv(path, infer_schema_length=10_000)


def load_patent_level_bundle(
    patent_level_dir: Path,
    model: str,
    *,
    meta_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    default_meta, default_embeddings = patent_level_paths(patent_level_dir, model)
    meta_path = meta_path or default_meta
    embeddings_path = embeddings_path or default_embeddings
    meta = read_csv(meta_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if meta.height != embeddings.shape[0]:
        raise ValueError(
            f"Patent-level row mismatch: {meta_path} has {meta.height} rows but {embeddings_path} has {embeddings.shape[0]} rows"
        )
    if PATENT_ROW_ID not in meta.columns:
        meta = meta.with_row_count(PATENT_ROW_ID)
    return meta, embeddings


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
    suffix = model_suffix(model_short)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape((embeddings.shape[0], 1))
    emb_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
    emb_df = pl.DataFrame(emb_cols) if emb_cols else pl.DataFrame()
    if emb_cols:
        emb_df = emb_df.with_columns(
            [
                pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
                for col in emb_df.columns
            ]
        )
    full = pl.concat([meta, emb_df], how="horizontal") if emb_cols else meta
    parquet_path = output_dir / f"{prefix}{suffix}_embeddings.parquet"
    full.write_parquet(parquet_path)
    logging.getLogger(__name__).info("Saved embeddings Parquet: %s", parquet_path)

    if save_npy:
        meta_path = output_dir / f"{prefix}{suffix}_meta.parquet"
        emb_path = output_dir / f"{prefix}{suffix}_embeddings.npy"
        meta.write_parquet(meta_path)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))
        logging.getLogger(__name__).info("Saved metadata Parquet: %s", meta_path)
        logging.getLogger(__name__).info("Saved embeddings NPY: %s", emb_path)
