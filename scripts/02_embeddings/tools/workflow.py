"""Patent-level embedding workflow."""

from __future__ import annotations

import logging
from itertools import islice
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from .config import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, TEXT_FIELD, model_short_name
from .io import patent_level_meta, patent_level_paths, prepare_patent_records
from .model import SBertEmbedder, get_gpu_info, recommend_batch_size


def iter_slices(df: pl.DataFrame, chunk_size: int | None):
    if chunk_size is None or chunk_size <= 0 or df.height <= chunk_size:
        yield df
        return
    for offset in range(0, df.height, chunk_size):
        yield df.slice(offset, chunk_size)


def choose_batch_size(batch_size: int | None, model_name: str) -> int:
    if batch_size is not None:
        return int(batch_size)
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        chosen = recommend_batch_size(gpu_info["memory_total"], model_name)
        logging.getLogger(__name__).info(
            "Auto-selected batch size %d for GPU %s %.1fGB",
            chosen,
            gpu_info["name"],
            gpu_info["memory_total"],
        )
        return chosen
    logging.getLogger(__name__).info("Auto-selected CPU batch size 64")
    return 64


def _limit_to_chunks(df: pl.DataFrame, row_chunk_size: int | None, max_chunks: int | None) -> pl.DataFrame:
    if max_chunks is None:
        return df
    parts = list(islice(iter_slices(df, row_chunk_size), max_chunks))
    return pl.concat(parts, how="vertical") if parts else df.head(0)


def run_patent_level_embedding_pipeline(
    *,
    input_path: Path,
    model_dir: Path = DEFAULT_MODELS_DIR,
    model_name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int | None = None,
    device: str | None = None,
    multi_gpu: bool = False,
    row_chunk_size: int | None = None,
    embed_backend: str = "overflow",
    max_seq_length: int | None = None,
    fp16: bool = False,
    tf32: bool = False,
    max_chunks: int | None = None,
) -> None:
    df = _limit_to_chunks(prepare_patent_records(input_path), row_chunk_size, max_chunks)
    short = model_short_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path, emb_path = patent_level_paths(output_dir, short)
    patent_level_meta(df).write_csv(meta_path)
    logging.getLogger(__name__).info("Saved patent-level metadata: %s", meta_path)

    if df.is_empty():
        np.save(emb_path, np.empty((0, 0), dtype=np.float32))
        return

    selected_batch_size = choose_batch_size(batch_size, model_name)
    embedder = SBertEmbedder(
        model_dir=model_dir,
        model_name=model_name,
        device=device,
        multi_gpu=multi_gpu,
        fp16=fp16,
        tf32=tf32,
        max_seq_length=max_seq_length,
        embed_backend=embed_backend,
    )

    memmap: np.memmap | None = None
    offset = 0
    try:
        for chunk in iter_slices(df, row_chunk_size):
            texts = chunk.get_column(TEXT_FIELD).to_list()
            embeddings = embedder.embed(texts, batch_size=selected_batch_size, show_progress=True)
            if memmap is None:
                memmap = open_memmap(
                    emb_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(df.height, embeddings.shape[1]),
                )
            next_offset = offset + embeddings.shape[0]
            memmap[offset:next_offset, :] = embeddings.astype(np.float32, copy=False)
            offset = next_offset
        if memmap is None:
            np.save(emb_path, np.empty((0, 0), dtype=np.float32))
        else:
            memmap.flush()
            del memmap
        logging.getLogger(__name__).info("Saved patent-level embeddings: %s rows=%d", emb_path, offset)
    finally:
        embedder.close()
