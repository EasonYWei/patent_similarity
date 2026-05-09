"""Shared entity-year embedding workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .aggregation import AggregateParts, build_aggregate_parts, finalize_aggregates
from .config import CITATION_COLUMN, DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, TEXT_FIELD, model_short_name
from .embedding_model import SBertEmbedder, get_gpu_info, recommend_batch_size
from .entities import EntitySpec
from .io import prepare_patent_data, save_embedding_bundle, save_patent_level_embeddings, select_existing
from .runtime import iter_slices


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


def _patent_level_meta(df: pl.DataFrame, spec: EntitySpec) -> pl.DataFrame:
    cols = [spec.id_col, spec.year_col, spec.key_col, "p_date", "text_is_empty", CITATION_COLUMN]
    return select_existing(df, cols)


def run_entity_embedding_pipeline(
    *,
    spec: EntitySpec,
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
    include_empty_in_agg: bool = False,
    save_npy: bool = False,
    save_patent_level: bool = False,
    max_chunks: int | None = None,
) -> None:
    df = prepare_patent_data(input_path, spec)
    short = model_short_name(model_name)
    if df.is_empty():
        empty_meta = pl.DataFrame({col: [] for col in spec.embedding_output_cols})
        save_embedding_bundle(output_dir, spec.output_prefix, empty_meta, np.empty((0, 0), dtype=np.float32), model_short=short, save_npy=save_npy)
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

    aggregate_parts: list[AggregateParts] = []
    patent_embeddings: list[np.ndarray] = []
    patent_meta_parts: list[pl.DataFrame] = []

    try:
        for chunk_index, chunk in enumerate(iter_slices(df, row_chunk_size), start=1):
            if max_chunks is not None and chunk_index > max_chunks:
                logging.getLogger(__name__).warning("Stopping after --max-chunks=%d", max_chunks)
                break
            texts = chunk.get_column(TEXT_FIELD).to_list()
            embeddings = embedder.embed(texts, batch_size=selected_batch_size, show_progress=True)
            aggregate_parts.append(
                build_aggregate_parts(
                    chunk,
                    embeddings,
                    spec,
                    include_empty_text=include_empty_in_agg,
                    citation_col=CITATION_COLUMN,
                )
            )
            if save_patent_level:
                patent_embeddings.append(embeddings.astype(np.float32, copy=False))
                patent_meta_parts.append(_patent_level_meta(chunk, spec))
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
        if save_patent_level:
            patent_meta = pl.concat(patent_meta_parts, how="vertical") if patent_meta_parts else _patent_level_meta(df, spec)
            save_patent_level_embeddings(output_dir, patent_meta, np.vstack(patent_embeddings), model_short=short)
    finally:
        embedder.close()
