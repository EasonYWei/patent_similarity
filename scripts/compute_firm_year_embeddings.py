#!/usr/bin/env python3
"""Compute firm-year patent embeddings."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
import os
from pathlib import Path

from patent_similarity.config import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PATENTS_FILE, resolve_model_name
from patent_similarity.embedding_workflow import run_entity_embedding_pipeline
from patent_similarity.entities import FIRM_SPEC
from patent_similarity.runtime import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute firm-year patent embeddings")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATENTS_FILE)
    parser.add_argument("--data-dir", type=Path, default=None, help="Deprecated alias for --input or data directory")
    parser.add_argument("--model", type=str, default=None, help="Model short name, e.g. minilm or distiluse")
    parser.add_argument("--model-name", type=str, default=None, help="Full local model directory name")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--row-chunk-size", type=int, default=None)
    parser.add_argument("--embed-backend", choices=["overflow", "legacy"], default="overflow")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--tokenizers-parallelism", type=str, default=None)
    parser.add_argument("--process-by-chunk", action="store_true", help="Deprecated; use --row-chunk-size")
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug limit when row chunking is enabled")
    parser.add_argument("--include-empty-in-agg", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--save-patent-level", action="store_true")
    parser.add_argument("--no-save-patent-level", action="store_false", dest="save_patent_level")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(save_patent_level=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    input_path = args.input
    if args.data_dir is not None:
        input_path = args.data_dir / "patents_cleaned.dta" if args.data_dir.is_dir() else args.data_dir
    if args.process_by_chunk and args.row_chunk_size is None:
        args.row_chunk_size = 100_000
    model_name = resolve_model_name(args.model, args.model_name)
    run_entity_embedding_pipeline(
        spec=FIRM_SPEC,
        input_path=input_path,
        model_dir=args.model_dir,
        model_name=model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        multi_gpu=args.multi_gpu,
        row_chunk_size=args.row_chunk_size,
        embed_backend=args.embed_backend,
        max_seq_length=args.max_seq_length,
        fp16=args.fp16,
        tf32=args.tf32,
        include_empty_in_agg=args.include_empty_in_agg,
        save_npy=args.save_npy,
        save_patent_level=args.save_patent_level,
        max_chunks=args.max_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
