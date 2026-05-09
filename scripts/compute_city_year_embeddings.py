#!/usr/bin/env python3
"""Compute city-year patent embeddings."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
import os
from pathlib import Path

from patent_similarity.config import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PATENTS_FILE, resolve_model_name
from patent_similarity.embedding_workflow import run_entity_embedding_pipeline
from patent_similarity.entities import CITY_SPEC
from patent_similarity.runtime import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute city-year patent embeddings")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATENTS_FILE)
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
    parser.add_argument("--include-empty-in-agg", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    model_name = resolve_model_name(args.model, args.model_name)
    run_entity_embedding_pipeline(
        spec=CITY_SPEC,
        input_path=args.input,
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
        save_patent_level=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
