"""Command-line parsing for patent-level embedding generation."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PATENT_LEVEL_INPUT


def setup_cli_runtime(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tokenizers_parallelism = getattr(args, "tokenizers_parallelism", None)
    if tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = tokenizers_parallelism


def parse_patent_embedding_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute patent-level text embeddings")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATENT_LEVEL_INPUT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --input or a directory containing patents_cleaned_with_city.dta",
    )
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
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug limit when row chunking is enabled")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def embedding_input_path(args: argparse.Namespace) -> Path:
    data_dir = getattr(args, "data_dir", None)
    if data_dir is None:
        return args.input
    return data_dir / "patents_cleaned_with_city.dta" if data_dir.is_dir() else data_dir
