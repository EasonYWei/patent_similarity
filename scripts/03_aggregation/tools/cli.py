"""Command-line parsing for aggregation scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_PATENT_LEVEL_DIR, DEFAULT_STKCD_INFO_FILE


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_aggregation_cli(
    description: str,
    argv: list[str] | None = None,
    *,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    default_patent_input_dir: Path = DEFAULT_PATENT_LEVEL_DIR,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="minilm", help="Model short name or full model name")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--patent-input-dir",
        type=Path,
        default=default_patent_input_dir,
        help="Directory containing patent-level metadata and embeddings",
    )
    parser.add_argument("--patent-meta", type=Path, default=None, help="Override patent-level metadata file")
    parser.add_argument("--patent-embeddings", type=Path, default=None, help="Override patent-level embeddings NPY")
    parser.add_argument(
        "--balance-path",
        type=Path,
        default=DEFAULT_STKCD_INFO_FILE,
        help="Firm-year metadata CSV used to build balanced city/industry panels",
    )
    parser.add_argument("--row-chunk-size", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug limit when row chunking is enabled")
    parser.add_argument("--include-empty-in-agg", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)
