#!/usr/bin/env python3
"""Compute industry-peer patent similarity metrics."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
from pathlib import Path

from patent_similarity.config import DEFAULT_INDUSTRY_FILE, DEFAULT_OUTPUT_DIR
from patent_similarity.industry_peer import process_models
from patent_similarity.runtime import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute industry-peer similarity for patent embeddings")
    parser.add_argument("--models", "-m", default="minilm,distiluse")
    parser.add_argument("--industry-path", type=Path, default=DEFAULT_INDUSTRY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", "-w", type=int, default=None, help="Accepted for compatibility; the Polars path is single-process vectorized")
    parser.add_argument("--clean", "-c", action="store_true", help="Accepted for compatibility; no files are deleted by the refactored script")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.clean:
        raise ValueError("--clean is disabled in the refactored script to avoid deleting generated outputs implicitly")
    process_models(args.models, args.industry_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
