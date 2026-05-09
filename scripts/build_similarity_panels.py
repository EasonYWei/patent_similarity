#!/usr/bin/env python3
"""Build merged firm-level and city-level similarity panels."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
from pathlib import Path

from patent_similarity.config import DEFAULT_CITY_PATENTS_FILE, DEFAULT_OUTPUT_DIR
from patent_similarity.panels import build_panels
from patent_similarity.runtime import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge firm, city, and industry-peer similarity outputs")
    parser.add_argument("--models", "-m", default="minilm,distiluse")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CITY_PATENTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    build_panels(args.models, args.data_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
