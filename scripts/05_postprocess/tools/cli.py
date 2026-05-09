"""Command-line parsing for postprocess scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_summary_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize firm, peer, and city similarity measures")
    parser.add_argument("--models", "-m", default="minilm,distiluse")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)
