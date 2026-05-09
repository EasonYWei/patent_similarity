"""Command-line parsing for similarity scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_CITY_PATENTS_FILE, DEFAULT_INDUSTRY_FILE, DEFAULT_OUTPUT_DIR


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def add_verbose(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--verbose", action="store_true")
    return parser


def add_model_list(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--models", "-m", default="minilm,distiluse")
    return parser


def parse_similarity_cli(description: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="distiluse", help="Model short name or full model name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return add_verbose(parser).parse_args(argv)


def parse_industry_peer_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute industry-peer similarity for patent embeddings")
    add_model_list(parser)
    parser.add_argument("--industry-path", type=Path, default=DEFAULT_INDUSTRY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Accepted for compatibility; the Polars path is single-process vectorized",
    )
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="Accepted for compatibility; no files are deleted by the refactored script",
    )
    return add_verbose(parser).parse_args(argv)


def parse_panel_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge firm, city, and industry-peer similarity outputs")
    add_model_list(parser)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CITY_PATENTS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return add_verbose(parser).parse_args(argv)
