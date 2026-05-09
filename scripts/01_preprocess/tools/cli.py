"""Command-line parsing for preprocessing scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_DATA_DIR


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def add_verbose(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_city_enrichment_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patents_cleaned_with_city.dta from raw patents.dta"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "patents.dta")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "patents_cleaned_with_city.dta",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Read raw Stata data in row chunks"
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Optional debug limit after filtering",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional debug limit on raw chunks read",
    )
    return add_verbose(parser).parse_args(argv)


def parse_parquet_split_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split data/patents.dta into 100-stock-code Parquet range files."
    )
    parser.add_argument("--input", type=Path, default=Path("data/patents.dta"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/patents_ranges"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--delete-source-after-verify", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output parquet files and temp parts before conversion.",
    )
    parser.add_argument(
        "--cleanup-failed-temp",
        action="store_true",
        help="Remove temporary parquet parts if conversion or validation fails.",
    )
    return add_verbose(parser).parse_args(argv)
