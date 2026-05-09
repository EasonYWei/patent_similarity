#!/usr/bin/env python3
"""Compute firm-year lagged similarity metrics from embedding CSVs."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
from pathlib import Path

from patent_similarity.config import DEFAULT_OUTPUT_DIR, model_short_name
from patent_similarity.entities import FIRM_SPEC
from patent_similarity.runtime import setup_logging
from patent_similarity.similarity import run_similarity_for_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute firm-year patent similarity")
    parser.add_argument("--model", default="distiluse", help="Model short name or full model name")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    run_similarity_for_model(FIRM_SPEC, model_short_name(args.model), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
