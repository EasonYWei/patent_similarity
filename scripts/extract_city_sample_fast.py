#!/usr/bin/env python3
"""Compatibility wrapper for `scripts/build_city_enriched_patents.py` fast sample mode."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import sys

from build_city_enriched_patents import main


if __name__ == "__main__":
    default_args = ["--target-rows", "500000", "--chunk-size", "100000", "--max-chunks", "10"]
    raise SystemExit(main(sys.argv[1:] or default_args))
