#!/usr/bin/env python3
"""Compatibility wrapper for `scripts/build_similarity_panels.py`."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

from build_similarity_panels import main


if __name__ == "__main__":
    raise SystemExit(main())
