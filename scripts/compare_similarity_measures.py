#!/usr/bin/env python3
"""Compatibility wrapper for `scripts/summarize_similarity_outputs.py`."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

from summarize_similarity_outputs import main


if __name__ == "__main__":
    raise SystemExit(main())
