#!/usr/bin/env python3
"""Compatibility wrapper for `scripts/compute_industry_peer_similarity.py`."""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

from compute_industry_peer_similarity import main


if __name__ == "__main__":
    raise SystemExit(main())
