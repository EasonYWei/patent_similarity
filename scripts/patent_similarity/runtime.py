"""Runtime helpers for logging and chunk iteration."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def iter_slices(df: pl.DataFrame, chunk_size: int | None) -> Iterator[pl.DataFrame]:
    if chunk_size is None or chunk_size <= 0 or df.height <= chunk_size:
        yield df
        return
    for offset in range(0, df.height, chunk_size):
        yield df.slice(offset, chunk_size)
