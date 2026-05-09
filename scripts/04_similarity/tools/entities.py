"""Entity-year specifications for similarity calculations."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CITY_CODE_COLUMN, CITY_COLUMN, STKCD_COLUMN, YEAR_COLUMN


@dataclass(frozen=True)
class EntitySpec:
    name: str
    id_col: str
    year_col: str
    sort_cols: tuple[str, ...]
    output_prefix: str
    similarity_metadata_cols: tuple[str, ...]


FIRM_SPEC = EntitySpec(
    name="firm",
    id_col=STKCD_COLUMN,
    year_col=YEAR_COLUMN,
    sort_cols=(STKCD_COLUMN, YEAR_COLUMN),
    output_prefix="stkcd_year",
    similarity_metadata_cols=(STKCD_COLUMN, YEAR_COLUMN, "n_patents", "n_texts_used"),
)

CITY_SPEC = EntitySpec(
    name="city",
    id_col=CITY_CODE_COLUMN,
    year_col=YEAR_COLUMN,
    sort_cols=(CITY_CODE_COLUMN, YEAR_COLUMN),
    output_prefix="city_year",
    similarity_metadata_cols=(CITY_CODE_COLUMN, YEAR_COLUMN, CITY_COLUMN, "n_patents", "n_texts_used"),
)
