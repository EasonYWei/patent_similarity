"""Entity-year specifications shared by firm and city workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    CITY_CODE_COLUMN,
    CITY_COLUMN,
    CITATION_COLUMN,
    PROVINCE_COLUMN,
    STKCD_COLUMN,
    TEXT_COLUMNS,
    YEAR_COLUMN,
)


@dataclass(frozen=True)
class EntitySpec:
    """Configuration for one entity-year aggregation and similarity workflow."""

    name: str
    id_col: str
    year_col: str
    key_col: str
    required_cols: tuple[str, ...]
    read_cols: tuple[str, ...]
    sort_cols: tuple[str, ...]
    output_prefix: str
    embedding_metadata_cols: tuple[str, ...]
    similarity_metadata_cols: tuple[str, ...]

    @property
    def embedding_output_cols(self) -> tuple[str, ...]:
        return self.embedding_metadata_cols + (
            "n_patents",
            "n_texts_used",
            "total_citations",
            "mean_citations",
        )


FIRM_SPEC = EntitySpec(
    name="firm",
    id_col=STKCD_COLUMN,
    year_col=YEAR_COLUMN,
    key_col="stkcd_year",
    required_cols=(STKCD_COLUMN, YEAR_COLUMN, *TEXT_COLUMNS),
    read_cols=(STKCD_COLUMN, YEAR_COLUMN, *TEXT_COLUMNS, "p_date", CITATION_COLUMN),
    sort_cols=(STKCD_COLUMN, YEAR_COLUMN),
    output_prefix="stkcd_year",
    embedding_metadata_cols=(STKCD_COLUMN, YEAR_COLUMN, "stkcd_year"),
    similarity_metadata_cols=(STKCD_COLUMN, YEAR_COLUMN, "n_patents", "n_texts_used"),
)

CITY_SPEC = EntitySpec(
    name="city",
    id_col=CITY_CODE_COLUMN,
    year_col=YEAR_COLUMN,
    key_col="city_year",
    required_cols=(CITY_CODE_COLUMN, YEAR_COLUMN, *TEXT_COLUMNS),
    read_cols=(
        CITY_COLUMN,
        CITY_CODE_COLUMN,
        PROVINCE_COLUMN,
        "province_code",
        YEAR_COLUMN,
        *TEXT_COLUMNS,
        "p_date",
        CITATION_COLUMN,
    ),
    sort_cols=(CITY_CODE_COLUMN, YEAR_COLUMN),
    output_prefix="city_year",
    embedding_metadata_cols=(CITY_COLUMN, CITY_CODE_COLUMN, PROVINCE_COLUMN, YEAR_COLUMN, "city_year"),
    similarity_metadata_cols=(CITY_CODE_COLUMN, YEAR_COLUMN, CITY_COLUMN, "n_patents", "n_texts_used"),
)

ENTITY_SPECS = {
    FIRM_SPEC.name: FIRM_SPEC,
    CITY_SPEC.name: CITY_SPEC,
}


def get_entity_spec(name: str) -> EntitySpec:
    try:
        return ENTITY_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ENTITY_SPECS))
        raise ValueError(f"Unknown entity {name!r}. Expected one of: {known}") from exc
