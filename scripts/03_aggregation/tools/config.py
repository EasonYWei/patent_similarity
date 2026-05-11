"""Aggregation stage constants and output naming helpers."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PATENT_LEVEL_DIR = DEFAULT_OUTPUT_DIR / "patent_embeddings"
DEFAULT_FIRM_YEAR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "firm_year_embeddings"
DEFAULT_CITY_YEAR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "city_year_embeddings"
DEFAULT_INDUSTRY_YEAR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "industry_year_embeddings"
DEFAULT_STKCD_INFO_FILE = PROJECT_ROOT / "data" / "stkcd_info.csv"

STKCD_COLUMN = "stkcd"
CITY_COLUMN = "city"
CITY_CODE_COLUMN = "city_code"
COUNTY_ID_COLUMN = "countyID"
PROVINCE_COLUMN = "province"
PROVINCE_CODE_COLUMN = "province_code"
INDUSTRY_COLUMN = "Ind"
YEAR_COLUMN = "p_year"
CITATION_COLUMN = "p_cite"
TEXT_IS_EMPTY_FIELD = "text_is_empty"
PATENT_ROW_ID = "patent_row_id"
ZERO_DIVISION_EPSILON = 1e-12

MODEL_SHORT_NAMES = {
    "paraphrase-multilingual-MiniLM-L12-v2": "minilm",
    "distiluse-base-multilingual-cased-v2": "distiluse",
}
MODEL_NAMES_BY_SHORT = {short: full for full, short in MODEL_SHORT_NAMES.items()}


def model_short_name(model_name_or_short: str) -> str:
    value = str(model_name_or_short).strip()
    if value in MODEL_NAMES_BY_SHORT:
        return value
    return MODEL_SHORT_NAMES.get(value, value.split("-")[0].lower())


def model_suffix(model_name_or_short: str) -> str:
    short = model_short_name(model_name_or_short)
    return f"_{short}" if short else ""


def patent_level_paths(patent_level_dir: Path, model_name_or_short: str) -> tuple[Path, Path]:
    suffix = model_suffix(model_name_or_short)
    return patent_level_dir / f"patent_level{suffix}_meta.csv", patent_level_dir / f"patent_level{suffix}_embeddings.npy"
