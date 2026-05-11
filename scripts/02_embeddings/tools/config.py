"""Embedding stage constants and model naming helpers."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "patent_embeddings"
DEFAULT_PATENTS_FILE = DEFAULT_DATA_DIR / "patents_cleaned.parquet"
DEFAULT_CITY_PATENTS_FILE = DEFAULT_DATA_DIR / "patents_cleaned_with_city.dta"
DEFAULT_PATENT_LEVEL_INPUT = DEFAULT_PATENTS_FILE

STKCD_COLUMN = "stkcd"
CITY_COLUMN = "city"
CITY_CODE_COLUMN = "city_code"
PROVINCE_COLUMN = "province"
PROVINCE_CODE_COLUMN = "province_code"
YEAR_COLUMN = "p_year"
TEXT_COLUMNS = ("p_tt", "p_abs")
CITATION_COLUMN = "p_cite"
TEXT_FIELD = "text"
TEXT_IS_EMPTY_FIELD = "text_is_empty"
PATENT_ROW_ID = "patent_row_id"

RAW_COLUMN_RENAMES = {
    "股票代码": STKCD_COLUMN,
    "年份": YEAR_COLUMN,
    "标题": "p_tt",
    "摘要": "p_abs",
    "申请日": "p_date",
    "专利类型": "p_type",
    "IPC": "p_ipc",
    "被引证次数": CITATION_COLUMN,
    "市": CITY_COLUMN,
    "市代码": CITY_CODE_COLUMN,
    "省": PROVINCE_COLUMN,
    "省代码": PROVINCE_CODE_COLUMN,
    "newipzlid": "p_id",
}

PATENT_LEVEL_COLUMNS = (
    PATENT_ROW_ID,
    "p_id",
    STKCD_COLUMN,
    YEAR_COLUMN,
    "stkcd_year",
    CITY_COLUMN,
    CITY_CODE_COLUMN,
    PROVINCE_COLUMN,
    PROVINCE_CODE_COLUMN,
    "Listdt",
    "county",
    "countyID",
    "Ind",
    "city_year",
    "p_date",
    "p_type",
    "p_ipc",
    CITATION_COLUMN,
    TEXT_IS_EMPTY_FIELD,
)

MODEL_SHORT_NAMES = {
    "paraphrase-multilingual-MiniLM-L12-v2": "minilm",
    "distiluse-base-multilingual-cased-v2": "distiluse",
}
MODEL_NAMES_BY_SHORT = {short: full for full, short in MODEL_SHORT_NAMES.items()}


def model_short_name(model_name_or_short: str) -> str:
    """Return the stable short model name used in output filenames."""
    value = str(model_name_or_short).strip()
    if value in MODEL_NAMES_BY_SHORT:
        return value
    return MODEL_SHORT_NAMES.get(value, value.split("-")[0].lower())


def resolve_model_name(model: str | None = None, model_name: str | None = None) -> str:
    """Resolve a CLI model alias or full model directory name."""
    if model_name:
        return str(model_name).strip()
    value = (model or "minilm").strip()
    return MODEL_NAMES_BY_SHORT.get(value, value)
