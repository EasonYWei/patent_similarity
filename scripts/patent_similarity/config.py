"""Project constants and naming helpers."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PATENTS_FILE = DEFAULT_DATA_DIR / "patents_cleaned.dta"
DEFAULT_CITY_PATENTS_FILE = DEFAULT_DATA_DIR / "patents_cleaned_with_city.dta"
DEFAULT_INDUSTRY_FILE = DEFAULT_DATA_DIR / "stkcd_info.xlsx"

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

SAFE_COSINE_TOLERANCE = 1e-12
ZERO_DIVISION_EPSILON = 1e-12

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


def model_suffix(model_name_or_short: str) -> str:
    """Return the output suffix, including the leading underscore."""
    short = model_short_name(model_name_or_short)
    return f"_{short}" if short else ""


def parse_models(models: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse a comma-separated model list into short names."""
    if models is None:
        return ["minilm", "distiluse"]
    if isinstance(models, str):
        parts = [part.strip() for part in models.split(",")]
    else:
        parts = [str(part).strip() for part in models]
    parsed = [model_short_name(part) for part in parts if part]
    if not parsed:
        raise ValueError("No models were provided")
    return parsed


def embedding_csv_name(prefix: str, model_name_or_short: str) -> str:
    return f"{prefix}{model_suffix(model_name_or_short)}_embeddings.csv"


def similarity_csv_name(prefix: str, model_name_or_short: str, *, weighted: bool = False, merged: bool = False) -> str:
    suffix = model_suffix(model_name_or_short)
    if merged:
        return f"{prefix}_similarity_merged{suffix}.csv"
    if weighted:
        return f"{prefix}_similarity_citweighted{suffix}.csv"
    return f"{prefix}_similarity{suffix}.csv"
