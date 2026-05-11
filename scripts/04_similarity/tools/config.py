"""Similarity stage constants and naming helpers."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_FIRM_YEAR_INPUT_DIR = DEFAULT_OUTPUT_DIR / "firm_year_embeddings"
DEFAULT_FIRM_SIMILARITY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "firm_similarity"
DEFAULT_IPC_SIMILARITY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ipc_similarity"
DEFAULT_INDUSTRY_PEER_SIMILARITY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "industry_peer_similarity"
DEFAULT_CITY_PATENTS_FILE = DEFAULT_DATA_DIR / "patents_cleaned_with_city.dta"
DEFAULT_INDUSTRY_FILE = DEFAULT_DATA_DIR / "stkcd_info.csv"
DEFAULT_PATENTS_PARQUET_FILE = DEFAULT_DATA_DIR / "patents_cleaned.parquet"

STKCD_COLUMN = "stkcd"
CITY_COLUMN = "city"
CITY_CODE_COLUMN = "city_code"
PROVINCE_COLUMN = "province"
YEAR_COLUMN = "p_year"
SAFE_COSINE_TOLERANCE = 1e-12

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


def parse_models(models: str | list[str] | tuple[str, ...] | None) -> list[str]:
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


def embedding_parquet_name(prefix: str, model_name_or_short: str) -> str:
    return f"{prefix}{model_suffix(model_name_or_short)}_embeddings.parquet"


def similarity_csv_name(prefix: str, model_name_or_short: str, *, weighted: bool = False, merged: bool = False) -> str:
    suffix = model_suffix(model_name_or_short)
    if merged:
        return f"{prefix}_similarity_merged{suffix}.csv"
    if weighted:
        return f"{prefix}_similarity_citweighted{suffix}.csv"
    return f"{prefix}_similarity{suffix}.csv"


def similarity_parquet_name(prefix: str, model_name_or_short: str, *, weighted: bool = False, merged: bool = False) -> str:
    suffix = model_suffix(model_name_or_short)
    if merged:
        return f"{prefix}_similarity_merged{suffix}.parquet"
    if weighted:
        return f"{prefix}_similarity_citweighted{suffix}.parquet"
    return f"{prefix}_similarity{suffix}.parquet"
