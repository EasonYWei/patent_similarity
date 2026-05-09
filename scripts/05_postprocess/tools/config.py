"""Postprocess stage constants and model helpers."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

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
