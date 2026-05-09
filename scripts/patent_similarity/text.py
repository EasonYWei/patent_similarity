"""Patent text preparation helpers."""

from __future__ import annotations

import polars as pl

from .config import TEXT_COLUMNS, TEXT_FIELD, TEXT_IS_EMPTY_FIELD


def with_combined_text(df: pl.DataFrame, text_cols: tuple[str, str] = TEXT_COLUMNS) -> pl.DataFrame:
    """Add normalized `text` and `text_is_empty` columns from title and abstract."""
    parts: list[pl.Expr] = []
    for col in text_cols:
        if col in df.columns:
            parts.append(pl.col(col).cast(pl.Utf8, strict=False).fill_null(""))
        else:
            parts.append(pl.lit(""))

    text_expr = pl.concat_str(parts, separator=" ").str.replace_all(r"\s+", " ").str.strip_chars()
    return df.with_columns(text_expr.alias(TEXT_FIELD)).with_columns(
        (pl.col(TEXT_FIELD).str.len_chars().fill_null(0) == 0).alias(TEXT_IS_EMPTY_FIELD)
    )
