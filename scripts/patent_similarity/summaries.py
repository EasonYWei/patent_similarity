"""Summary comparisons across firm, peer, and city similarity outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import DEFAULT_OUTPUT_DIR, model_short_name, parse_models
from .io import read_csv, write_csv


def _numeric_diff(df: pl.DataFrame, left: str, right: str, out: str) -> pl.Expr | None:
    if left not in df.columns or right not in df.columns:
        return None
    return (pl.col(left) - pl.col(right)).alias(out)


def calculate_differences(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        _numeric_diff(df, "firm_cos_sim_lag1", "peer_sim_t1", "diff_firm_peer"),
        _numeric_diff(df, "firm_cos_sim_lag1", "city_cos_sim_lag1", "diff_firm_city"),
        _numeric_diff(df, "peer_sim_t1", "city_cos_sim_lag1", "diff_peer_city"),
        _numeric_diff(df, "firm_cos_sim_lag1_citw", "peer_sim_t1_citw", "diff_firm_peer_citw"),
        _numeric_diff(df, "firm_cos_sim_lag1_citw", "city_cos_sim_lag1_citw", "diff_firm_city_citw"),
        _numeric_diff(df, "peer_sim_t1_citw", "city_cos_sim_lag1_citw", "diff_peer_city_citw"),
    ]
    df = df.with_columns([expr for expr in exprs if expr is not None])
    if "diff_firm_peer" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("diff_firm_peer").is_null())
            .then(None)
            .when(pl.col("diff_firm_peer") < -0.1)
            .then(pl.lit("follower"))
            .when(pl.col("diff_firm_peer") <= 0.1)
            .then(pl.lit("neutral"))
            .otherwise(pl.lit("leader"))
            .alias("firm_vs_peer_type")
        )
    return df


def correlation_matrix(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    cols = [col for col in columns if col in df.columns]
    values = df.select(cols).to_numpy().astype(np.float64, copy=False) if cols else np.empty((0, 0))
    out = np.full((len(cols), len(cols)), np.nan, dtype=np.float64)
    for i, left in enumerate(cols):
        for j, right in enumerate(cols):
            pair = values[:, [i, j]]
            mask = np.isfinite(pair).all(axis=1)
            if mask.sum() >= 2:
                out[i, j] = float(np.corrcoef(pair[mask, 0], pair[mask, 1])[0, 1])
    data = {"": cols}
    for idx, col in enumerate(cols):
        data[col] = out[:, idx]
    return pl.DataFrame(data)


def _stats_exprs(columns: list[str]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for col in columns:
        exprs.extend(
            [
                pl.col(col).mean().round(4).alias(f"{col}_mean"),
                pl.col(col).std().round(4).alias(f"{col}_std"),
                pl.col(col).count().alias(f"{col}_count"),
            ]
        )
    return exprs


def analyze_by_industry(df: pl.DataFrame) -> pl.DataFrame:
    if "Ind" not in df.columns:
        return pl.DataFrame()
    cols = ["firm_cos_sim_lag1", "peer_sim_t1", "city_cos_sim_lag1"]
    exprs = _stats_exprs([col for col in cols if col in df.columns])
    for col in ["diff_firm_peer", "diff_firm_city"]:
        if col in df.columns:
            exprs.extend([pl.col(col).mean().round(4).alias(f"{col}_mean"), pl.col(col).std().round(4).alias(f"{col}_std")])
    if "stkcd" in df.columns:
        exprs.append(pl.col("stkcd").n_unique().alias("n_firms"))
    return df.group_by("Ind").agg(exprs).sort("Ind")


def analyze_by_city(df: pl.DataFrame) -> pl.DataFrame:
    if "city_code" not in df.columns:
        return pl.DataFrame()
    cols = ["firm_cos_sim_lag1", "peer_sim_t1", "city_cos_sim_lag1"]
    exprs = _stats_exprs([col for col in cols if col in df.columns])
    if "diff_firm_peer" in df.columns:
        exprs.extend([pl.col("diff_firm_peer").mean().round(4).alias("diff_firm_peer_mean"), pl.col("diff_firm_peer").std().round(4).alias("diff_firm_peer_std")])
    if "stkcd" in df.columns:
        exprs.append(pl.col("stkcd").n_unique().alias("n_firms"))
    if "city" in df.columns:
        exprs.append(pl.col("city").drop_nulls().first().alias("city"))
    return df.group_by("city_code").agg(exprs).sort("city_code")


def summarize_model(model: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    short = model_short_name(model)
    input_file = output_dir / f"merged_similarity_by_firm_{short}.csv"
    if not input_file.exists():
        logging.getLogger(__name__).warning("Input file not found, skipping: %s", input_file)
        return
    df = calculate_differences(read_csv(input_file))

    base_cols = ["stkcd", "p_year", "city_code", "city", "Ind", "n_patents_firm", "n_texts_used_firm"]
    sim_cols = ["firm_cos_sim_lag1", "firm_cos_sim_lag3", "peer_sim_t1", "peer_sim_t2", "peer_sim_t3", "city_cos_sim_lag1", "city_cos_sim_lag3"]
    diff_cols = ["diff_firm_peer", "diff_firm_city", "diff_peer_city", "diff_firm_peer_citw", "diff_firm_city_citw", "diff_peer_city_citw", "firm_vs_peer_type"]
    output_cols = [col for col in base_cols + sim_cols + diff_cols if col in df.columns]
    write_csv(df.select(output_cols), output_dir / f"similarity_comparison_{short}.csv")

    corr_cols = ["firm_cos_sim_lag1", "firm_cos_sim_lag3", "peer_sim_t1", "peer_sim_t2", "peer_sim_t3", "city_cos_sim_lag1", "city_cos_sim_lag3"]
    write_csv(correlation_matrix(df, corr_cols), output_dir / f"similarity_correlation_{short}.csv")

    industry = analyze_by_industry(df)
    if industry.height:
        write_csv(industry, output_dir / f"similarity_by_industry_{short}.csv")
    city = analyze_by_city(df)
    if city.height:
        write_csv(city, output_dir / f"similarity_by_city_summary_{short}.csv")
    logging.getLogger(__name__).info("Saved comparison summaries for %s", short)


def summarize_models(models: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    for model in parse_models(models):
        summarize_model(model, output_dir)
