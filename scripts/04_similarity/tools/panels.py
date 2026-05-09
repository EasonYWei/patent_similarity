"""Build firm-level and city-level merged similarity panels."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .config import DEFAULT_CITY_PATENTS_FILE, DEFAULT_OUTPUT_DIR, model_short_name, parse_models
from .io import cast_common_keys, read_csv, read_stata, select_existing, write_csv


def _rename_existing(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    return df.rename({old: new for old, new in mapping.items() if old in df.columns})


def load_stkcd_city_mapping(dta_path: Path = DEFAULT_CITY_PATENTS_FILE) -> pl.DataFrame:
    df = read_stata(dta_path, columns=["stkcd", "p_year", "city", "city_code"])
    df = cast_common_keys(df, has_firm=True, has_city=True).filter(
        pl.col("stkcd").is_not_null() & pl.col("p_year").is_not_null() & pl.col("city_code").is_not_null()
    )
    counts = (
        df.group_by(["stkcd", "p_year", "city_code", "city"])
        .len()
        .sort(["stkcd", "p_year", "len"], descending=[False, False, True], maintain_order=True)
    )
    mapping = counts.group_by(["stkcd", "p_year"]).first().select("stkcd", "p_year", "city_code", "city")
    logging.getLogger(__name__).info(
        "Loaded city mapping rows=%d cities=%d",
        mapping.height,
        mapping.select(pl.col("city_code").n_unique()).item(),
    )
    return mapping


def merge_firm_level_similarities(model: str, mapping: pl.DataFrame, output_dir: Path = DEFAULT_OUTPUT_DIR) -> pl.DataFrame:
    short = model_short_name(model)
    firm_file = output_dir / f"stkcd_year_similarity_merged_{short}.csv"
    peer_file = output_dir / f"industry_peer_similarity_merged_{short}.csv"
    city_file = output_dir / f"city_year_similarity_merged_{short}.csv"

    firm = _rename_existing(
        cast_common_keys(read_csv(firm_file), has_firm=True),
        {
            "cos_sim_lag1": "firm_cos_sim_lag1",
            "cos_sim_lag3": "firm_cos_sim_lag3",
            "cos_sim_cumulative": "firm_cos_sim_cumulative",
            "cos_sim_lag1_citw": "firm_cos_sim_lag1_citw",
            "cos_sim_lag3_citw": "firm_cos_sim_lag3_citw",
            "cos_sim_cumulative_citw": "firm_cos_sim_cumulative_citw",
            "n_patents": "n_patents_firm",
            "n_texts_used": "n_texts_used_firm",
        },
    )
    peer_cols = [
        "stkcd",
        "p_year",
        "Ind",
        "n_peers_t1",
        "n_peers_t2",
        "n_peers_t3",
        "peer_sim_t1",
        "peer_sim_t2",
        "peer_sim_t3",
        "n_peers_t1_citw",
        "n_peers_t2_citw",
        "n_peers_t3_citw",
        "peer_sim_t1_citw",
        "peer_sim_t2_citw",
        "peer_sim_t3_citw",
    ]
    peer = select_existing(cast_common_keys(read_csv(peer_file), has_firm=True), peer_cols)
    merged = firm.join(peer, on=["stkcd", "p_year"], how="left")
    merged = merged.join(mapping, on=["stkcd", "p_year"], how="left")

    city = _rename_existing(
        cast_common_keys(read_csv(city_file), has_city=True),
        {
            "cos_sim_lag1": "city_cos_sim_lag1",
            "cos_sim_lag3": "city_cos_sim_lag3",
            "cos_sim_cumulative": "city_cos_sim_cumulative",
            "cos_sim_lag1_citw": "city_cos_sim_lag1_citw",
            "cos_sim_lag3_citw": "city_cos_sim_lag3_citw",
            "cos_sim_cumulative_citw": "city_cos_sim_cumulative_citw",
            "n_patents": "n_patents_city",
            "n_texts_used": "n_texts_used_city",
        },
    )
    city_cols = [
        "city_code",
        "p_year",
        "city_cos_sim_lag1",
        "city_cos_sim_lag3",
        "city_cos_sim_cumulative",
        "city_cos_sim_lag1_citw",
        "city_cos_sim_lag3_citw",
        "city_cos_sim_cumulative_citw",
        "n_patents_city",
        "n_texts_used_city",
    ]
    merged = merged.join(select_existing(city, city_cols), on=["city_code", "p_year"], how="left")
    return merged.sort(["stkcd", "p_year"], maintain_order=True)


def create_city_level_file(model: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> pl.DataFrame:
    short = model_short_name(model)
    city_file = output_dir / f"city_year_similarity_merged_{short}.csv"
    city = cast_common_keys(read_csv(city_file), has_city=True)
    city_cols = [
        "city_code",
        "p_year",
        "city",
        "n_patents",
        "n_texts_used",
        "cos_sim_lag1",
        "cos_sim_lag3",
        "cos_sim_cumulative",
        "cos_sim_lag1_citw",
        "cos_sim_lag3_citw",
        "cos_sim_cumulative_citw",
    ]
    return select_existing(city, city_cols).rename(
        {"n_patents": "n_patents_city", "n_texts_used": "n_texts_used_city"}
    )


def build_panels(models: str, data_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    mapping = load_stkcd_city_mapping(data_path)
    for model in parse_models(models):
        short = model_short_name(model)
        firm = merge_firm_level_similarities(short, mapping, output_dir)
        firm_output = output_dir / f"merged_similarity_by_firm_{short}.csv"
        write_csv(firm, firm_output)
        logging.getLogger(__name__).info("Saved firm panel: %s", firm_output)

        city = create_city_level_file(short, output_dir)
        city_output = output_dir / f"merged_similarity_by_city_{short}.csv"
        write_csv(city, city_output)
        logging.getLogger(__name__).info("Saved city panel: %s", city_output)
