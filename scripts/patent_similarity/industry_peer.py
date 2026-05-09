"""Industry-peer similarity calculation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import DEFAULT_INDUSTRY_FILE, DEFAULT_OUTPUT_DIR, embedding_csv_name, model_short_name, parse_models
from .io import embedding_columns, read_csv, read_excel, write_csv
from .similarity import safe_cosine_similarity

PEER_COUNT_COLUMNS = ("n_peers_t1", "n_peers_t2", "n_peers_t3")
PEER_SIM_COLUMNS = ("peer_sim_t1", "peer_sim_t2", "peer_sim_t3")


def load_industry_info(path: Path = DEFAULT_INDUSTRY_FILE) -> pl.DataFrame:
    df = read_excel(path)
    df = df.rename({col: col.lower() for col in df.columns})
    if "ind" in df.columns:
        df = df.rename({"ind": "Ind"})
    required = {"stkcd", "year", "Ind"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df.select(
        pl.col("stkcd").cast(pl.Utf8, strict=False).str.strip_chars().alias("stkcd"),
        pl.col("year").cast(pl.Int64, strict=False).alias("p_year"),
        pl.col("Ind").cast(pl.Utf8, strict=False).alias("Ind"),
    )


def attach_industry(embeddings: pl.DataFrame, industry_info: pl.DataFrame) -> pl.DataFrame:
    embeddings = embeddings.with_columns(
        pl.col("stkcd").cast(pl.Utf8, strict=False).str.strip_chars().alias("stkcd"),
        pl.col("p_year").cast(pl.Int64, strict=False).alias("p_year"),
    )
    merged = embeddings.join(industry_info, on=["stkcd", "p_year"], how="left")
    missing = merged.filter(pl.col("Ind").is_null()).height
    if missing:
        logging.getLogger(__name__).warning("%d rows missing industry info; excluding them", missing)
    return merged.filter(pl.col("Ind").is_not_null())


def _build_lookup(df: pl.DataFrame, emb_cols: list[str]) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, group in df.partition_by(["Ind", "p_year"], as_dict=True).items():
        if not isinstance(key, tuple):
            key = (key,)
        ind, year = str(key[0]), int(key[1])
        matrix = group.select(emb_cols).to_numpy().astype(np.float64, copy=False)
        norms = np.sqrt(np.sum(matrix * matrix, axis=1))
        stkcds = np.asarray(group.get_column("stkcd").to_list(), dtype=object)
        lookup[(ind, year)] = (matrix, norms, stkcds)
    return lookup


def calculate_peer_similarity(df: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    df = df.sort(["Ind", "stkcd", "p_year"], maintain_order=True)
    lookup = _build_lookup(df, emb_cols)
    matrix = df.select(emb_cols).to_numpy().astype(np.float64, copy=False)
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    stkcds = np.asarray(df.get_column("stkcd").to_list(), dtype=object)
    years = df.get_column("p_year").to_numpy().astype(np.int64, copy=False)
    industries = np.asarray(df.get_column("Ind").to_list(), dtype=object)

    n_rows = df.height
    counts = {col: np.zeros(n_rows, dtype=np.int64) for col in PEER_COUNT_COLUMNS}
    sims = {col: np.full(n_rows, np.nan, dtype=np.float64) for col in PEER_SIM_COLUMNS}

    for i in range(n_rows):
        if norms[i] <= 1e-12 or not np.isfinite(norms[i]):
            continue
        for lag in (1, 2, 3):
            key = (str(industries[i]), int(years[i] - lag))
            if key not in lookup:
                continue
            peer_matrix, peer_norms, peer_stkcds = lookup[key]
            peer_mask = peer_stkcds != stkcds[i]
            if not np.any(peer_mask):
                continue
            peer_matrix = peer_matrix[peer_mask]
            peer_norms = peer_norms[peer_mask]
            n_peers = len(peer_matrix)
            valid = (peer_norms > 1e-12) & np.isfinite(peer_norms)
            count_col = f"n_peers_t{lag}"
            sim_col = f"peer_sim_t{lag}"
            counts[count_col][i] = n_peers
            if not np.any(valid):
                continue
            dot = peer_matrix[valid] @ matrix[i]
            values = dot / (peer_norms[valid] * norms[i])
            values = values[np.isfinite(values)]
            if values.size:
                sims[sim_col][i] = float(np.max(values))

    output = df.select(["stkcd", "p_year", "Ind", "n_patents", "n_texts_used"]).with_columns(
        [pl.Series(name, values) for name, values in {**counts, **sims}.items()]
    )
    return output.sort(["stkcd", "p_year"], maintain_order=True)


def _process_one(input_path: Path, industry_info: pl.DataFrame) -> pl.DataFrame:
    df = attach_industry(read_csv(input_path), industry_info)
    emb_cols = embedding_columns(df)
    logging.getLogger(__name__).info(
        "Loaded %s rows=%d dimensions=%d industries=%d",
        input_path,
        df.height,
        len(emb_cols),
        df.select(pl.col("Ind").n_unique()).item(),
    )
    return calculate_peer_similarity(df, emb_cols)


def process_model(model: str, industry_info: pl.DataFrame, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    short = model_short_name(model)
    simple_input = output_dir / embedding_csv_name("stkcd_year", short)
    cit_input = output_dir / embedding_csv_name("stkcd_year_citweighted", short)
    simple_output = output_dir / f"industry_peer_similarity_{short}.csv"
    cit_output = output_dir / f"industry_peer_similarity_citweighted_{short}.csv"
    merged_output = output_dir / f"industry_peer_similarity_merged_{short}.csv"

    if not simple_input.exists():
        logging.getLogger(__name__).warning("Input file not found, skipping: %s", simple_input)
        return
    result_simple = _process_one(simple_input, industry_info)
    write_csv(result_simple, simple_output)
    logging.getLogger(__name__).info("Saved simple peer similarity: %s", simple_output)

    if not cit_input.exists():
        logging.getLogger(__name__).warning("Citation-weighted input not found, skipping: %s", cit_input)
        return
    result_cit = _process_one(cit_input, industry_info).rename(
        {
            "n_peers_t1": "n_peers_t1_citw",
            "n_peers_t2": "n_peers_t2_citw",
            "n_peers_t3": "n_peers_t3_citw",
            "peer_sim_t1": "peer_sim_t1_citw",
            "peer_sim_t2": "peer_sim_t2_citw",
            "peer_sim_t3": "peer_sim_t3_citw",
        }
    )
    write_csv(result_cit, cit_output)
    merged = result_simple.join(
        result_cit.select(
            [
                "stkcd",
                "p_year",
                "Ind",
                "n_peers_t1_citw",
                "n_peers_t2_citw",
                "n_peers_t3_citw",
                "peer_sim_t1_citw",
                "peer_sim_t2_citw",
                "peer_sim_t3_citw",
            ]
        ),
        on=["stkcd", "p_year", "Ind"],
        how="outer",
    ).sort(["stkcd", "p_year"], maintain_order=True)
    write_csv(merged, merged_output)
    logging.getLogger(__name__).info("Saved merged peer similarity: %s", merged_output)


def process_models(models: str, industry_path: Path, output_dir: Path) -> None:
    industry_info = load_industry_info(industry_path)
    for model in parse_models(models):
        process_model(model, industry_info, output_dir)
