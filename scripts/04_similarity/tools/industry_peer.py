"""Industry-peer similarity calculation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .config import (
    DEFAULT_FIRM_YEAR_INPUT_DIR,
    DEFAULT_INDUSTRY_FILE,
    DEFAULT_INDUSTRY_PEER_SIMILARITY_OUTPUT_DIR,
    SAFE_COSINE_TOLERANCE,
    embedding_parquet_name,
    model_short_name,
    parse_models,
    similarity_parquet_name,
)
from .io import embedding_columns, read_csv, read_excel, read_frame, write_parquet

PEER_COUNT_COLUMNS = ("n_peers_t1", "n_peers_t2", "n_peers_t3")
PEER_SIM_COLUMNS = ("peer_sim_t1", "peer_sim_t2", "peer_sim_t3")
PEER_LAGS = (1, 2, 3)
PEER_CHUNK_SIZE = 20_000
PEER_CITW_RENAME = {
    "n_peers_t1": "n_peers_t1_citw",
    "n_peers_t2": "n_peers_t2_citw",
    "n_peers_t3": "n_peers_t3_citw",
    "peer_sim_t1": "peer_sim_t1_citw",
    "peer_sim_t2": "peer_sim_t2_citw",
    "peer_sim_t3": "peer_sim_t3_citw",
}
PEER_CITW_SIM_COLUMNS = tuple(PEER_CITW_RENAME[col] for col in PEER_SIM_COLUMNS)


def _read_industry_source(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return read_excel(path)
    raise ValueError(f"Unsupported industry metadata format for {path}. Expected .csv, .xls, or .xlsx")


def _normalized_stock_expr() -> pl.Expr:
    stock = pl.col("stkcd").cast(pl.Utf8, strict=False).str.strip_chars()
    stock = pl.when(stock.str.contains(r"^\d+\.0+$")).then(
        stock.str.replace(r"\.0+$", "")
    ).otherwise(stock)
    return stock.str.zfill(6).alias("stkcd")


def load_industry_info(path: Path = DEFAULT_INDUSTRY_FILE) -> pl.DataFrame:
    df = _read_industry_source(path)
    df = df.rename({col: col.strip() for col in df.columns if col != col.strip()})
    df = df.rename({col: col.lower() for col in df.columns})
    if "ind" in df.columns:
        df = df.rename({"ind": "Ind"})
    required = {"stkcd", "year", "Ind"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    selected = df.select(
        _normalized_stock_expr(),
        pl.col("year").cast(pl.Int64, strict=False).alias("p_year"),
        pl.col("Ind").cast(pl.Utf8, strict=False).alias("Ind"),
    ).filter(
        pl.col("stkcd").is_not_null()
        & pl.col("p_year").is_not_null()
        & pl.col("Ind").is_not_null()
    )
    duplicate_keys = (
        selected.group_by(["stkcd", "p_year"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_keys:
        logging.getLogger(__name__).warning(
            "%d duplicate firm-year industry rows found; keeping first mapping",
            duplicate_keys,
        )
    return selected.unique(["stkcd", "p_year"], keep="first", maintain_order=True)


def attach_industry(embeddings: pl.DataFrame, industry_info: pl.DataFrame) -> pl.DataFrame:
    embeddings = embeddings.with_columns(
        _normalized_stock_expr(),
        pl.col("p_year").cast(pl.Int64, strict=False).alias("p_year"),
    )
    merged = embeddings.join(industry_info, on=["stkcd", "p_year"], how="left")
    missing = merged.filter(pl.col("Ind").is_null()).height
    if missing:
        logging.getLogger(__name__).warning("%d rows missing industry info; excluding them", missing)
    return merged.filter(pl.col("Ind").is_not_null())


def _embedding_matrix(df: pl.DataFrame, emb_cols: list[str]) -> np.ndarray:
    return (
        df.select(
            [
                pl.col(col)
                .cast(pl.Float64, strict=False)
                .fill_null(float("nan"))
                .alias(col)
                for col in emb_cols
            ]
        )
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def _valid_embedding_mask(matrix: np.ndarray, norms: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(matrix).all(axis=1)
        & np.isfinite(norms)
        & (norms > SAFE_COSINE_TOLERANCE)
    )


def _build_peer_summaries(
    matrix: np.ndarray,
    valid_rows: np.ndarray,
    stkcds: np.ndarray,
    years: np.ndarray,
    industries: np.ndarray,
) -> tuple[
    dict[tuple[str, int], int],
    dict[tuple[str, int, str], int],
    np.ndarray,
    np.ndarray,
]:
    valid_idx = np.flatnonzero(valid_rows)
    group_key_to_idx: dict[tuple[str, int], int] = {}
    valid_row_by_firm_year: dict[tuple[str, int, str], int] = {}
    group_ids = np.empty(valid_idx.size, dtype=np.int64)

    for pos, row in enumerate(valid_idx):
        ind = str(industries[row])
        year = int(years[row])
        stkcd = str(stkcds[row])
        group_key = (ind, year)
        group_idx = group_key_to_idx.setdefault(group_key, len(group_key_to_idx))
        group_ids[pos] = group_idx
        valid_row_by_firm_year[(ind, year, stkcd)] = int(row)

    group_sums = np.zeros((len(group_key_to_idx), matrix.shape[1]), dtype=np.float64)
    if valid_idx.size:
        np.add.at(group_sums, group_ids, matrix[valid_idx])
    group_counts = np.bincount(group_ids, minlength=len(group_key_to_idx)).astype(np.int64)
    return group_key_to_idx, valid_row_by_firm_year, group_sums, group_counts


def _fill_lag_peer_values(
    *,
    lag: int,
    matrix: np.ndarray,
    norms: np.ndarray,
    valid_rows: np.ndarray,
    stkcds: np.ndarray,
    years: np.ndarray,
    industries: np.ndarray,
    group_key_to_idx: dict[tuple[str, int], int],
    valid_row_by_firm_year: dict[tuple[str, int, str], int],
    group_sums: np.ndarray,
    group_counts: np.ndarray,
    counts: np.ndarray,
    sims: np.ndarray,
) -> None:
    n_rows = matrix.shape[0]
    for start in range(0, n_rows, PEER_CHUNK_SIZE):
        end = min(start + PEER_CHUNK_SIZE, n_rows)
        chunk_size = end - start
        target_groups = np.full(chunk_size, -1, dtype=np.int64)
        self_rows = np.full(chunk_size, -1, dtype=np.int64)

        for offset, row in enumerate(range(start, end)):
            ind = str(industries[row])
            target_year = int(years[row] - lag)
            target_groups[offset] = group_key_to_idx.get((ind, target_year), -1)
            self_rows[offset] = valid_row_by_firm_year.get(
                (ind, target_year, str(stkcds[row])),
                -1,
            )

        has_group = target_groups >= 0
        if not np.any(has_group):
            continue

        positions = np.flatnonzero(has_group)
        peer_counts = group_counts[target_groups[positions]].copy()
        peer_sums = group_sums[target_groups[positions]].copy()

        has_self_peer = self_rows[positions] >= 0
        if np.any(has_self_peer):
            peer_sums[has_self_peer] -= matrix[self_rows[positions][has_self_peer]]
            peer_counts[has_self_peer] -= 1

        peer_counts = np.maximum(peer_counts, 0)
        global_positions = start + positions
        counts[global_positions] = peer_counts

        usable = (peer_counts > 0) & valid_rows[global_positions]
        if not np.any(usable):
            continue

        sim_positions = global_positions[usable]
        sim_peer_sums = peer_sums[usable]
        peer_norms = np.sqrt(np.sum(sim_peer_sums * sim_peer_sums, axis=1))
        valid_peer_sums = np.isfinite(peer_norms) & (peer_norms > SAFE_COSINE_TOLERANCE)
        if not np.any(valid_peer_sums):
            continue

        sim_positions = sim_positions[valid_peer_sums]
        sim_peer_sums = sim_peer_sums[valid_peer_sums]
        peer_norms = peer_norms[valid_peer_sums]
        dots = np.sum(matrix[sim_positions] * sim_peer_sums, axis=1)
        values = dots / (norms[sim_positions] * peer_norms)
        values = np.where(np.isfinite(values), values, np.nan)
        sims[sim_positions] = values


def calculate_peer_similarity(df: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Compare each firm-year vector with lagged same-industry peer centroids."""
    df = df.sort(["Ind", "stkcd", "p_year"], maintain_order=True)
    matrix = _embedding_matrix(df, emb_cols)
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    valid_rows = _valid_embedding_mask(matrix, norms)
    stkcds = np.asarray(df.get_column("stkcd").to_list(), dtype=object)
    years = df.get_column("p_year").to_numpy().astype(np.int64, copy=False)
    industries = np.asarray(df.get_column("Ind").to_list(), dtype=object)
    group_key_to_idx, valid_row_by_firm_year, group_sums, group_counts = _build_peer_summaries(
        matrix,
        valid_rows,
        stkcds,
        years,
        industries,
    )

    n_rows = df.height
    counts = {col: np.zeros(n_rows, dtype=np.int64) for col in PEER_COUNT_COLUMNS}
    sims = {col: np.full(n_rows, np.nan, dtype=np.float64) for col in PEER_SIM_COLUMNS}

    for lag in PEER_LAGS:
        _fill_lag_peer_values(
            lag=lag,
            matrix=matrix,
            norms=norms,
            valid_rows=valid_rows,
            stkcds=stkcds,
            years=years,
            industries=industries,
            group_key_to_idx=group_key_to_idx,
            valid_row_by_firm_year=valid_row_by_firm_year,
            group_sums=group_sums,
            group_counts=group_counts,
            counts=counts[f"n_peers_t{lag}"],
            sims=sims[f"peer_sim_t{lag}"],
        )

    output = df.select(["stkcd", "p_year", "Ind", "n_patents", "n_texts_used"]).with_columns(
        [pl.Series(name, values) for name, values in {**counts, **sims}.items()]
    )
    return output.sort(["stkcd", "p_year"], maintain_order=True)


def _replace_nan_with_nulls(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    exprs = [
        pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        for col in columns
        if col in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _process_one(input_path: Path, industry_info: pl.DataFrame) -> pl.DataFrame:
    df = attach_industry(read_frame(input_path), industry_info)
    emb_cols = embedding_columns(df)
    logging.getLogger(__name__).info(
        "Loaded %s rows=%d dimensions=%d industries=%d",
        input_path,
        df.height,
        len(emb_cols),
        df.select(pl.col("Ind").n_unique()).item(),
    )
    return calculate_peer_similarity(df, emb_cols)


def process_model(
    model: str,
    industry_info: pl.DataFrame,
    input_dir: Path = DEFAULT_FIRM_YEAR_INPUT_DIR,
    output_dir: Path = DEFAULT_INDUSTRY_PEER_SIMILARITY_OUTPUT_DIR,
) -> None:
    short = model_short_name(model)
    simple_input = input_dir / embedding_parquet_name("stkcd_year", short)
    cit_input = input_dir / embedding_parquet_name("stkcd_year_citweighted", short)
    simple_output = output_dir / similarity_parquet_name("industry_peer", short)
    cit_output = output_dir / similarity_parquet_name("industry_peer", short, weighted=True)
    merged_output = output_dir / similarity_parquet_name("industry_peer", short, merged=True)

    result_simple = _replace_nan_with_nulls(
        _process_one(simple_input, industry_info),
        PEER_SIM_COLUMNS,
    )
    write_parquet(result_simple, simple_output)
    logging.getLogger(__name__).info("Saved simple peer similarity: %s", simple_output)

    if not cit_input.exists():
        logging.getLogger(__name__).warning("Citation-weighted input not found, skipping: %s", cit_input)
        return
    result_cit = _replace_nan_with_nulls(
        _process_one(cit_input, industry_info).rename(PEER_CITW_RENAME),
        PEER_CITW_SIM_COLUMNS,
    )
    write_parquet(result_cit, cit_output)
    logging.getLogger(__name__).info("Saved citation-weighted peer similarity: %s", cit_output)

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
        how="left",
    ).sort(["stkcd", "p_year"], maintain_order=True)
    write_parquet(merged, merged_output)
    logging.getLogger(__name__).info("Saved merged peer similarity: %s", merged_output)


def process_models(
    models: str,
    industry_path: Path,
    input_dir: Path = DEFAULT_FIRM_YEAR_INPUT_DIR,
    output_dir: Path = DEFAULT_INDUSTRY_PEER_SIMILARITY_OUTPUT_DIR,
) -> None:
    industry_info = load_industry_info(industry_path)
    for model in parse_models(models):
        process_model(model, industry_info, input_dir, output_dir)
