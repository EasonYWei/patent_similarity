#!/usr/bin/env python3
"""Patent SBERT embeddings + city-year aggregation pipeline.

This script reads a cleaned patent file (default: `data/patents_cleaned.dta`),
computes patent embeddings with SBERT, aggregates by city-year, and writes:
- output/city_year_{model}_embeddings.csv
- output/city_year_citweighted_{model}_embeddings.csv

Citation weighting is based on `p_cite` in the cleaned file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Import shared components from patents_embeddings.py
sys.path.insert(0, str(Path(__file__).parent))
from patents_embeddings import (
    SBertEmbedder,
    build_text_field,
    coerce_citations,
    divide_rows,
    get_model_short_name,
    load_single_file,
    recommend_batch_size,
    setup_logging,
    validate_required_columns,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "patents_cleaned.dta"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# City-level constants
CITY_COLUMN = "city"
CITY_CODE_COLUMN = "city_code"
PROVINCE_COLUMN = "province"
PROVINCE_CODE_COLUMN = "province_code"
YEAR_COLUMN = "p_year"
CITY_KEY_COLUMN = "city_year"  # Composite key: city_code + year
TEXT_COLUMNS = ("p_tt", "p_abs")
CITATION_COLUMN = "p_cite"

# Required columns for city-level analysis
CITY_REQUIRED_COLUMNS = (CITY_CODE_COLUMN, YEAR_COLUMN, TEXT_COLUMNS[0], TEXT_COLUMNS[1])

ZERO_DIVISION_EPSILON = 1e-12

CITY_EMBEDDING_OUTPUT_COLUMNS = [
    CITY_COLUMN,
    CITY_CODE_COLUMN,
    PROVINCE_COLUMN,
    YEAR_COLUMN,
    CITY_KEY_COLUMN,
    "n_patents",
    "n_texts_used",
    "total_citations",
    "mean_citations",
]


def validate_city_required_columns(
    df: pd.DataFrame, data_path: Path, required: Iterable[str]
) -> None:
    """Validate that required columns exist for city-level analysis."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input {data_path} missing required columns for city-level analysis: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Please ensure 'pre.do' includes city fields (市, 市代码)."
        )


def load_and_prepare_city_data(data_path: Path) -> pd.DataFrame:
    """Load data and prepare city-year keys."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        df = pd.read_stata(data_path, convert_categoricals=False)
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to read %s: %s", data_path, exc)
        raise

    # Validate required columns for city-level analysis
    validate_city_required_columns(df, data_path, CITY_REQUIRED_COLUMNS)

    # Keep memory use reasonable by dropping unused columns early
    keep = [
        CITY_COLUMN,
        CITY_CODE_COLUMN,
        PROVINCE_COLUMN,
        PROVINCE_CODE_COLUMN,
        YEAR_COLUMN,
        "p_tt",
        "p_abs",
        "p_date",
        CITATION_COLUMN,
    ]
    keep = [c for c in keep if c in df.columns]
    if len(keep) < len(df.columns):
        missing = sorted(set(df.columns) - set(keep))
        logging.getLogger(__name__).debug(
            "Dropping non-required columns from %s: %s", data_path.name, missing
        )
        df = df[keep].copy()

    # Filter out rows with missing city_code or year
    before = len(df)
    df = df[df[CITY_CODE_COLUMN].notna() & df[YEAR_COLUMN].notna()].copy()
    dropped = before - len(df)
    if dropped:
        logging.getLogger(__name__).warning(
            "Dropped %d rows with missing city_code or year in %s", dropped, data_path.name
        )

    # Convert data types
    df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors="coerce").astype("Int32")
    df = df[df[YEAR_COLUMN].notna()].copy()

    df[CITY_CODE_COLUMN] = df[CITY_CODE_COLUMN].astype("string").str.strip()
    if CITY_COLUMN in df.columns:
        df[CITY_COLUMN] = df[CITY_COLUMN].astype("string")
    if PROVINCE_COLUMN in df.columns:
        df[PROVINCE_COLUMN] = df[PROVINCE_COLUMN].astype("string")

    # Sort by city and date/year
    if "p_date" in df.columns:
        df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
        df = df.sort_values([CITY_CODE_COLUMN, "p_date"], ascending=True, kind="mergesort")
    else:
        df = df.sort_values([CITY_CODE_COLUMN, YEAR_COLUMN], ascending=True, kind="mergesort")

    df = df.reset_index(drop=True)

    # Build text field
    df["text"], df["text_is_empty"] = build_text_field(df, text_cols=TEXT_COLUMNS)

    # Create city_year composite key
    df[CITY_KEY_COLUMN] = (
        df[CITY_CODE_COLUMN].astype("string") + "_" + df[YEAR_COLUMN].astype("Int32").astype(str)
    )

    # Process citations
    if CITATION_COLUMN in df.columns:
        df[CITATION_COLUMN] = (
            pd.to_numeric(df[CITATION_COLUMN], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

    if df.empty:
        logging.getLogger(__name__).warning("No valid rows in %s after preprocessing", data_path.name)
        return df

    logging.getLogger(__name__).info(
        "Loaded %s -> rows=%d | cities=%d | city-years=%d | empty_text=%d",
        data_path.name,
        len(df),
        df[CITY_CODE_COLUMN].nunique(),
        df[CITY_KEY_COLUMN].nunique(),
        int(df["text_is_empty"].sum()),
    )
    return df


def aggregate_chunk_by_city(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    key_col: str = CITY_KEY_COLUMN,
    exclude_empty_text: bool = True,
    citation_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chunk-level aggregation into sufficient statistics for city-year groups.
    Returns:
      - meta (one row per key in this chunk)
      - sum_embedding (sum over rows that contribute to mean)
      - n_text_rows (counts used in mean)
      - sum_citation_weighted_embedding (sum(citation * embedding))
    """
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: df has {len(df)} rows but embeddings has {embeddings.shape[0]} rows"
        )

    if len(df) == 0:
        return (
            pd.DataFrame(columns=CITY_EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
        )

    log = logging.getLogger(__name__)
    t0 = time.time()

    keys = df[key_col].astype("string").to_numpy()
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    inv = inv.astype(np.int64)
    dim = embeddings.shape[1]
    n_groups = len(uniq_keys)

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    # Handle empty text exclusion
    if exclude_empty_text and "text_is_empty" in df.columns:
        text_weights = (~df["text_is_empty"].to_numpy()).astype(np.float32)
    else:
        text_weights = np.ones(len(df), dtype=np.float32)

    # Compute sums for simple mean
    sum_embedding = np.zeros((n_groups, dim), dtype=np.float32)
    np.add.at(sum_embedding, inv, emb * text_weights[:, None])
    n_text_rows = np.bincount(inv, weights=text_weights).astype(np.float64)

    # Fallback: if all text is empty for a group, use all rows
    all_counts = np.bincount(inv).astype(np.float64)
    if np.any(n_text_rows == 0):
        fallback_mask = n_text_rows == 0
        all_sum = np.zeros((n_groups, dim), dtype=np.float32)
        np.add.at(all_sum, inv, emb)
        sum_embedding[fallback_mask] = all_sum[fallback_mask]
        n_text_rows[fallback_mask] = all_counts[fallback_mask]
        if np.any(fallback_mask):
            log.warning(
                "Some city-year groups have only empty text; using all rows for those groups."
            )

    # Compute citation-weighted sums
    citations, invalid_count = coerce_citations(df, citation_col)
    if invalid_count:
        log.warning(
            "Detected %d invalid citation values in %s for chunk aggregation.",
            invalid_count,
            citation_col,
        )
    sum_cit_weight = np.zeros((n_groups, dim), dtype=np.float32)
    np.add.at(sum_cit_weight, inv, emb * citations[:, None].astype(np.float32))
    total_citations = np.bincount(inv, weights=citations).astype(np.float64)

    # Build metadata
    meta = (
        df.groupby(key_col, sort=True)[[CITY_COLUMN, CITY_CODE_COLUMN, PROVINCE_COLUMN, YEAR_COLUMN]]
        .first()
        .reindex(uniq_keys)
        .reset_index()
    )
    meta["n_patents"] = np.bincount(inv).astype(np.int64)
    meta["n_texts_used"] = n_text_rows.astype(np.int64)
    meta["total_citations"] = total_citations
    denom = meta["n_patents"].to_numpy(dtype=np.float64)
    meta["mean_citations"] = np.divide(
        meta["total_citations"].to_numpy(dtype=np.float64),
        denom,
        out=np.full(n_groups, np.nan),
        where=denom > 0,
    )

    log.info("City chunk aggregate built for %d groups in %.2fs", n_groups, time.time() - t0)
    return meta, sum_embedding, n_text_rows, sum_cit_weight


def finalize_city_aggregates(
    chunk_meta_parts: List[pd.DataFrame],
    chunk_sum_embeddings: List[np.ndarray],
    chunk_text_counts: List[np.ndarray],
    chunk_cit_weight_sums: List[np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Finalize city-year aggregates from chunks."""
    if not chunk_meta_parts:
        return (
            pd.DataFrame(columns=CITY_EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, 0), dtype=np.float32),
            None,
        )

    meta = pd.concat(chunk_meta_parts, ignore_index=True)
    if meta.empty:
        return (
            pd.DataFrame(columns=CITY_EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, 0), dtype=np.float32),
            None,
        )

    if not chunk_sum_embeddings or not chunk_text_counts:
        raise ValueError("Chunk aggregate components are incomplete")

    dim = chunk_sum_embeddings[0].shape[1] if chunk_sum_embeddings[0].size else 0
    global_meta = (
        meta.groupby(CITY_KEY_COLUMN, sort=True)
        .agg(
            {
                CITY_COLUMN: "first",
                CITY_CODE_COLUMN: "first",
                PROVINCE_COLUMN: "first",
                YEAR_COLUMN: "first",
                "n_patents": "sum",
                "n_texts_used": "sum",
                "total_citations": "sum",
            }
        )
        .reset_index()
    )
    global_meta["mean_citations"] = np.divide(
        global_meta["total_citations"].to_numpy(dtype=np.float64),
        global_meta["n_patents"].to_numpy(dtype=np.float64),
        out=np.full(len(global_meta), np.nan),
        where=global_meta["n_patents"].to_numpy() > 0,
    )

    key_to_idx = pd.Series(
        np.arange(len(global_meta), dtype=np.int64),
        index=global_meta[CITY_KEY_COLUMN].astype("string"),
    )
    n_groups = len(global_meta)
    global_sum_embedding = np.zeros((n_groups, dim), dtype=np.float64)
    global_text_counts = np.zeros(n_groups, dtype=np.float64)
    global_cit_weight_sum = np.zeros((n_groups, dim), dtype=np.float64)

    for part_meta, part_sum, part_counts, part_cit_sum in zip(
        chunk_meta_parts, chunk_sum_embeddings, chunk_text_counts, chunk_cit_weight_sums
    ):
        if len(part_meta) == 0:
            continue
        chunk_key_idx = (
            key_to_idx.reindex(part_meta[CITY_KEY_COLUMN].astype("string")).to_numpy(dtype=np.int64)
        )
        if np.any(np.isnan(chunk_key_idx)):
            raise RuntimeError("Failed to map chunk keys to global key space")
        np.add.at(global_sum_embedding, chunk_key_idx, part_sum)
        np.add.at(global_text_counts, chunk_key_idx, part_counts)
        np.add.at(global_cit_weight_sum, chunk_key_idx, part_cit_sum)

    global_sum_embedding = np.asarray(global_sum_embedding, dtype=np.float64)
    global_text_counts = np.asarray(global_text_counts, dtype=np.float64)
    global_meta["n_texts_used"] = global_text_counts.astype(np.int64)

    final_sum_embedding = divide_rows(global_sum_embedding, global_meta["n_texts_used"].to_numpy())

    cit_denom = global_meta["total_citations"].to_numpy(dtype=np.float64)
    final_cit_weight = divide_rows(global_cit_weight_sum, cit_denom)
    fallback = cit_denom <= ZERO_DIVISION_EPSILON
    if np.any(fallback):
        final_cit_weight[fallback] = final_sum_embedding[fallback]

    final_meta = global_meta[CITY_EMBEDDING_OUTPUT_COLUMNS].copy()
    return final_meta, final_sum_embedding.astype(np.float32), final_cit_weight.astype(np.float32)


def save_city_embeddings_bundle(
    output_dir: Path,
    prefix: str,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    save_npy: bool = False,
    model_short: str = "",
) -> None:
    """Save city-year embeddings to CSV and optionally NPY."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)

    emb_shape = embeddings.shape
    if len(emb_shape) != 2:
        embeddings = np.asarray(embeddings, dtype=np.float32).reshape((emb_shape[0], -1))

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    full = pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(embeddings, columns=emb_cols)],
        axis=1,
    )
    suffix = f"_{model_short}" if model_short else ""
    csv_path = output_dir / f"{prefix}{suffix}_embeddings.csv"
    full.to_csv(csv_path, index=False)
    log.info("Saved city embeddings CSV: %s", csv_path)

    if save_npy:
        meta_path = output_dir / f"{prefix}{suffix}_meta.csv"
        emb_path = output_dir / f"{prefix}{suffix}_embeddings.npy"
        meta.to_csv(meta_path, index=False)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))
        log.info("Saved city meta: %s", meta_path)
        log.info("Saved city embeddings NPY: %s", emb_path)


def write_city_embedding_outputs(
    output_dir: Path,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    citation_embeddings: Optional[np.ndarray],
    save_npy: bool = False,
    model_short: str = "",
) -> None:
    """Write city-year embedding outputs."""
    save_city_embeddings_bundle(
        output_dir=output_dir,
        prefix="city_year",
        meta=meta,
        embeddings=embeddings,
        save_npy=save_npy,
        model_short=model_short,
    )
    if citation_embeddings is None:
        return
    save_city_embeddings_bundle(
        output_dir=output_dir,
        prefix="city_year_citweighted",
        meta=meta,
        embeddings=citation_embeddings,
        save_npy=save_npy,
        model_short=model_short,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="City-level Patent SBERT embedding + aggregation")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input cleaned patent file (default: data/patents_cleaned.dta)",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument(
        "--model-name",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding (auto-detect if not specified)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--multi-gpu", action="store_true", help="Use multi-GPU encode path")
    p.add_argument(
        "--row-chunk-size",
        type=int,
        default=None,
        help="Process embeddings + aggregation in row chunks of this size",
    )
    p.add_argument(
        "--embed-backend",
        choices=["overflow", "legacy"],
        default="overflow",
    )
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--fp16", action="store_true", help="CUDA only: use fp16 weights")
    p.add_argument("--tf32", action="store_true", help="CUDA only: allow TF32 matmul")
    p.add_argument(
        "--include-empty-in-agg",
        action="store_true",
        help="Do not exclude empty text rows from aggregation",
    )
    p.add_argument("--save-npy", action="store_true", help="Also save .npy outputs")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def empty_city_embedding_outputs(output_dir: Path, save_npy: bool) -> None:
    write_city_embedding_outputs(
        output_dir=output_dir,
        meta=pd.DataFrame(columns=CITY_EMBEDDING_OUTPUT_COLUMNS),
        embeddings=np.empty((0, 0), dtype=np.float32),
        citation_embeddings=None,
        save_npy=save_npy,
    )


def process_city_embeddings(args: argparse.Namespace) -> int:
    log = logging.getLogger(__name__)
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_and_prepare_city_data(input_path)
    if df.empty:
        log.warning("No data loaded after preprocessing; writing empty outputs.")
        empty_city_embedding_outputs(output_dir=args.output_dir, save_npy=args.save_npy)
        return 0

    model_short = get_model_short_name(args.model_name)

    # Auto-recommend batch size if not specified
    batch_size = args.batch_size
    if batch_size is None:
        from patents_embeddings import get_gpu_info
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            batch_size = recommend_batch_size(gpu_info["memory_total"], args.model_name)
            log.info(
                "Auto-selected batch size: %d (GPU: %s, %.1fGB)",
                batch_size,
                gpu_info["name"],
                gpu_info["memory_total"],
            )
        else:
            batch_size = 64
            log.info("Auto-selected batch size: %d (CPU mode)", batch_size)
    else:
        log.info("Using specified batch size: %d", batch_size)

    embedder = SBertEmbedder(
        model_dir=args.model_dir,
        model_name=args.model_name,
        device=args.device,
        multi_gpu=args.multi_gpu,
        fp16=bool(getattr(args, "fp16", False)),
        tf32=bool(getattr(args, "tf32", False)),
        max_seq_length=getattr(args, "max_seq_length", None),
        embed_backend=getattr(args, "embed_backend", "overflow"),
    )

    try:
        row_chunk_size = getattr(args, "row_chunk_size", None)
        if row_chunk_size is not None:
            row_chunk_size = int(row_chunk_size)
            if row_chunk_size <= 0:
                row_chunk_size = None

        # Streaming row-chunk path
        if row_chunk_size is not None and len(df) > row_chunk_size:
            log.info(
                "Processing in row chunks: chunk_size=%d | total_rows=%d",
                row_chunk_size,
                len(df),
            )

            chunk_meta_parts: List[pd.DataFrame] = []
            chunk_sum_embeddings: List[np.ndarray] = []
            chunk_text_counts: List[np.ndarray] = []
            chunk_cit_weight_sums: List[np.ndarray] = []

            n_rows = len(df)
            n_chunks = (n_rows + row_chunk_size - 1) // row_chunk_size

            for chunk_id in range(n_chunks):
                start = chunk_id * row_chunk_size
                end = min((chunk_id + 1) * row_chunk_size, n_rows)
                df_chunk = df.iloc[start:end]

                log.info(
                    "Embedding chunk %d/%d: rows [%d:%d)",
                    chunk_id + 1,
                    n_chunks,
                    start,
                    end,
                )
                texts = df_chunk["text"].tolist()
                emb_chunk = embedder.embed(texts, batch_size=batch_size, show_progress=False)

                meta, simple_sum, text_counts, cit_weight_sum = aggregate_chunk_by_city(
                    df_chunk,
                    emb_chunk,
                    key_col=CITY_KEY_COLUMN,
                    exclude_empty_text=not args.include_empty_in_agg,
                    citation_col=CITATION_COLUMN,
                )
                chunk_meta_parts.append(meta)
                chunk_sum_embeddings.append(simple_sum)
                chunk_text_counts.append(text_counts)
                chunk_cit_weight_sums.append(cit_weight_sum)

            final_meta, city_year_embeddings, cit_weighted = finalize_city_aggregates(
                chunk_meta_parts,
                chunk_sum_embeddings,
                chunk_text_counts,
                chunk_cit_weight_sums,
            )

            write_city_embedding_outputs(
                output_dir=args.output_dir,
                meta=final_meta,
                embeddings=city_year_embeddings,
                citation_embeddings=cit_weighted,
                save_npy=args.save_npy,
                model_short=model_short,
            )

            log.info("Done (streaming). Outputs in: %s", args.output_dir)
            return 0

        # Full-file path
        texts = df["text"].tolist()
        patent_embeddings = embedder.embed(texts, batch_size=batch_size, show_progress=True)

        meta, simple_sum, text_counts, cit_weight_sum = aggregate_chunk_by_city(
            df,
            patent_embeddings,
            key_col=CITY_KEY_COLUMN,
            exclude_empty_text=not args.include_empty_in_agg,
            citation_col=CITATION_COLUMN,
        )

        final_meta, city_year_embeddings, cit_weighted = finalize_city_aggregates(
            [meta],
            [simple_sum],
            [text_counts],
            [cit_weight_sum],
        )

        write_city_embedding_outputs(
            output_dir=args.output_dir,
            meta=final_meta,
            embeddings=city_year_embeddings,
            citation_embeddings=cit_weighted,
            save_npy=args.save_npy,
            model_short=model_short,
        )

        log.info("Done. City-level outputs in: %s", args.output_dir)
        return 0
    finally:
        embedder.close()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info(
        "PyTorch: %s | CUDA available: %s",
        torch.__version__,
        torch.cuda.is_available(),
    )

    return process_city_embeddings(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
