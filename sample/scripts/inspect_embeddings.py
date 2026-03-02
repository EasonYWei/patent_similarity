#!/usr/bin/env python3
"""
Inspect embeddings and aggregation results for sample patents.
逐条检查样本专利的embeddings和聚合结果。

This script provides detailed inspection capabilities for understanding
how patent embeddings are computed and aggregated at firm-year level.

Usage:
    python inspect_embeddings.py --step all          # Run full pipeline
    python inspect_embeddings.py --step load         # Just load and show data
    python inspect_embeddings.py --step embed        # Compute embeddings
    python inspect_embeddings.py --step aggregate    # Show aggregation details
    python inspect_embeddings.py --company 600808    # Focus on one company
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path for importing
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SAMPLE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Now we can import from the main script
try:
    from patents_embeddings import (
        SBertEmbedder,
        aggregate_chunk,
        build_text_field,
        coerce_citations,
        divide_rows,
        finalize_chunk_aggregates,
        write_embedding_outputs,
        STKCD_COLUMN,
        YEAR_COLUMN,
        KEY_COLUMN,
        TEXT_COLUMNS,
        CITATION_COLUMN,
        EMBEDDING_OUTPUT_COLUMNS,
    )
except ImportError as e:
    print(f"Error importing from main script: {e}")
    print("Make sure patents_embeddings.py exists in the main scripts folder")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_sample_data(data_path: Path) -> pd.DataFrame:
    """Load sample patent data from CSV or pickle."""
    if data_path.suffix == ".pkl":
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Reset index to ensure 0-based contiguous indexing for embeddings alignment
    df = df.reset_index(drop=True)
    
    # Ensure stkcd is string
    df[STKCD_COLUMN] = df[STKCD_COLUMN].astype("string").str.strip()
    df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors="coerce").astype("Int32")
    
    # Build text field
    df["text"], df["text_is_empty"] = build_text_field(df)
    df[KEY_COLUMN] = (
        df[STKCD_COLUMN].astype("string") + "_" + df[YEAR_COLUMN].astype("Int32").astype(str)
    )
    
    # Handle citations
    if CITATION_COLUMN in df.columns:
        df[CITATION_COLUMN] = (
            pd.to_numeric(df[CITATION_COLUMN], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    
    return df


def show_data_summary(df: pd.DataFrame, focus_company: Optional[str] = None):
    """Display summary of loaded data."""
    print("\n" + "="*70)
    print("DATA SUMMARY / 数据摘要")
    print("="*70)
    
    print(f"\nTotal patents: {len(df):,}")
    print(f"Companies: {df[STKCD_COLUMN].nunique()}")
    print(f"Firm-years: {df[KEY_COLUMN].nunique()}")
    print(f"Empty texts: {df['text_is_empty'].sum():,}")
    
    # Company breakdown
    print("\n--- Company Breakdown / 公司分布 ---")
    company_stats = df.groupby(STKCD_COLUMN).agg(
        n_patents=(YEAR_COLUMN, "count"),
        year_min=(YEAR_COLUMN, "min"),
        year_max=(YEAR_COLUMN, "max"),
        n_years=(YEAR_COLUMN, "nunique"),
    ).reset_index()
    
    for _, row in company_stats.iterrows():
        marker = " <--" if focus_company and str(row[STKCD_COLUMN]) == focus_company else ""
        print(f"  {row[STKCD_COLUMN]}: {row['n_patents']:>4} patents, "
              f"{row['n_years']:>2} years ({row['year_min']:.0f}-{row['year_max']:.0f}){marker}")
    
    # Firm-year breakdown
    print("\n--- Firm-Year Breakdown / 公司-年份分布 ---")
    fy_stats = df.groupby([STKCD_COLUMN, YEAR_COLUMN]).agg(
        n_patents=("text", "count"),
        n_empty=("text_is_empty", "sum"),
        total_citations=(CITATION_COLUMN, "sum") if CITATION_COLUMN in df.columns else ("text", lambda x: 0),
    ).reset_index()
    
    for _, row in fy_stats.head(20).iterrows():  # Show first 20
        marker = ""
        if focus_company and str(row[STKCD_COLUMN]) == focus_company:
            marker = " <--"
        cite_info = f", citations={row['total_citations']:.0f}" if CITATION_COLUMN in df.columns else ""
        empty_info = f", empty={int(row['n_empty'])}" if row['n_empty'] > 0 else ""
        print(f"  {row[STKCD_COLUMN]}_{row[YEAR_COLUMN]:.0f}: "
              f"{row['n_patents']:>3} patents{empty_info}{cite_info}{marker}")
    
    if len(fy_stats) > 20:
        print(f"  ... and {len(fy_stats) - 20} more firm-years")


def show_sample_texts(df: pd.DataFrame, company: Optional[str] = None, n: int = 3):
    """Display sample patent texts for inspection."""
    print("\n" + "="*70)
    print("SAMPLE PATENT TEXTS / 样本专利文本")
    print("="*70)
    
    if company:
        df = df[df[STKCD_COLUMN] == company]
    
    sample = df.head(n)
    
    for idx, row in sample.iterrows():
        print(f"\n--- Patent {idx+1} ---")
        print(f"Company: {row[STKCD_COLUMN]}, Year: {row[YEAR_COLUMN]}")
        if "p_id" in row:
            print(f"Patent ID: {row['p_id']}")
        if CITATION_COLUMN in row:
            print(f"Citations: {row[CITATION_COLUMN]}")
        print(f"Text empty: {row['text_is_empty']}")
        print(f"Combined text (first 200 chars): {row['text'][:200]}...")


def compute_and_inspect_embeddings(
    df: pd.DataFrame,
    model_dir: Path,
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings with detailed logging."""
    print("\n" + "="*70)
    print("EMBEDDING COMPUTATION / 嵌入计算")
    print("="*70)
    
    embedder = SBertEmbedder(
        model_dir=model_dir,
        model_name=model_name,
        device=device,
    )
    
    texts = df["text"].tolist()
    print(f"\nComputing embeddings for {len(texts)} patents...")
    print(f"Model: {model_name}")
    print(f"Device: {embedder.device}")
    print(f"Batch size: {batch_size}")
    
    t0 = time.time()
    embeddings = embedder.embed(texts, batch_size=batch_size, show_progress=True)
    elapsed = time.time() - t0
    
    print(f"\nCompleted in {elapsed:.2f}s ({elapsed/len(texts):.4f}s per patent)")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    
    # Show sample embedding vector
    print(f"\nSample embedding (first patent, first 10 dims):")
    print(f"  {embeddings[0, :10]}")
    
    return embeddings


def inspect_aggregation(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    focus_company: Optional[str] = None,
):
    """Show detailed aggregation process."""
    print("\n" + "="*70)
    print("AGGREGATION DETAILS / 聚合详情")
    print("="*70)
    
    # First show the aggregation by firm-year
    print("\n--- Step 1: Group by firm-year / 按公司-年份分组 ---")
    
    keys = df[KEY_COLUMN].unique()
    print(f"Total unique firm-years: {len(keys)}")
    
    # Show aggregation for a specific company if requested
    # Note: If focus_company is specified, df and embeddings should already
    # be filtered in main(). This is a safety check.
    if focus_company and len(df[STKCD_COLUMN].unique()) > 1:
        company_mask = df[STKCD_COLUMN] == focus_company
        company_indices = df[company_mask].index.to_numpy()
        df = df[company_mask].copy().reset_index(drop=True)
        embeddings = embeddings[company_indices, :]
    
    # Get unique keys and inverse mapping
    keys_arr = df[KEY_COLUMN].astype("string").to_numpy()
    uniq_keys, inv = np.unique(keys_arr, return_inverse=True)
    
    print(f"\nProcessing {len(uniq_keys)} firm-year groups...")
    
    # Simple mean aggregation (excluding empty texts)
    text_weights = (~df["text_is_empty"].to_numpy()).astype(np.float32)
    dim = embeddings.shape[1]
    n_groups = len(uniq_keys)
    
    sum_embedding = np.zeros((n_groups, dim), dtype=np.float32)
    np.add.at(sum_embedding, inv, embeddings * text_weights[:, None])
    n_text_rows = np.bincount(inv, weights=text_weights).astype(np.float64)
    
    # Compute mean
    mean_embeddings = divide_rows(sum_embedding, n_text_rows)
    
    # Show details for each group
    print("\n--- Step 2: Simple Mean Aggregation / 简单平均聚合 ---")
    for i, key in enumerate(uniq_keys):
        group_mask = inv == i
        group_size = group_mask.sum()
        n_non_empty = int(n_text_rows[i])
        
        print(f"\n  Firm-Year: {key}")
        print(f"    Total patents: {group_size}")
        print(f"    Non-empty texts: {n_non_empty}")
        print(f"    Mean embedding (first 5 dims): {mean_embeddings[i, :5]}")
        print(f"    Norm of mean: {np.linalg.norm(mean_embeddings[i]):.4f}")
    
    # Citation-weighted aggregation
    if CITATION_COLUMN in df.columns:
        print("\n--- Step 3: Citation-Weighted Aggregation / 引证加权聚合 ---")
        
        citations = df[CITATION_COLUMN].to_numpy()
        sum_cit_weight = np.zeros((n_groups, dim), dtype=np.float32)
        np.add.at(sum_cit_weight, inv, embeddings * citations[:, None].astype(np.float32))
        total_citations = np.bincount(inv, weights=citations).astype(np.float64)
        
        cit_weighted_embeddings = divide_rows(sum_cit_weight, total_citations)
        
        for i, key in enumerate(uniq_keys):
            total_cite = total_citations[i]
            print(f"\n  Firm-Year: {key}")
            print(f"    Total citations: {total_cite:.1f}")
            print(f"    Cit-weighted mean (first 5 dims): {cit_weighted_embeddings[i, :5]}")
            print(f"    Norm of weighted mean: {np.linalg.norm(cit_weighted_embeddings[i]):.4f}")
    
    return mean_embeddings


def save_inspection_results(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    model_short: str,
):
    """Save inspection results for further analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save patent-level embeddings
    patent_output = output_dir / f"patent_level_{model_short}_inspection.csv"
    
    # Create DataFrame with metadata and embeddings
    meta_cols = [STKCD_COLUMN, YEAR_COLUMN, KEY_COLUMN, "text", "text_is_empty"]
    if CITATION_COLUMN in df.columns:
        meta_cols.append(CITATION_COLUMN)
    if "p_id" in df.columns:
        meta_cols.append("p_id")
    
    result_df = df[meta_cols].copy()
    
    # Add embedding columns
    for i in range(embeddings.shape[1]):
        result_df[f"emb_{i}"] = embeddings[:, i]
    
    result_df.to_csv(patent_output, index=False, encoding="utf-8-sig")
    print(f"\nSaved patent-level embeddings to: {patent_output}")
    
    # Also save as numpy for easy loading
    np_output = output_dir / f"patent_level_{model_short}_embeddings.npy"
    np.save(np_output, embeddings)
    print(f"Saved embeddings array to: {np_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect embeddings and aggregation for sample patents"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SAMPLE_DIR / "data" / "sample_patents_raw.pkl",
        help="Path to sample patent data (CSV or pickle)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory containing SBERT models"
    )
    parser.add_argument(
        "--model-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SBERT model name"
    )
    parser.add_argument(
        "--step",
        choices=["load", "embed", "aggregate", "all"],
        default="all",
        help="Which step to run (default: all)"
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Focus on specific company (stkcd)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for computation (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAMPLE_DIR / "output",
        help="Output directory for inspection results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths
    data_path = args.data
    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            SAMPLE_DIR / "data" / "sample_patents_raw.csv",
            SAMPLE_DIR / "data" / "sample_patents_raw.pkl",
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            print(f"Error: Data file not found: {args.data}")
            print("Please run extract_sample_patents.py first")
            sys.exit(1)
    
    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Run requested steps
    embeddings = None
    df = None
    
    if args.step in ("load", "embed", "aggregate", "all"):
        print(f"\nLoading data from: {data_path}")
        df = load_sample_data(data_path)
        show_data_summary(df, focus_company=args.company)
        show_sample_texts(df, company=args.company, n=3)
    
    if args.step in ("embed", "aggregate", "all"):
        if df is None:
            df = load_sample_data(data_path)
        
        # If company is specified, only compute embeddings for that company
        if args.company:
            company_mask = df[STKCD_COLUMN] == args.company
            df_compute = df[company_mask].copy().reset_index(drop=True)
            print(f"\n[Filter] Computing embeddings only for company {args.company} "
                  f"({len(df_compute)} patents)")
        else:
            df_compute = df
        
        embeddings = compute_and_inspect_embeddings(
            df_compute,
            model_dir=model_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        
        # Update df to match the computed embeddings
        if args.company:
            df = df_compute
    
    if args.step in ("aggregate", "all"):
        if df is None:
            df = load_sample_data(data_path)
        if embeddings is None:
            # Load pre-computed embeddings if available
            model_short = args.model_name.split("-")[0].lower()
            np_path = args.output_dir / f"patent_level_{model_short}_embeddings.npy"
            if np_path.exists():
                embeddings = np.load(np_path)
                print(f"\nLoaded pre-computed embeddings from: {np_path}")
            else:
                print("\nError: Embeddings not computed and no saved file found")
                print("Please run with --step embed or --step all first")
                sys.exit(1)
        
        inspect_aggregation(df, embeddings, focus_company=args.company)
    
    # Save results
    if args.step == "all" and embeddings is not None:
        model_short = args.model_name.split("-")[0].lower()
        save_inspection_results(df, embeddings, args.output_dir, model_short)
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE / 检查完成")
    print("="*70)


if __name__ == "__main__":
    main()
