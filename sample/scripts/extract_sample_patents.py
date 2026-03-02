#!/usr/bin/env python3
"""
Extract sample patent data from original cleaned data for manual inspection.
从原始清洗后的数据中提取样本专利数据用于人工检查。

This script extracts raw patent text data for specific sample companies
identified in create_sample.R (600808, 2, 61, 4).
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install: pip install pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample companies selected in create_sample.R
# Order: many years (>=20), medium (5-10), few (2-3), single (1)
# Note: stkcd in raw data is 6-digit zero-padded string
SAMPLE_STKCD = ["600808", "000002", "000061", "000004"]


def extract_sample_patents(
    input_path: Path,
    output_dir: Path,
    sample_stkcd: list[str] | None = None
) -> Path:
    """
    Extract patent data for sample companies from the cleaned dataset.
    
    Args:
        input_path: Path to patents_cleaned.dta
        output_dir: Directory to save extracted samples
        sample_stkcd: List of company codes to extract (default: SAMPLE_STKCD)
    
    Returns:
        Path to the saved sample CSV file
    """
    if sample_stkcd is None:
        sample_stkcd = SAMPLE_STKCD
    
    logger.info(f"Reading original data from: {input_path}")
    
    try:
        # Read only necessary columns to save memory
        df = pd.read_stata(
            input_path,
            columns=["stkcd", "p_year", "p_tt", "p_abs", "p_id", "p_cite", "p_date", "p_type"],
            convert_categoricals=False
        )
    except Exception as exc:
        logger.error(f"Failed to read {input_path}: {exc}")
        raise
    
    logger.info(f"Original data: {len(df):,} rows")
    
    # Filter for sample companies
    # stkcd might be numeric or string, handle both
    df["stkcd"] = df["stkcd"].astype(str).str.strip()
    sample_mask = df["stkcd"].isin(sample_stkcd)
    df_sample = df[sample_mask].copy()
    
    logger.info(f"Sample companies: {sample_stkcd}")
    logger.info(f"Extracted {len(df_sample):,} rows for sample companies")
    
    # Show breakdown by company
    logger.info("\nBreakdown by company:")
    for stkcd in sample_stkcd:
        count = len(df_sample[df_sample["stkcd"] == stkcd])
        year_range = ""
        if count > 0:
            years = df_sample[df_sample["stkcd"] == stkcd]["p_year"].dropna().unique()
            if len(years) > 0:
                year_range = f" ({min(years):.0f}-{max(years):.0f})"
        logger.info(f"  stkcd={stkcd}: {count:,} patents{year_range}")
    
    # Sort for easier inspection
    df_sample = df_sample.sort_values(["stkcd", "p_year", "p_id" if "p_id" in df_sample.columns else "p_tt"])
    
    # Save to CSV (more readable than Stata)
    output_path = output_dir / "sample_patents_raw.csv"
    df_sample.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"\nSaved sample patents to: {output_path}")
    
    # Also save as pickle for Python compatibility (faster loading)
    output_pkl = output_dir / "sample_patents_raw.pkl"
    df_sample.to_pickle(output_pkl)
    logger.info(f"Saved sample patents to: {output_pkl}")
    
    # Create a summary file
    summary = df_sample.groupby(["stkcd", "p_year"]).agg(
        n_patents=("p_tt", "count"),
        n_with_citations=("p_cite", lambda x: x.notna().sum()),
        total_citations=("p_cite", "sum"),
    ).reset_index()
    
    summary_path = output_dir / "sample_firm_year_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved firm-year summary to: {summary_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract sample patent data for manual inspection"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/patents_cleaned.dta"),
        help="Path to patents_cleaned.dta (default: ../data/patents_cleaned.dta)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sample/data relative to script)"
    )
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help=f"List of company codes to extract (default: {SAMPLE_STKCD})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent  # sample/scripts -> sample -> project_root
    
    if not args.input.is_absolute():
        input_path = project_root / args.input
    else:
        input_path = args.input
    
    if args.output_dir is None:
        output_dir = script_dir.parent / "data"  # sample/scripts/../data
    elif not args.output_dir.is_absolute():
        output_dir = project_root / args.output_dir
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        # Try alternative path
        alt_path = project_root / "data" / "patents_cleaned.dta"
        if alt_path.exists():
            input_path = alt_path
        else:
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
    
    sample_stkcd = args.companies if args.companies else SAMPLE_STKCD
    
    try:
        extract_sample_patents(input_path, output_dir, sample_stkcd)
        logger.info("\nExtraction complete!")
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
