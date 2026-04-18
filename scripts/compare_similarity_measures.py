#!/usr/bin/env python3
# =============================================================================
# Compare Similarity Measures
# Analyzes and compares firm-year, city-year, and industry-peer similarities
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate differences between similarity measures."""
    
    result = df.copy()
    
    # Simple mean differences
    result['diff_firm_peer'] = result['firm_cos_sim_lag1'] - result['peer_sim_t1']
    result['diff_firm_city'] = result['firm_cos_sim_lag1'] - result['city_cos_sim_lag1']
    result['diff_peer_city'] = result['peer_sim_t1'] - result['city_cos_sim_lag1']
    
    # Citation-weighted differences
    result['diff_firm_peer_citw'] = result['firm_cos_sim_lag1_citw'] - result['peer_sim_t1_citw']
    result['diff_firm_city_citw'] = result['firm_cos_sim_lag1_citw'] - result['city_cos_sim_lag1_citw']
    result['diff_peer_city_citw'] = result['peer_sim_t1_citw'] - result['city_cos_sim_lag1_citw']
    
    # Classification based on differences
    # Positive diff_firm_peer: firm is more similar to itself than peers (technological leader/uniqueness)
    # Negative diff_firm_peer: firm is more similar to peers than itself (technological follower)
    result['firm_vs_peer_type'] = pd.cut(
        result['diff_firm_peer'],
        bins=[-np.inf, -0.1, 0.1, np.inf],
        labels=['follower', 'neutral', 'leader']
    )
    
    return result


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for similarity measures."""
    
    sim_cols = ['firm_cos_sim_lag1', 'firm_cos_sim_lag3',
                'peer_sim_t1', 'peer_sim_t2', 'peer_sim_t3',
                'city_cos_sim_lag1', 'city_cos_sim_lag3']
    
    # Select columns that exist
    available_cols = [c for c in sim_cols if c in df.columns]
    
    corr_matrix = df[available_cols].corr()
    
    return corr_matrix


def analyze_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze similarity patterns by industry."""
    
    if 'Ind' not in df.columns:
        print("Warning: No industry column found, skipping industry analysis")
        return pd.DataFrame()
    
    industry_stats = df.groupby('Ind').agg({
        'firm_cos_sim_lag1': ['mean', 'std', 'count'],
        'peer_sim_t1': ['mean', 'std', 'count'],
        'city_cos_sim_lag1': ['mean', 'std', 'count'],
        'diff_firm_peer': ['mean', 'std'],
        'diff_firm_city': ['mean', 'std'],
        'stkcd': 'nunique'
    }).round(4)
    
    # Flatten column names
    industry_stats.columns = ['_'.join(col).strip() for col in industry_stats.columns]
    industry_stats = industry_stats.rename(columns={'stkcd_nunique': 'n_firms'})
    
    return industry_stats.reset_index()


def analyze_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze similarity patterns by city."""
    
    if 'city_code' not in df.columns:
        print("Warning: No city_code column found, skipping city analysis")
        return pd.DataFrame()
    
    city_stats = df.groupby('city_code').agg({
        'firm_cos_sim_lag1': ['mean', 'std', 'count'],
        'peer_sim_t1': ['mean', 'std', 'count'],
        'city_cos_sim_lag1': ['mean', 'std', 'count'],
        'diff_firm_peer': ['mean', 'std'],
        'stkcd': 'nunique'
    }).round(4)
    
    # Flatten column names
    city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns]
    city_stats = city_stats.rename(columns={'stkcd_nunique': 'n_firms'})
    
    # Add city name if available
    if 'city' in df.columns:
        city_names = df[['city_code', 'city']].drop_duplicates()
        city_stats = city_stats.reset_index().merge(city_names, on='city_code', how='left')
    
    return city_stats


def create_summary_report(df: pd.DataFrame, model_suffix: str) -> dict:
    """Create comprehensive summary report."""
    
    report = {
        'model': model_suffix,
        'total_observations': len(df),
        'unique_firms': df['stkcd'].nunique(),
        'year_range': f"{df['p_year'].min()}-{df['p_year'].max()}",
    }
    
    # Basic statistics for each measure
    measures = {
        'firm_cos_sim_lag1': 'Firm Self-Similarity (t-1)',
        'peer_sim_t1': 'Industry Peer Similarity (t-1)',
        'city_cos_sim_lag1': 'City Similarity (t-1)'
    }
    
    for col, name in measures.items():
        if col in df.columns:
            valid = df[col].notna()
            report[f'{col}_mean'] = df.loc[valid, col].mean()
            report[f'{col}_std'] = df.loc[valid, col].std()
            report[f'{col}_n'] = valid.sum()
    
    # Difference statistics
    diff_measures = ['diff_firm_peer', 'diff_firm_city', 'diff_peer_city']
    for col in diff_measures:
        if col in df.columns:
            valid = df[col].notna()
            report[f'{col}_mean'] = df.loc[valid, col].mean()
            report[f'{col}_std'] = df.loc[valid, col].std()
    
    # Correlation between measures
    sim_cols = ['firm_cos_sim_lag1', 'peer_sim_t1', 'city_cos_sim_lag1']
    available_cols = [c for c in sim_cols if c in df.columns]
    if len(available_cols) >= 2:
        corr = df[available_cols].corr()
        for i in range(len(available_cols)):
            for j in range(i+1, len(available_cols)):
                col1, col2 = available_cols[i], available_cols[j]
                report[f'corr_{col1}_{col2}'] = corr.loc[col1, col2]
    
    return report


def process_model(model_suffix: str):
    """Process a single model's merged data."""
    
    input_file = f"./output/merged_similarity_by_firm{model_suffix}.csv"
    
    if not os.path.exists(input_file):
        print(f"Warning: Input file not found: {input_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Comparing similarities for model: {model_suffix}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading merged data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded: {len(df)} rows")
    
    # Calculate differences
    print("Calculating differences...")
    df = calculate_differences(df)
    
    # Save comparison file
    comparison_output = f"./output/similarity_comparison{model_suffix}.csv"
    
    # Select columns for output
    base_cols = ['stkcd', 'p_year', 'city_code', 'city', 'Ind', 
                 'n_patents_firm', 'n_texts_used_firm']
    sim_cols = ['firm_cos_sim_lag1', 'firm_cos_sim_lag3',
                'peer_sim_t1', 'peer_sim_t2', 'peer_sim_t3',
                'city_cos_sim_lag1', 'city_cos_sim_lag3']
    diff_cols = ['diff_firm_peer', 'diff_firm_city', 'diff_peer_city',
                 'diff_firm_peer_citw', 'diff_firm_city_citw', 'diff_peer_city_citw',
                 'firm_vs_peer_type']
    
    output_cols = [c for c in base_cols + sim_cols + diff_cols if c in df.columns]
    df_output = df[output_cols]
    df_output.to_csv(comparison_output, index=False)
    print(f"\nSaved comparison: {comparison_output}")
    
    # Create summary report
    print("\n" + "-"*60)
    print("OVERALL SUMMARY")
    print("-"*60)
    
    report = create_summary_report(df, model_suffix)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Correlation matrix
    print("\n" + "-"*60)
    print("CORRELATION MATRIX")
    print("-"*60)
    corr_matrix = calculate_correlations(df)
    print(corr_matrix.round(4))
    
    # Save correlation matrix
    corr_output = f"./output/similarity_correlation{model_suffix}.csv"
    corr_matrix.to_csv(corr_output)
    print(f"\nSaved correlation matrix: {corr_output}")
    
    # Industry analysis
    print("\n" + "-"*60)
    print("INDUSTRY-LEVEL ANALYSIS")
    print("-"*60)
    industry_stats = analyze_by_industry(df)
    if not industry_stats.empty:
        print(industry_stats.head(10))
        industry_output = f"./output/similarity_by_industry{model_suffix}.csv"
        industry_stats.to_csv(industry_output, index=False)
        print(f"\nSaved industry stats: {industry_output}")
    
    # City analysis
    print("\n" + "-"*60)
    print("CITY-LEVEL ANALYSIS (Top 10)")
    print("-"*60)
    city_stats = analyze_by_city(df)
    if not city_stats.empty:
        # Sort by number of firms and show top 10
        city_stats_sorted = city_stats.sort_values('n_firms', ascending=False).head(10)
        print(city_stats_sorted)
        city_output = f"./output/similarity_by_city_summary{model_suffix}.csv"
        city_stats.to_csv(city_output, index=False)
        print(f"\nSaved city stats: {city_output}")
    
    # Distribution of firm types
    if 'firm_vs_peer_type' in df.columns:
        print("\n" + "-"*60)
        print("FIRM TYPE DISTRIBUTION (Firm vs Peer)")
        print("-"*60)
        type_dist = df['firm_vs_peer_type'].value_counts()
        print(type_dist)
        print(f"\nPercentages:")
        print((type_dist / type_dist.sum() * 100).round(2))


def main():
    parser = argparse.ArgumentParser(
        description='Compare similarity measures across different dimensions'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='minilm,distiluse',
        help='Comma-separated list of models to process (default: minilm,distiluse)'
    )
    
    args = parser.parse_args()
    
    # Process each model
    models = args.models.split(',')
    for model in models:
        model = model.strip()
        if model == 'minilm':
            process_model('_minilm')
        elif model == 'distiluse':
            process_model('_distiluse')
        else:
            print(f"Unknown model: {model}")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)


if __name__ == '__main__':
    main()
