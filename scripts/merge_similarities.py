#!/usr/bin/env python3
# =============================================================================
# Merge Similarity Measures
# Combines firm-year, city-year, and industry-peer similarities into unified files
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_stkcd_city_mapping(dta_path: str) -> pd.DataFrame:
    """Load stkcd to city_code mapping from patents data."""
    print(f"Loading stkcd-city mapping from {dta_path}...")
    
    # Read only necessary columns
    df = pd.read_stata(dta_path, columns=['stkcd', 'p_year', 'city', 'city_code'])
    
    # Convert stkcd to string to match similarity files
    df['stkcd'] = df['stkcd'].astype(str).str.strip()
    df['p_year'] = df['p_year'].astype(int)
    
    # Create unique mapping: for each stkcd-p_year, get the most frequent city
    # Group by stkcd and p_year, get mode of city_code
    mapping = df.groupby(['stkcd', 'p_year']).agg({
        'city_code': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'city': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    print(f"Loaded mapping: {len(mapping)} stkcd-year observations, "
          f"{mapping['city_code'].nunique()} unique cities")
    
    return mapping


def merge_firm_level_similarities(model_suffix: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """Merge firm-year, industry-peer, and city-year similarities at firm level."""
    
    # File paths
    firm_file = f"./output/stkcd_year_similarity_merged{model_suffix}.csv"
    peer_file = f"./output/industry_peer_similarity_merged{model_suffix}.csv"
    city_file = f"./output/city_year_similarity_merged{model_suffix}.csv"
    
    print(f"\n{'='*60}")
    print(f"Processing model: {model_suffix}")
    print(f"{'='*60}")
    
    # 1. Load firm-year similarity
    print("Loading firm-year similarity...")
    firm_sim = pd.read_csv(firm_file)
    firm_sim['stkcd'] = firm_sim['stkcd'].astype(str).str.strip()
    firm_sim['p_year'] = firm_sim['p_year'].astype(int)
    firm_sim = firm_sim.rename(columns={
        'cos_sim_lag1': 'firm_cos_sim_lag1',
        'cos_sim_lag3': 'firm_cos_sim_lag3',
        'cos_sim_cumulative': 'firm_cos_sim_cumulative',
        'cos_sim_lag1_citw': 'firm_cos_sim_lag1_citw',
        'cos_sim_lag3_citw': 'firm_cos_sim_lag3_citw',
        'cos_sim_cumulative_citw': 'firm_cos_sim_cumulative_citw',
        'n_patents': 'n_patents_firm',
        'n_texts_used': 'n_texts_used_firm'
    })
    print(f"  Loaded: {len(firm_sim)} rows")
    
    # 2. Load industry-peer similarity
    print("Loading industry-peer similarity...")
    peer_sim = pd.read_csv(peer_file)
    peer_sim['stkcd'] = peer_sim['stkcd'].astype(str).str.strip()
    peer_sim['p_year'] = peer_sim['p_year'].astype(int)
    peer_cols = ['stkcd', 'p_year', 'Ind', 'n_peers_t1', 'n_peers_t2', 'n_peers_t3',
                 'peer_sim_t1', 'peer_sim_t2', 'peer_sim_t3',
                 'n_peers_t1_citw', 'n_peers_t2_citw', 'n_peers_t3_citw',
                 'peer_sim_t1_citw', 'peer_sim_t2_citw', 'peer_sim_t3_citw']
    peer_sim = peer_sim[peer_cols]
    print(f"  Loaded: {len(peer_sim)} rows")
    
    # 3. Merge firm and peer
    print("Merging firm-year with industry-peer...")
    merged = firm_sim.merge(peer_sim, on=['stkcd', 'p_year'], how='outer')
    print(f"  After merge: {len(merged)} rows")
    
    # 4. Add city mapping
    print("Adding city mapping...")
    merged = merged.merge(mapping, on=['stkcd', 'p_year'], how='left')
    missing_city = merged['city_code'].isna().sum()
    if missing_city > 0:
        print(f"  Warning: {missing_city} rows missing city mapping")
    
    # 5. Load city-year similarity
    print("Loading city-year similarity...")
    city_sim = pd.read_csv(city_file)
    city_sim = city_sim.rename(columns={
        'cos_sim_lag1': 'city_cos_sim_lag1',
        'cos_sim_lag3': 'city_cos_sim_lag3',
        'cos_sim_cumulative': 'city_cos_sim_cumulative',
        'cos_sim_lag1_citw': 'city_cos_sim_lag1_citw',
        'cos_sim_lag3_citw': 'city_cos_sim_lag3_citw',
        'cos_sim_cumulative_citw': 'city_cos_sim_cumulative_citw',
        'n_patents': 'n_patents_city',
        'n_texts_used': 'n_texts_used_city'
    })
    # Keep only necessary columns for merging
    city_cols = ['city_code', 'p_year', 'city_cos_sim_lag1', 'city_cos_sim_lag3', 
                 'city_cos_sim_cumulative', 'city_cos_sim_lag1_citw', 
                 'city_cos_sim_lag3_citw', 'city_cos_sim_cumulative_citw',
                 'n_patents_city', 'n_texts_used_city']
    city_sim = city_sim[city_cols]
    print(f"  Loaded: {len(city_sim)} rows")
    
    # 6. Merge city similarity
    print("Merging city-year similarity...")
    merged = merged.merge(city_sim, on=['city_code', 'p_year'], how='left', suffixes=('', '_city'))
    # Fix duplicate city column if exists
    if 'city_city' in merged.columns:
        merged = merged.drop(columns=['city_city'])
    print(f"  Final: {len(merged)} rows")
    
    return merged


def create_city_level_file(model_suffix: str) -> pd.DataFrame:
    """Create city-level merged file."""
    
    city_file = f"./output/city_year_similarity_merged{model_suffix}.csv"
    
    print(f"\nCreating city-level file for {model_suffix}...")
    
    # Load city-year similarity
    city_sim = pd.read_csv(city_file)
    
    # Select and rename columns
    city_cols = ['city_code', 'p_year', 'city', 'n_patents', 'n_texts_used',
                 'cos_sim_lag1', 'cos_sim_lag3', 'cos_sim_cumulative',
                 'cos_sim_lag1_citw', 'cos_sim_lag3_citw', 'cos_sim_cumulative_citw']
    city_df = city_sim[city_cols].copy()
    city_df = city_df.rename(columns={
        'n_patents': 'n_patents_city',
        'n_texts_used': 'n_texts_used_city'
    })
    
    print(f"  Created: {len(city_df)} rows")
    
    return city_df


def print_summary(df: pd.DataFrame, name: str):
    """Print summary statistics."""
    print(f"\n{'-'*60}")
    print(f"Summary: {name}")
    print(f"{'-'*60}")
    print(f"Total rows: {len(df)}")
    print(f"Unique firms: {df['stkcd'].nunique() if 'stkcd' in df.columns else 'N/A'}")
    print(f"Year range: {df['p_year'].min()}-{df['p_year'].max()}")
    
    # Similarity columns summary
    sim_cols = [c for c in df.columns if 'sim' in c and 'citw' not in c]
    if sim_cols:
        print("\nSimilarity measures (simple):")
        for col in sim_cols[:6]:  # Limit to first 6
            valid = df[col].notna().sum()
            if valid > 0:
                print(f"  {col}: mean={df[col].mean():.4f}, sd={df[col].std():.4f}, n={valid}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge similarity measures into unified files'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='minilm,distiluse',
        help='Comma-separated list of models to process (default: minilm,distiluse)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/patents_cleaned_with_city.dta',
        help='Path to patents data with city info'
    )
    
    args = parser.parse_args()
    
    # Load mapping once
    mapping = load_stkcd_city_mapping(args.data_path)
    
    # Process each model
    models = args.models.split(',')
    for model in models:
        model = model.strip()
        if model == 'minilm':
            suffix = '_minilm'
        elif model == 'distiluse':
            suffix = '_distiluse'
        else:
            print(f"Unknown model: {model}")
            continue
        
        # Create firm-level merged file
        firm_merged = merge_firm_level_similarities(suffix, mapping)
        firm_output = f"./output/merged_similarity_by_firm{suffix}.csv"
        firm_merged.to_csv(firm_output, index=False)
        print(f"\nSaved: {firm_output}")
        print_summary(firm_merged, f"Firm-level ({model})")
        
        # Create city-level file
        city_merged = create_city_level_file(suffix)
        city_output = f"./output/merged_similarity_by_city{suffix}.csv"
        city_merged.to_csv(city_output, index=False)
        print(f"\nSaved: {city_output}")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)


if __name__ == '__main__':
    main()
