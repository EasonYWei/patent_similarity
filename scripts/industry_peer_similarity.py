#!/usr/bin/env python3
# =============================================================================
# Industry Peer Similarity Calculation (Optimized Parallel Version)
# Computes cosine similarity between a firm and its industry peers from t-1, t-2, t-3 years
# =============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Constants
SAFE_COSINE_TOLERANCE = 1e-12


def safe_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate safe cosine similarity between two vectors."""
    if len(v1) == 0 or len(v2) == 0:
        return np.nan
    if len(v1) != len(v2):
        return np.nan
    
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        return np.nan
    
    n1 = np.sqrt(np.sum(v1 * v1))
    n2 = np.sqrt(np.sum(v2 * v2))
    
    if n1 <= SAFE_COSINE_TOLERANCE or n2 <= SAFE_COSINE_TOLERANCE:
        return np.nan
    
    return np.sum(v1 * v2) / (n1 * n2)


def calculate_max_peer_similarity(args) -> dict:
    """
    Calculate max similarity with peers for a single firm-year.
    This function is designed to be used with multiprocessing.
    """
    (stkcd, year, ind, emb_vec, industry_data_dict, embedding_cols) = args
    
    results = {
        'stkcd': stkcd,
        'p_year': year,
        'Ind': ind,
        'n_peers_t1': 0,
        'n_peers_t2': 0,
        'n_peers_t3': 0,
        'peer_sim_t1': np.nan,
        'peer_sim_t2': np.nan,
        'peer_sim_t3': np.nan,
    }
    
    for lag in [1, 2, 3]:
        target_year = year - lag
        
        # Get peers from same industry in target year, excluding self
        key = (ind, target_year)
        if key not in industry_data_dict:
            continue
            
        peers_df = industry_data_dict[key]
        peers_df = peers_df[peers_df['stkcd'] != stkcd]
        
        if len(peers_df) == 0:
            continue
        
        # Get peer embeddings as matrix
        peer_matrix = peers_df[embedding_cols].values.astype(np.float64)
        n_peers = len(peer_matrix)
        
        # Calculate cosine similarity with each peer
        # Vectorized computation for efficiency
        norms = np.sqrt(np.sum(peer_matrix ** 2, axis=1))
        emb_norm = np.sqrt(np.sum(emb_vec ** 2))
        
        if emb_norm <= SAFE_COSINE_TOLERANCE:
            continue
            
        # Filter out zero-norm peers
        valid_peers = norms > SAFE_COSINE_TOLERANCE
        if not np.any(valid_peers):
            continue
            
        peer_matrix_valid = peer_matrix[valid_peers]
        norms_valid = norms[valid_peers]
        
        # Compute dot products
        dot_products = np.dot(peer_matrix_valid, emb_vec)
        similarities = dot_products / (norms_valid * emb_norm)
        
        # Filter finite values and get max
        valid_sims = similarities[np.isfinite(similarities)]
        if len(valid_sims) > 0:
            max_sim = np.max(valid_sims)
        else:
            max_sim = np.nan
        
        if lag == 1:
            results['n_peers_t1'] = n_peers
            results['peer_sim_t1'] = max_sim
        elif lag == 2:
            results['n_peers_t2'] = n_peers
            results['peer_sim_t2'] = max_sim
        else:
            results['n_peers_t3'] = n_peers
            results['peer_sim_t3'] = max_sim
    
    return results


def process_industry_batch(industry_group, embedding_cols):
    """Process all firm-years within a single industry."""
    results = []
    
    # Build lookup dictionary for faster access
    industry_data_dict = {}
    for (ind, year), group in industry_group.groupby(['Ind', 'p_year']):
        industry_data_dict[(ind, year)] = group
    
    # Process each row
    for _, row in industry_group.iterrows():
        stkcd = row['stkcd']
        year = row['p_year']
        ind = row['Ind']
        emb_vec = row[embedding_cols].values.astype(np.float64)
        
        args = (stkcd, year, ind, emb_vec, industry_data_dict, embedding_cols)
        result = calculate_max_peer_similarity(args)
        results.append(result)
    
    return results


def calculate_industry_peer_similarity(embeddings_df: pd.DataFrame, 
                                       embedding_cols: list,
                                       n_workers: int = None) -> pd.DataFrame:
    """
    Calculate industry peer similarities using parallel processing.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Using {n_workers} workers for parallel processing...")
    
    # Sort data
    embeddings_df = embeddings_df.sort_values(['Ind', 'stkcd', 'p_year']).copy()
    
    # Get unique industries
    industries = embeddings_df['Ind'].unique()
    print(f"Processing {len(industries)} industries...")
    
    all_results = []
    
    # Prepare arguments for parallel processing by industry
    industry_groups = [group for _, group in embeddings_df.groupby('Ind')]
    
    # Process industries in parallel
    process_func = partial(process_industry_batch, embedding_cols=embedding_cols)
    
    with Pool(processes=n_workers) as pool:
        results_iter = pool.imap(process_func, industry_groups)
        
        for results in tqdm(results_iter, total=len(industry_groups), desc="Processing industries"):
            all_results.extend(results)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(all_results)
    
    # Merge with original metadata
    output_cols = ['stkcd', 'p_year', 'Ind', 'n_patents', 'n_texts_used',
                   'n_peers_t1', 'n_peers_t2', 'n_peers_t3',
                   'peer_sim_t1', 'peer_sim_t2', 'peer_sim_t3']
    
    result_df = result_df.merge(
        embeddings_df[['stkcd', 'p_year', 'n_patents', 'n_texts_used']],
        on=['stkcd', 'p_year'],
        how='left'
    )
    
    return result_df[output_cols]


def process_model(model_suffix: str, industry_info: pd.DataFrame, n_workers: int = None):
    """Process a single model's embeddings."""
    simple_input = f"./output/stkcd_year{model_suffix}_embeddings.csv"
    cit_input = f"./output/stkcd_year_citweighted{model_suffix}_embeddings.csv"
    simple_output = f"./output/industry_peer_similarity{model_suffix}.csv"
    cit_output = f"./output/industry_peer_similarity_citweighted{model_suffix}.csv"
    merged_output = f"./output/industry_peer_similarity_merged{model_suffix}.csv"
    
    if not os.path.exists(simple_input):
        print(f"Warning: Input file not found: {simple_input}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing model: {model_suffix}")
    print(f"{'='*60}")
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = pd.read_csv(simple_input)
    
    # Merge with industry info
    print("Merging with industry info...")
    industry_info_renamed = industry_info.rename(columns={'year': 'p_year'})
    embeddings = embeddings.merge(
        industry_info_renamed[['stkcd', 'p_year', 'Ind']],
        on=['stkcd', 'p_year'],
        how='left'
    )
    
    # Check for missing industry info
    missing_ind = embeddings['Ind'].isna().sum()
    if missing_ind > 0:
        print(f"Warning: {missing_ind} rows missing industry info, excluding them")
        embeddings = embeddings.dropna(subset=['Ind'])
    
    embedding_cols = [col for col in embeddings.columns if col.startswith('emb_')]
    print(f"Loaded: {len(embeddings)} rows, {len(embedding_cols)} dimensions, "
          f"{embeddings['Ind'].nunique()} unique industries")
    
    # Calculate simple peer similarities
    print("\nCalculating simple peer similarities...")
    result_simple = calculate_industry_peer_similarity(embeddings, embedding_cols, n_workers)
    result_simple.to_csv(simple_output, index=False)
    print(f"Saved: {simple_output}")
    
    # Process citation-weighted if available
    if os.path.exists(cit_input):
        print("\nProcessing citation-weighted embeddings...")
        embeddings_cit = pd.read_csv(cit_input)
        embeddings_cit = embeddings_cit.merge(
            industry_info_renamed[['stkcd', 'p_year', 'Ind']],
            on=['stkcd', 'p_year'],
            how='left'
        )
        embeddings_cit = embeddings_cit.dropna(subset=['Ind'])
        
        embedding_cols_cit = [col for col in embeddings_cit.columns if col.startswith('emb_')]
        print(f"Calculating citation-weighted peer similarities...")
        result_cit = calculate_industry_peer_similarity(embeddings_cit, embedding_cols_cit, n_workers)
        
        # Rename columns
        result_cit = result_cit.rename(columns={
            'n_peers_t1': 'n_peers_t1_citw',
            'n_peers_t2': 'n_peers_t2_citw',
            'n_peers_t3': 'n_peers_t3_citw',
            'peer_sim_t1': 'peer_sim_t1_citw',
            'peer_sim_t2': 'peer_sim_t2_citw',
            'peer_sim_t3': 'peer_sim_t3_citw',
        })
        
        result_cit.to_csv(cit_output, index=False)
        print(f"Saved: {cit_output}")
        
        # Merge and save
        merged = result_simple.merge(
            result_cit[['stkcd', 'p_year', 'Ind',
                       'n_peers_t1_citw', 'n_peers_t2_citw', 'n_peers_t3_citw',
                       'peer_sim_t1_citw', 'peer_sim_t2_citw', 'peer_sim_t3_citw']],
            on=['stkcd', 'p_year', 'Ind'],
            how='outer'
        )
        merged.to_csv(merged_output, index=False)
        print(f"Saved: {merged_output}")
    
    # Print summary
    print("\n" + "-"*40)
    print("Summary Statistics")
    print("-"*40)
    print(f"Simple peer_sim_t1: mean={result_simple['peer_sim_t1'].mean():.4f}, "
          f"sd={result_simple['peer_sim_t1'].std():.4f}, "
          f"n={result_simple['peer_sim_t1'].notna().sum()}")
    print(f"Simple peer_sim_t2: mean={result_simple['peer_sim_t2'].mean():.4f}, "
          f"sd={result_simple['peer_sim_t2'].std():.4f}, "
          f"n={result_simple['peer_sim_t2'].notna().sum()}")
    print(f"Simple peer_sim_t3: mean={result_simple['peer_sim_t3'].mean():.4f}, "
          f"sd={result_simple['peer_sim_t3'].std():.4f}, "
          f"n={result_simple['peer_sim_t3'].notna().sum()}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate industry peer similarity for patent embeddings'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='minilm,distiluse',
        help='Comma-separated list of models to process (default: minilm,distiluse)'
    )
    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Clean existing output files before processing'
    )
    
    args = parser.parse_args()
    
    # Clean existing files if requested
    if args.clean:
        print("Cleaning existing output files...")
        patterns = [
            'output/industry_peer_similarity*.csv',
        ]
        for pattern in patterns:
            for f in Path('.').glob(pattern):
                f.unlink()
                print(f"  Removed: {f}")
    
    # Load industry info
    print("Loading industry info from data/stkcd_info.xlsx...")
    industry_info = pd.read_excel('./data/stkcd_info.xlsx')
    industry_info.columns = [c.lower() for c in industry_info.columns]
    industry_info = industry_info.rename(columns={'ind': 'Ind'})
    industry_info['stkcd'] = industry_info['stkcd'].astype(int)
    industry_info['year'] = industry_info['year'].astype(int)
    
    print(f"Loaded industry info: {len(industry_info)} rows, "
          f"{industry_info['Ind'].nunique()} unique industries")
    
    # Process each model
    models = args.models.split(',')
    for model in models:
        model = model.strip()
        if model == 'minilm':
            process_model('_minilm', industry_info, args.workers)
        elif model == 'distiluse':
            process_model('_distiluse', industry_info, args.workers)
        else:
            print(f"Unknown model: {model}")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)


if __name__ == '__main__':
    main()
