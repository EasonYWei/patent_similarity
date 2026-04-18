#!/usr/bin/env python3
# =============================================================================
# Industry Peer Similarity Calculation (Highly Optimized Version)
# Uses efficient matrix operations and pre-computed lookup tables
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Constants
SAFE_COSINE_TOLERANCE = 1e-12


def precompute_industry_matrices(embeddings_df, embedding_cols):
    """
    Pre-compute matrices and norms for each (industry, year) combination.
    Returns a dictionary mapping (ind, year) -> (matrix, norms, stkcd_list)
    """
    lookup = {}
    groups = embeddings_df.groupby(['Ind', 'p_year'])
    
    for (ind, year), group in tqdm(groups, desc="Precomputing matrices"):
        matrix = group[embedding_cols].values.astype(np.float64)
        norms = np.sqrt(np.sum(matrix ** 2, axis=1))
        stkcds = group['stkcd'].values
        lookup[(ind, year)] = (matrix, norms, stkcds)
    
    return lookup


def calculate_peer_similarities_vectorized(embeddings_df, embedding_cols, lookup):
    """
    Calculate max peer similarities using pre-computed matrices.
    Vectorized operations for efficiency.
    """
    n_rows = len(embeddings_df)
    results = {
        'stkcd': embeddings_df['stkcd'].values,
        'p_year': embeddings_df['p_year'].values,
        'Ind': embeddings_df['Ind'].values,
        'n_patents': embeddings_df['n_patents'].values,
        'n_texts_used': embeddings_df['n_texts_used'].values,
        'n_peers_t1': np.zeros(n_rows, dtype=int),
        'n_peers_t2': np.zeros(n_rows, dtype=int),
        'n_peers_t3': np.zeros(n_rows, dtype=int),
        'peer_sim_t1': np.full(n_rows, np.nan),
        'peer_sim_t2': np.full(n_rows, np.nan),
        'peer_sim_t3': np.full(n_rows, np.nan),
    }
    
    # Convert to matrices for fast access
    emb_matrix = embeddings_df[embedding_cols].values.astype(np.float64)
    emb_norms = np.sqrt(np.sum(emb_matrix ** 2, axis=1))
    
    for i in tqdm(range(n_rows), desc="Calculating similarities"):
        stkcd = results['stkcd'][i]
        year = results['p_year'][i]
        ind = results['Ind'][i]
        emb_vec = emb_matrix[i]
        emb_norm = emb_norms[i]
        
        if emb_norm <= SAFE_COSINE_TOLERANCE:
            continue
        
        for lag in [1, 2, 3]:
            target_year = year - lag
            key = (ind, target_year)
            
            if key not in lookup:
                continue
            
            peer_matrix, peer_norms, peer_stkcds = lookup[key]
            
            # Exclude self
            mask = peer_stkcds != stkcd
            if not np.any(mask):
                continue
            
            peer_matrix_filtered = peer_matrix[mask]
            peer_norms_filtered = peer_norms[mask]
            n_peers = len(peer_matrix_filtered)
            
            # Filter valid peers (non-zero norms)
            valid_mask = peer_norms_filtered > SAFE_COSINE_TOLERANCE
            if not np.any(valid_mask):
                continue
            
            peer_matrix_valid = peer_matrix_filtered[valid_mask]
            peer_norms_valid = peer_norms_filtered[valid_mask]
            
            # Vectorized cosine similarity
            dot_products = np.dot(peer_matrix_valid, emb_vec)
            similarities = dot_products / (peer_norms_valid * emb_norm)
            
            # Get max finite similarity
            finite_sims = similarities[np.isfinite(similarities)]
            if len(finite_sims) > 0:
                max_sim = np.max(finite_sims)
            else:
                max_sim = np.nan
            
            if lag == 1:
                results['n_peers_t1'][i] = n_peers
                results['peer_sim_t1'][i] = max_sim
            elif lag == 2:
                results['n_peers_t2'][i] = n_peers
                results['peer_sim_t2'][i] = max_sim
            else:
                results['n_peers_t3'][i] = n_peers
                results['peer_sim_t3'][i] = max_sim
    
    return pd.DataFrame(results)


def process_model(model_suffix: str, industry_info: pd.DataFrame):
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
    
    # Pre-compute matrices
    print("\nPre-computing industry-year matrices...")
    lookup = precompute_industry_matrices(embeddings, embedding_cols)
    
    # Calculate simple peer similarities
    print("\nCalculating simple peer similarities...")
    result_simple = calculate_peer_similarities_vectorized(embeddings, embedding_cols, lookup)
    
    # Sort output
    result_simple = result_simple.sort_values(['stkcd', 'p_year']).reset_index(drop=True)
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
        
        print("Pre-computing citation-weighted matrices...")
        lookup_cit = precompute_industry_matrices(embeddings_cit, embedding_cols_cit)
        
        print("Calculating citation-weighted peer similarities...")
        result_cit = calculate_peer_similarities_vectorized(embeddings_cit, embedding_cols_cit, lookup_cit)
        
        # Rename columns
        result_cit = result_cit.rename(columns={
            'n_peers_t1': 'n_peers_t1_citw',
            'n_peers_t2': 'n_peers_t2_citw',
            'n_peers_t3': 'n_peers_t3_citw',
            'peer_sim_t1': 'peer_sim_t1_citw',
            'peer_sim_t2': 'peer_sim_t2_citw',
            'peer_sim_t3': 'peer_sim_t3_citw',
        })
        
        result_cit = result_cit.sort_values(['stkcd', 'p_year']).reset_index(drop=True)
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
        merged = merged.sort_values(['stkcd', 'p_year']).reset_index(drop=True)
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
            process_model('_minilm', industry_info)
        elif model == 'distiluse':
            process_model('_distiluse', industry_info)
        else:
            print(f"Unknown model: {model}")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)


if __name__ == '__main__':
    main()
