#!/usr/bin/env python3
"""
Extract a sample from patents.dta with city fields for testing.
Uses chunked reading to handle large files.
"""

import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "patents.dta"
    output_file = project_root / "data" / "patents_cleaned_with_city.dta"
    
    print(f"Reading {input_file} in chunks...")
    
    # Read only necessary columns
    columns = [
        '股票代码', 'newipzlid', '年份', '标题', '摘要', '申请日', 
        '专利类型', 'IPC', '被引证次数',
        '市', '市代码', '省', '省代码'
    ]
    
    chunks = []
    chunk_size = 100000
    target_rows = 500000  # Target ~500K rows for testing
    
    try:
        # Read first few chunks
        for i, chunk in enumerate(pd.read_stata(
            input_file, 
            columns=columns, 
            convert_categoricals=False,
            chunksize=chunk_size
        )):
            print(f"Processing chunk {i+1}, rows so far: {len(chunks) * chunk_size + len(chunk)}")
            
            # Rename columns
            column_mapping = {
                '股票代码': 'stkcd',
                'newipzlid': 'p_id',
                '年份': 'p_year',
                '标题': 'p_tt',
                '摘要': 'p_abs',
                '申请日': 'p_date',
                '专利类型': 'p_type',
                'IPC': 'p_ipc',
                '被引证次数': 'p_cite',
                '市': 'city',
                '市代码': 'city_code',
                '省': 'province',
                '省代码': 'province_code'
            }
            chunk = chunk.rename(columns=column_mapping)
            
            # Filter by patent type
            patent_types = ['发明申请', '发明授权', '实用新型']
            chunk = chunk[chunk['p_type'].isin(patent_types)]
            
            # Filter by stock code prefix
            chunk['stkcd_str'] = chunk['stkcd'].astype(str).str.strip()
            chunk = chunk[chunk['stkcd_str'].str[0].isin(['0', '3', '6'])]
            chunk = chunk.drop(columns=['stkcd_str'])
            
            # Fill missing citations
            chunk['p_cite'] = chunk['p_cite'].fillna(0)
            
            # Remove rows with missing city_code or year
            chunk = chunk.dropna(subset=['city_code', 'p_year'])
            
            chunks.append(chunk)
            
            # Stop after reaching target
            total_rows = sum(len(c) for c in chunks)
            if total_rows >= target_rows:
                print(f"Reached target of {target_rows} rows")
                break
            
            if i >= 9:  # Max 10 chunks (~1M raw rows)
                break
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        print(f"\nCombined dataset: {len(df)} rows")
        
        # Show city distribution
        print("\nTop 10 cities by patent count:")
        print(df['city'].value_counts().head(10))
        
        print(f"\nUnique cities: {df['city_code'].nunique()}")
        df['city_year'] = df['city_code'].astype(str) + '_' + df['p_year'].astype(str)
        print(f"Unique city-years: {df['city_year'].nunique()}")
        df = df.drop(columns=['city_year'])
        
        # Save to new file
        print(f"\nSaving to {output_file}...")
        df.to_stata(output_file, write_index=False)
        print(f"Saved successfully!")
        print(f"Final dataset: {len(df)} rows, {df['city_code'].nunique()} cities")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
