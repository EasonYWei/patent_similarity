#!/usr/bin/env python3
"""
Extract a sample from patents.dta with city fields for testing.
This creates a small dataset with city information to test city-level analysis.
"""

import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "patents.dta"
    output_file = project_root / "data" / "patents_cleaned_with_city.dta"
    
    print(f"Reading {input_file}...")
    print("This may take a while as the file is ~27GB...")
    
    # Read only necessary columns
    columns = [
        '股票代码', 'newipzlid', '年份', '标题', '摘要', '申请日', 
        '专利类型', 'IPC', '被引证次数',
        '市', '市代码', '省', '省代码'
    ]
    
    try:
        df = pd.read_stata(input_file, columns=columns, convert_categoricals=False)
        print(f"Loaded {len(df)} rows")
        
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
        df = df.rename(columns=column_mapping)
        
        # Filter by patent type (same as pre.do)
        patent_types = ['发明申请', '发明授权', '实用新型']
        df = df[df['p_type'].isin(patent_types)]
        print(f"After patent type filter: {len(df)} rows")
        
        # Filter by stock code prefix (0, 3, or 6)
        df['stkcd_str'] = df['stkcd'].astype(str).str.strip()
        df = df[df['stkcd_str'].str[0].isin(['0', '3', '6'])]
        df = df.drop(columns=['stkcd_str'])
        print(f"After stock code filter: {len(df)} rows")
        
        # Fill missing citations with 0
        df['p_cite'] = df['p_cite'].fillna(0)
        
        # Remove rows with missing city_code or year
        df = df.dropna(subset=['city_code', 'p_year'])
        print(f"After removing missing city/year: {len(df)} rows")
        
        # Show city distribution
        print("\nTop 10 cities by patent count:")
        print(df['city'].value_counts().head(10))
        
        print(f"\nUnique cities: {df['city_code'].nunique()}")
        print(f"Unique city-years: {df['city_code'].astype(str) + '_' + df['p_year'].astype(str)}.nunique()")
        
        # Save to new file
        df.to_stata(output_file, write_index=False)
        print(f"\nSaved to {output_file}")
        print(f"Final dataset: {len(df)} rows, {df['city_code'].nunique()} cities")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
