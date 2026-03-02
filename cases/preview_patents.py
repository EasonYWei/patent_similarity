#!/usr/bin/env python3
"""
快速预览 patents_cleaned.dta 文件的前N条记录

用法:
    python preview_patents.py              # 显示前10条
    python preview_patents.py -n 5         # 显示前5条
    python preview_patents.py -n 20 --save # 保存前20条到sample.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).parent.parent / "data" / "patents_cleaned.dta"


def preview(n_rows: int = 10, save_to: str = None):
    """预览数据文件的前N行"""
    
    if not DATA_FILE.exists():
        print(f"错误: 数据文件不存在: {DATA_FILE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"正在读取 {DATA_FILE.name} 的前 {n_rows} 条记录...")
    print()
    
    try:
        # 使用iterator只读取前n_rows行
        reader = pd.read_stata(DATA_FILE, iterator=True, chunksize=n_rows)
        df = next(reader)
        
        # 如果只需要部分行
        if len(df) > n_rows:
            df = df.head(n_rows)
        
        # 显示基本信息
        print("="*100)
        print(f"总列数: {len(df.columns)}")
        print(f"显示行数: {len(df)}")
        print("="*100)
        print()
        
        # 显示列名
        print("列名列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        print()
        
        # 选择关键列显示
        key_cols = ['stkcd', 'p_year', 'p_id', 'p_tt', 'p_abs', 'p_type', 'p_cite']
        available_cols = [c for c in key_cols if c in df.columns]
        
        print("="*100)
        print("数据预览（关键列）:")
        print("="*100)
        
        for idx, row in df.iterrows():
            print(f"\n--- 记录 {idx + 1} ---")
            for col in available_cols:
                value = row.get(col, 'N/A')
                # 截断长文本
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {col}: {value}")
        
        # 保存选项
        if save_to:
            df.to_csv(save_to, index=False, encoding='utf-8-sig')
            print(f"\n✓ 已保存到: {save_to}")
        
        return df
        
    except Exception as e:
        print(f"读取数据时出错: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='快速预览 patents_cleaned.dta 文件的前N条记录'
    )
    parser.add_argument('-n', '--rows', type=int, default=10,
                        help='要显示的行数 (默认: 10)')
    parser.add_argument('--save', metavar='FILE',
                        help='将预览数据保存到CSV文件')
    
    args = parser.parse_args()
    
    preview(n_rows=args.rows, save_to=args.save)
    return 0


if __name__ == '__main__':
    sys.exit(main())
