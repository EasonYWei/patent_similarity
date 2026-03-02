#!/usr/bin/env python3
"""
批量提取多个公司的专利数据（从 patents_cleaned.dta）

用法:
    python batch_extract.py --companies 000002,000012,000518 --year 2010
    python batch_extract.py --companies 000002,000012,000518 --start-year 2008 --end-year 2013 -o batch_output/
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).parent.parent / "data" / "patents_cleaned.dta"

# 列名（清理后数据使用英文列名）
COL_STKCD = 'stkcd'
COL_YEAR = 'p_year'
COL_PID = 'p_id'
COL_TITLE = 'p_tt'
COL_ABSTRACT = 'p_abs'
COL_DATE = 'p_date'
COL_TYPE = 'p_type'
COL_IPC = 'p_ipc'
COL_CITE = 'p_cite'


def batch_extract(companies: list, start_year: int = None, end_year: int = None,
                  specific_year: int = None, output_dir: str = None,
                  combine: bool = False):
    """
    批量提取多个公司的专利数据
    
    Args:
        companies: 股票代码列表
        start_year: 起始年份
        end_year: 结束年份
        specific_year: 指定年份
        output_dir: 输出目录
        combine: 是否合并为一个文件
    """
    
    if not DATA_FILE.exists():
        print(f"错误: 数据文件不存在: {DATA_FILE}", file=sys.stderr)
        sys.exit(1)
    
    # 确定年份范围描述
    if specific_year is not None:
        year_desc = f"{specific_year}年"
        year_filter = lambda y: y == specific_year
    elif start_year is not None and end_year is not None:
        year_desc = f"{start_year}-{end_year}年"
        year_filter = lambda y: start_year <= y <= end_year
    elif start_year is not None:
        year_desc = f"{start_year}年及以后"
        year_filter = lambda y: y >= start_year
    elif end_year is not None:
        year_desc = f"{end_year}年及以前"
        year_filter = lambda y: y <= end_year
    else:
        year_desc = "所有年份"
        year_filter = lambda y: True
    
    print(f"批量提取 {len(companies)} 家公司的专利数据 ({year_desc})")
    print(f"目标公司: {', '.join(companies)}")
    print()
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 单次扫描，提取所有公司的数据
    print("开始扫描数据文件...")
    company_set = set(companies)
    
    try:
        reader = pd.read_stata(DATA_FILE, iterator=True, chunksize=100000)
        
        filtered_chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(reader):
            total_rows += len(chunk)
            
            # 转换股票代码为字符串
            chunk[COL_STKCD] = chunk[COL_STKCD].astype(str)
            
            # 过滤：匹配任一公司 + 年份范围
            stkcd_match = chunk[COL_STKCD].isin(company_set)
            
            # 也尝试去除前导零的匹配
            if not stkcd_match.any():
                chunk_codes = chunk[COL_STKCD].str.lstrip('0')
                query_codes = [c.lstrip('0') for c in company_set if c.lstrip('0')]
                stkcd_match = chunk_codes.isin(query_codes)
            
            year_match = chunk[COL_YEAR].apply(year_filter)
            mask = stkcd_match & year_match
            
            filtered = chunk[mask].copy()
            if len(filtered) > 0:
                filtered_chunks.append(filtered)
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {total_rows:,} 行...", end='\r')
        
        print(f"\n扫描完成！共处理 {total_rows:,} 行")
        
        if not filtered_chunks:
            print("\n未找到匹配的专利数据")
            return
        
        # 合并所有数据
        all_data = pd.concat(filtered_chunks, ignore_index=True)
        all_data = all_data.sort_values([COL_STKCD, COL_YEAR, COL_DATE])
        
        print(f"\n✓ 共找到 {len(all_data)} 条专利记录")
        
        # 按公司统计
        for stkcd in companies:
            # 处理前导零的情况
            company_data = all_data[
                (all_data[COL_STKCD] == stkcd) | 
                (all_data[COL_STKCD].str.lstrip('0') == stkcd.lstrip('0'))
            ]
            print(f"  公司 {stkcd}: {len(company_data)} 件专利")
        
        # 输出结果
        if output_dir:
            if combine:
                # 合并输出
                output_file = Path(output_dir) / f"batch_{year_desc.replace(' ', '_')}.csv"
                all_data.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"\n✓ 合并文件已保存: {output_file}")
            else:
                # 分开输出
                for stkcd in companies:
                    company_data = all_data[
                        (all_data[COL_STKCD] == stkcd) | 
                        (all_data[COL_STKCD].str.lstrip('0') == stkcd.lstrip('0'))
                    ]
                    if len(company_data) > 0:
                        output_file = Path(output_dir) / f"company_{stkcd}_{year_desc.replace(' ', '_')}.csv"
                        company_data.to_csv(output_file, index=False, encoding='utf-8-sig')
                        print(f"✓ 公司 {stkcd}: {output_file}")
        else:
            # 输出预览
            print("\n" + "="*80)
            print("数据预览（每家公司的前3条）:")
            print("="*80)
            for stkcd in companies:
                company_data = all_data[
                    (all_data[COL_STKCD] == stkcd) | 
                    (all_data[COL_STKCD].str.lstrip('0') == stkcd.lstrip('0'))
                ]
                if len(company_data) > 0:
                    print(f"\n--- 公司 {stkcd} ---")
                    preview_cols = [COL_PID, COL_YEAR, COL_TITLE, COL_TYPE]
                    available_cols = [c for c in preview_cols if c in company_data.columns]
                    print(company_data[available_cols].head(3).to_string(index=False))
        
        return all_data
        
    except Exception as e:
        print(f"\n处理数据时出错: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='批量提取多个公司的专利数据（从 patents_cleaned.dta）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取多家公司2010年的专利
  python batch_extract.py --companies 000002,000012,000518 --year 2010
  
  # 提取多家公司2008-2013年的专利，分别保存
  python batch_extract.py --companies 000002,000012,000518 --start-year 2008 --end-year 2013 -o output/
  
  # 提取并合并为一个文件
  python batch_extract.py --companies 000002,000012,000518 --year 2021 --combine -o output/
  
  # 使用简写股票代码
  python batch_extract.py --companies 2,12,518 --year 2010 -o output/
        """
    )
    
    parser.add_argument('--companies', '-c', required=True,
                        help='股票代码列表，用逗号分隔（如 000002,000012,000518）')
    parser.add_argument('--year', '-y', type=int,
                        help='指定年份')
    parser.add_argument('--start-year', type=int,
                        help='起始年份')
    parser.add_argument('--end-year', type=int,
                        help='结束年份')
    parser.add_argument('--output', '-o',
                        help='输出目录')
    parser.add_argument('--combine', action='store_true',
                        help='将所有公司数据合并为一个文件')
    
    args = parser.parse_args()
    
    # 解析公司列表
    companies = [c.strip() for c in args.companies.split(',')]
    
    # 执行批量提取
    result = batch_extract(
        companies=companies,
        start_year=args.start_year,
        end_year=args.end_year,
        specific_year=args.year,
        output_dir=args.output,
        combine=args.combine
    )
    
    return 0 if result is not None else 1


if __name__ == '__main__':
    sys.exit(main())
