#!/usr/bin/env python3
"""
从 patents_cleaned.dta 中提取指定公司、指定年份范围的专利文本

用法:
    python extract_patent_texts.py --stkcd 000002 --start-year 2008 --end-year 2010
    python extract_patent_texts.py --stkcd 000012 --year 2021 --output company12_2021.csv
    python extract_patent_texts.py --stkcd 000518 --start-year 2008 --end-year 2013 --format markdown

注意:
    数据文件约2GB，扫描时间约1-3分钟。
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# 数据文件路径
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


def extract_patents(stkcd: str, start_year: int = None, end_year: int = None, 
                    specific_year: int = None, output_file: str = None,
                    output_format: str = 'csv', chunk_size: int = 100000):
    """
    从Stata文件中提取指定条件的专利数据
    
    Args:
        stkcd: 股票代码（如 '2', '000002', '000012' 等）
        start_year: 起始年份（可选）
        end_year: 结束年份（可选）
        specific_year: 指定年份（可选，优先级高于start/end year）
        output_file: 输出文件路径（可选）
        output_format: 输出格式 ('csv' 或 'markdown')
        chunk_size: 分块读取大小
    """
    
    if not DATA_FILE.exists():
        print(f"错误: 数据文件不存在: {DATA_FILE}", file=sys.stderr)
        print("请确保数据文件位于: data/patents_cleaned.dta", file=sys.stderr)
        sys.exit(1)
    
    # 确定年份范围
    if specific_year is not None:
        year_filter = lambda y: y == specific_year
        year_desc = f"{specific_year}年"
    elif start_year is not None and end_year is not None:
        year_filter = lambda y: start_year <= y <= end_year
        year_desc = f"{start_year}-{end_year}年"
    elif start_year is not None:
        year_filter = lambda y: y >= start_year
        year_desc = f"{start_year}年及以后"
    elif end_year is not None:
        year_filter = lambda y: y <= end_year
        year_desc = f"{end_year}年及以前"
    else:
        year_filter = lambda y: True
        year_desc = "所有年份"
    
    print(f"正在提取公司 {stkcd} 在 {year_desc} 的专利数据...")
    print(f"数据文件: {DATA_FILE} (约2GB)")
    print("提示: 扫描数据文件可能需要1-3分钟，请耐心等待...")
    print()
    
    start_time = time.time()
    
    filtered_chunks = []
    total_rows = 0
    match_count = 0
    
    try:
        # 使用iterator分块读取大文件
        reader = pd.read_stata(DATA_FILE, iterator=True, chunksize=chunk_size)
        
        for i, chunk in enumerate(reader):
            total_rows += len(chunk)
            
            # 过滤当前块
            # 股票代码可能是字符串或数字，统一转为字符串处理
            chunk[COL_STKCD] = chunk[COL_STKCD].astype(str)
            stkcd_str = str(stkcd)
            
            # 精确匹配
            stkcd_match = chunk[COL_STKCD] == stkcd_str
            
            # 也尝试去除前导零的匹配（如 '000002' vs '2'）
            if not stkcd_match.any() and stkcd_str.lstrip('0'):
                stkcd_match = chunk[COL_STKCD].str.lstrip('0') == stkcd_str.lstrip('0')
            
            # 尝试添加前导零的匹配（如 '2' vs '000002'）
            if not stkcd_match.any() and stkcd_str.isdigit():
                stkcd_padded = stkcd_str.zfill(6)
                stkcd_match = chunk[COL_STKCD] == stkcd_padded
            
            year_match = chunk[COL_YEAR].apply(year_filter)
            
            mask = stkcd_match & year_match
            filtered = chunk[mask].copy()
            
            if len(filtered) > 0:
                filtered_chunks.append(filtered)
                match_count += len(filtered)
            
            # 每处理10个块显示一次进度
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = total_rows / elapsed if elapsed > 0 else 0
                print(f"  已处理 {total_rows:,} 行，找到 {match_count} 条匹配记录... "
                      f"({speed:,.0f} 行/秒)", end='\r')
        
        elapsed = time.time() - start_time
        print(f"\n  扫描完成！共处理 {total_rows:,} 行，耗时 {elapsed:.1f} 秒")
        
    except Exception as e:
        print(f"\n读取数据时出错: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not filtered_chunks:
        print(f"\n未找到公司 {stkcd} 在 {year_desc} 的专利数据")
        return None
    
    # 合并所有匹配的数据
    result = pd.concat(filtered_chunks, ignore_index=True)
    result = result.sort_values([COL_YEAR, COL_DATE])
    
    total_time = time.time() - start_time
    print(f"\n✓ 提取完成！共找到 {len(result)} 条专利记录 (用时 {total_time:.1f} 秒)")
    print(f"  年份分布: {result[COL_YEAR].min()} - {result[COL_YEAR].max()}")
    
    # 输出结果
    if output_file:
        if output_format == 'markdown':
            save_as_markdown(result, output_file, stkcd, year_desc)
        else:
            result.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 结果已保存到: {output_file}")
    else:
        # 默认输出到控制台（仅显示部分列）
        display_cols = [COL_PID, COL_YEAR, COL_TITLE, COL_ABSTRACT, COL_TYPE, COL_CITE]
        available_cols = [c for c in display_cols if c in result.columns]
        print("\n" + "="*80)
        print("专利列表预览（前5条）：")
        print("="*80)
        print(result[available_cols].head().to_string(index=False))
        
        if len(result) > 5:
            print(f"\n... 共 {len(result)} 条记录，使用 --output 参数保存完整结果")
    
    return result


def save_as_markdown(df: pd.DataFrame, output_file: str, stkcd: str, year_desc: str):
    """将结果保存为Markdown格式"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 公司 {stkcd} 专利文本提取报告\n\n")
        f.write(f"**提取范围**: {year_desc}  \n")
        f.write(f"**专利总数**: {len(df)} 件  \n")
        f.write(f"**数据日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("---\n\n")
        
        # 按年份分组输出
        for year in sorted(df[COL_YEAR].unique()):
            year_df = df[df[COL_YEAR] == year]
            f.write(f"## {year}年 ({len(year_df)}件专利)\n\n")
            
            for idx, row in year_df.iterrows():
                patent_id = row.get(COL_PID, 'N/A')
                title = row.get(COL_TITLE, 'N/A')
                abstract = row.get(COL_ABSTRACT, 'N/A')
                patent_type = row.get(COL_TYPE, 'N/A')
                citations = row.get(COL_CITE, 'N/A')
                apply_date = row.get(COL_DATE, 'N/A')
                ipc = row.get(COL_IPC, 'N/A')
                
                f.write(f"### 专利 {patent_id}\n\n")
                f.write(f"- **申请日**: {apply_date}  \n")
                f.write(f"- **专利类型**: {patent_type}  \n")
                f.write(f"- **被引证次数**: {citations}  \n")
                f.write(f"- **IPC分类**: {ipc}\n\n")
                f.write(f"**标题**: {title}\n\n")
                f.write(f"**摘要**: {abstract}\n\n")
                f.write("---\n\n")
    
    print(f"\n✓ Markdown报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='从 patents_cleaned.dta 中提取指定公司、指定年份范围的专利文本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取公司000002在2010年的专利
  python extract_patent_texts.py --stkcd 000002 --year 2010
  
  # 提取公司000012在2020-2021年的专利，保存为CSV
  python extract_patent_texts.py --stkcd 000012 --start-year 2020 --end-year 2021 -o output.csv
  
  # 提取公司000518在2008-2013年的专利，保存为Markdown报告
  python extract_patent_texts.py --stkcd 000518 --start-year 2008 --end-year 2013 -f markdown -o report.md
  
  # 使用简写股票代码（自动补零匹配）
  python extract_patent_texts.py --stkcd 2 --year 2010
        """
    )
    
    parser.add_argument('--stkcd', '-s', required=True, 
                        help='股票代码（如 000002, 2 等）')
    parser.add_argument('--year', '-y', type=int, 
                        help='指定年份（优先级高于start-year/end-year）')
    parser.add_argument('--start-year', type=int, 
                        help='起始年份')
    parser.add_argument('--end-year', type=int, 
                        help='结束年份')
    parser.add_argument('--output', '-o', 
                        help='输出文件路径（不指定则输出到控制台）')
    parser.add_argument('--format', '-f', choices=['csv', 'markdown'], default='csv',
                        help='输出格式（默认: csv）')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='分块读取大小（默认: 100000）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.year is None and args.start_year is None and args.end_year is None:
        print("警告: 未指定年份范围，将提取该公司的所有专利数据", file=sys.stderr)
    
    # 执行提取
    result = extract_patents(
        stkcd=args.stkcd,
        start_year=args.start_year,
        end_year=args.end_year,
        specific_year=args.year,
        output_file=args.output,
        output_format=args.format,
        chunk_size=args.chunk_size
    )
    
    if result is not None and len(result) > 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
