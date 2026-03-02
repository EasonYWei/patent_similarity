#!/usr/bin/env python3
"""
分析专利相似度数据，找出公司技术转型的典型案例。
转型特征：cos_sim_lag1（与前一年的相似度）显著下降
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 读取数据
df_mini = pd.read_csv("output/stkcd_year_similarity_merged_minilm.csv")
df_dist = pd.read_csv("output/stkcd_year_similarity_merged_distiluse.csv")

# 过滤有效数据（有lag1相似度的）
df_mini_valid = df_mini[df_mini['cos_sim_lag1'].notna()].copy()
df_dist_valid = df_dist[df_dist['cos_sim_lag1'].notna()].copy()

print(f"MiniLM有效记录数: {len(df_mini_valid)}")
print(f"DistilUSE有效记录数: {len(df_dist_valid)}")

# 计算相似度下降幅度
df_mini_valid['sim_drop'] = 1 - df_mini_valid['cos_sim_lag1']
df_dist_valid['sim_drop'] = 1 - df_dist_valid['cos_sim_lag1']

# 筛选条件：
# 1. 有一定专利数量的公司（至少5个专利）
# 2. 有连续两年数据
# 3. 相似度下降幅度较大

min_patents = 5

# MiniLM模型结果
df_mini_filtered = df_mini_valid[df_mini_valid['n_patents'] >= min_patents].copy()
df_mini_filtered = df_mini_filtered.sort_values(['stkcd', 'p_year'])

# DistilUSE模型结果  
df_dist_filtered = df_dist_valid[df_dist_valid['n_patents'] >= min_patents].copy()
df_dist_filtered = df_dist_filtered.sort_values(['stkcd', 'p_year'])

print("\n=== MiniLM模型统计 ===")
print(f"平均相似度: {df_mini_filtered['cos_sim_lag1'].mean():.4f}")
print(f"相似度中位数: {df_mini_filtered['cos_sim_lag1'].median():.4f}")
print(f"相似度标准差: {df_mini_filtered['cos_sim_lag1'].std():.4f}")

print("\n=== DistilUSE模型统计 ===")
print(f"平均相似度: {df_dist_filtered['cos_sim_lag1'].mean():.4f}")
print(f"相似度中位数: {df_dist_filtered['cos_sim_lag1'].median():.4f}")
print(f"相似度标准差: {df_dist_filtered['cos_sim_lag1'].std():.4f}")

# 找出转型案例（相似度显著下降）
# 定义转型：相似度低于0.5（即下降超过0.5）
transform_threshold = 0.5

print(f"\n=== 寻找转型案例（cos_sim_lag1 < {transform_threshold}）===")

# MiniLM转型案例
transform_mini = df_mini_filtered[df_mini_filtered['cos_sim_lag1'] < transform_threshold].copy()
transform_mini = transform_mini.sort_values('cos_sim_lag1')
print(f"\nMiniLM转型案例数: {len(transform_mini)}")

# DistilUSE转型案例
transform_dist = df_dist_filtered[df_dist_filtered['cos_sim_lag1'] < transform_threshold].copy()
transform_dist = transform_dist.sort_values('cos_sim_lag1')
print(f"DistilUSE转型案例数: {len(transform_dist)}")

# 找出在两个模型中都显示转型的案例（交叉验证）
transform_mini_keys = set(zip(transform_mini['stkcd'], transform_mini['p_year']))
transform_dist_keys = set(zip(transform_dist['stkcd'], transform_dist['p_year']))
common_transforms = transform_mini_keys & transform_dist_keys

print(f"\n两个模型共同识别的转型案例数: {len(common_transforms)}")

# 输出典型案例
print("\n=== 典型案例（两个模型都显示低相似度）===")
common_cases = []
for stkcd, year in sorted(common_transforms)[:20]:
    mini_row = df_mini[(df_mini['stkcd'] == stkcd) & (df_mini['p_year'] == year)].iloc[0]
    dist_row = df_dist[(df_dist['stkcd'] == stkcd) & (df_dist['p_year'] == year)].iloc[0]
    print(f"\n公司 {stkcd}, 年份 {year}:")
    print(f"  专利数: {mini_row['n_patents']}")
    print(f"  MiniLM相似度: {mini_row['cos_sim_lag1']:.4f}")
    print(f"  DistilUSE相似度: {dist_row['cos_sim_lag1']:.4f}")
    common_cases.append({
        'stkcd': stkcd,
        'year': year,
        'n_patents': mini_row['n_patents'],
        'minilm_sim': mini_row['cos_sim_lag1'],
        'distiluse_sim': dist_row['cos_sim_lag1']
    })

# 寻找公司轨迹（连续多年数据，有转型特征）
print("\n=== 寻找公司转型轨迹 ===")

def find_company_trajectory(df, model_name):
    """找出一个公司连续多年数据中有转型特征的案例"""
    trajectories = []
    
    for stkcd in df['stkcd'].unique()[:100]:  # 检查前100个公司
        company_data = df[df['stkcd'] == stkcd].sort_values('p_year')
        if len(company_data) >= 4:  # 至少4年数据
            sim_values = company_data['cos_sim_lag1'].tolist()
            years = company_data['p_year'].tolist()
            patents = company_data['n_patents'].tolist()
            
            # 检查是否有明显的下降后回升或持续低相似度
            low_sim_years = [(years[i], sim_values[i], patents[i]) 
                            for i in range(len(sim_values)) 
                            if sim_values[i] < 0.5]
            
            if len(low_sim_years) >= 1:
                trajectories.append({
                    'stkcd': stkcd,
                    'years': years,
                    'similarities': sim_values,
                    'patents': patents,
                    'low_sim_years': low_sim_years
                })
    
    return trajectories

traj_mini = find_company_trajectory(df_mini_filtered, "MiniLM")
print(f"\n找到 {len(traj_mini)} 个有转型轨迹的公司 (MiniLM)")

# 选择几个典型案例
case_stocks = []
for traj in traj_mini[:10]:
    stkcd = traj['stkcd']
    print(f"\n公司 {stkcd}:")
    print(f"  年份: {traj['years']}")
    print(f"  相似度: {[round(s, 3) for s in traj['similarities']]}")
    print(f"  专利数: {traj['patents']}")
    print(f"  低相似度年份: {traj['low_sim_years']}")
    case_stocks.append(stkcd)

# 保存案例数据
print("\n=== 保存案例数据 ===")

# 保存转型案例列表
pd.DataFrame(common_cases).to_csv("cases/transformation_cases.csv", index=False)

# 保存公司完整轨迹
for stkcd in case_stocks[:5]:
    company_mini = df_mini[df_mini['stkcd'] == stkcd].sort_values('p_year')
    company_dist = df_dist[df_dist['stkcd'] == stkcd].sort_values('p_year')
    
    # 合并两个模型的数据
    merged = company_mini.merge(
        company_dist[['stkcd', 'p_year', 'cos_sim_lag1']], 
        on=['stkcd', 'p_year'], 
        suffixes=('_minilm', '_distiluse')
    )
    merged.to_csv(f"cases/company_{stkcd}_trajectory.csv", index=False)
    print(f"保存公司 {stkcd} 轨迹数据")

print("\n案例数据已保存到 cases/ 文件夹")
