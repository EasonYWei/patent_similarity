# Sample Data for Patent Embeddings Inspection
# 专利嵌入检查样本数据

This folder contains sample data and scripts for manually inspecting and understanding the patent embeddings and aggregation pipeline. It is separated from the main pipeline to avoid confusion.

本文件夹包含用于手动检查和理解专利嵌入及聚合流程的样本数据和脚本。与主流程分离以避免混淆。

## Folder Structure / 文件夹结构

```
sample/
├── README.md                          # This file
├── data/                              # Sample data files
│   ├── sample_patents_raw.csv         # Raw patent text data (4,938 patents)
│   ├── sample_patents_raw.pkl         # Same data in pickle format (faster loading)
│   ├── sample_firm_year_summary.csv   # Firm-year summary statistics
│   ├── sample_minilm_embeddings.csv   # Extracted from main output
│   └── sample_citweighted_minilm_embeddings.csv  # Citation-weighted embeddings
├── scripts/                           # Inspection scripts
│   ├── ps_self.R                      # 专利级别自相似度计算 (Patent-level self-similarity)
│   ├── extract_sample_patents.py      # Extract sample from main data
│   ├── inspect_embeddings.py          # Main inspection script (Python)
│   ├── create_sample_embeddings.R     # Create sample from main embeddings
│   ├── calculate_sample_similarity.R  # Calculate similarity for sample
│   └── minimal_similarity_demo.R      # Detailed demo with step-by-step output
└── output/                            # Inspection outputs (generated)
    ├── sample_similarity_minilm.csv
    ├── sample_similarity_citweighted_minilm.csv
    └── sample_similarity_merged_minilm.csv
```

## Sample Companies / 样本公司

| stkcd  | Patents | Years | Type |
|--------|---------|-------|------|
| 600808 | 4,820   | 40 (1985-2024) | Many years - for testing lag-3 and cumulative |
| 000002 | 110     | 10 (2002-2014) | Medium years - normal case |
| 000061 | 6       | 3 (2009-2012)  | Few years - lag-3 should be NA |
| 000004 | 2       | 1 (2002)       | Single year - all similarities NA |

## Patent-Level Self-Similarity / 专利级别自相似度

### ps_self.R - 计算专利级别自相似度

该脚本计算每家公司的每个专利与其之前所有专利的余弦相似度，输出三个变量：
- **sim_max**: 最大余弦相似度
- **sim_max_d**: 与最相似专利的日期间隔（天）
- **sim_ave**: 平均余弦相似度

```bash
cd sample/scripts
Rscript ps_self.R
```

**输入文件**:
- `output/patent_level_paraphrase_inspection.csv` - 专利元数据及384维embeddings
- `output/patent_level_paraphrase_embeddings.npy` - 专利embeddings (npy格式，可选)

**输出文件**:
- `output/patent_self_similarity.csv` - 包含 sim_max, sim_max_d, sim_ave 的结果

**示例输出**:
```
stkcd,p_year,p_id,date,sim_max,sim_max_d,sim_ave
2,2002,2002071026,2002-01-01,,,
2,2002,2002122379,2002-01-01,0.9550683,0,0.9550683
2,2004,2004089389,2004-01-01,0.8007952,730,0.7841423
```

**注意**: 每家公司的第一条专利没有相似度计算结果（因为之前没有专利），所以 sim_max/sim_max_d/sim_ave 为 NA。

---

## Workflow / 工作流程

### Step 1: Extract Raw Patent Data / 提取原始专利数据

Extract raw patent text for sample companies from the main data:

```bash
cd sample/scripts
python extract_sample_patents.py
```

Output: `sample/data/sample_patents_raw.csv` and `sample_patents_raw.pkl`

### Step 2: Create Sample Embeddings / 创建样本嵌入

Extract firm-year embeddings for sample companies from main output:

```bash
cd sample/scripts
Rscript create_sample_embeddings.R
```

Prerequisite: Main pipeline must have generated `output/stkcd_year_minilm_embeddings.csv`

Output: `sample/data/sample_minilm_embeddings.csv`

### Step 3: Calculate Similarities / 计算相似度

Run similarity calculation on sample data:

```bash
cd sample/scripts
Rscript calculate_sample_similarity.R
```

Output: `sample/output/sample_similarity_*.csv`

到这一步可以用 Excel 结合 `sample/data/sample_minilm_embeddings.csv` 手动进行检验

### Step 4: Detailed Inspection (Optional) / 详细检查（可选）

For detailed step-by-step similarity calculation:

```bash
cd sample/scripts
Rscript minimal_similarity_demo.R
```

For Python-based embedding inspection:

```bash
cd sample/scripts
python inspect_embeddings.py --step all --company 000002
```

## Usage Examples / 使用示例

### Quick Start / 快速开始

```bash
cd sample/scripts

# 1. Extract sample from main embeddings
Rscript create_sample_embeddings.R

# 2. Calculate similarities
Rscript calculate_sample_similarity.R

# 3. View results
cat ../output/sample_similarity_merged_minilm.csv | head -20
```

### Python Inspection / Python检查

```bash
cd sample/scripts

# Load and view data only
python inspect_embeddings.py --step load

# Compute embeddings (requires model)
python inspect_embeddings.py --step embed --batch-size 32

# Inspect aggregation
python inspect_embeddings.py --step aggregate

# Focus on one company
python inspect_embeddings.py --company 000002 --step all
```

### R Demo / R演示

```bash
cd sample/scripts

# Detailed similarity calculation demo
Rscript minimal_similarity_demo.R
```

## Output Files / 输出文件

### Data Files / 数据文件

- `sample_patents_raw.csv`: Raw patent texts with metadata
- `sample_minilm_embeddings.csv`: Firm-year embeddings (from main output)
- `sample_citweighted_minilm_embeddings.csv`: Citation-weighted embeddings

### Similarity Results / 相似度结果

- `sample_similarity_minilm.csv`: Simple mean similarity results
- `sample_similarity_citweighted_minilm.csv`: Citation-weighted similarity results
- `sample_similarity_merged_minilm.csv`: Combined results

## Comparing with Main Pipeline / 与主流程对比

The main pipeline (`scripts/patents_embeddings.py`) processes all ~2.3M patents and outputs:
- `output/stkcd_year_{model}_embeddings.csv` (simple mean)
- `output/stkcd_year_citweighted_{model}_embeddings.csv` (citation-weighted)

You can compare the firm-year results from this sample inspection with the main pipeline output to verify correctness.

Example:
```python
import pandas as pd

# From main pipeline
main_result = pd.read_csv("output/stkcd_year_minilm_embeddings.csv")

# Filter for sample company
sample_from_main = main_result[main_result["stkcd"] == "600808"]
print(sample_from_main[["stkcd_year", "n_patents", "emb_0", "emb_1", "emb_2"]])
```

## Troubleshooting / 故障排除

**Sample file not found**
- Run `create_sample_embeddings.R` first to extract from main output

**ImportError: No module named 'patents_embeddings'**
- Make sure you're running from `sample/scripts/` directory

**Model not found**
- Check that models are in `../models/` (relative to project root)

**Out of memory**
- Reduce `--batch-size` (default: 32, try 16 or 8)
- Use CPU with `--device cpu`

## Notes / 注意事项

1. This sample is for **inspection and debugging only**, not for production use
2. The sample companies were selected to cover different year spans for testing edge cases
3. All embeddings should match between this inspection script and the main pipeline
4. The citation-weighted aggregation uses the same logic as the main pipeline
