# Patent Similarity Calculation Project / 专利相似度计算项目

基于 Sentence-BERT 的专利文本相似度计算工具，用于分析企业技术转型。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行嵌入计算
python scripts/patents_embeddings.py

# 运行相似度计算
Rscript scripts/patents_similarity.R
```

## 数据文件

本项目使用专利数据文件进行分析。由于数据文件较大（~2GB），请从以下位置下载：

### 📥 下载地址

- **GitHub Releases**: https://github.com/EasonYWei/patent_similarity/releases
  - 下载 `patents_cleaned.dta`（清理后的专利数据，约 1.9GB）

### 数据说明

下载后请将数据文件放置在 `data/` 目录下：

```
data/
└── patents_cleaned.dta   # 清理后的专利数据 (~2GB)
```

### 数据结构

| 字段 | 说明 |
|------|------|
| `stkcd` | 股票代码 |
| `p_year` | 专利年份 |
| `p_tt` | 专利标题 |
| `p_abs` | 专利摘要 |
| `p_cite` | 被引证次数 |
| `p_type` | 专利类型 |
| `p_date` | 申请日 |
| `p_ipc` | IPC 分类 |

## 项目结构

```
├── scripts/           # 主流程脚本
│   ├── patents_embeddings.py    # 嵌入计算
│   ├── patents_similarity.R     # 相似度计算
│   └── pre.do                   # 数据预处理 (Stata)
├── sample/            # 示例数据和调试脚本
├── cases/             # 技术转型案例分析
├── models/            # SBERT 模型目录
├── data/              # 数据文件（需单独下载）
└── output/            # 输出目录
```

## 技术栈

- Python 3.x + PyTorch + Sentence-Transformers
- R (相似度计算)
- Stata (数据预处理)

## 详细文档

- [AGENTS.md](./AGENTS.md) - 项目详细说明和开发规范
- [sample/README.md](./sample/README.md) - 示例数据使用说明
- [cases/README.md](./cases/README.md) - 案例分析报告（中文）

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{patent_similarity_2025,
  title = {Patent Similarity Calculation Project},
  author = {Eason Wei},
  year = {2025},
  url = {https://github.com/EasonYWei/patent_similarity}
}
```

## License

MIT License
