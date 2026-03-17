# Patent Similarity Calculation Project / 专利相似度计算项目

## Project Overview

This project computes semantic embeddings for patent data using Sentence-BERT (SBERT) models and aggregates them by firm-year for similarity analysis. It processes Chinese (and potentially multilingual) patent text to generate vector representations suitable for measuring patent similarity, innovation analysis, and research in technological change.

The pipeline reads patent data (titles and abstracts), computes dense vector embeddings using pre-trained multilingual SBERT models, and aggregates these embeddings at the firm-year level with both simple mean and citation-weighted approaches. Additionally, the project includes tools for identifying and analyzing firm technology transformation cases.

## Technology Stack

- **Language**: Python 3.x, R, Stata
- **Deep Learning**: PyTorch >= 2.0.0, Transformers >= 4.30.0
- **Embedding Models**: sentence-transformers >= 2.2.0 (SBERT)
- **Data Processing**: pandas >= 2.0.0, numpy >= 1.24.0
- **Progress Tracking**: tqdm >= 4.65.0
- **Input Data Format**: Stata (.dta) files

## Project Structure

```
.
├── scripts/                       # Main pipeline scripts
│   ├── patents_embeddings.py      # Main embedding pipeline (Python)
│   ├── patents_similarity.R       # Similarity calculation (R)
│   └── pre.do                     # Stata data preprocessing script
├── sample/                        # Sample data for testing/debugging
│   ├── data/                      # Sample datasets
│   │   ├── sample_patents_raw.csv         # Raw patent texts (~5K patents)
│   │   ├── sample_patents_raw.pkl         # Same data in pickle format
│   │   ├── sample_minilm_embeddings.csv   # Firm-year embeddings sample
│   │   ├── sample_citweighted_minilm_embeddings.csv  # Citation-weighted
│   │   └── sample_firm_year_summary.csv   # Summary statistics
│   ├── scripts/                   # Sample inspection scripts
│   │   ├── extract_sample_patents.py      # Extract from main data
│   │   ├── inspect_embeddings.py          # Detailed inspection (Python)
│   │   ├── create_sample_embeddings.R     # Create sample embeddings
│   │   ├── calculate_sample_similarity.R  # Sample similarity calc
│   │   ├── minimal_similarity_demo.R      # Step-by-step demo
│   │   └── ps_self.R                      # Patent-level self-similarity
│   ├── output/                    # Sample outputs (generated)
│   └── README.md                  # Sample folder documentation
├── cases/                         # Technology transformation case analysis
│   ├── README.md                  # Case study documentation (Chinese)
│   ├── transformation_cases.csv   # 247 identified transformation cases
│   ├── find_transformation_cases.py     # Identify transformation cases
│   ├── extract_patent_texts.py    # Extract patent texts for analysis
│   ├── batch_extract.py           # Batch extraction tool
│   ├── preview_patents.py         # Preview data structure
│   ├── company_2_trajectory.csv   # Case: Company 2 trajectory
│   ├── company_12_trajectory.csv  # Case: Company 12 trajectory
│   ├── company_423_trajectory.csv # Case: Company 423 trajectory
│   ├── company_518_trajectory.csv # Case: Company 518 trajectory
│   ├── company_538_trajectory.csv # Case: Company 538 trajectory
│   └── details/                   # Detailed patent texts for cases
├── models/                        # Pre-trained SBERT models (local storage)
│   ├── paraphrase-multilingual-MiniLM-L12-v2/  # 384-dim embeddings
│   └── distiluse-base-multilingual-cased-v2/   # 512-dim embeddings
├── data/                          # Input data directory
│   ├── patents.dta               # Raw patent data (~27GB)
│   └── patents_cleaned.dta       # Cleaned input data (~2GB)
├── output/                        # Generated outputs (created at runtime)
│   ├── stkcd_year_{model}_embeddings.csv              # Simple mean embeddings
│   ├── stkcd_year_citweighted_{model}_embeddings.csv  # Citation-weighted embeddings
│   ├── stkcd_year_similarity_{model}.csv              # Similarity results
│   ├── patent_level_{model}_meta.csv                  # Patent-level metadata (optional)
│   └── patent_level_{model}_embeddings.npy            # Patent-level embeddings (optional)
├── requirements.txt               # Python dependencies
└── presentation.tex               # LaTeX presentation

Note: {model} is the model short name (e.g., "minilm" for paraphrase-multilingual-MiniLM-L12-v2,
"distiluse" for distiluse-base-multilingual-cased-v2).
```

## Data Schema

### Input Data (patents_cleaned.dta)

Required columns:
- `stkcd` (string): Company stock code / 股票代码
- `p_year` (integer): Patent year / 年份
- `p_tt` (string): Patent title / 标题
- `p_abs` (string): Patent abstract / 摘要

Optional columns:
- `p_id` (string): Patent ID / newipzlid
- `p_cite` (numeric): Citation count for weighting / 被引证次数
- `p_date` (date): Application date / 申请日
- `p_type` (string): Patent type / 专利类型
- `p_ipc` (string): IPC classification / IPC

### Output Data

**Firm-Year Embeddings** (`stkcd_year_{model}_embeddings.csv`):
- `stkcd`: Company stock code
- `p_year`: Year
- `stkcd_year`: Composite key (stkcd_p_year)
- `n_patents`: Number of patents in group
- `n_texts_used`: Number of patents with non-empty text
- `total_citations`: Sum of citations
- `mean_citations`: Average citations per patent
- `emb_0` to `emb_N`: Embedding vector components (384 or 512 dimensions)

## Build and Run Commands

### Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing (Stata)

```stata
# Run the Stata preprocessing script to clean raw data
do scripts/pre.do

# This will:
# 1. Load patents.dta and select required columns
# 2. Rename Chinese columns to English abbreviations
# 3. Filter by patent type (invention applications, granted inventions, utility models)
# 4. Filter by stock code prefix (0, 3, or 6)
# 5. Handle missing citations (fill with 0)
# 6. Save to data/patents_cleaned.dta
```

### Running the Pipeline

```bash
# Basic usage (uses default settings)
python scripts/patents_embeddings.py

# With specific model and options
python scripts/patents_embeddings.py \
    --input data/patents_cleaned.dta \
    --model-dir models \
    --model-name paraphrase-multilingual-MiniLM-L12-v2 \
    --output-dir output \
    --batch-size 256 \
    --save-npy \
    --verbose

# Use GPU if available
python scripts/patents_embeddings.py --device cuda

# Multi-GPU processing
python scripts/patents_embeddings.py --multi-gpu

# Include empty text rows in aggregation (default excludes them)
python scripts/patents_embeddings.py --include-empty-in-agg

# Disable patent-level output (enabled by default)
python scripts/patents_embeddings.py --no-save-patent-level
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data/patents_cleaned.dta` | Input Stata file path |
| `--model-dir` | `models` | Directory containing SBERT models |
| `--model-name` | `paraphrase-multilingual-MiniLM-L12-v2` | Model subdirectory name |
| `--output-dir` | `output` | Output directory |
| `--batch-size` | 256 | Batch size for encoding |
| `--device` | auto (cuda if available) | Compute device (cuda/cpu) |
| `--multi-gpu` | False | Enable multi-GPU encoding |
| `--save-npy` | False | Also save .npy format outputs |
| `--save-patent-level` | False | Save patent-level embeddings |
| `--include-empty-in-agg` | False | Include empty texts in aggregation |
| `--verbose` | False | Enable debug logging |

### Similarity Calculation (R)

```bash
# Run similarity calculation on embeddings
Rscript scripts/patents_similarity.R
```

### Patent-Level Self-Similarity (Sample)

```bash
cd sample/scripts
Rscript ps_self.R
```

This calculates for each patent:
- `sim_max`: Maximum cosine similarity with previous patents
- `sim_max_d`: Days since the most similar patent
- `sim_ave`: Average cosine similarity with all previous patents

## Code Organization

### Main Script: `scripts/patents_embeddings.py`

The script is organized into these functional components:

1. **Data Loading** (`load_single_file`, `load_and_prepare_data`):
   - Reads Stata (.dta) files using pandas
   - Validates required columns
   - Normalizes legacy column names
   - Builds combined text fields (title + abstract)
   - Creates composite `stkcd_year` keys

2. **Embedding Model** (`SBertEmbedder` class):
   - Loads SBERT models from local paths
   - Handles device selection (CPU/CUDA/Multi-GPU)
   - Implements text chunking for long documents exceeding token limits
   - Provides fallback encoding for oversized texts
   - Uses sentence-level splitting with regex: `(?<=[。；;!?！？。!?.])\s+|\n+`

3. **Aggregation Logic** (`aggregate_chunk`, `finalize_chunk_aggregates`):
   - Groups embeddings by `stkcd_year`
   - Computes simple mean embeddings
   - Computes citation-weighted embeddings
   - Handles empty text exclusion

4. **Output Writing** (`save_embeddings_bundle`, `write_embedding_outputs`):
   - Writes CSV files with metadata and embedding vectors
   - Optionally saves NPY format for efficient array storage

### Data Preprocessing: `scripts/pre.do`

Stata script for cleaning raw patent data:
1. Loads `patents.dta` with selected columns
2. Renames Chinese variable names to English abbreviations
3. Filters by patent type (发明申请, 发明授权, 实用新型)
4. Filters by stock code prefix (0, 3, or 6)
5. Fills missing citation counts with 0
6. Reports duplicates and saves to `patents_cleaned.dta`

### Model Configuration

Two pre-trained models are included locally:

1. **paraphrase-multilingual-MiniLM-L12-v2** (default):
   - Architecture: MiniLM-L12 (distilled BERT)
   - Output dimension: 384
   - Max sequence length: 128 tokens
   - Supports 50+ languages including Chinese

2. **distiluse-base-multilingual-cased-v2**:
   - Architecture: DistilBERT + Dense layer
   - Output dimension: 512
   - Max sequence length: 128 tokens
   - Dense projection: 768 → 512 with Tanh activation

## Cases: Technology Transformation Analysis

The `cases/` directory contains tools for identifying and analyzing firm technology transformation cases based on patent similarity trajectories.

### Key Files

- `find_transformation_cases.py`: Identifies firms with significant similarity drops (cos_sim_lag1 < 0.5)
- `extract_patent_texts.py`: Extracts patent texts for specific firms and years
- `batch_extract.py`: Batch extraction for multiple firms
- `preview_patents.py`: Preview data structure and sample records

### Usage

```bash
# Find transformation cases
cd cases
python find_transformation_cases.py

# Extract patent texts for a specific firm and year
python extract_patent_texts.py --stkcd 000002 --year 2010 --output company2_2010.csv

# Batch extract for multiple firms
python batch_extract.py --companies 000002,000012,000518 --year 2010

# Preview data
python preview_patents.py -n 10
```

### Transformation Identification Criteria

- **Threshold**: cos_sim_lag1 < 0.5 (similarity below 50%)
- **Cross-validation**: Both models (MiniLM and DistilUSE) identify low similarity
- **Sample size**: At least 5 patents in the year (ensures statistical reliability)

## Sample: Inspection and Debugging

The `sample/` directory provides sample data and scripts for manually inspecting and understanding the patent embeddings and aggregation pipeline.

### Workflow

```bash
cd sample/scripts

# 1. Extract sample from main data
python extract_sample_patents.py

# 2. Create sample embeddings from main output
Rscript create_sample_embeddings.R

# 3. Calculate similarities
Rscript calculate_sample_similarity.R

# 4. Run patent-level self-similarity analysis
Rscript ps_self.R
```

### Sample Companies

| stkcd  | Patents | Years | Type |
|--------|---------|-------|------|
| 600808 | 4,820   | 40 (1985-2024) | Many years - for testing lag-3 and cumulative |
| 000002 | 110     | 10 (2002-2014) | Medium years - normal case |
| 000061 | 6       | 3 (2009-2012)  | Few years - lag-3 should be NA |
| 000004 | 2       | 1 (2002)       | Single year - all similarities NA |

## Development Conventions

### Code Style

- **Type hints**: Extensive use of Python type annotations (`typing` module)
- **Docstrings**: Module and function docstrings follow Google style
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Constants**: UPPER_CASE for module-level constants
- **Path handling**: Uses `pathlib.Path` for filesystem operations

### Key Constants

```python
STKCD_COLUMN = "stkcd"           # Company identifier
YEAR_COLUMN = "p_year"           # Year column
KEY_COLUMN = "stkcd_year"        # Composite aggregation key
TEXT_COLUMNS = ("p_tt", "p_abs") # Title and abstract
CITATION_COLUMN = "p_cite"       # Citation count
```

## Performance Considerations

- **GPU Recommended**: Embedding computation is significantly faster on CUDA
- **Batch Size**: Default 256; increase if GPU memory permits
- **Multi-GPU**: Use `--multi-gpu` for systems with multiple CUDA devices
- **Memory**: Full dataset (~2GB Stata) requires substantial RAM; the script filters columns early
- **Chunking**: Long patents are automatically split at sentence boundaries, then token boundaries if needed

## Security Considerations

- Input file paths are not sanitized; ensure trusted input only
- No network operations; all models are loaded from local filesystem
- No sensitive credential handling
- Output files are written with default permissions

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch-size`, or use CPU with `--device cpu` |
| Model not found | Verify model directory exists: `models/{model_name}` |
| Missing columns | Check input file has required columns: stkcd, p_year, p_tt, p_abs |
| CUDA errors | Update PyTorch to match CUDA version; or use CPU mode |
| Slow processing | Enable GPU; increase batch size; consider `--multi-gpu` |
| Stata preprocessing fails | Ensure `patents.dta` exists in `data/` directory |

## References

- Sentence-BERT paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- SBERT documentation: https://www.sbert.net/
- Models from Hugging Face:
  - https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
