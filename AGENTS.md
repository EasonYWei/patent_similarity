# Remote-First Guide for `patent_similarity_new`

## Default Working Rule

- Unless the user explicitly asks for local-only work, do the work on the remote host:
  ```bash
  ssh eason
  cd ~/patent_similarity_new
  ```
- Treat `/home/ubuntu/patent_similarity_new` as the source of truth. This local folder is a wrapper for notes, synced outputs, and agent instructions.
- The remote host does not appear to have `rg`; prefer `find`, `grep`, `sed`, and `ls` there.
- In non-interactive SSH shells, activate conda explicitly before running project commands:
  ```bash
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate patent_sim
  ```

## Project Overview

This project computes semantic embeddings for patent title and abstract text using multilingual Sentence-BERT models. It aggregates patent embeddings at firm-year and city-year levels, then computes lag-1, lag-3, and cumulative cosine similarities for innovation and technology-change analysis.

The active pipeline is Python-first and organized into numbered folders that sort in execution order: `01_preprocess`, `02_embeddings`, `03_aggregation`, and `04_similarity`. Historical R and Stata scripts are kept only under `scripts/archive/` for reference.

## Stack And Models

- Python 3 with `pandas`, `polars`, `numpy`, `torch`, `transformers`, `sentence-transformers`, and `tqdm`.
- R scripts remain in `sample/` and `cases/` workflows where documented; old production R similarity scripts are archived.
- Input data is stored as raw Stata `.dta` files plus Parquet range and cleaned outputs.
- Local model directories on the remote host:
  - `models/paraphrase-multilingual-MiniLM-L12-v2` -> short name `minilm`, 384 dimensions.
  - `models/distiluse-base-multilingual-cased-v2` -> short name `distiluse`, 512 dimensions.

## Remote Repo Layout

- `README.md`: active quick start and project description.
- `AGENTS.md`: this remote-first operating guide.
- `requirements.txt`: Python dependency list.
- `scripts/`: active Python pipeline folders.
  - `01_preprocess/`: raw-data preparation, Parquet cleaning, and Stata-to-Parquet chunking.
  - `02_embeddings/`: patent-level SBERT model inference only.
  - `03_aggregation/`: firm-year and city-year embedding aggregation from patent-level vectors.
  - `04_similarity/`: firm, city, industry-peer, and merged-panel similarity calculations.
  - `archive/r/`: historical pre-refactor production R similarity scripts.
  - `archive/stata/`: historical Stata preprocessing script.
- `cases/`: technology-transformation case identification and patent-text extraction.
- `sample/`: small inspection and debugging workflow.
- `data/`: large `.dta` inputs and mapping files.
- `models/`: local SBERT model directories.
- `output/`: generated embeddings, similarities, and merged panels.

## Data Schema And Required Inputs

Core remote files:

- `data/patents.dta`: raw source data.
- `data/patents_ranges/`: stock-code range Parquet inputs.
- `data/patents_cleaned.parquet`: cleaned firm-year Parquet input built from range files.
- `data/patents_cleaned.dta`: legacy cleaned main input.
- `data/patents_cleaned_with_city.dta`: city-enriched input for city and merge workflows.
- `data/stkcd_info.csv`: firm-year industry mapping.

Minimum columns for firm-year embeddings:

- `stkcd`: company stock code.
- `p_year`: patent year.
- `p_tt`: patent title.
- `p_abs`: patent abstract.

City workflow columns:

- `city`: city name.
- `city_code`: city code.
- `province`: province name.
- `province_code`: province code.

Common optional columns used downstream:

- `p_id`: patent ID, often from `newipzlid`.
- `p_cite`: citation count used for weighting.
- `p_date`: application date.
- `p_type`: patent type.
- `p_ipc`: IPC classification.

Each numbered stage keeps its own `tools/` helpers. Do not reintroduce a shared `patent_similarity/` package or conda auto-reexec helper unless explicitly requested.

## Main Commands

Always start on the remote host unless local-only work was explicitly requested:

```bash
ssh eason
cd ~/patent_similarity_new
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate patent_sim
```

Refresh Python dependencies only when needed:

```bash
pip install -r requirements.txt
```

Build cleaned firm-year Parquet data from range files:

```bash
python scripts/01_preprocess/build_parquet_patents.py
```

Generate patent-level embeddings, aggregate them, and calculate similarities:

```bash
python scripts/02_embeddings/compute_patent_level_embeddings.py --model minilm
python scripts/02_embeddings/compute_patent_level_embeddings.py --model distiluse

python scripts/03_aggregation/aggregate_firm_year_embeddings.py --model minilm
python scripts/03_aggregation/aggregate_firm_year_embeddings.py --model distiluse
python scripts/03_aggregation/aggregate_city_year_embeddings.py --model minilm
python scripts/03_aggregation/aggregate_city_year_embeddings.py --model distiluse

python scripts/04_similarity/compute_firm_year_similarity.py --model minilm
python scripts/04_similarity/compute_firm_year_similarity.py --model distiluse
python scripts/04_similarity/compute_city_year_similarity.py --model minilm
python scripts/04_similarity/compute_city_year_similarity.py --model distiluse
```

Run industry and merged-panel workflows:

```bash
python scripts/04_similarity/compute_industry_peer_similarity.py --models minilm,distiluse
python scripts/04_similarity/build_similarity_panels.py --data-path ./data/patents_cleaned_with_city.dta --models minilm,distiluse
```

## Active CLI Notes

Patent-level embedding script options:

- `--input`: input `.dta` or `.parquet` file; defaults to `data/patents_cleaned_with_city.dta`.
- `--model`: model short name, usually `minilm` or `distiluse`.
- `--model-name`: full local model directory name.
- `--model-dir`: directory containing local SBERT models.
- `--output-dir`: output directory.
- `--batch-size`: embedding batch size; auto-selected when omitted.
- `--device`: compute device, usually CUDA if available.
- `--multi-gpu`: use SentenceTransformers multi-process multi-GPU path.
- `--row-chunk-size`: embed patent rows in chunks.
- `--embed-backend`: `overflow` by default; `legacy` remains available for comparison.
- `--max-seq-length`, `--fp16`, `--tf32`, `--max-chunks`, `--verbose`.

Aggregation scripts consume `output/patent_level_{model}_meta.csv` and `output/patent_level_{model}_embeddings.npy`, then produce the existing firm-year and city-year embedding filenames. They support `--patent-meta`, `--patent-embeddings`, `--row-chunk-size`, `--include-empty-in-agg`, and `--save-npy`.

## Expected Outputs

Use `{model}` as `minilm` or `distiluse`.

Patent-level embeddings:

- `output/patent_level_{model}_meta.csv`
- `output/patent_level_{model}_embeddings.npy`

Firm-year embeddings and similarities:

- `output/stkcd_year_{model}_embeddings.csv`
- `output/stkcd_year_citweighted_{model}_embeddings.csv`
- `output/stkcd_year_similarity_{model}.csv`
- `output/stkcd_year_similarity_citweighted_{model}.csv`
- `output/stkcd_year_similarity_merged_{model}.csv`

City-year embeddings and similarities:

- `output/city_year_{model}_embeddings.csv`
- `output/city_year_citweighted_{model}_embeddings.csv`
- `output/city_year_similarity_{model}.csv`
- `output/city_year_similarity_citweighted_{model}.csv`
- `output/city_year_similarity_merged_{model}.csv`

Downstream merged analysis:

- `output/industry_peer_similarity_{model}.csv`
- `output/industry_peer_similarity_citweighted_{model}.csv`
- `output/industry_peer_similarity_merged_{model}.csv`
- `output/merged_similarity_by_firm_{model}.csv`
- `output/merged_similarity_by_city_{model}.csv`

## Development Conventions

- Use Python type annotations where practical.
- Keep module and function docstrings in the existing style.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for module-level constants.
- Use `pathlib.Path` for filesystem paths.
- Prefer structured data handling through Polars, pandas, data.table, or Stata rather than ad hoc text parsing.
- Keep generated outputs in `output/`, sample outputs in `sample/output/`, and case artifacts under `cases/`.
- Do not reintroduce top-level compatibility wrappers for removed scripts unless explicitly requested.
- Ignored generated files such as `__pycache__/` and `.pyc` files should not be left in the scripts tree after cleanup work.

## Performance And Troubleshooting

- GPU is strongly recommended for full embedding runs.
- Default batch size is auto-detected; tune `--batch-size` if memory or throughput is poor.
- Use `--row-chunk-size` to reduce peak RAM/VRAM on large data.
- Use `--multi-gpu` only when multiple CUDA devices are available and the multi-GPU path is desired.
- `--fp16` and `--tf32` can speed up CUDA runs but may introduce small numeric drift.
- Full cleaned data is large; check `df -h ~/patent_similarity_new` before expensive regeneration.
- If city similarity output is empty, confirm `scripts/02_embeddings/compute_patent_level_embeddings.py` ran on an input with city columns, then confirm `scripts/03_aggregation/aggregate_city_year_embeddings.py` ran for the selected model.
- If a model is missing, verify `models/{model_name}` exists on the remote host.

## References

- Sentence-BERT paper: https://arxiv.org/abs/1908.10084
- SBERT documentation: https://www.sbert.net/
- Model: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Model: https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
