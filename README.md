# Patent Similarity Calculation Project

This project computes semantic embeddings for patent title and abstract text,
aggregates them at firm-year and city-year levels, and calculates lagged cosine
similarity measures for innovation and technology-change analysis.

The current pipeline is Python-first. The old R similarity scripts have been
moved to `scripts/archive/r/`; use the Python entrypoints in `scripts/` for new
runs.

## Quick Start

On the remote project host:

```bash
ssh eason
cd /home/ubuntu/patent_similarity_new
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate patent_sim
```

Before expensive runs, check that data, models, and disk space are available:

```bash
ls data models
df -h .
```

Run a reproducible full workflow for both supported models. The folder names are numbered so they sort in pipeline order:

```bash
# Patent-level model inference
python scripts/02_embeddings/compute_patent_level_embeddings.py --model minilm
python scripts/02_embeddings/compute_patent_level_embeddings.py --model distiluse

# Aggregate patent-level vectors to firm-year, city-year, and industry-year vectors
python scripts/03_aggregation/aggregate_firm_year_embeddings.py --model minilm
python scripts/03_aggregation/aggregate_firm_year_embeddings.py --model distiluse
python scripts/03_aggregation/aggregate_city_year_embeddings.py --model minilm
python scripts/03_aggregation/aggregate_city_year_embeddings.py --model distiluse
python scripts/03_aggregation/aggregate_industry_year_embeddings.py --model minilm
python scripts/03_aggregation/aggregate_industry_year_embeddings.py --model distiluse

# Similarity metrics, merged panels, and comparison summaries
python scripts/04_similarity/compute_firm_year_similarity.py --model minilm
python scripts/04_similarity/compute_firm_year_similarity.py --model distiluse
python scripts/04_similarity/compute_city_year_similarity.py --model minilm
python scripts/04_similarity/compute_city_year_similarity.py --model distiluse
python scripts/04_similarity/compute_industry_peer_similarity.py --models minilm,distiluse
python scripts/04_similarity/build_similarity_panels.py --models minilm,distiluse
python scripts/05_postprocess/summarize_similarity_outputs.py --models minilm,distiluse
```

Pass `--model` explicitly. Patent-level embedding defaults to the city-enriched cleaned file so one run can feed both firm and city aggregation.

## Models

Local model directories are expected under `models/`.

| Alias | Local directory | Dimensions |
| --- | --- | --- |
| `minilm` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 |
| `distiluse` | `distiluse-base-multilingual-cased-v2` | 512 |

Both models are loaded from the local filesystem; normal pipeline runs should
not require network access.

## Data Inputs

Main input files:

- `data/patents_ranges/`: stock-code range Parquet files generated from raw data.
- `data/patents_cleaned.parquet`: cleaned firm-year patent input built from
  `data/patents_ranges/`.
- `data/patents_cleaned.dta`: legacy cleaned firm-year patent input.
- `data/patents_cleaned_with_city.dta`: cleaned input with city fields.
- `data/patents.dta`: raw source data, used when rebuilding range files.
- `data/stkcd_info.csv`: firm-year industry mapping used by peer similarity.

The embedding scripts can read `.dta` and single `.parquet` inputs. Required
columns for firm-year runs are:

| Column | Meaning |
| --- | --- |
| `stkcd` | Company stock code |
| `p_year` | Patent year |
| `p_tt` | Patent title |
| `p_abs` | Patent abstract |

Optional columns used by aggregation and outputs:

| Column | Meaning |
| --- | --- |
| `p_id` | Patent ID |
| `p_cite` | Citation count for citation-weighted aggregation |
| `p_date` | Application date; used for stable sorting when present |
| `p_type` | Patent type |
| `p_ipc` | IPC classification |

City-year runs additionally require firm-year metadata with `countyID`; the
aggregation derives `city_code` from the first four digits of zero-padded
`countyID` and balances the panel against `data/stkcd_info.csv`.

| Column | Meaning |
| --- | --- |
| `city` | City name |
| `countyID` | County/district code; first four digits define `city_code` |
| `province` | Province name |

Use `data/patents_cleaned_with_city.dta` for city workflows that need city
fields. Build the cleaned firm-year Parquet input from the range files with:

```bash
python scripts/01_preprocess/build_parquet_patents.py
```

The historical Stata preprocessing script has been archived at
`scripts/archive/stata/pre.do`. New preprocessing work should use the Python
entry points in `scripts/`.

## Main Entry Points

| Stage | Script | Purpose |
| --- | --- | --- |
| `01_preprocess` | `scripts/01_preprocess/build_parquet_patents.py` | Build cleaned firm-year patent Parquet data from range files |
| `01_preprocess` | `scripts/01_preprocess/split_patents_dta_to_parquet.py` | Split raw Stata data into stock-range Parquet files |
| `02_embeddings` | `scripts/02_embeddings/compute_patent_level_embeddings.py` | Run SBERT model inference and save patent-level vectors |
| `03_aggregation` | `scripts/03_aggregation/aggregate_firm_year_embeddings.py` | Aggregate patent-level vectors to firm-year embeddings |
| `03_aggregation` | `scripts/03_aggregation/aggregate_city_year_embeddings.py` | Aggregate patent-level vectors to city-year embeddings |
| `03_aggregation` | `scripts/03_aggregation/aggregate_industry_year_embeddings.py` | Aggregate patent-level vectors to industry-year embeddings |
| `04_similarity` | `scripts/04_similarity/compute_firm_year_similarity.py` | Compute firm-year lag-1, lag-3, and cumulative similarity |
| `04_similarity` | `scripts/04_similarity/compute_city_year_similarity.py` | Compute city-year lag-1, lag-3, and cumulative similarity |
| `04_similarity` | `scripts/04_similarity/compute_industry_peer_similarity.py` | Compare each firm-year with prior-year industry peers |
| `04_similarity` | `scripts/04_similarity/build_similarity_panels.py` | Merge firm, city, and industry-peer outputs |
| `05_postprocess` | `scripts/05_postprocess/summarize_similarity_outputs.py` | Build comparison, correlation, industry, and city summaries |

Legacy top-level wrappers and pre-refactor R/Stata scripts have been removed from the active API. Historical code lives under `scripts/archive/`.

## Embedding and Aggregation Options

Patent-level embedding options:

| Option | Description |
| --- | --- |
| `--input PATH` | Patent input file. Defaults to `data/patents_cleaned.parquet`. |
| `--model minilm|distiluse` | Stable model alias. |
| `--model-name NAME` | Full local model directory name. |
| `--model-dir PATH` | Model parent directory. Defaults to `models`. |
| `--output-dir PATH` | Output directory. Defaults to `output/patent_embeddings`. |
| `--batch-size N` | Override auto-selected model batch size. |
| `--device cuda|cpu` | Force compute device. |
| `--multi-gpu` | Use SentenceTransformers multi-process multi-GPU mode when available. |
| `--row-chunk-size N` | Process patent texts in row chunks. |
| `--embed-backend overflow|legacy` | Long-text embedding strategy. Default is `overflow`. |
| `--max-seq-length N` | Override tokenizer/model max sequence length. |
| `--fp16` | Use fp16 model weights on CUDA. |
| `--tf32` | Enable TF32 matmul on supported CUDA GPUs. |
| `--max-chunks N` | Debug limit for chunked runs. |
| `--verbose` | Enable debug logging. |

Aggregation options:

| Option | Description |
| --- | --- |
| `--model minilm|distiluse` | Select the patent-level bundle and output suffix. |
| `--output-dir PATH` | Directory containing patent-level inputs and aggregate outputs. |
| `--patent-meta PATH` | Override patent-level metadata CSV. |
| `--patent-embeddings PATH` | Override patent-level embedding `.npy`. |
| `--row-chunk-size N` | Aggregate patent-level rows in chunks. |
| `--include-empty-in-agg` | Include empty title/abstract rows in aggregation. |
| `--save-npy` | Also save aggregate metadata CSV plus NumPy embedding arrays. |

## Output Files

Use `{model}` as `minilm` or `distiluse`.

Firm-year embedding outputs:

- `output/stkcd_year_{model}_embeddings.csv`
- `output/stkcd_year_citweighted_{model}_embeddings.csv`

City-year embedding outputs:

- `output/city_year_embeddings/city_year_{model}_embeddings.parquet`
- `output/city_year_embeddings/city_year_citweighted_{model}_embeddings.parquet`

Industry-year embedding outputs:

- `output/industry_year_embeddings/industry_year_{model}_embeddings.parquet`
- `output/industry_year_embeddings/industry_year_citweighted_{model}_embeddings.parquet`

If `--save-npy` is used, each embedding bundle also writes:

- `output/{prefix}_{model}_meta.csv`
- `output/{prefix}_{model}_embeddings.npy`

If firm `--save-patent-level` is used:

- `output/patent_embeddings/patent_level_{model}_meta.csv`
- `output/patent_embeddings/patent_level_{model}_embeddings.npy`

Similarity outputs:

- `output/stkcd_year_similarity_{model}.csv`
- `output/stkcd_year_similarity_citweighted_{model}.csv`
- `output/stkcd_year_similarity_merged_{model}.csv`
- `output/city_year_similarity_{model}.csv`
- `output/city_year_similarity_citweighted_{model}.csv`
- `output/city_year_similarity_merged_{model}.csv`

Industry, merged-panel, and summary outputs:

- `output/industry_peer_similarity_{model}.csv`
- `output/industry_peer_similarity_citweighted_{model}.csv`
- `output/industry_peer_similarity_merged_{model}.csv`
- `output/merged_similarity_by_firm_{model}.csv`
- `output/merged_similarity_by_city_{model}.csv`
- `output/similarity_comparison_{model}.csv`
- `output/similarity_correlation_{model}.csv`
- `output/similarity_by_industry_{model}.csv`
- `output/similarity_by_city_summary_{model}.csv`

Embedding CSV metadata columns include:

- Firm-year: `stkcd`, `p_year`, `stkcd_year`, `n_patents`,
  `n_texts_used`, `total_citations`, `mean_citations`, `emb_0` ...
- City-year: `city`, `city_code`, `province`, `p_year`, `city_year`,
  `n_patents`, `n_texts_used`, `total_citations`, `mean_citations`, `emb_0`
  ...
- Industry-year: `Ind`, `p_year`, `industry_year`, `n_patents`,
  `n_texts_used`, `total_citations`, `mean_citations`, `emb_0` ...

Similarity CSVs include `cos_sim_lag1`, `cos_sim_lag3`, and
`cos_sim_cumulative`. Merged files also include citation-weighted versions:
`cos_sim_lag1_citw`, `cos_sim_lag3_citw`, and
`cos_sim_cumulative_citw`.

## Similarity Definitions

Patent text is built from `p_tt + " " + p_abs`. The embedding scripts aggregate
patent vectors by entity-year using:

- simple mean embeddings;
- citation-weighted embeddings using `p_cite`.

For each firm or city, the similarity scripts compute:

- `cos_sim_lag1`: current year versus previous observed year;
- `cos_sim_lag3`: current year versus the mean of the previous three observed
  years;
- `cos_sim_cumulative`: current year versus the mean of all previous observed
  years.

Rows with insufficient history or undefined vector norms receive missing
similarity values. Citation-weighted embeddings fall back to simple embeddings
when a group has zero total citations.

Industry-peer similarity compares a firm-year vector against other firms in the
same industry from years `t-1`, `t-2`, and `t-3`, excluding the same firm. The
reported `peer_sim_t*` values are cosine similarities to the valid peer centroid
for the corresponding lag year.

## Project Layout

```text
.
├── scripts/
│   ├── 01_preprocess/           # Raw-data preparation and chunking
│   ├── 02_embeddings/           # Patent-level SBERT model inference only
│   ├── 03_aggregation/          # Firm-year and city-year embedding aggregation
│   ├── 04_similarity/           # Firm, city, peer, and panel similarity calculations
│   ├── 05_postprocess/          # Summaries, comparisons, and descriptive outputs
│   └── archive/                 # Historical R/Stata scripts
├── data/                        # Large local inputs and mappings
├── models/                      # Local SBERT model directories
├── output/                      # Generated embeddings, similarities, panels, and summaries
├── sample/                      # Small sample/debug workflows
├── cases/                       # Technology-transformation case extraction
├── requirements.txt
└── AGENTS.md                    # Longer operational notes for coding agents
```

## Sample and Case Workflows

Sample scripts live under `sample/scripts/`:

```bash
cd sample/scripts
python extract_sample_patents.py
Rscript create_sample_embeddings.R
Rscript calculate_sample_similarity.R
Rscript ps_self.R
```

Technology-transformation case tools live under `cases/`:

```bash
cd cases
python find_transformation_cases.py
python extract_patent_texts.py --stkcd 000002 --year 2010 --output company2_2010.csv
python batch_extract.py --companies 000002,000012,000518 --year 2010 -o output/
python preview_patents.py -n 10
```

## Troubleshooting

| Issue | Check |
| --- | --- |
| `conda: command not found` | Source `/home/ubuntu/miniconda3/etc/profile.d/conda.sh` before `conda activate patent_sim`. |
| `python: command not found` | Activate `patent_sim` or call `/home/ubuntu/miniconda3/envs/patent_sim/bin/python`. |
| Model not found | Verify `models/{model_name}` exists on the remote host. |
| Missing firm columns | Check for `stkcd`, `p_year`, `p_tt`, and `p_abs`. |
| Missing city columns | Use `data/patents_cleaned_with_city.dta` or rebuild it from raw data. |
| CUDA out of memory | Reduce `--batch-size`, set `--row-chunk-size`, or use `--device cpu`. |
| Slow embedding runs | Use CUDA, increase batch size if memory allows, and consider `--fp16` or `--tf32`. |
| Empty similarity output | Confirm the matching embedding CSVs exist for the selected `--model`. |
| Excel industry mapping fails | Ensure dependencies from `requirements.txt` are installed, including `fastexcel` and `openpyxl`. |

## References

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SBERT documentation](https://www.sbert.net/)
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)

## Citation

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
