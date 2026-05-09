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

Run a reproducible full workflow for both supported models:

```bash
# Firm-year embeddings and similarities
python scripts/compute_firm_year_embeddings.py --model minilm
python scripts/compute_firm_year_embeddings.py --model distiluse
python scripts/compute_firm_year_similarity.py --model minilm
python scripts/compute_firm_year_similarity.py --model distiluse

# City-year embeddings and similarities
python scripts/compute_city_year_embeddings.py --input data/patents_cleaned_with_city.dta --model minilm
python scripts/compute_city_year_embeddings.py --input data/patents_cleaned_with_city.dta --model distiluse
python scripts/compute_city_year_similarity.py --model minilm
python scripts/compute_city_year_similarity.py --model distiluse

# Industry-peer metrics, merged panels, and comparison summaries
python scripts/compute_industry_peer_similarity.py --models minilm,distiluse
python scripts/build_similarity_panels.py --models minilm,distiluse
python scripts/summarize_similarity_outputs.py --models minilm,distiluse
```

Pass `--model` explicitly. Embedding scripts default to `minilm`, while the
similarity scripts default to `distiluse` for compatibility.

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

- `data/patents_cleaned.dta`: cleaned firm-year patent input.
- `data/patents_cleaned_with_city.dta`: cleaned input with city fields.
- `data/patents.dta`: raw source data, used only when preprocessing or rebuilding
  city-enriched data.
- `data/stkcd_info.xlsx`: firm-year industry mapping used by peer similarity.

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

City-year runs additionally require:

| Column | Meaning |
| --- | --- |
| `city` | City name |
| `city_code` | City code |
| `province` | Province name |
| `province_code` | Province code |

If `data/patents_cleaned.dta` lacks city fields, use
`data/patents_cleaned_with_city.dta` for city workflows or rebuild it:

```bash
python scripts/build_city_enriched_patents.py \
  --input data/patents.dta \
  --output data/patents_cleaned_with_city.dta \
  --chunk-size 100000
```

The Stata preprocessing script is still available:

```stata
do scripts/pre.do
```

It renames raw columns, keeps invention applications, granted inventions, and
utility models, filters stock codes starting with `0`, `3`, or `6`, fills missing
citations with `0`, and writes `data/patents_cleaned.dta`.

## Main Entry Points

| Script | Purpose |
| --- | --- |
| `scripts/compute_firm_year_embeddings.py` | Compute firm-year mean and citation-weighted embeddings |
| `scripts/compute_firm_year_similarity.py` | Compute firm-year lag-1, lag-3, and cumulative similarity |
| `scripts/compute_city_year_embeddings.py` | Compute city-year mean and citation-weighted embeddings |
| `scripts/compute_city_year_similarity.py` | Compute city-year lag-1, lag-3, and cumulative similarity |
| `scripts/compute_industry_peer_similarity.py` | Compare each firm-year with prior-year industry peers |
| `scripts/build_similarity_panels.py` | Merge firm, city, and industry-peer outputs |
| `scripts/summarize_similarity_outputs.py` | Build comparison, correlation, industry, and city summaries |
| `scripts/build_city_enriched_patents.py` | Build city-enriched cleaned patent data from raw data |
| `scripts/split_patents_dta_to_parquet.py` | Split raw Stata data into stock-range Parquet files |
| `scripts/compare_validation_embeddings.py` | Compare validation embeddings against baseline outputs |

Compatibility wrappers remain available:

| Wrapper | Current target |
| --- | --- |
| `scripts/patents_embeddings.py` | `compute_firm_year_embeddings.py` |
| `scripts/city_embeddings.py` | `compute_city_year_embeddings.py` |
| `scripts/industry_peer_similarity.py` | `compute_industry_peer_similarity.py` |
| `scripts/industry_peer_similarity_v2.py` | `compute_industry_peer_similarity.py` |
| `scripts/merge_similarities.py` | `build_similarity_panels.py` |
| `scripts/compare_similarity_measures.py` | `summarize_similarity_outputs.py` |

## Embedding Options

Common embedding options:

| Option | Description |
| --- | --- |
| `--input PATH` | Patent input file. Defaults to `data/patents_cleaned.dta`. |
| `--model minilm|distiluse` | Stable model alias. |
| `--model-name NAME` | Full local model directory name. |
| `--model-dir PATH` | Model parent directory. Defaults to `models`. |
| `--output-dir PATH` | Output directory. Defaults to `output`. |
| `--batch-size N` | Override auto-selected embedding batch size. |
| `--device cuda|cpu` | Force compute device. |
| `--multi-gpu` | Use SentenceTransformers multi-process multi-GPU mode when available. |
| `--row-chunk-size N` | Embed and aggregate rows in chunks to reduce peak memory. |
| `--embed-backend overflow|legacy` | Long-text embedding strategy. Default is `overflow`. |
| `--max-seq-length N` | Override tokenizer/model max sequence length. |
| `--fp16` | Use fp16 model weights on CUDA. |
| `--tf32` | Enable TF32 matmul on supported CUDA GPUs. |
| `--include-empty-in-agg` | Include empty title/abstract rows in aggregation. |
| `--save-npy` | Also save metadata CSV plus NumPy embedding arrays. |
| `--verbose` | Enable debug logging. |

Firm-year embeddings also support:

| Option | Description |
| --- | --- |
| `--data-dir PATH` | Deprecated alias for `--input` or a directory containing `patents_cleaned.dta`. |
| `--tokenizers-parallelism VALUE` | Set `TOKENIZERS_PARALLELISM` before model loading. |
| `--process-by-chunk` | Deprecated compatibility flag; maps to `--row-chunk-size 100000` when unset. |
| `--max-chunks N` | Debug limit when row chunking is enabled. |
| `--save-patent-level` | Save patent-level metadata and embedding `.npy` output. |

## Output Files

Use `{model}` as `minilm` or `distiluse`.

Firm-year embedding outputs:

- `output/stkcd_year_{model}_embeddings.csv`
- `output/stkcd_year_citweighted_{model}_embeddings.csv`

City-year embedding outputs:

- `output/city_year_{model}_embeddings.csv`
- `output/city_year_citweighted_{model}_embeddings.csv`

If `--save-npy` is used, each embedding bundle also writes:

- `output/{prefix}_{model}_meta.csv`
- `output/{prefix}_{model}_embeddings.npy`

If firm `--save-patent-level` is used:

- `output/patent_level_{model}_meta.csv`
- `output/patent_level_{model}_embeddings.npy`

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
reported `peer_sim_t*` values are the maximum valid peer cosine similarities.

## Project Layout

```text
.
├── scripts/
│   ├── patent_similarity/       # Shared Python package for I/O, embeddings, aggregation, similarity, panels
│   ├── compute_*                # Current Python pipeline entrypoints
│   ├── build_*                  # Data and panel build scripts
│   ├── *_embeddings.py          # Compatibility wrappers
│   └── archive/r/               # Legacy R similarity scripts
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
