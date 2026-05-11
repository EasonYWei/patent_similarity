# Scripts Directory

Active code is organized into numbered pipeline folders so a directory listing sorts in execution order.

## Folder Naming Rule

- `01_preprocess/`: raw data preparation before model inference.
- `02_embeddings/`: patent-level SBERT model calls only.
- `03_aggregation/`: aggregate patent-level vectors into firm-year and city-year embeddings.
- `04_similarity/`: calculate firm, city, industry-peer, and merged-panel similarities.
- `archive/`: historical R/Stata scripts; reference-only.

Each active folder may have a local `tools/` subfolder for helpers used by that stage. Do not add a new top-level shared package without an explicit design decision.

## Active Entry Points

```bash
python scripts/01_preprocess/build_parquet_patents.py --help
python scripts/01_preprocess/split_patents_dta_to_parquet.py --help
python scripts/02_embeddings/compute_patent_level_embeddings.py --help
python scripts/03_aggregation/aggregate_firm_year_embeddings.py --help
python scripts/03_aggregation/aggregate_city_year_embeddings.py --help
python scripts/04_similarity/compute_firm_year_similarity.py --help
python scripts/04_similarity/compute_city_year_similarity.py --help
python scripts/04_similarity/compute_industry_peer_similarity.py --help
python scripts/04_similarity/build_similarity_panels.py --help
```

## Patent-Level First Workflow

`02_embeddings/compute_patent_level_embeddings.py` writes:

- `output/patent_embeddings/patent_level_{model}_meta.csv`
- `output/patent_embeddings/patent_level_{model}_embeddings.npy`

The `03_aggregation/` scripts read those files and write the existing downstream-compatible aggregate files:

- `output/stkcd_year_{model}_embeddings.csv`
- `output/stkcd_year_citweighted_{model}_embeddings.csv`
- `output/city_year_{model}_embeddings.csv`
- `output/city_year_citweighted_{model}_embeddings.csv`

## Archived Code

`archive/r/` and `archive/stata/` contain historical pre-refactor scripts. They are reference-only and should not be edited for active pipeline behavior.

Ignored generated files such as `__pycache__/` and `.pyc` files should not be kept in this tree.
