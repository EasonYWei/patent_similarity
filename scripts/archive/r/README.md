# Archived R Similarity Scripts

These R scripts are kept as legacy references for the pre-refactor workflow:

- `patents_similarity.R` -> replaced by `scripts/compute_firm_year_similarity.py`
- `city_similarity.R` -> replaced by `scripts/compute_city_year_similarity.py`
- `industry_peer_similarity.R` -> replaced by `scripts/compute_industry_peer_similarity.py`

The active workflow now uses Python with Polars for data analysis. These archived files
should not be edited for new pipeline behavior; use them only to compare historical
logic while validating the Python refactor.
