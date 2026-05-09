# Archived R Similarity Scripts

These scripts are historical references from the pre-refactor production workflow.
The active production similarity pipeline is Python with Polars:

- `patents_similarity.R` -> `scripts/compute_firm_year_similarity.py`
- `city_similarity.R` -> `scripts/compute_city_year_similarity.py`
- `industry_peer_similarity.R` -> `scripts/compute_industry_peer_similarity.py`

Do not edit these files for new pipeline behavior. Use them only to compare
historical logic while validating the Python implementation.
