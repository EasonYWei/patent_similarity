# Archived Stata Preprocessing

`pre.do` is the historical Stata preprocessing script. It is kept for reference
and for comparing old preprocessing behavior, but it is not part of the active
top-level script API.

For new work, use the Python data-loading and preprocessing helpers in
`scripts/patent_similarity/io.py`. To rebuild city-enriched cleaned data, use:

```bash
python scripts/build_city_enriched_patents.py \
  --input data/patents.dta \
  --output data/patents_cleaned_with_city.dta \
  --chunk-size 100000
```
