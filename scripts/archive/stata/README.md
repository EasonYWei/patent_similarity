# Archived Stata Preprocessing

`pre.do` is the historical Stata preprocessing script. It is kept for reference
and for comparing old preprocessing behavior, but it is not part of the active
top-level script API.

For new work, use the Python data-loading and preprocessing helpers in
`scripts/01_preprocess/`. To rebuild cleaned firm-year Parquet data, use:

```bash
python scripts/01_preprocess/build_parquet_patents.py
```
