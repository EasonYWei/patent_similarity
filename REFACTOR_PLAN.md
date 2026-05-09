# Patent Similarity Refactor And Publication Plan

Status: planning artifact only. This document does not authorize immediate refactor implementation. Do not restructure code, regenerate outputs, run formatters, or change data/model artifacts until a later explicit implementation request.

Remote source of truth: `/home/ubuntu/patent_similarity_new`

## 1. Executive Summary

The project has moved beyond a prototype. The remote repository already contains completed generated outputs for two embedding models (`minilm` and `distiluse`) across firm-year embeddings, city-year embeddings, firm/city similarities, industry-peer similarities, merged panels, comparison summaries, and patent-level arrays. The next step should be a controlled refactor that preserves the current research behavior while turning the codebase into a maintainable and publishable package.

The refactor should happen in two broad stages:

1. Freeze and verify existing behavior using small golden fixtures.
2. Extract reusable package modules and modern CLI entrypoints while keeping current scripts and output schemas compatible during the transition.

The main technical risks are silent numeric drift, duplicated firm/city logic, hard-coded model suffixes in R workflows, and limited free disk space on the remote host.

## 2. Current Project Snapshot

Observed remote state before writing this plan:

- Git branch: `master`.
- Git worktree was clean before adding this plan file.
- Large generated and input artifacts are ignored rather than tracked.
- Project size is approximately `58G`:
  - `data/`: approximately `30G`
  - `models/`: approximately `18G`
  - `output/`: approximately `9.3G`
  - `.git/`: less than `1M`
- Remote root filesystem was tight: about `16G` free and `92%` used.
- Key current output sizes include multi-GB patent-level `.npy` arrays.
- Existing generated output row counts include approximately:
  - `56,967` firm-year rows for each model embedding output
  - `5,713` city-year rows for each model embedding output
  - `2,292,391` patent-level metadata rows for MiniLM
- There are no visible automated tests, `pyproject.toml`, `LICENSE`, `CITATION.cff`, or lock/environment file.
- Current active implementation is script-centered:
  - `scripts/patents_embeddings.py` is the largest core script and contains firm embedding, text preparation, model loading, chunking, aggregation, and output writing logic.
  - `scripts/city_embeddings.py` mirrors part of the firm embedding flow with city-specific grouping.
  - `scripts/patents_similarity.R` and `scripts/city_similarity.R` duplicate similarity logic and currently rely on manually edited `model_suffix` values.
  - Downstream Python scripts handle industry peers, merged panels, comparisons, sample helpers, and case extraction.

## 3. Refactor Goals

Primary goals:

- Preserve current numeric behavior and output schemas.
- Make the project reproducible for other researchers.
- Reduce duplicated firm/city code by introducing a shared entity-year pipeline.
- Replace manual model switching with CLI/config options.
- Move core similarity logic into tested Python code while keeping R scripts as legacy compatibility or validation references.
- Package reusable logic under `src/patent_similarity/`.
- Add a publication-ready repository surface: README, license, citation metadata, tests, and data/model setup instructions.

Secondary goals:

- Improve error messages and validation for missing columns, missing model directories, and missing outputs.
- Make sample/debug workflows easy to run without full data regeneration.
- Keep expensive full-data execution opt-in and clearly documented.

## 4. Non-Goals And Out Of Scope

The following are explicitly out of scope for the initial refactor unless separately requested:

- IPC-based patent similarity calculation.
- Changing the semantic model choices or adding new embedding models.
- Changing the mathematical definitions of lag-1, lag-3, cumulative, or citation-weighted similarity.
- Regenerating full outputs as part of the planning/documentation step.
- Reprocessing raw Stata data unless schema changes require it later.
- Replacing Stata preprocessing in `scripts/pre.do`.
- Publishing private or restricted data.
- Uploading large data/model/output files to git.
- Optimizing GPU throughput beyond preserving the current working behavior.
- Changing research conclusions, thresholds, or transformation-case criteria.

## 5. Guiding Compatibility Rules

During refactor implementation, compatibility is more important than architectural neatness.

Required compatibility rules:

- Existing generated CSV filename conventions remain supported:
  - `output/stkcd_year_{model}_embeddings.csv`
  - `output/stkcd_year_citweighted_{model}_embeddings.csv`
  - `output/city_year_{model}_embeddings.csv`
  - `output/city_year_citweighted_{model}_embeddings.csv`
  - `output/stkcd_year_similarity_{model}.csv`
  - `output/stkcd_year_similarity_citweighted_{model}.csv`
  - `output/stkcd_year_similarity_merged_{model}.csv`
  - `output/city_year_similarity_{model}.csv`
  - `output/city_year_similarity_citweighted_{model}.csv`
  - `output/city_year_similarity_merged_{model}.csv`
  - `output/industry_peer_similarity_{model}.csv`
  - `output/industry_peer_similarity_citweighted_{model}.csv`
  - `output/industry_peer_similarity_merged_{model}.csv`
  - `output/merged_similarity_by_firm_{model}.csv`
  - `output/merged_similarity_by_city_{model}.csv`
  - `output/similarity_comparison_{model}.csv`
  - `output/similarity_correlation_{model}.csv`
- Existing model short names remain supported:
  - `minilm` maps to `paraphrase-multilingual-MiniLM-L12-v2`
  - `distiluse` maps to `distiluse-base-multilingual-cased-v2`
- Existing key columns remain supported:
  - firm key: `stkcd_year`
  - city key: `city_year`
- Existing embedding vector columns remain named `emb_0`, `emb_1`, ..., `emb_N`.
- Missing history continues to emit missing similarity values.
- Zero, non-finite, or invalid vectors continue to emit missing similarity values.
- Citation-weighted outputs continue to use `p_cite` with current zero/missing handling semantics.
- Legacy script commands should keep working during migration, even if they become thin wrappers.

## 6. Proposed Package Structure

Create a Python package under `src/patent_similarity/`.

Proposed modules and responsibilities:

- `config.py`
  - Project constants, default paths, supported model metadata, model short-name mapping, common column names, and default tolerance values.

- `entities.py`
  - `EntitySpec` dataclass describing firm/city grouping behavior.
  - Built-in specs for firm-year and city-year workflows.
  - Metadata column rules and key construction policies.

- `io.py`
  - Stata and CSV loading.
  - Required-column validation.
  - Output directory handling.
  - Embedding CSV and optional `.npy` writing.
  - Schema checks for existing output files.

- `text.py`
  - Title/abstract text construction.
  - Empty text normalization.
  - Optional text length diagnostics.

- `models.py`
  - Local SBERT model path resolution.
  - Device selection helpers.
  - GPU diagnostics and batch-size recommendations.

- `embedding.py`
  - `SBertEmbedder` and embedding backends.
  - Overflow-window and legacy encode paths.
  - Patent-level embedding save support.

- `aggregation.py`
  - Entity-year simple mean aggregation.
  - Entity-year citation-weighted aggregation.
  - Chunked aggregation utilities.
  - `n_patents`, `n_texts_used`, `total_citations`, and `mean_citations` calculation.

- `similarity.py`
  - Safe cosine similarity.
  - Lag-1, lag-3, and cumulative metrics.
  - Simple and citation-weighted similarity merging.
  - Python parity implementation for current R scripts.

- `peer.py`
  - Industry-peer similarity calculation.
  - Peer group construction from `data/stkcd_info.xlsx`.
  - Parallel execution controls.

- `merge.py`
  - Firm/city/peer panel merging.
  - City-level summary output generation.

- `compare.py`
  - Summary comparisons.
  - Correlation outputs.
  - Industry and city summaries.

- `cases.py` or `cases/` package modules
  - Transformation-case identification.
  - Patent text extraction.
  - Batch extraction helpers.
  - Existing `cases/` scripts can remain as wrappers initially.

- `cli.py`
  - Main command registration and argument parsing.
  - Console entrypoint exposed as `patent-sim`.

- `logging.py` or `runtime.py`
  - Logging setup, progress messages, and runtime diagnostics.

## 7. Entity-Year Abstraction

Firm-year and city-year processing should use one shared aggregation and similarity engine. The difference should be configuration, not duplicated logic.

Proposed interface:

```python
@dataclass(frozen=True)
class EntitySpec:
    name: str
    id_col: str
    year_col: str
    key_col: str
    metadata_cols: tuple[str, ...]
    required_cols: tuple[str, ...]
    output_prefix: str
```

Built-in specs:

```python
FIRM_SPEC = EntitySpec(
    name="firm",
    id_col="stkcd",
    year_col="p_year",
    key_col="stkcd_year",
    metadata_cols=("stkcd",),
    required_cols=("stkcd", "p_year", "p_tt", "p_abs"),
    output_prefix="stkcd_year",
)

CITY_SPEC = EntitySpec(
    name="city",
    id_col="city_code",
    year_col="p_year",
    key_col="city_year",
    metadata_cols=("city_code", "city", "province", "province_code"),
    required_cols=("city_code", "p_year", "p_tt", "p_abs"),
    output_prefix="city_year",
)
```

Implementation notes:

- City metadata columns should be preserved when available.
- City workflows should fail clearly when `city_code` is missing.
- Firm workflows should preserve stock code formatting, including leading zeros.
- Key construction should be deterministic and centralized.

Acceptance criteria:

- One aggregation implementation can produce both firm-year and city-year embedding CSVs.
- Output column names match current files.
- Golden fixture outputs match current behavior.

## 8. CLI Design

Proposed top-level command: `patent-sim`.

Initial command groups:

```bash
patent-sim embed firm --model minilm
patent-sim embed firm --model distiluse
patent-sim embed city --model minilm --input data/patents_cleaned.dta
patent-sim similarity firm --model minilm
patent-sim similarity city --model distiluse
patent-sim peer --models minilm,distiluse
patent-sim merge --models minilm,distiluse
patent-sim compare --models minilm,distiluse
patent-sim cases find
patent-sim cases extract --stkcd 000002 --year 2010
```

Common options:

- `--input`
- `--output-dir`
- `--model-dir`
- `--model`
- `--models`
- `--device`
- `--batch-size`
- `--row-chunk-size`
- `--embed-backend`
- `--max-seq-length`
- `--fp16`
- `--tf32`
- `--save-npy`
- `--save-patent-level`
- `--include-empty-in-agg`
- `--verbose`

Compatibility wrappers:

- Keep `python scripts/patents_embeddings.py ...` working initially.
- Keep `python scripts/city_embeddings.py ...` working initially.
- Keep existing downstream script commands working initially.
- R scripts may remain but should not be the preferred active path once Python parity is verified.

Acceptance criteria:

- `patent-sim --help` is clear and stable.
- Current documented commands either still work or have documented replacements.
- No workflow requires manual source editing to switch models.

## 9. Similarity Semantics To Preserve

Current similarity behavior must be treated as the baseline.

Definitions:

- Lag-1 similarity compares current entity-year vector with the immediately previous available entity-year vector for the same entity.
- Lag-3 similarity compares current vector with the mean of the previous three available entity-year vectors for the same entity.
- Cumulative similarity compares current vector with the mean of all previous available entity-year vectors for the same entity.
- Missing history emits missing similarity.
- Invalid cosine inputs emit missing similarity.

Important parity details:

- Data should be sorted by entity id and year before calculation.
- Previous available rows are used, not necessarily calendar-contiguous years, unless a future research decision explicitly changes this.
- Citation-weighted similarities are calculated from citation-weighted yearly aggregate vectors, not by weighting the final cosine values.
- Merged outputs combine simple and citation-weighted metrics by entity id and year.

Acceptance criteria:

- Python similarity output matches current R similarity output on golden fixtures.
- Differences, if any, are documented and explicitly approved before full adoption.

## 10. Golden Fixture Strategy

Before implementation, freeze a small deterministic fixture set.

Suggested firm fixtures:

- `600808`: many years, useful for lag-3 and cumulative behavior.
- `000002`: medium normal case.
- `000061`: few years, lag-3 should be missing where history is too short.
- `000004`: single-year case, all similarity values should be missing.

Suggested city fixtures:

- Select 2 to 4 cities from existing city-year outputs with varied year coverage.
- Include at least one city with enough years for lag-3.
- Include one sparse city with insufficient history.

Fixture layers:

1. Minimal synthetic unit fixtures with hand-computable embeddings.
2. Small real-data fixtures extracted from current outputs.
3. Optional patent text fixture for embedding pipeline smoke tests.

Expected outputs to freeze:

- Aggregated firm simple embeddings.
- Aggregated firm citation-weighted embeddings.
- Firm similarity merged output.
- Aggregated city simple embeddings.
- Aggregated city citation-weighted embeddings.
- City similarity merged output.
- Industry-peer output for a minimal fixture if practical.

Fixture storage:

```text
tests/fixtures/input/
tests/fixtures/expected/
tests/fixtures/README.md
```

Rules:

- Fixtures must be small enough to commit.
- Fixtures must not include restricted or private records unless publishing rights are clear.
- If real data cannot be committed, use synthetic fixtures and provide a private fixture generation script for local validation.

Acceptance criteria:

- Tests can run quickly on CPU.
- Tests do not require local SBERT models unless explicitly marked as integration tests.
- Core aggregation and similarity tests run without GPU or full data.

## 11. Phase-By-Phase Roadmap

### Phase 0: Safety Freeze

Purpose: establish a reliable baseline before changing implementation.

Tasks:

- Record current command inventory and output schemas.
- Extract or build golden fixtures.
- Create tests for cosine, aggregation, and similarity semantics.
- Add a short baseline report with row counts and selected checksum/hash values for fixture outputs.

Acceptance criteria:

- `pytest` can validate core math behavior against fixtures.
- Current output schema assumptions are documented.
- No full output regeneration is required.

### Phase 1: Packaging Skeleton

Purpose: create a modern package without changing runtime behavior.

Tasks:

- Add `pyproject.toml`.
- Create `src/patent_similarity/`.
- Add package metadata and console script placeholder.
- Move no major logic yet, or move only low-risk constants/helpers.

Acceptance criteria:

- Package imports locally.
- Existing scripts still work.
- Tests still pass.

### Phase 2: Core Extraction

Purpose: move reusable logic out of scripts.

Tasks:

- Extract config, entity specs, I/O validation, text preparation, and safe cosine functions.
- Keep script wrappers delegating to package functions.
- Avoid changing output formats.

Acceptance criteria:

- Existing scripts produce fixture-equivalent outputs.
- No user-facing CLI behavior breaks.

### Phase 3: Shared Aggregation Engine

Purpose: remove firm/city aggregation duplication.

Tasks:

- Implement entity-year aggregation using `EntitySpec`.
- Port firm aggregation to shared engine.
- Port city aggregation to shared engine.
- Preserve metadata and key columns.

Acceptance criteria:

- Firm fixture outputs match baseline.
- City fixture outputs match baseline.
- Current embedding output schemas remain unchanged.

### Phase 4: Python Similarity Engine

Purpose: replace active R dependency for similarity calculation.

Tasks:

- Implement Python lag/cumulative similarity logic.
- Add CLI commands for firm and city similarity.
- Compare fixture outputs against current R script outputs.
- Keep R scripts as legacy references or wrappers.

Acceptance criteria:

- Python similarity matches R baseline on fixtures.
- Model selection is controlled by CLI/config, not source edits.
- Similarity outputs keep current names and columns.

### Phase 5: Downstream Workflow Consolidation

Purpose: clean up peer, merge, compare, sample, and case workflows.

Tasks:

- Move industry-peer logic into package modules.
- Move merge and comparison logic into package modules.
- Make existing scripts delegate to package commands.
- Review older alternate scripts such as `industry_peer_similarity_v2.py` for archive/removal after compatibility is confirmed.

Acceptance criteria:

- Current downstream outputs can be reproduced on fixtures or small samples.
- Users have one documented path for each workflow.
- Legacy/alternate scripts are clearly labeled.

### Phase 6: Documentation And Publication Prep

Purpose: make the project publishable.

Tasks:

- Rewrite README around the full current workflow.
- Add `LICENSE` matching the MIT claim.
- Add `CITATION.cff`.
- Add data/model setup instructions.
- Add output schema documentation.
- Add performance and disk-space warnings.
- Add release checklist.

Acceptance criteria:

- A new user can understand what is included in git versus external artifacts.
- Reproduction commands are copy-pasteable.
- Publication metadata is complete.

### Phase 7: Final Validation And Release

Purpose: verify before publishing.

Tasks:

- Run unit and fixture tests.
- Run CLI smoke tests.
- Confirm no large files are tracked.
- Confirm README instructions match actual CLI.
- Tag a release candidate.

Acceptance criteria:

- Tests pass.
- Git status contains only intended release changes.
- Large files remain outside git.
- Release notes clearly describe refactor compatibility.

## 12. Migration Sequence That Avoids Breaking Scripts

Recommended migration order:

1. Add tests and fixtures first.
2. Add package skeleton with no behavior change.
3. Extract constants and pure helpers.
4. Make existing scripts import helpers from the package.
5. Extract aggregation while preserving script CLIs.
6. Add Python similarity engine while keeping R scripts available.
7. Add the new `patent-sim` CLI.
8. Update docs to prefer `patent-sim`.
9. Mark old scripts as compatibility wrappers.
10. Only after a release cycle, decide whether to remove or archive legacy scripts.

Do not combine high-risk changes in one commit. Suggested commit boundaries:

- `docs: add refactor plan`
- `test: add golden fixtures for similarity behavior`
- `build: add package skeleton`
- `refactor: extract shared config and io helpers`
- `refactor: unify entity-year aggregation`
- `feat: add python similarity cli`
- `docs: update publication workflow`

## 13. Disk-Space Precautions

Remote disk is currently tight. Refactor validation should avoid full regeneration unless space is cleared first.

Rules:

- Do not regenerate full patent-level `.npy` arrays during early refactor.
- Do not run both full embedding models as part of normal tests.
- Use fixtures for automated tests.
- Before expensive regeneration, run:

```bash
df -h ~/patent_similarity_new
du -sh ~/patent_similarity_new/data ~/patent_similarity_new/models ~/patent_similarity_new/output
```

- Consider moving or archiving old generated outputs before full reruns.
- Write temporary outputs to a clearly named scratch directory, not over current baseline outputs.
- Prefer `--output-dir output_refactor_check` for validation runs.
- Delete only files that are explicitly confirmed as generated and no longer needed.

Acceptance criteria:

- No full-data validation step starts with insufficient disk space.
- Baseline outputs are not overwritten during refactor validation.

## 14. Publication Checklist

Repository files to add or update:

- `README.md`: full current workflow, not just early quick start.
- `LICENSE`: MIT, matching current README claim.
- `CITATION.cff`: software citation metadata.
- `pyproject.toml`: package metadata and console script.
- `requirements.txt`: keep broad dependency list if desired.
- `environment.yml` or `requirements.lock.txt`: reproducible environment option.
- `tests/`: unit and fixture tests.
- `docs/` or expanded markdown docs for output schemas and reproduction.
- `.gitignore`: confirm large artifacts stay ignored.

Release notes should state:

- What data is not included in git.
- Where to obtain required cleaned data.
- Where to obtain or place local SBERT model directories.
- Expected disk requirements.
- Expected runtime/GPU requirements.
- Which outputs are reproducible and which are optional.

Before publication:

```bash
git status --short
git ls-files | grep -E "(data/|models/|output/|\\.dta$|\\.npy$|\\.csv$)"
python -m pytest
python -m build
```

The `git ls-files` command should not show large generated artifacts. Small committed fixtures are acceptable if they are intentionally stored under `tests/fixtures/`.

## 15. Testing Plan

Unit tests:

- Required-column validation.
- Model short-name mapping.
- Text construction from title and abstract.
- Empty text handling.
- Citation coercion.
- Simple mean aggregation.
- Citation-weighted aggregation.
- Safe cosine with zero vectors.
- Safe cosine with non-finite values.
- Safe cosine with mismatched dimensions.
- Lag-1 similarity.
- Lag-3 similarity.
- Cumulative similarity.
- Missing-history output behavior.

Golden tests:

- Firm aggregation fixture equals expected output.
- City aggregation fixture equals expected output.
- Firm Python similarity equals current R baseline fixture.
- City Python similarity equals current R baseline fixture.
- Merged similarity output columns and row counts match expected fixtures.

CLI smoke tests:

- `patent-sim --help`
- `patent-sim embed firm --help`
- `patent-sim similarity firm --help`
- Fixture-only firm similarity run.
- Fixture-only city similarity run.

Integration tests, optional and not default:

- Local model loading.
- Small embedding generation on CPU.
- GPU embedding smoke test when CUDA is available.

## 16. Documentation Plan

README should cover:

- Project purpose.
- Current workflow diagram or ordered command list.
- Data requirements and schemas.
- Model setup.
- Installation.
- Firm-year workflow.
- City-year workflow.
- Industry-peer workflow.
- Merge and comparison workflow.
- Case extraction workflow.
- Output schema reference.
- Performance and disk-space notes.
- Citation and license.

Separate docs, if added:

```text
docs/
  data.md
  models.md
  outputs.md
  workflows.md
  development.md
  release.md
```

Documentation acceptance criteria:

- No command requires manual source-code edits.
- Model suffixes and output suffixes are explained once and reused consistently.
- Full-data commands are clearly separated from fixture/test commands.

## 17. Risk Register

Risk: Numeric drift after refactor.

Mitigation: Freeze golden fixtures before moving code. Compare outputs at controlled precision. Require explicit approval for intentional behavior changes.

Risk: Output schema changes break downstream analysis.

Mitigation: Treat existing filenames and columns as compatibility contracts. Add schema tests.

Risk: Disk exhaustion during validation.

Mitigation: Use fixtures by default, write scratch outputs to separate directories, and check disk before full runs.

Risk: R/Python behavior mismatch.

Mitigation: Keep R scripts as parity references until Python similarity output is verified.

Risk: Leading zeros in stock codes are lost.

Mitigation: Add tests for stock code formatting and treat `stkcd` as a string identifier.

Risk: City metadata is inconsistent or missing.

Mitigation: Validate required city columns, preserve optional metadata when available, and document accepted inputs.

Risk: Publishing restricted data.

Mitigation: Keep full data ignored and commit only approved synthetic or safe fixtures.

## 18. Deferred Decisions

These should be decided before implementation begins:

- Whether real-data fixtures are safe to commit, or whether fixtures must be synthetic only.
- Whether R scripts should become wrappers, move to `legacy/`, or remain side-by-side for one release.
- Whether the package should require Python 3.10, 3.11, or a broader range.
- Whether to use Typer, Click, or argparse for the new CLI. Conservative default: use argparse unless richer CLI ergonomics are worth an added dependency.
- Whether to use a lock file generated by pip-tools, uv, or conda. Conservative default: keep `requirements.txt` plus add `environment.yml` for the remote conda workflow.

## 19. Immediate Next Step After This Plan

The next implementation request should start with Phase 0 only:

1. Create a tiny fixture plan.
2. Extract or synthesize fixture data.
3. Add tests for current similarity and aggregation semantics.
4. Do not change production pipeline logic until those tests exist.

This keeps the refactor anchored to the current research outputs instead of relying on visual inspection or manual spot checks.
