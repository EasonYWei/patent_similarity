"""IPC portfolio construction and firm-year similarity calculations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
import re

import numpy as np
import polars as pl

from .config import (
    DEFAULT_IPC_SIMILARITY_OUTPUT_DIR,
    DEFAULT_PATENTS_PARQUET_FILE,
    STKCD_COLUMN,
    YEAR_COLUMN,
)
from .entities import EntitySpec
from .io import ensure_columns, pandas_to_polars, write_parquet
from .similarity import SIMILARITY_COLUMNS, calculate_entity_similarities

IPC3_RE = re.compile(r"\b([A-HY]\d{2})", re.IGNORECASE)
IPC_VECTOR_PREFIX = "ipc_"
IPC_SIMILARITY_COLUMNS = tuple(
    col.replace("cos_sim", "ipc_sim") for col in SIMILARITY_COLUMNS
)

IPC_FIRM_SPEC = EntitySpec(
    name="firm IPC portfolio",
    id_col=STKCD_COLUMN,
    year_col=YEAR_COLUMN,
    sort_cols=(STKCD_COLUMN, YEAR_COLUMN),
    output_prefix="stkcd_year_ipc",
    similarity_metadata_cols=(STKCD_COLUMN, YEAR_COLUMN, "n_patents", "n_ipc_assignments"),
)


@dataclass(frozen=True)
class IpcPortfolioData:
    annual_weights: dict[tuple[str, int, str], float]
    annual_patents: dict[tuple[str, int], int]
    annual_assignments: dict[tuple[str, int], int]
    ipc_codes: tuple[str, ...]
    input_rows: int
    used_patents: int
    skipped_missing_keys: int
    skipped_missing_ipc: int


def parse_ipc3_codes(value: object) -> list[str]:
    """Return unique IPC3 classes from an IPC string, preserving first-seen order."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip().upper()
    if not text:
        return []

    codes: list[str] = []
    seen: set[str] = set()
    for match in IPC3_RE.finditer(text):
        code = match.group(1).upper()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _normalize_stock(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text.zfill(6) if text.isdigit() else text


def _normalize_year(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _patent_type_filter_enabled(patent_type: str | None) -> bool:
    return bool(patent_type) and str(patent_type).strip().lower() != "all"


def _select_input_columns(patent_type: str | None) -> list[str]:
    cols = [STKCD_COLUMN, YEAR_COLUMN, "p_ipc"]
    if _patent_type_filter_enabled(patent_type):
        cols.append("p_type")
    return cols


def _iter_parquet_batches(
    path: Path,
    columns: list[str],
    row_chunk_size: int,
    max_chunks: int | None,
) -> Iterator[pl.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to stream Parquet inputs") from exc

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = [col for col in columns if col not in available]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    for chunk_idx, batch in enumerate(
        parquet_file.iter_batches(batch_size=row_chunk_size, columns=columns),
        start=1,
    ):
        if max_chunks is not None and chunk_idx > max_chunks:
            break
        yield pl.from_arrow(batch)


def _iter_csv_batches(
    path: Path,
    columns: list[str],
    row_chunk_size: int,
    max_chunks: int | None,
) -> Iterator[pl.DataFrame]:
    df = pl.read_csv(path, columns=columns, infer_schema_length=10_000)
    for start in range(0, df.height, row_chunk_size):
        chunk_idx = start // row_chunk_size + 1
        if max_chunks is not None and chunk_idx > max_chunks:
            break
        yield df.slice(start, row_chunk_size)


def _iter_stata_batches(
    path: Path,
    columns: list[str],
    row_chunk_size: int,
    max_chunks: int | None,
) -> Iterator[pl.DataFrame]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for Stata .dta compatibility I/O") from exc

    reader = pd.read_stata(
        path,
        columns=columns,
        convert_categoricals=False,
        chunksize=row_chunk_size,
    )
    for chunk_idx, pdf in enumerate(reader, start=1):
        if max_chunks is not None and chunk_idx > max_chunks:
            break
        yield pandas_to_polars(pdf)


def iter_input_batches(
    path: str | Path,
    patent_type: str | None,
    row_chunk_size: int,
    max_chunks: int | None = None,
) -> Iterator[pl.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if row_chunk_size <= 0:
        raise ValueError("--row-chunk-size must be positive")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("--max-chunks must be positive when provided")

    columns = _select_input_columns(patent_type)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from _iter_parquet_batches(path, columns, row_chunk_size, max_chunks)
    elif suffix == ".csv":
        yield from _iter_csv_batches(path, columns, row_chunk_size, max_chunks)
    elif suffix == ".dta":
        yield from _iter_stata_batches(path, columns, row_chunk_size, max_chunks)
    else:
        raise ValueError(f"Unsupported input format for {path}. Expected .parquet, .csv, or .dta")


def _allocation_weight(codes: list[str], ipc_allocation: str) -> float:
    if ipc_allocation == "fractional":
        return 1.0 / len(codes)
    return 1.0


def _select_codes_for_allocation(codes: list[str], ipc_allocation: str) -> list[str]:
    if ipc_allocation == "first":
        return codes[:1]
    return codes


def collect_ipc_portfolios(
    input_path: str | Path = DEFAULT_PATENTS_PARQUET_FILE,
    *,
    patent_type: str = "all",
    ipc_allocation: str = "fractional",
    row_chunk_size: int = 250_000,
    max_chunks: int | None = None,
) -> IpcPortfolioData:
    """Aggregate patent rows into annual firm IPC3 portfolio counts."""
    if ipc_allocation not in {"fractional", "first", "full"}:
        raise ValueError(f"Unsupported IPC allocation method: {ipc_allocation}")

    filter_enabled = _patent_type_filter_enabled(patent_type)
    annual_weights: defaultdict[tuple[str, int, str], float] = defaultdict(float)
    annual_patents: defaultdict[tuple[str, int], int] = defaultdict(int)
    annual_assignments: defaultdict[tuple[str, int], int] = defaultdict(int)
    ipc_codes: set[str] = set()
    input_rows = 0
    used_patents = 0
    skipped_missing_keys = 0
    skipped_missing_ipc = 0

    for batch_idx, batch in enumerate(
        iter_input_batches(input_path, patent_type, row_chunk_size, max_chunks),
        start=1,
    ):
        ensure_columns(batch, _select_input_columns(patent_type), input_path)
        input_rows += batch.height
        if filter_enabled:
            batch = batch.filter(pl.col("p_type").cast(pl.Utf8, strict=False) == str(patent_type))
        if batch.is_empty():
            logging.getLogger(__name__).info("Chunk %d has no rows after filtering", batch_idx)
            continue

        stocks = batch.get_column(STKCD_COLUMN).to_list()
        years = batch.get_column(YEAR_COLUMN).to_list()
        ipcs = batch.get_column("p_ipc").to_list()

        for stock_value, year_value, ipc_value in zip(stocks, years, ipcs, strict=True):
            stock = _normalize_stock(stock_value)
            year = _normalize_year(year_value)
            if stock is None or year is None:
                skipped_missing_keys += 1
                continue

            codes = parse_ipc3_codes(ipc_value)
            codes = _select_codes_for_allocation(codes, ipc_allocation)
            if not codes:
                skipped_missing_ipc += 1
                continue

            key = (stock, year)
            weight = _allocation_weight(codes, ipc_allocation)
            annual_patents[key] += 1
            annual_assignments[key] += len(codes)
            used_patents += 1
            ipc_codes.update(codes)
            for code in codes:
                annual_weights[(stock, year, code)] += weight

        logging.getLogger(__name__).info(
            "Processed chunk %d rows=%d cumulative_used_patents=%d",
            batch_idx,
            batch.height,
            used_patents,
        )

    return IpcPortfolioData(
        annual_weights=dict(annual_weights),
        annual_patents=dict(annual_patents),
        annual_assignments=dict(annual_assignments),
        ipc_codes=tuple(sorted(ipc_codes)),
        input_rows=input_rows,
        used_patents=used_patents,
        skipped_missing_keys=skipped_missing_keys,
        skipped_missing_ipc=skipped_missing_ipc,
    )


def build_cumulative_ipc_vectors(portfolios: IpcPortfolioData) -> pl.DataFrame:
    """Convert annual IPC counts into cumulative firm-year IPC share vectors."""
    if not portfolios.annual_weights:
        raise ValueError("No valid patent IPC observations were found")

    ipc_codes = list(portfolios.ipc_codes)
    ipc_index = {code: idx for idx, code in enumerate(ipc_codes)}
    firm_years = sorted(portfolios.annual_patents)
    row_index = {key: idx for idx, key in enumerate(firm_years)}
    annual_matrix = np.zeros((len(firm_years), len(ipc_codes)), dtype=np.float64)

    for (stock, year, code), weight in portfolios.annual_weights.items():
        annual_matrix[row_index[(stock, year)], ipc_index[code]] = weight

    rows: list[dict[str, object]] = []
    start = 0
    while start < len(firm_years):
        stock = firm_years[start][0]
        end = start + 1
        while end < len(firm_years) and firm_years[end][0] == stock:
            end += 1

        cumulative_counts = np.zeros(len(ipc_codes), dtype=np.float64)
        cumulative_patents = 0
        cumulative_assignments = 0
        for row_pos in range(start, end):
            key = firm_years[row_pos]
            cumulative_counts += annual_matrix[row_pos]
            cumulative_patents += portfolios.annual_patents[key]
            cumulative_assignments += portfolios.annual_assignments[key]
            total_weight = float(np.sum(cumulative_counts))
            if total_weight <= 0:
                shares = np.zeros(len(ipc_codes), dtype=np.float64)
            else:
                shares = cumulative_counts / total_weight
            row = {
                STKCD_COLUMN: key[0],
                YEAR_COLUMN: key[1],
                "n_patents": cumulative_patents,
                "n_ipc_assignments": cumulative_assignments,
            }
            row.update({f"{IPC_VECTOR_PREFIX}{code}": float(shares[idx]) for idx, code in enumerate(ipc_codes)})
            rows.append(row)
        start = end

    return pl.DataFrame(rows).sort([STKCD_COLUMN, YEAR_COLUMN], maintain_order=True)


def ipc_vector_columns(df: pl.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith(IPC_VECTOR_PREFIX)]
    if not cols:
        raise ValueError(f"No IPC vector columns found. Expected columns named {IPC_VECTOR_PREFIX}<IPC3>")
    return cols


def calculate_ipc_similarities(vectors: pl.DataFrame) -> pl.DataFrame:
    """Calculate lagged and cumulative IPC portfolio similarities by firm."""
    result = calculate_entity_similarities(vectors, IPC_FIRM_SPEC, ipc_vector_columns(vectors))
    return result.rename(dict(zip(SIMILARITY_COLUMNS, IPC_SIMILARITY_COLUMNS)))


def run_ipc_similarity(
    input_path: str | Path = DEFAULT_PATENTS_PARQUET_FILE,
    *,
    output_dir: str | Path = DEFAULT_IPC_SIMILARITY_OUTPUT_DIR,
    patent_type: str = "all",
    ipc_allocation: str = "fractional",
    row_chunk_size: int = 250_000,
    max_chunks: int | None = None,
) -> None:
    """Build cumulative IPC vectors, calculate similarities, and write Parquet outputs."""
    output_dir = Path(output_dir)
    portfolios = collect_ipc_portfolios(
        input_path,
        patent_type=patent_type,
        ipc_allocation=ipc_allocation,
        row_chunk_size=row_chunk_size,
        max_chunks=max_chunks,
    )
    logging.getLogger(__name__).info(
        "IPC portfolio input_rows=%d used_patents=%d skipped_missing_keys=%d skipped_missing_ipc=%d ipc_classes=%d",
        portfolios.input_rows,
        portfolios.used_patents,
        portfolios.skipped_missing_keys,
        portfolios.skipped_missing_ipc,
        len(portfolios.ipc_codes),
    )

    vectors = build_cumulative_ipc_vectors(portfolios)
    vector_output = output_dir / "stkcd_year_ipc_vectors.parquet"
    write_parquet(vectors, vector_output)
    logging.getLogger(__name__).info("Saved IPC vectors: %s", vector_output)

    similarities = calculate_ipc_similarities(vectors)
    similarity_output = output_dir / "stkcd_year_ipc_similarity.parquet"
    write_parquet(similarities, similarity_output)
    logging.getLogger(__name__).info("Saved IPC similarity: %s", similarity_output)
