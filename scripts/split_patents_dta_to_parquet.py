#!/usr/bin/env python3
"""Split raw patent Stata data into stock-range Parquet files.

The script streams the source `.dta` file in chunks, writes temporary parquet
parts by 100-stock-code ranges, validates the consolidated result, then can
delete the source file only after all checks pass.
"""

from __future__ import annotations

from _ensure_conda_env import ensure_patent_sim_env

ensure_patent_sim_env()

import argparse
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


STOCK_COLUMN = "股票代码"
RANGE_COLUMN = "__stkcd_range"
STOCK_CODE_PATTERN = re.compile(r"^\d{1,6}$")


@dataclass(frozen=True)
class RangeInfo:
    key: str
    begin: int
    end: int

    @property
    def filename(self) -> str:
        return f"{self.begin:06d}_{self.end:06d}.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split data/patents.dta into 100-stock-code Parquet range files."
    )
    parser.add_argument("--input", type=Path, default=Path("data/patents.dta"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/patents_ranges"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--delete-source-after-verify", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output parquet files and temp parts before conversion.",
    )
    parser.add_argument(
        "--cleanup-failed-temp",
        action="store_true",
        help="Remove temporary parquet parts if conversion or validation fails.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pandas_to_polars(pdf: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas to Polars without requiring pandas' pyarrow bridge."""
    return pl.DataFrame({str(col): pdf[col].to_list() for col in pdf.columns})


def infer_stable_schema(pdf: pd.DataFrame) -> dict[str, pl.DataType]:
    """Infer one stable Polars schema from the first Stata chunk."""
    schema: dict[str, pl.DataType] = {}
    for column, dtype in pdf.dtypes.items():
        name = str(column)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            schema[name] = pl.Datetime("us")
        elif pd.api.types.is_integer_dtype(dtype):
            schema[name] = pl.Int64
        elif pd.api.types.is_float_dtype(dtype):
            schema[name] = pl.Float64
        elif pd.api.types.is_bool_dtype(dtype):
            schema[name] = pl.Boolean
        else:
            schema[name] = pl.Utf8
    return schema


def align_to_schema(df: pl.DataFrame, columns: list[str], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Cast every chunk to the same output schema and column order."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Chunk is missing columns from the source schema: {missing}")
    return df.select(
        [
            pl.col(column).cast(schema[column], strict=False).alias(column)
            for column in columns
        ]
    )


def normalize_stock_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not STOCK_CODE_PATTERN.fullmatch(text):
        return None
    return text.zfill(6)


def range_for_stock(stock_code: str) -> RangeInfo:
    value = int(stock_code)
    begin = (value // 100) * 100
    end = begin + 99
    return RangeInfo(key=f"{begin:06d}_{end:06d}", begin=begin, end=end)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_tmp_parts"
    existing_final = list(output_dir.glob("*.parquet"))
    existing_temp = temp_dir.exists() and any(temp_dir.iterdir())
    if (existing_final or existing_temp) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains parquet output or temp parts. "
            "Use --overwrite to rebuild it."
        )
    if overwrite:
        for path in existing_final:
            path.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def source_columns(input_path: Path) -> list[str]:
    reader = pd.read_stata(input_path, convert_categoricals=False, chunksize=1)
    first = next(reader)
    return [str(col) for col in first.columns]


def add_range_column(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    codes = df.get_column(STOCK_COLUMN).to_list()
    normalized = [normalize_stock_code(value) for value in codes]
    invalid_count = sum(code is None for code in normalized)
    range_keys = [range_for_stock(code).key if code is not None else None for code in normalized]
    return df.with_columns(pl.Series(RANGE_COLUMN, range_keys)), invalid_count


def write_temp_parts(
    input_path: Path,
    temp_dir: Path,
    chunk_size: int,
    compression: str,
) -> tuple[int, int, list[str]]:
    total_rows = 0
    invalid_rows = 0
    columns: list[str] | None = None
    schema: dict[str, pl.DataType] | None = None
    reader = pd.read_stata(input_path, convert_categoricals=False, chunksize=chunk_size)

    for chunk_index, pdf in enumerate(reader, start=1):
        if columns is None:
            columns = [str(col) for col in pdf.columns]
            schema = infer_stable_schema(pdf)
            if STOCK_COLUMN not in columns:
                raise ValueError(f"Input file is missing required stock column: {STOCK_COLUMN}")

        if schema is None:
            raise RuntimeError("Stable schema was not initialized")

        df = align_to_schema(pandas_to_polars(pdf), columns, schema)
        df, invalid_count = add_range_column(df)
        invalid_rows += invalid_count
        if invalid_count:
            examples = (
                df.filter(pl.col(RANGE_COLUMN).is_null())
                .select(STOCK_COLUMN)
                .head(10)
                .to_series()
                .to_list()
            )
            raise ValueError(f"Found {invalid_count} rows with invalid stock codes in chunk {chunk_index}: {examples}")

        total_rows += df.height
        for (range_key,), group in df.partition_by(RANGE_COLUMN, as_dict=True).items():
            range_dir = temp_dir / range_key
            range_dir.mkdir(parents=True, exist_ok=True)
            part_path = range_dir / f"part-{chunk_index:06d}.parquet"
            group.drop(RANGE_COLUMN).write_parquet(part_path, compression=compression)

        logging.info("Wrote chunk %d rows=%d total_rows=%d", chunk_index, df.height, total_rows)

    if columns is None:
        raise ValueError(f"Input file has no rows: {input_path}")
    return total_rows, invalid_rows, columns


def consolidate_range(range_dir: Path, output_dir: Path, columns: list[str], compression: str) -> dict[str, Any]:
    info = RangeInfo(
        key=range_dir.name,
        begin=int(range_dir.name.split("_", 1)[0]),
        end=int(range_dir.name.split("_", 1)[1]),
    )
    output_path = output_dir / info.filename
    part_paths = sorted(range_dir.glob("part-*.parquet"))
    if not part_paths:
        raise ValueError(f"No part files found in {range_dir}")

    writer: pq.ParquetWriter | None = None
    row_count = 0
    observed_codes: set[str] = set()

    try:
        for part_path in part_paths:
            parquet_file = pq.ParquetFile(part_path)
            if parquet_file.schema_arrow.names != columns:
                raise ValueError(f"Schema mismatch in {part_path}")
            for batch in parquet_file.iter_batches(batch_size=50_000):
                table = pa.Table.from_batches([batch])
                stock_values = table.column(STOCK_COLUMN).to_pylist()
                for value in stock_values:
                    code = normalize_stock_code(value)
                    if code is None:
                        raise ValueError(f"Invalid stock code found during consolidation: {value!r}")
                    code_int = int(code)
                    if code_int < info.begin or code_int > info.end:
                        raise ValueError(f"Stock code {code} is outside range {info.key}")
                    observed_codes.add(code)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                writer.write_table(table)
                row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return {
        "file": output_path.name,
        "range_begin": f"{info.begin:06d}",
        "range_end": f"{info.end:06d}",
        "rows": row_count,
        "unique_stkcd": len(observed_codes),
        "size_bytes": output_path.stat().st_size,
    }


def consolidate_all(temp_dir: Path, output_dir: Path, columns: list[str], compression: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for range_dir in sorted(path for path in temp_dir.iterdir() if path.is_dir()):
        row = consolidate_range(range_dir, output_dir, columns, compression)
        manifest.append(row)
        logging.info("Consolidated %s rows=%d", row["file"], row["rows"])
    if not manifest:
        raise ValueError("No non-empty stock ranges were written")
    return manifest


def validate_outputs(output_dir: Path, manifest: list[dict[str, Any]], columns: list[str], total_rows: int) -> None:
    manifest_rows = sum(int(row["rows"]) for row in manifest)
    if manifest_rows != total_rows:
        raise ValueError(f"Row count mismatch: read={total_rows}, manifest={manifest_rows}")

    for row in manifest:
        path = output_dir / row["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing final parquet file: {path}")
        parquet_file = pq.ParquetFile(path)
        if parquet_file.schema_arrow.names != columns:
            raise ValueError(f"Final schema mismatch: {path}")
        begin = int(row["range_begin"])
        end = int(row["range_end"])
        checked_rows = 0
        for batch in parquet_file.iter_batches(columns=[STOCK_COLUMN], batch_size=100_000):
            table = pa.Table.from_batches([batch])
            for value in table.column(STOCK_COLUMN).to_pylist():
                code = normalize_stock_code(value)
                if code is None:
                    raise ValueError(f"Invalid stock code in {path}: {value!r}")
                code_int = int(code)
                if code_int < begin or code_int > end:
                    raise ValueError(f"Stock code {code} outside file range {path.name}")
            checked_rows += table.num_rows
        if checked_rows != int(row["rows"]):
            raise ValueError(f"Validated row count mismatch for {path}: {checked_rows} != {row['rows']}")


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    pl.DataFrame(manifest).write_csv(output_dir / "conversion_manifest.csv")


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def remove_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    started_at = time.time()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    temp_dir = prepare_output_dir(args.output_dir, args.overwrite)
    source_size = args.input.stat().st_size
    report: dict[str, Any] = {
        "input": str(args.input),
        "source_size_bytes": source_size,
        "output_dir": str(args.output_dir),
        "chunk_size": args.chunk_size,
        "compression": args.compression,
        "deleted_source": False,
        "status": "started",
    }

    try:
        total_rows, invalid_rows, columns = write_temp_parts(
            args.input,
            temp_dir,
            args.chunk_size,
            args.compression,
        )
        manifest = consolidate_all(temp_dir, args.output_dir, columns, args.compression)
        validate_outputs(args.output_dir, manifest, columns, total_rows)
        write_manifest(args.output_dir, manifest)
        remove_temp_dir(temp_dir)

        if args.delete_source_after_verify:
            args.input.unlink()
            report["deleted_source"] = True

        report.update(
            {
                "status": "success",
                "source_rows": total_rows,
                "output_rows": sum(int(row["rows"]) for row in manifest),
                "invalid_stock_rows": invalid_rows,
                "source_columns": columns,
                "range_files": len(manifest),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        )
        write_report(args.output_dir, report)
        logging.info("Conversion completed successfully. range_files=%d rows=%d", len(manifest), total_rows)
        if report["deleted_source"]:
            logging.info("Deleted source after verification: %s", args.input)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        )
        write_report(args.output_dir, report)
        if args.cleanup_failed_temp:
            remove_temp_dir(temp_dir)
        logging.exception("Conversion failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
