"""Build cleaned patent Parquet data from range Parquet inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "patents_ranges"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "patents_cleaned.parquet"
DEFAULT_STKCD_INFO = PROJECT_ROOT / "data" / "stkcd_info.csv"

APPLICATION_TYPE = "发明申请"
GRANTED_TYPE = "发明授权"
PATENT_TYPES = (APPLICATION_TYPE, GRANTED_TYPE)
STOCK_PREFIXES = ("0", "3", "6")

SOURCE_COLUMNS = (
    "股票代码",
    "年份",
    "标题",
    "摘要",
    "申请日",
    "专利类型",
    "IPC",
    "被引证次数",
)

COLUMN_RENAMES = {
    "股票代码": "stkcd",
    "年份": "p_year",
    "标题": "p_tt",
    "摘要": "p_abs",
    "申请日": "p_date",
    "专利类型": "p_type",
    "IPC": "p_ipc",
    "被引证次数": "p_cite",
}

STKCD_INFO_COLUMNS = (
    "stkcd",
    "year",
    "Listdt",
    "province",
    "city",
    "county",
    "countyID",
    "Ind",
)

STKCD_INFO_METADATA_COLUMNS = ("Listdt", "province", "city", "county", "countyID", "Ind")

PATENT_OUTPUT_COLUMNS = (
    "stkcd",
    "p_year",
    "p_tt",
    "p_abs",
    "p_date",
    "p_type",
    "p_ipc",
    "p_cite",
    "is_granted",
)

OUTPUT_COLUMNS = (
    *PATENT_OUTPUT_COLUMNS,
    *STKCD_INFO_METADATA_COLUMNS,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleaned patent Parquet data from data/patents_ranges/*.parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing patent range Parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--stkcd-info",
        type=Path,
        default=DEFAULT_STKCD_INFO,
        help="Firm-year metadata CSV to merge on stkcd and year.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec passed to Polars.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional debug limit on the number of input files read.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def patent_files(input_dir: Path, max_files: int | None = None) -> list[Path]:
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be positive when provided")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files[:max_files] if max_files is not None else files


def ensure_required_columns(path: Path) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in SOURCE_COLUMNS if column not in schema.names()]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def normalized_stock_expr(column: str = "stkcd") -> pl.Expr:
    stock = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    stock = pl.when(stock.str.contains(r"^\d+\.0+$")).then(
        stock.str.replace(r"\.0+$", "")
    ).otherwise(stock)
    return stock.str.zfill(6).alias("stkcd")


def clean_text_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.replace_all(r"\s+", "").alias(column)


def read_stkcd_info(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stock metadata file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pl.read_csv(path, columns=list(STKCD_INFO_COLUMNS), infer_schema_length=10_000)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pl.read_excel(path)
        missing = [column for column in STKCD_INFO_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df = df.select(STKCD_INFO_COLUMNS)
    else:
        raise ValueError(f"Unsupported stock metadata format for {path}. Expected .csv, .xls, or .xlsx")

    info = (
        df.with_columns(
            normalized_stock_expr("stkcd"),
            pl.col("year").cast(pl.Int64, strict=False).alias("p_year"),
            pl.col("Listdt").cast(pl.Utf8, strict=False).alias("Listdt"),
            pl.col("province").cast(pl.Utf8, strict=False).alias("province"),
            pl.col("city").cast(pl.Utf8, strict=False).alias("city"),
            pl.col("county").cast(pl.Utf8, strict=False).alias("county"),
            pl.col("countyID").cast(pl.Int64, strict=False).alias("countyID"),
            pl.col("Ind").cast(pl.Utf8, strict=False).alias("Ind"),
        )
        .select("stkcd", "p_year", *STKCD_INFO_METADATA_COLUMNS)
        .filter(pl.col("stkcd").is_not_null() & pl.col("p_year").is_not_null())
    )
    duplicate_keys = info.height - info.select(pl.struct(["stkcd", "p_year"]).n_unique()).item()
    if duplicate_keys:
        logging.getLogger(__name__).warning(
            "%d duplicate stkcd-year rows in %s; keeping the first",
            duplicate_keys,
            path,
        )
        info = info.unique(subset=["stkcd", "p_year"], keep="first", maintain_order=True)
    return info


def attach_stkcd_info(lf: pl.LazyFrame, stkcd_info_path: Path) -> pl.LazyFrame:
    return lf.join(read_stkcd_info(stkcd_info_path).lazy(), on=["stkcd", "p_year"], how="left").select(
        OUTPUT_COLUMNS
    )


def scan_clean_patent_file(path: Path, file_index: int) -> pl.LazyFrame:
    ensure_required_columns(path)
    return (
        pl.scan_parquet(path)
        .select(SOURCE_COLUMNS)
        .with_row_index("_source_row")
        .with_columns(pl.lit(file_index).alias("_source_file_index"))
        .rename(COLUMN_RENAMES)
        .with_columns(
            normalized_stock_expr(),
            pl.col("p_year").cast(pl.Int64, strict=False).alias("p_year"),
            clean_text_expr("p_tt"),
            clean_text_expr("p_abs"),
            pl.col("p_date").cast(pl.Datetime("us"), strict=False).alias("p_date"),
            pl.col("p_type").cast(pl.Utf8, strict=False).str.strip_chars().alias("p_type"),
            pl.col("p_ipc").cast(pl.Utf8, strict=False).alias("p_ipc"),
            pl.col("p_cite")
            .cast(pl.Float64, strict=False)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("p_cite"),
        )
        .filter(pl.col("p_type").is_in(PATENT_TYPES))
        .filter(pl.col("stkcd").str.slice(0, 1).is_in(STOCK_PREFIXES))
    )


def deduplicate_patents(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns(
            (pl.col("p_type") == GRANTED_TYPE)
            .cast(pl.Int8)
            .max()
            .over("p_abs")
            .alias("is_granted"),
            (pl.col("p_type") == APPLICATION_TYPE).any().over("p_abs").alias("_has_application"),
        )
        .filter(pl.col("_has_application") & (pl.col("p_type") == APPLICATION_TYPE))
        .sort(["p_abs", "p_date", "stkcd", "p_year", "_source_file_index", "_source_row"], nulls_last=True)
        .unique(subset=["p_abs"], keep="first", maintain_order=True)
        .select(PATENT_OUTPUT_COLUMNS)
        .sort(["stkcd", "p_year", "p_date", "p_abs"], nulls_last=True)
    )


def build_parquet_patents(
    *,
    input_dir: Path,
    output_path: Path,
    compression: str = "zstd",
    max_files: int | None = None,
    stkcd_info_path: Path = DEFAULT_STKCD_INFO,
) -> None:
    files = patent_files(input_dir, max_files=max_files)
    logging.getLogger(__name__).info("Reading %d parquet files from %s", len(files), input_dir)

    cleaned_files = [scan_clean_patent_file(path, idx) for idx, path in enumerate(files)]
    cleaned = pl.concat(cleaned_files, how="vertical")
    result = attach_stkcd_info(deduplicate_patents(cleaned), stkcd_info_path).collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path, compression=compression)

    stats = result.select(
        pl.len().alias("rows"),
        pl.col("is_granted").sum().alias("granted_rows"),
        pl.col("p_cite").is_null().sum().alias("null_citations"),
        pl.col("p_abs").n_unique().alias("unique_abstracts"),
        pl.col("Ind").is_null().sum().alias("missing_stkcd_info"),
    ).to_dicts()[0]
    logging.getLogger(__name__).info("Saved %s", output_path)
    logging.getLogger(__name__).info("Output stats: %s", stats)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    build_parquet_patents(
        input_dir=args.input_dir,
        output_path=args.output,
        compression=args.compression,
        max_files=args.max_files,
        stkcd_info_path=args.stkcd_info,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
