"""Split raw patent Stata data into stock-range Parquet files."""

from __future__ import annotations

from tools.cli import parse_parquet_split_cli, setup_logging
from tools.parquet_split import split_patents_dta_to_parquet


def main(argv: list[str] | None = None) -> int:
    args = parse_parquet_split_cli(argv)
    setup_logging(args.verbose)
    return split_patents_dta_to_parquet(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        compression=args.compression,
        delete_source_after_verify=args.delete_source_after_verify,
        overwrite=args.overwrite,
        cleanup_failed_temp=args.cleanup_failed_temp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
