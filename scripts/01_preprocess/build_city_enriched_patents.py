"""Build a city-enriched cleaned patent Stata file from raw patent data."""

from __future__ import annotations

from tools.city_enrichment import build_city_enriched_patents
from tools.cli import parse_city_enrichment_cli, setup_logging


def main(argv: list[str] | None = None) -> int:
    args = parse_city_enrichment_cli(argv)
    setup_logging(args.verbose)
    build_city_enriched_patents(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        target_rows=args.target_rows,
        max_chunks=args.max_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
