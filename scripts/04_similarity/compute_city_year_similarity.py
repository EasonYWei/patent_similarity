"""Compute city-year lagged similarity metrics from aggregate embedding Parquet files."""

from __future__ import annotations

from tools.cli import parse_similarity_cli, setup_logging
from tools.config import model_short_name
from tools.entities import CITY_SPEC
from tools.similarity import run_similarity_for_model


def main(argv: list[str] | None = None) -> int:
    args = parse_similarity_cli("Compute city-year patent similarity", argv)
    setup_logging(args.verbose)
    run_similarity_for_model(
        CITY_SPEC,
        model_short_name(args.model),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
