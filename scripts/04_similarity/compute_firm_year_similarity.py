"""Compute firm-year lagged similarity metrics from aggregate embedding Parquet files."""

from __future__ import annotations

from tools.cli import parse_similarity_cli, setup_logging
from tools.config import DEFAULT_FIRM_SIMILARITY_OUTPUT_DIR, DEFAULT_FIRM_YEAR_INPUT_DIR, model_short_name
from tools.entities import FIRM_SPEC
from tools.similarity import run_similarity_for_model


def main(argv: list[str] | None = None) -> int:
    args = parse_similarity_cli(
        "Compute firm-year patent similarity",
        argv,
        default_input_dir=DEFAULT_FIRM_YEAR_INPUT_DIR,
        default_output_dir=DEFAULT_FIRM_SIMILARITY_OUTPUT_DIR,
    )
    setup_logging(args.verbose)
    run_similarity_for_model(
        FIRM_SPEC,
        model_short_name(args.model),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
