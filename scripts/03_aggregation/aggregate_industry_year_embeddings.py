"""Aggregate patent-level embeddings into industry-year embeddings."""

from __future__ import annotations

from tools.cli import parse_aggregation_cli, setup_logging
from tools.config import DEFAULT_INDUSTRY_YEAR_OUTPUT_DIR
from tools.entities import INDUSTRY_SPEC
from tools.workflow import run_entity_aggregation_pipeline


def main(argv: list[str] | None = None) -> int:
    args = parse_aggregation_cli(
        "Aggregate industry-year patent embeddings",
        argv,
        default_output_dir=DEFAULT_INDUSTRY_YEAR_OUTPUT_DIR,
    )
    setup_logging(args.verbose)
    run_entity_aggregation_pipeline(
        spec=INDUSTRY_SPEC,
        output_dir=args.output_dir,
        patent_input_dir=args.patent_input_dir,
        model=args.model,
        patent_meta_path=args.patent_meta,
        patent_embeddings_path=args.patent_embeddings,
        row_chunk_size=args.row_chunk_size,
        max_chunks=args.max_chunks,
        include_empty_in_agg=args.include_empty_in_agg,
        save_npy=args.save_npy,
        balance_path=args.balance_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
