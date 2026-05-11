"""Compute firm-year similarity from cumulative IPC technology portfolios."""

from __future__ import annotations

from tools.cli import parse_ipc_similarity_cli, setup_logging
from tools.ipc_similarity import run_ipc_similarity


def main(argv: list[str] | None = None) -> int:
    args = parse_ipc_similarity_cli(argv)
    setup_logging(args.verbose)
    run_ipc_similarity(
        input_path=args.input,
        output_dir=args.output_dir,
        patent_type=args.patent_type,
        ipc_allocation=args.ipc_allocation,
        row_chunk_size=args.row_chunk_size,
        max_chunks=args.max_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
