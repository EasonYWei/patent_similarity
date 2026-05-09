"""Compute industry-peer patent similarity metrics."""

from __future__ import annotations

from tools.cli import parse_industry_peer_cli, setup_logging
from tools.industry_peer import process_models


def main(argv: list[str] | None = None) -> int:
    args = parse_industry_peer_cli(argv)
    setup_logging(args.verbose)
    if args.clean:
        raise ValueError("--clean is disabled to avoid deleting generated outputs implicitly")
    process_models(args.models, args.industry_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
