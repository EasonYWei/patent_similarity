"""Summarize and compare merged similarity outputs."""

from __future__ import annotations

from tools.cli import parse_summary_cli, setup_logging
from tools.summaries import summarize_models


def main(argv: list[str] | None = None) -> int:
    args = parse_summary_cli(argv)
    setup_logging(args.verbose)
    summarize_models(args.models, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
