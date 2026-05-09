"""Build merged firm-level and city-level similarity panels."""

from __future__ import annotations

from tools.cli import parse_panel_cli, setup_logging
from tools.panels import build_panels


def main(argv: list[str] | None = None) -> int:
    args = parse_panel_cli(argv)
    setup_logging(args.verbose)
    build_panels(args.models, args.data_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
