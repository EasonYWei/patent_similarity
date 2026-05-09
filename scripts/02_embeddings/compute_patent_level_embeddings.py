"""Compute patent-level text embeddings."""

from __future__ import annotations

from tools.cli import embedding_input_path, parse_patent_embedding_cli, setup_cli_runtime
from tools.config import resolve_model_name
from tools.workflow import run_patent_level_embedding_pipeline


def main(argv: list[str] | None = None) -> int:
    args = parse_patent_embedding_cli(argv)
    setup_cli_runtime(args)
    model_name = resolve_model_name(args.model, args.model_name)
    run_patent_level_embedding_pipeline(
        input_path=embedding_input_path(args),
        model_dir=args.model_dir,
        model_name=model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        multi_gpu=args.multi_gpu,
        row_chunk_size=args.row_chunk_size,
        embed_backend=args.embed_backend,
        max_seq_length=args.max_seq_length,
        fp16=args.fp16,
        tf32=args.tf32,
        max_chunks=args.max_chunks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
