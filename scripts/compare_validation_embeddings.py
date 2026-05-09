#!/usr/bin/env python3
"""Compare validation embedding outputs against the existing baseline outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from patent_similarity.config import model_short_name

KEY_COLUMN = "stkcd_year"
METADATA_COLUMNS = (
    "stkcd",
    "p_year",
    "n_patents",
    "n_texts_used",
    "total_citations",
    "mean_citations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare validation embeddings with baseline outputs")
    parser.add_argument("--new-dir", type=Path, required=True, help="Directory containing validation embedding CSVs")
    parser.add_argument("--baseline-dir", type=Path, default=Path("output"), help="Directory containing baseline CSVs")
    parser.add_argument("--model", default="minilm", help="Model short name or full model name")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Maximum allowed absolute embedding difference")
    parser.add_argument("--metadata-tolerance", type=float, default=1e-8, help="Maximum allowed numeric metadata difference")
    parser.add_argument("--cosine-threshold", type=float, default=0.999999, help="Minimum allowed row-level cosine similarity")
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum mismatch examples to print per check")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path, dtype={KEY_COLUMN: "string", "stkcd": "string"})


def embedding_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("emb_")]
    return sorted(cols, key=lambda col: int(col.split("_", 1)[1]))


def check_unique_keys(df: pd.DataFrame, label: str) -> list[str]:
    duplicated = df.loc[df[KEY_COLUMN].duplicated(), KEY_COLUMN].dropna().unique()
    if len(duplicated) == 0:
        return []
    examples = ", ".join(map(str, duplicated[:10]))
    return [f"{label} has duplicate {KEY_COLUMN} values: {examples}"]


def compare_metadata(merged: pd.DataFrame, max_examples: int, tolerance: float) -> list[str]:
    issues: list[str] = []
    for col in METADATA_COLUMNS:
        left = f"{col}_new"
        right = f"{col}_base"
        if left not in merged.columns or right not in merged.columns:
            issues.append(f"Missing metadata column for comparison: {col}")
            continue

        if col == "stkcd":
            new_values = merged[left].astype("string").str.strip().fillna("<NA>")
            base_values = merged[right].astype("string").str.strip().fillna("<NA>")
            bad = new_values != base_values
        else:
            new_values = pd.to_numeric(merged[left], errors="coerce")
            base_values = pd.to_numeric(merged[right], errors="coerce")
            both_missing = new_values.isna() & base_values.isna()
            close = (new_values - base_values).abs() <= tolerance
            bad = ~(both_missing | close)

        if bad.any():
            keys = merged.loc[bad, KEY_COLUMN].astype(str).head(max_examples).tolist()
            issues.append(f"Metadata mismatch in {col}: {int(bad.sum())} rows; examples={keys}")
    return issues


def compare_one(
    *,
    label: str,
    new_path: Path,
    baseline_path: Path,
    tolerance: float,
    metadata_tolerance: float,
    cosine_threshold: float,
    max_examples: int,
) -> bool:
    print(f"\n== {label} ==")
    print(f"new:      {new_path}")
    print(f"baseline: {baseline_path}")

    new_df = read_csv(new_path)
    baseline_df = read_csv(baseline_path)
    issues = check_unique_keys(new_df, "new") + check_unique_keys(baseline_df, "baseline")

    new_keys = set(new_df[KEY_COLUMN].dropna().astype(str))
    baseline_keys = set(baseline_df[KEY_COLUMN].dropna().astype(str))
    missing_in_baseline = sorted(new_keys - baseline_keys)
    ignored_baseline_count = len(baseline_keys - new_keys)
    if missing_in_baseline:
        issues.append(
            f"Generated keys missing from baseline: {len(missing_in_baseline)}; "
            f"examples={missing_in_baseline[:max_examples]}"
        )

    new_emb_cols = embedding_columns(new_df)
    baseline_emb_cols = embedding_columns(baseline_df)
    if new_emb_cols != baseline_emb_cols:
        issues.append(
            f"Embedding columns differ: new={len(new_emb_cols)} baseline={len(baseline_emb_cols)}"
        )

    baseline_subset = baseline_df[baseline_df[KEY_COLUMN].astype(str).isin(new_keys)]
    merged = new_df.merge(
        baseline_subset,
        on=KEY_COLUMN,
        how="left",
        suffixes=("_new", "_base"),
        validate="one_to_one",
    )
    issues.extend(compare_metadata(merged, max_examples=max_examples, tolerance=metadata_tolerance))

    stats: dict[str, float | int] = {
        "new_rows": len(new_df),
        "baseline_matching_rows": len(baseline_subset),
        "baseline_extra_keys_ignored": ignored_baseline_count,
        "embedding_dimensions": len(new_emb_cols),
    }

    if new_emb_cols == baseline_emb_cols and not missing_in_baseline:
        new_matrix = merged[[f"{col}_new" for col in new_emb_cols]].to_numpy(dtype=np.float64)
        base_matrix = merged[[f"{col}_base" for col in baseline_emb_cols]].to_numpy(dtype=np.float64)
        diff = np.abs(new_matrix - base_matrix)
        row_max = np.nanmax(diff, axis=1) if len(diff) else np.array([], dtype=np.float64)
        bad_diff = np.isnan(row_max) | (row_max > tolerance)

        dot = np.sum(new_matrix * base_matrix, axis=1)
        denom = np.linalg.norm(new_matrix, axis=1) * np.linalg.norm(base_matrix, axis=1)
        cosine = np.divide(dot, denom, out=np.full_like(dot, np.nan), where=denom > 0)
        bad_cosine = np.isnan(cosine) | (cosine < cosine_threshold)

        stats.update(
            {
                "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
                "mean_abs_diff": float(np.nanmean(diff)) if diff.size else 0.0,
                "rows_over_tolerance": int(bad_diff.sum()),
                "min_cosine": float(np.nanmin(cosine)) if cosine.size else 1.0,
                "rows_below_cosine_threshold": int(bad_cosine.sum()),
            }
        )

        if bad_diff.any():
            keys = merged.loc[bad_diff, KEY_COLUMN].astype(str).head(max_examples).tolist()
            issues.append(f"Embedding absolute difference exceeds {tolerance}: {int(bad_diff.sum())} rows; examples={keys}")
        if bad_cosine.any():
            keys = merged.loc[bad_cosine, KEY_COLUMN].astype(str).head(max_examples).tolist()
            issues.append(
                f"Embedding cosine below {cosine_threshold}: {int(bad_cosine.sum())} rows; examples={keys}"
            )

    for name, value in stats.items():
        print(f"{name}: {value}")

    if issues:
        print("FAILED")
        for issue in issues:
            print(f"- {issue}")
        return False

    print("PASSED")
    return True


def main() -> int:
    args = parse_args()
    model = model_short_name(args.model)
    comparisons = (
        ("simple", f"stkcd_year_{model}_embeddings.csv"),
        ("citation-weighted", f"stkcd_year_citweighted_{model}_embeddings.csv"),
    )

    results = [
        compare_one(
            label=label,
            new_path=args.new_dir / filename,
            baseline_path=args.baseline_dir / filename,
            tolerance=args.tolerance,
            metadata_tolerance=args.metadata_tolerance,
            cosine_threshold=args.cosine_threshold,
            max_examples=args.max_examples,
        )
        for label, filename in comparisons
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
