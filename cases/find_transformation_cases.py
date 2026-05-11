#!/usr/bin/env python3
"""Find representative firm-year technology transformation cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MINI_FILE = Path("output/stkcd_year_similarity_merged_minilm.csv")
DEFAULT_DIST_FILE = Path("output/stkcd_year_similarity_merged_distiluse.csv")
DEFAULT_META_FILE = Path("data/stkcd_info.csv")
DEFAULT_CANDIDATE_OUTPUT = Path("cases/transformation_case_candidates.csv")
DEFAULT_TOP_OUTPUT = Path("cases/transformation_case_top5.csv")

SIM_COLS = [
    "cos_sim_lag1",
    "cos_sim_lag3",
    "cos_sim_cumulative",
    "cos_sim_lag1_citw",
    "cos_sim_lag3_citw",
    "cos_sim_cumulative_citw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify representative firm-year transformation cases from similarity outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top firm-year cases to keep.")
    parser.add_argument(
        "--min-event-patents",
        type=int,
        default=8,
        help="Minimum patents and non-empty texts required in the event year.",
    )
    parser.add_argument(
        "--min-prev-patents",
        type=int,
        default=3,
        help="Minimum patents required in each of the previous two observed years.",
    )
    parser.add_argument(
        "--max-lag1-mini",
        type=float,
        default=0.50,
        help="Maximum allowed MiniLM lag-1 similarity in the event year.",
    )
    parser.add_argument(
        "--max-lag1-dist",
        type=float,
        default=0.50,
        help="Maximum allowed DistilUSE lag-1 similarity in the event year.",
    )
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.20,
        help="Minimum drop from the previous-two-year mean required in both models.",
    )
    parser.add_argument(
        "--require-consecutive-years",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the previous two and next one observed rows to be consecutive calendar years.",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT,
        help="Path for the full filtered candidate table.",
    )
    parser.add_argument(
        "--top-output",
        type=Path,
        default=DEFAULT_TOP_OUTPUT,
        help="Path for the top-ranked firm-level cases.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def to_numeric_stkcd(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    stripped = text.lstrip("0")
    normalized = stripped if stripped else "0"
    if normalized.isdigit():
        return int(normalized)
    return None


def format_stkcd(value: object) -> str:
    numeric = to_numeric_stkcd(value)
    if numeric is None:
        return ""
    return f"{numeric:06d}"


def load_similarity(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"stkcd", "p_year", "n_patents", "n_texts_used", *SIM_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")

    keep_cols = ["stkcd", "p_year", "n_patents", "n_texts_used", *SIM_COLS]
    df = df[keep_cols].copy()
    df["stkcd_num"] = df["stkcd"].map(to_numeric_stkcd)
    df = df[df["stkcd_num"].notna()].copy()
    df["stkcd_num"] = df["stkcd_num"].astype(int)
    df["p_year"] = pd.to_numeric(df["p_year"], errors="coerce").astype("Int64")
    df = df[df["p_year"].notna()].copy()
    df["p_year"] = df["p_year"].astype(int)

    rename_map = {"n_patents": f"n_patents_{suffix}", "n_texts_used": f"n_texts_used_{suffix}"}
    rename_map.update({col: f"{col}_{suffix}" for col in SIM_COLS})
    return df.rename(columns=rename_map)


def load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported metadata format for {path}. Expected .csv, .xls, or .xlsx")
    required = {"stkcd", "year", "province", "city", "Ind"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")

    df = df[["stkcd", "year", "province", "city", "Ind"]].copy()
    df["stkcd_num"] = df["stkcd"].map(to_numeric_stkcd)
    df = df[df["stkcd_num"].notna()].copy()
    df["stkcd_num"] = df["stkcd_num"].astype(int)
    df["p_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["p_year"].notna()].copy()
    df["p_year"] = df["p_year"].astype(int)
    return df.rename(columns={"Ind": "industry_code"})[
        ["stkcd_num", "p_year", "province", "city", "industry_code"]
    ]


def merge_inputs(mini_path: Path, dist_path: Path, meta_path: Path) -> pd.DataFrame:
    mini = load_similarity(mini_path, "minilm")
    dist = load_similarity(dist_path, "distiluse")
    merged = mini.merge(dist, on=["stkcd", "stkcd_num", "p_year"], how="inner")

    if not merged["n_patents_minilm"].equals(merged["n_patents_distiluse"]):
        raise ValueError("n_patents mismatch between MiniLM and DistilUSE files")
    if not merged["n_texts_used_minilm"].equals(merged["n_texts_used_distiluse"]):
        raise ValueError("n_texts_used mismatch between MiniLM and DistilUSE files")

    merged["n_patents"] = merged["n_patents_minilm"]
    merged["n_texts_used"] = merged["n_texts_used_minilm"]
    merged["stkcd"] = merged["stkcd_num"].map(format_stkcd)

    meta = load_metadata(meta_path)
    merged = merged.merge(meta, on=["stkcd_num", "p_year"], how="left")
    return merged.sort_values(["stkcd_num", "p_year"]).reset_index(drop=True)


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("stkcd_num", sort=False)

    shift_cols = {
        "p_year": "year",
        "n_patents": "n_patents",
        "n_texts_used": "n_texts_used",
        "cos_sim_lag1_minilm": "cos_sim_lag1_minilm",
        "cos_sim_lag1_distiluse": "cos_sim_lag1_distiluse",
    }

    for col, prefix in shift_cols.items():
        for k in (1, 2):
            out[f"prev_{prefix}_{k}"] = group[col].shift(k)
        out[f"next_{prefix}_1"] = group[col].shift(-1)

    out["year_gap_prev1"] = out["p_year"] - out["prev_year_1"]
    out["year_gap_prev2"] = out["prev_year_1"] - out["prev_year_2"]
    out["year_gap_next1"] = out["next_year_1"] - out["p_year"]

    out["prev_lag1_complete"] = out[
        [
            "prev_cos_sim_lag1_minilm_1",
            "prev_cos_sim_lag1_minilm_2",
            "prev_cos_sim_lag1_distiluse_1",
            "prev_cos_sim_lag1_distiluse_2",
        ]
    ].notna().all(axis=1)
    out["next_lag1_complete"] = out[
        ["next_cos_sim_lag1_minilm_1", "next_cos_sim_lag1_distiluse_1"]
    ].notna().all(axis=1)

    out["prev_mean_lag1_minilm"] = (
        out["prev_cos_sim_lag1_minilm_1"] + out["prev_cos_sim_lag1_minilm_2"]
    ) / 2
    out["prev_mean_lag1_distiluse"] = (
        out["prev_cos_sim_lag1_distiluse_1"] + out["prev_cos_sim_lag1_distiluse_2"]
    ) / 2

    out["drop_from_prev_mean_minilm"] = out["prev_mean_lag1_minilm"] - out["cos_sim_lag1_minilm"]
    out["drop_from_prev_mean_distiluse"] = (
        out["prev_mean_lag1_distiluse"] - out["cos_sim_lag1_distiluse"]
    )

    out["severity_score"] = (1 - out["cos_sim_lag1_minilm"]) + (1 - out["cos_sim_lag1_distiluse"])
    out["break_score"] = (
        out["drop_from_prev_mean_minilm"].clip(lower=0)
        + out["drop_from_prev_mean_distiluse"].clip(lower=0)
    ) / 2
    out["score"] = 2 * out["severity_score"] + 1.5 * out["break_score"] + np.log1p(out["n_patents"])

    rebound = (
        (out["next_cos_sim_lag1_minilm_1"] >= 0.65)
        & (out["next_cos_sim_lag1_distiluse_1"] >= 0.50)
    )
    persist_low = (
        (out["next_cos_sim_lag1_minilm_1"] < 0.60)
        & (out["next_cos_sim_lag1_distiluse_1"] < 0.60)
    )
    out["post_confirmation_type"] = np.select(
        [rebound, persist_low],
        ["rebound", "persistent_low"],
        default="none",
    )
    return out


def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    mask &= df["n_patents"] >= args.min_event_patents
    mask &= df["n_texts_used"] >= args.min_event_patents
    mask &= df["cos_sim_lag1_minilm"] < args.max_lag1_mini
    mask &= df["cos_sim_lag1_distiluse"] < args.max_lag1_dist
    mask &= df["prev_lag1_complete"]
    mask &= df["next_lag1_complete"]
    mask &= df["prev_n_patents_1"] >= args.min_prev_patents
    mask &= df["prev_n_patents_2"] >= args.min_prev_patents
    mask &= df["prev_mean_lag1_minilm"] >= 0.70
    mask &= df["prev_mean_lag1_distiluse"] >= 0.55
    mask &= df["drop_from_prev_mean_minilm"] >= args.min_drop
    mask &= df["drop_from_prev_mean_distiluse"] >= args.min_drop
    mask &= df["post_confirmation_type"].isin(["rebound", "persistent_low"])

    if args.require_consecutive_years:
        mask &= df["year_gap_prev1"] == 1
        mask &= df["year_gap_prev2"] == 1
        mask &= df["year_gap_next1"] == 1

    candidates = df[mask].copy()
    candidates = candidates.sort_values(
        ["score", "n_patents", "p_year", "stkcd_num"],
        ascending=[False, False, True, True],
    )
    return candidates


def firm_level_top(candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    deduped = candidates.sort_values(
        ["stkcd_num", "score", "n_patents", "p_year"],
        ascending=[True, False, False, True],
    ).drop_duplicates(subset=["stkcd_num"], keep="first")
    return deduped.sort_values(
        ["score", "n_patents", "p_year", "stkcd_num"],
        ascending=[False, False, True, True],
    ).head(top_n)


def build_output_table(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output = output.rename(columns={"p_year": "event_year"})

    ordered_cols = [
        "stkcd",
        "event_year",
        "n_patents",
        "n_texts_used",
        "province",
        "city",
        "industry_code",
        "prev_year_2",
        "prev_year_1",
        "next_year_1",
        "year_gap_prev2",
        "year_gap_prev1",
        "year_gap_next1",
        "prev_n_patents_2",
        "prev_n_patents_1",
        "next_n_patents_1",
        "prev_mean_lag1_minilm",
        "prev_mean_lag1_distiluse",
        "cos_sim_lag1_minilm",
        "cos_sim_lag3_minilm",
        "cos_sim_cumulative_minilm",
        "cos_sim_lag1_distiluse",
        "cos_sim_lag3_distiluse",
        "cos_sim_cumulative_distiluse",
        "cos_sim_lag1_citw_minilm",
        "cos_sim_lag3_citw_minilm",
        "cos_sim_cumulative_citw_minilm",
        "cos_sim_lag1_citw_distiluse",
        "cos_sim_lag3_citw_distiluse",
        "cos_sim_cumulative_citw_distiluse",
        "drop_from_prev_mean_minilm",
        "drop_from_prev_mean_distiluse",
        "severity_score",
        "break_score",
        "score",
        "post_confirmation_type",
    ]

    available_cols = [col for col in ordered_cols if col in output.columns]
    extra_cols = [col for col in output.columns if col not in available_cols]
    return output[available_cols + extra_cols]


def write_trajectories(full_df: pd.DataFrame, selected: pd.DataFrame) -> None:
    cases_dir = repo_root() / "cases"
    for stkcd_num in selected["stkcd_num"].unique():
        stkcd = format_stkcd(stkcd_num)
        trajectory = full_df[full_df["stkcd_num"] == stkcd_num].copy()
        trajectory = trajectory.sort_values("p_year")
        trajectory.to_csv(cases_dir / f"company_{stkcd}_trajectory.csv", index=False)


def print_summary(candidates: pd.DataFrame, selected: pd.DataFrame) -> None:
    print(f"候选案例数: {len(candidates)}")
    print(f"候选企业数: {candidates['stkcd_num'].nunique()}")
    print()
    if selected.empty:
        print("未找到满足条件的代表性案例。")
        return

    print("Top cases:")
    preview_cols = [
        "stkcd",
        "p_year",
        "n_patents",
        "cos_sim_lag1_minilm",
        "cos_sim_lag1_distiluse",
        "drop_from_prev_mean_minilm",
        "drop_from_prev_mean_distiluse",
        "score",
        "post_confirmation_type",
    ]
    print(selected[preview_cols].to_string(index=False))


def main() -> int:
    args = parse_args()
    mini_path = resolve_path(DEFAULT_MINI_FILE)
    dist_path = resolve_path(DEFAULT_DIST_FILE)
    meta_path = resolve_path(DEFAULT_META_FILE)
    candidate_output = resolve_path(args.candidate_output)
    top_output = resolve_path(args.top_output)

    try:
        full_df = merge_inputs(mini_path, dist_path, meta_path)
        full_df = add_context_columns(full_df)
        candidates = filter_candidates(full_df, args)
        selected = firm_level_top(candidates, args.top_n)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    top_output.parent.mkdir(parents=True, exist_ok=True)

    build_output_table(candidates).to_csv(candidate_output, index=False)
    build_output_table(selected).to_csv(top_output, index=False)
    write_trajectories(full_df, selected)
    print_summary(candidates, selected)
    print()
    print(f"候选案例已写入: {candidate_output}")
    print(f"Top案例已写入: {top_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
