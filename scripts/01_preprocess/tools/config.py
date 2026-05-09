"""Preprocessing stage constants."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

STKCD_COLUMN = "stkcd"
CITATION_COLUMN = "p_cite"

RAW_COLUMNS = [
    "股票代码",
    "newipzlid",
    "年份",
    "标题",
    "摘要",
    "申请日",
    "专利类型",
    "IPC",
    "被引证次数",
    "市",
    "市代码",
    "省",
    "省代码",
]

COLUMN_MAPPING = {
    "股票代码": "stkcd",
    "newipzlid": "p_id",
    "年份": "p_year",
    "标题": "p_tt",
    "摘要": "p_abs",
    "申请日": "p_date",
    "专利类型": "p_type",
    "IPC": "p_ipc",
    "被引证次数": "p_cite",
    "市": "city",
    "市代码": "city_code",
    "省": "province",
    "省代码": "province_code",
}

PATENT_TYPES = ("发明申请", "发明授权", "实用新型")
STOCK_PREFIXES = ("0", "3", "6")
