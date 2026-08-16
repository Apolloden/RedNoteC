"""Dataset audit summaries and report writers.

Runs before any split exists. Its job is to surface the things that would
quietly invalidate a result: metadata columns that are label-determined,
duplicate or cross-label texts, and formatting skew such as title presence.
Nothing here changes the data — it only describes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rednote_aigt.utils.io import ensure_dir, write_json

SUSPICIOUS_MODEL_FEATURE_COLUMNS = [
    "local_time",
    "likes",
    "collections",
    "comments",
    "domain",
    "model_family",
    "model",
]


def build_audit_report(df: pd.DataFrame, load_reports: list[Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_rows": int(len(df)),
        "load_reports": [_dataclass_to_dict(item) for item in (load_reports or [])],
        "label_counts": _value_counts(df, "label"),
        "domain_counts_by_label": _nested_counts(df, ["label", "domain"]) if "domain" in df else {},
        "model_family_counts_aigc": _value_counts(df[df["label"] == 1], "model_family") if "model_family" in df else {},
        "model_counts_aigc": _value_counts(df[df["label"] == 1], "model") if "model" in df else {},
        "missing_field_counts": _missing_counts(df),
        "empty_counts": _empty_counts(df),
        "empty_title_counts_by_label": _empty_title_counts_by_label(df),
        "duplicate_text_counts": _duplicate_counts(df),
        "human_aigc_exact_text_overlap": _text_overlap(df),
        "length_stats_by_label": _length_stats(df, ["label"]),
        "length_stats_by_domain": _length_stats(df, ["domain"]) if "domain" in df else {},
        "suspicious_columns_not_for_model_features": [col for col in SUSPICIOUS_MODEL_FEATURE_COLUMNS if col in df.columns],
    }
    return report


def save_audit_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    ensure_dir(reports_dir)
    json_path = reports_dir / "data_audit.json"
    md_path = reports_dir / "data_audit.md"
    write_json(report, json_path)
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Data Audit",
        "",
        f"- Total rows: {report.get('total_rows', 0)}",
        f"- Label counts: `{report.get('label_counts', {})}`",
        f"- Duplicate texts: `{report.get('duplicate_text_counts', {})}`",
        f"- Human/AIGC exact text overlap: `{report.get('human_aigc_exact_text_overlap', {})}`",
        f"- Empty titles by label (0=human, 1=AI): `{report.get('empty_title_counts_by_label', {})}`",
        f"- Suspicious columns excluded from model features: `{report.get('suspicious_columns_not_for_model_features', [])}`",
        "",
        "## Missing Fields",
        "",
        "```json",
        _format_jsonish(report.get("missing_field_counts", {})),
        "```",
        "",
        "## Domain Counts By Label",
        "",
        "```json",
        _format_jsonish(report.get("domain_counts_by_label", {})),
        "```",
        "",
        "## AIGC Model Counts",
        "",
        "```json",
        _format_jsonish(
            {
                "model_family": report.get("model_family_counts_aigc", {}),
                "model": report.get("model_counts_aigc", {}),
            }
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("<missing>").value_counts(dropna=False).to_dict().items()}


def _nested_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    if not all(col in df for col in columns):
        return {}
    grouped = df.groupby(columns, dropna=False).size()
    result: dict[str, dict[str, int]] = {}
    for key_tuple, count in grouped.items():
        outer, inner = key_tuple
        result.setdefault(str(outer), {})[str(inner)] = int(count)
    return result


def _missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _empty_title_counts_by_label(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """How often each label is missing a title.

    The canonical text only gets a ``标题：`` line when a title exists, so a
    label-skewed title rate is a formatting cue a model can learn instead of
    writing style. This is the number that quantifies it.
    """
    if not {"note_title", "label"}.issubset(df.columns):
        return {}
    empty = df["note_title"].fillna("").astype(str).str.len().eq(0)
    result: dict[str, dict[str, float]] = {}
    for label, group in empty.groupby(df["label"]):
        rows = int(len(group))
        empty_rows = int(group.sum())
        result[str(label)] = {
            "rows": rows,
            "empty_title_rows": empty_rows,
            "empty_title_share": round(empty_rows / rows, 6) if rows else 0.0,
        }
    return result


def _empty_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for col in ["note_title", "note_content", "text"]:
        if col in df:
            counts[col] = int(df[col].fillna("").astype(str).str.len().eq(0).sum())
    return counts


def _duplicate_counts(df: pd.DataFrame) -> dict[str, Any]:
    if "text" not in df:
        return {}
    duplicated = df.duplicated("text", keep=False)
    by_label = df.loc[duplicated].groupby("label").size().astype(int).to_dict() if "label" in df and duplicated.any() else {}
    return {
        "duplicate_rows_overall": int(duplicated.sum()),
        "duplicate_text_values_overall": int(df["text"].duplicated().sum()),
        "duplicate_rows_by_label": {str(k): int(v) for k, v in by_label.items()},
    }


def _text_overlap(df: pd.DataFrame) -> dict[str, int]:
    if not {"text", "label"}.issubset(df.columns):
        return {}
    human = set(df.loc[df["label"] == 0, "text"])
    aigc = set(df.loc[df["label"] == 1, "text"])
    overlap = human & aigc
    return {
        "overlap_text_values": len(overlap),
        "rows_in_overlap": int(df["text"].isin(overlap).sum()),
    }


def _length_stats(df: pd.DataFrame, group_cols: list[str]) -> dict[str, dict[str, float]]:
    if "text_len_chars" not in df or not all(col in df for col in group_cols):
        return {}
    grouped = df.groupby(group_cols, dropna=False)["text_len_chars"].agg(["count", "mean", "median", "min", "max"])
    result: dict[str, dict[str, float]] = {}
    for key, row in grouped.iterrows():
        result[str(key)] = {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "min": int(row["min"]),
            "max": int(row["max"]),
        }
    return result


def _dataclass_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "__dataclass_fields__"):
        return {field: getattr(item, field) for field in item.__dataclass_fields__}
    if isinstance(item, dict):
        return item
    return {"value": str(item)}


def _format_jsonish(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
