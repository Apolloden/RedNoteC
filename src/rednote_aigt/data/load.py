"""Load RedNote-Vibe JSONL files into a canonical dataframe."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

CONTENT_ALIASES = ("note_content", "desc")
ENGAGEMENT_ALIASES = {
    "likes": ("likes", "liked_count"),
    "collections": ("collections", "collected_count"),
    "comments": ("comments", "comments_count"),
}
OPTIONAL_COLUMNS = [
    "note_id",
    "local_time",
    "likes",
    "collections",
    "comments",
    "domain",
    "model_family",
    "model",
]
CANONICAL_COLUMNS = [
    "note_title",
    "note_content",
    "label",
    "source_file",
    "source_line",
    *OPTIONAL_COLUMNS,
]


@dataclass(frozen=True)
class LoadReport:
    path: str
    label: int
    rows_loaded: int
    malformed_rows: int
    missing_required_rows: int
    aliases_used: dict[str, str]


def load_rednote_file(path: Path, label: int, sample: int | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Load one JSONL file and attach a binary label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required raw file not found: {path}. Download RedNote-Vibe and place it under data/raw/."
        )

    rows: list[dict[str, Any]] = []
    malformed: list[tuple[int, str]] = []
    missing_required: list[tuple[int, str]] = []
    aliases_used: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if sample is not None and len(rows) >= sample:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append((line_no, str(exc)))
                continue
            if not isinstance(raw, dict):
                malformed.append((line_no, "JSON value is not an object"))
                continue

            try:
                rows.append(_canonicalize_record(raw, path.name, line_no, label, aliases_used))
            except ValueError as exc:
                missing_required.append((line_no, str(exc)))

    if malformed:
        preview = "; ".join(f"line {line}: {msg}" for line, msg in malformed[:3])
        raise ValueError(f"Malformed JSONL rows in {path}: {len(malformed)} rows. Examples: {preview}")
    if missing_required:
        preview = "; ".join(f"line {line}: {msg}" for line, msg in missing_required[:3])
        raise ValueError(f"Missing required fields in {path}: {len(missing_required)} rows. Examples: {preview}")

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    LOGGER.info("Loaded %s rows from %s with label=%s", len(df), path, label)
    report = LoadReport(
        path=str(path),
        label=label,
        rows_loaded=len(df),
        malformed_rows=len(malformed),
        missing_required_rows=len(missing_required),
        aliases_used=aliases_used,
    )
    return df, report


def load_training_data(human_file: Path, aigc_file: Path, sample: int | None = None) -> tuple[pd.DataFrame, list[LoadReport]]:
    """Load human and AIGC supervised training files."""
    human_df, human_report = load_rednote_file(human_file, label=0, sample=sample)
    aigc_df, aigc_report = load_rednote_file(aigc_file, label=1, sample=sample)
    df = pd.concat([human_df, aigc_df], ignore_index=True)
    return df, [human_report, aigc_report]


def _canonicalize_record(
    raw: dict[str, Any],
    source_file: str,
    source_line: int,
    label: int,
    aliases_used: dict[str, str],
) -> dict[str, Any]:
    if "note_title" not in raw:
        raise ValueError("required field 'note_title' is absent")
    content_key = _first_present(raw, CONTENT_ALIASES)
    if content_key is None:
        raise ValueError("required field 'note_content' is absent and alias 'desc' was not found")
    if content_key != "note_content":
        aliases_used["note_content"] = content_key

    row = {
        "note_title": raw.get("note_title"),
        "note_content": raw.get(content_key),
        "label": label,
        "source_file": source_file,
        "source_line": source_line,
        "note_id": raw.get("note_id"),
        "local_time": raw.get("local_time"),
        "domain": raw.get("domain"),
        "model_family": raw.get("model_family"),
        "model": raw.get("model"),
    }
    for canonical, aliases in ENGAGEMENT_ALIASES.items():
        key = _first_present(raw, aliases)
        row[canonical] = raw.get(key) if key is not None else None
        if key is not None and key != canonical:
            aliases_used[canonical] = key
    return row


def _first_present(raw: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in raw:
            return key
    return None
