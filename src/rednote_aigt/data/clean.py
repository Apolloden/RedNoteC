"""Conservative text cleaning and canonical text construction."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

SPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(value: object) -> str:
    """Normalize whitespace without removing punctuation, hashtags, or emojis."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return NEWLINE_RE.sub("\n\n", text)


def build_canonical_text(title: object, content: object) -> tuple[str, bool]:
    """Build the model input from title and content only."""
    clean_title = normalize_text(title)
    clean_content = normalize_text(content)
    if clean_title and clean_content:
        return f"标题：{clean_title}\n正文：{clean_content}", False
    if clean_content:
        return f"正文：{clean_content}", False
    if clean_title:
        return f"标题：{clean_title}", True
    return "", False


def clean_posts(df: pd.DataFrame, min_title_only_chars: int = 1) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean title/content and drop rows with no usable text."""
    cleaned = df.copy()
    cleaned["note_title"] = cleaned["note_title"].map(normalize_text)
    cleaned["note_content"] = cleaned["note_content"].map(normalize_text)

    built = cleaned.apply(lambda row: build_canonical_text(row["note_title"], row["note_content"]), axis=1)
    cleaned["text"] = [item[0] for item in built]
    cleaned["title_only"] = [item[1] for item in built]
    cleaned["text_len_chars"] = cleaned["text"].str.len()

    both_empty = cleaned["text"].eq("")
    short_title_only = cleaned["title_only"] & (cleaned["text_len_chars"] < min_title_only_chars)
    drop_mask = both_empty | short_title_only
    report = {
        "input_rows": int(len(df)),
        "dropped_empty_text_rows": int(both_empty.sum()),
        "dropped_short_title_only_rows": int(short_title_only.sum()),
        "kept_title_only_rows": int((cleaned["title_only"] & ~drop_mask).sum()),
        "output_rows": int((~drop_mask).sum()),
    }
    cleaned = cleaned.loc[~drop_mask].copy()
    cleaned["id"] = cleaned.apply(lambda row: stable_row_id(row["label"], row["text"]), axis=1)
    return cleaned, report


def stable_row_id(label: int, text: str) -> str:
    digest = hashlib.sha1(f"{label}\n{text}".encode("utf-8")).hexdigest()[:16]
    prefix = "ai" if int(label) == 1 else "human"
    return f"{prefix}_{digest}"
