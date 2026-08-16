"""Train/validation/test splitting and leakage checks.

Two guarantees back the held-out numbers:

1. ``validate_splits`` raises if the *same* text value appears in two splits.
2. ``near_duplicate_leakage`` reports texts that are identical once whitespace,
   punctuation, and emoji are stripped — cheap protection against a "duplicate"
   that differs only by a trailing 😊. It reports rather than raises, because
   the right response depends on how many rows are involved.

Neither catches paraphrase-level overlap; see the limitations in README.md.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from rednote_aigt.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

SPLIT_COLUMNS = [
    "id",
    "text",
    "label",
    "note_title",
    "note_content",
    "domain",
    "model_family",
    "model",
    "source_file",
    "split",
    "text_len_chars",
    "title_only",
    "note_id",
    "local_time",
    "likes",
    "collections",
    "comments",
]
GROUP_CANDIDATES = ["original_id", "seed_id", "post_id", "note_id", "id"]

# Everything that is not a Unicode word character: whitespace, punctuation,
# emoji. CJK characters are word characters, so they survive.
NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    prefer_domain: bool = True,
    min_stratum_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    working = df.copy()
    group_col, group_note = detect_group_column(working)

    if group_col:
        LOGGER.info("Using group-aware split with column %s", group_col)
        train_df, val_df, test_df = _group_split(working, group_col, train_ratio, val_ratio, test_ratio, seed)
        split_strategy = "group"
        # Stratification is not applied on this path; do not claim it in the manifest.
        stratify_col, stratify_note = None, "not applicable: group-aware split"
    else:
        stratify_col, stratify_note = choose_stratification_column(working, prefer_domain, min_stratum_count)
        LOGGER.info("Using random stratified split: %s", stratify_col)
        train_df, val_df, test_df = _stratified_split(
            working,
            stratify_col,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
        )
        split_strategy = "stratified"

    train_df = _assign_split(train_df, "train")
    val_df = _assign_split(val_df, "val")
    test_df = _assign_split(test_df, "test")
    validate_splits(train_df, val_df, test_df)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    manifest = build_manifest(
        splits,
        seed=seed,
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        split_strategy=split_strategy,
        stratify_column=stratify_col,
        stratification_note=stratify_note,
        group_column=group_col,
        group_note=group_note,
    )
    manifest["near_duplicate_leakage"] = near_duplicate_leakage(splits)
    return train_df, val_df, test_df, manifest


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: dict[str, Any],
    processed_dir: Path,
    force: bool = False,
) -> None:
    ensure_dir(processed_dir)
    targets = [
        processed_dir / "train.csv",
        processed_dir / "val.csv",
        processed_dir / "test.csv",
        processed_dir / "dataset_manifest.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError("Processed outputs already exist. Use --force to overwrite: " + ", ".join(str(path) for path in existing))

    _select_columns(train_df).to_csv(processed_dir / "train.csv", index=False)
    _select_columns(val_df).to_csv(processed_dir / "val.csv", index=False)
    _select_columns(test_df).to_csv(processed_dir / "test.csv", index=False)
    write_json(manifest, processed_dir / "dataset_manifest.json")


def near_duplicate_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Count texts shared across splits once punctuation/whitespace is stripped.

    Exact duplicates are already removed before splitting, so any hit here is a
    near-duplicate: same words, different decoration. Reported in the dataset
    manifest so the number is visible instead of assumed to be zero.
    """
    fingerprints = {name: split_df["text"].map(text_fingerprint) for name, split_df in splits.items()}
    unique = {name: set(values) for name, values in fingerprints.items()}
    pairs = {
        "train_val": unique["train"] & unique["val"],
        "train_test": unique["train"] & unique["test"],
        "val_test": unique["val"] & unique["test"],
    }
    report: dict[str, Any] = {f"{name}_shared_fingerprints": len(values) for name, values in pairs.items()}
    shared = set().union(*pairs.values())
    report["rows_involved"] = int(sum(int(values.isin(shared).sum()) for values in fingerprints.values()))
    if shared:
        LOGGER.warning(
            "Near-duplicate texts cross split boundaries: %s fingerprints, %s rows. Held-out metrics may be optimistic.",
            len(shared),
            report["rows_involved"],
        )
    return report


def text_fingerprint(text: object) -> str:
    """Hash a text after removing whitespace, punctuation, and emoji."""
    stripped = NON_WORD_RE.sub("", str(text)).casefold()
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise if a split is empty, missing a label, or shares an exact text with another."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    empty = [name for name, split_df in splits.items() if split_df.empty]
    if empty:
        raise ValueError(f"Empty split(s): {empty}")

    text_sets = {name: set(split_df["text"]) for name, split_df in splits.items()}
    overlaps = {
        "train_val": text_sets["train"] & text_sets["val"],
        "train_test": text_sets["train"] & text_sets["test"],
        "val_test": text_sets["val"] & text_sets["test"],
    }
    bad = {name: len(values) for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(f"Exact duplicate text leakage across splits: {bad}")

    for name, split_df in splits.items():
        labels = set(split_df["label"])
        if labels != {0, 1}:
            raise ValueError(f"Split {name} does not contain both labels: {sorted(labels)}")


def choose_stratification_column(
    df: pd.DataFrame,
    prefer_domain: bool,
    min_stratum_count: int,
) -> tuple[str, str]:
    working = df.copy()
    working["_label_stratum"] = working["label"].astype(str)
    if prefer_domain and "domain" in working.columns:
        working["_label_domain_stratum"] = working["label"].astype(str) + "__" + working["domain"].fillna("<missing>").astype(str)
        counts = working["_label_domain_stratum"].value_counts()
        if not counts.empty and int(counts.min()) >= max(4, min_stratum_count):
            return "_label_domain_stratum", "stratified by label+domain"
        LOGGER.warning("Falling back to label-only stratification because label+domain strata are too small.")
    return "_label_stratum", "stratified by label only"


def detect_group_column(df: pd.DataFrame) -> tuple[str | None, str]:
    """Find a column that ties related rows together, so they stay in one split.

    A column qualifies only if it is populated for essentially every row and
    actually repeats — a unique id per row groups nothing. Every candidate is
    examined before giving up, and the reasons are returned for the manifest.
    """
    rejected: list[str] = []
    for column in GROUP_CANDIDATES:
        if column not in df.columns or column == "id":
            continue
        values = df[column].fillna("").astype(str)
        non_empty_mask = values.str.len().gt(0)
        coverage = float(non_empty_mask.mean()) if len(values) else 0.0
        duplicate_groups = int(values.loc[non_empty_mask].duplicated().sum())
        if coverage >= 0.95 and duplicate_groups > 0:
            return column, f"using group column {column}"
        if coverage > 0:
            rejected.append(f"{column} (coverage={coverage:.3f}, duplicate_groups={duplicate_groups})")
    if rejected:
        return None, ("no usable group column; candidates need coverage >= 0.95 and repeated values, but found: " + "; ".join(rejected))
    return None, "no stable original post id/seed id/post id column found for group-aware splitting"


def build_manifest(
    splits: dict[str, pd.DataFrame],
    seed: int,
    ratios: dict[str, float],
    split_strategy: str,
    stratify_column: str | None,
    stratification_note: str,
    group_column: str | None,
    group_note: str,
) -> dict[str, Any]:
    """Describe how the splits were produced, for the record written next to them."""
    return {
        "seed": seed,
        "ratios": ratios,
        "split_strategy": split_strategy,
        "stratify_column": stratify_column,
        "stratification_note": stratification_note,
        "group_column": group_column,
        "group_note": group_note,
        "counts": {name: int(len(df)) for name, df in splits.items()},
        "label_counts": {name: _counts(df, "label") for name, df in splits.items()},
        "domain_counts": {name: _counts(df, "domain") for name, df in splits.items()},
        "model_family_counts_aigc": {name: _counts(df[df["label"] == 1], "model_family") for name, df in splits.items()},
        "model_counts_aigc": {name: _counts(df[df["label"] == 1], "model") for name, df in splits.items()},
        "model_input_columns": ["text"],
        "label_column": "label",
        "metadata_columns_not_for_model_features": [
            "local_time",
            "likes",
            "collections",
            "comments",
            "domain",
            "model_family",
            "model",
        ],
    }


def _stratified_split(
    df: pd.DataFrame,
    stratify_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    if stratify_col not in working:
        if stratify_col == "_label_stratum":
            working[stratify_col] = working["label"].astype(str)
        elif stratify_col == "_label_domain_stratum":
            working[stratify_col] = working["label"].astype(str) + "__" + working["domain"].fillna("<missing>").astype(str)

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        working,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=working[stratify_col],
    )
    relative_test = test_ratio / temp_ratio
    temp_stratify = temp_df[stratify_col]
    if temp_stratify.value_counts().min() < 2:
        LOGGER.warning("Validation/test split falls back to label stratification after first split.")
        temp_stratify = temp_df["label"]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )
    return _drop_internal_cols(train_df), _drop_internal_cols(val_df), _drop_internal_cols(test_df)


def _group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_ratio = val_ratio + test_ratio
    splitter = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    relative_test = test_ratio / temp_ratio
    temp_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    val_idx, test_idx = next(temp_splitter.split(temp_df, groups=temp_df[group_col]))
    return train_df.copy(), temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()


def _assign_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_name
    return out


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive.")


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in SPLIT_COLUMNS if col in df.columns]
    return df.loc[:, cols]


def _drop_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in df.columns if col.startswith("_")], errors="ignore")


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("<missing>").value_counts(dropna=False).to_dict().items()}
