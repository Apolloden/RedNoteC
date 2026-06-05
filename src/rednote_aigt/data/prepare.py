"""End-to-end data preparation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rednote_aigt.data.audit import build_audit_report, save_audit_report
from rednote_aigt.data.clean import clean_posts
from rednote_aigt.data.load import load_training_data
from rednote_aigt.data.split import create_splits, save_splits
from rednote_aigt.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def prepare_data(config: dict[str, Any], sample: int | None = None, force: bool = False) -> dict[str, Any]:
    paths = config.get("paths", {})
    split_cfg = config.get("split", {})
    cleaning_cfg = config.get("cleaning", {})

    human_file = Path(paths.get("human_file", "data/raw/training_set_human.jsonl"))
    aigc_file = Path(paths.get("aigc_file", "data/raw/training_set_aigc.jsonl"))
    interim_dir = ensure_dir(Path(paths.get("interim_dir", "data/interim")))
    processed_dir = ensure_dir(Path(paths.get("processed_dir", "data/processed")))
    reports_dir = ensure_dir(Path(paths.get("reports_dir", "outputs/reports")))

    raw_df, load_reports = load_training_data(human_file, aigc_file, sample=sample)
    cleaned_df, cleaning_report = clean_posts(
        raw_df,
        min_title_only_chars=int(cleaning_cfg.get("min_title_only_chars", 1)),
    )

    audit_report = build_audit_report(cleaned_df, load_reports=load_reports)
    audit_report["cleaning_report"] = cleaning_report
    audit_json, audit_md = save_audit_report(audit_report, reports_dir)

    deduped_df, dedupe_report = deduplicate_texts(cleaned_df, interim_dir)
    cleaned_all_path = interim_dir / "cleaned_all.csv"
    deduped_df.to_csv(cleaned_all_path, index=False)

    train_df, val_df, test_df, manifest = create_splits(
        deduped_df,
        train_ratio=float(split_cfg.get("train", 0.70)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        test_ratio=float(split_cfg.get("test", 0.15)),
        seed=int(config.get("seed", 42)),
        prefer_domain=bool(split_cfg.get("prefer_domain_stratification", True)),
        min_stratum_count=int(split_cfg.get("min_stratum_count", 2)),
    )
    manifest["load_reports"] = [report.__dict__ for report in load_reports]
    manifest["cleaning_report"] = cleaning_report
    manifest["dedupe_report"] = dedupe_report
    save_splits(train_df, val_df, test_df, manifest, processed_dir, force=force)

    summary = {
        "raw_rows": int(len(raw_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "deduped_rows": int(len(deduped_df)),
        "splits": manifest["counts"],
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "cleaned_all": str(cleaned_all_path),
        "processed_dir": str(processed_dir),
        "dedupe_report": dedupe_report,
    }
    write_json(summary, reports_dir / "prepare_summary.json")
    return summary


def audit_data(config: dict[str, Any], sample: int | None = None) -> dict[str, Any]:
    paths = config.get("paths", {})
    reports_dir = ensure_dir(Path(paths.get("reports_dir", "outputs/reports")))
    raw_df, load_reports = load_training_data(
        Path(paths.get("human_file", "data/raw/training_set_human.jsonl")),
        Path(paths.get("aigc_file", "data/raw/training_set_aigc.jsonl")),
        sample=sample,
    )
    cleaned_df, cleaning_report = clean_posts(raw_df)
    report = build_audit_report(cleaned_df, load_reports=load_reports)
    report["cleaning_report"] = cleaning_report
    save_audit_report(report, reports_dir)
    return report


def deduplicate_texts(df: pd.DataFrame, interim_dir: Path) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Remove label conflicts and exact duplicate texts within labels."""
    ensure_dir(interim_dir)
    work = df.copy()

    label_counts_per_text = work.groupby("text")["label"].nunique()
    conflict_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    conflicts = work[work["text"].isin(conflict_texts)].copy()
    conflict_path = interim_dir / "label_conflicts.csv"
    if not conflicts.empty:
        conflicts.to_csv(conflict_path, index=False)
        LOGGER.warning("Quarantined %s rows with cross-label text conflicts to %s", len(conflicts), conflict_path)
    else:
        pd.DataFrame(columns=work.columns).to_csv(conflict_path, index=False)

    without_conflicts = work[~work["text"].isin(conflict_texts)].copy()
    before_dedupe = len(without_conflicts)
    deduped = without_conflicts.drop_duplicates(subset=["label", "text"], keep="first").copy()
    report: dict[str, int | str] = {
        "input_rows": int(len(df)),
        "conflict_text_values": int(len(conflict_texts)),
        "conflict_rows_removed": int(len(conflicts)),
        "within_label_duplicate_rows_removed": int(before_dedupe - len(deduped)),
        "output_rows": int(len(deduped)),
        "label_conflicts_path": str(conflict_path),
    }
    return deduped.reset_index(drop=True), report
