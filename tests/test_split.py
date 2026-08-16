import pandas as pd
import pytest

from rednote_aigt.data.prepare import deduplicate_texts
from rednote_aigt.data.split import (
    create_splits,
    detect_group_column,
    near_duplicate_leakage,
    text_fingerprint,
    validate_splits,
)


def make_df(rows_per_label=10):
    rows = []
    for label in [0, 1]:
        for i in range(rows_per_label):
            rows.append(
                {
                    "id": f"{label}_{i}",
                    "text": f"text {label} {i}",
                    "label": label,
                    "note_title": f"title {i}",
                    "note_content": f"content {i}",
                    "domain": "美食" if i % 2 == 0 else "穿搭",
                    "model_family": "gpt" if label == 1 else None,
                    "model": "gpt-o3" if label == 1 else None,
                    "source_file": "fixture.jsonl",
                    "text_len_chars": 10,
                    "title_only": False,
                }
            )
    return pd.DataFrame(rows)


def test_create_splits_preserves_both_labels_and_no_overlap():
    train_df, val_df, test_df, manifest = create_splits(make_df(), seed=7)
    assert set(train_df["label"]) == {0, 1}
    assert set(val_df["label"]) == {0, 1}
    assert set(test_df["label"]) == {0, 1}
    assert manifest["counts"]["train"] + manifest["counts"]["val"] + manifest["counts"]["test"] == 20
    validate_splits(train_df, val_df, test_df)


def test_validate_splits_rejects_duplicate_text_across_splits():
    df = make_df(rows_per_label=3)
    train_df = df.iloc[:2].copy()
    val_df = df.iloc[2:4].copy()
    test_df = df.iloc[4:].copy()
    test_df.loc[test_df.index[0], "text"] = train_df.iloc[0]["text"]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(train_df, val_df, test_df)


def test_manifest_reports_near_duplicate_leakage():
    manifest = create_splits(make_df(), seed=7)[3]
    assert manifest["near_duplicate_leakage"]["rows_involved"] == 0
    assert manifest["stratify_column"] == "_label_domain_stratum"


def test_text_fingerprint_ignores_punctuation_whitespace_and_emoji():
    assert text_fingerprint("今天  真的，绝了！😊") == text_fingerprint("今天真的绝了")
    assert text_fingerprint("今天真的绝了") != text_fingerprint("昨天真的绝了")


def test_near_duplicate_leakage_finds_decoration_only_differences():
    train = pd.DataFrame({"text": ["标题：今天真的绝了", "另一条"]})
    val = pd.DataFrame({"text": ["标题：今天真的绝了！！😊"]})
    test = pd.DataFrame({"text": ["完全不同的内容"]})
    report = near_duplicate_leakage({"train": train, "val": val, "test": test})
    assert report["train_val_shared_fingerprints"] == 1
    assert report["train_test_shared_fingerprints"] == 0
    assert report["rows_involved"] == 2


def test_detect_group_column_checks_every_candidate():
    # A sparse post_id must not stop the scan before the usable note_id.
    df = pd.DataFrame(
        {
            "post_id": ["p1", "", "", ""],
            "note_id": ["n1", "n1", "n2", "n2"],
        }
    )
    assert detect_group_column(df)[0] == "note_id"


def test_detect_group_column_rejects_row_unique_ids():
    df = pd.DataFrame({"note_id": ["n1", "n2", "n3"]})
    column, note = detect_group_column(df)
    assert column is None
    assert "duplicate_groups=0" in note


def test_deduplicate_texts_removes_conflicting_labels(tmp_path):
    df = make_df(rows_per_label=3)
    conflict = df.iloc[[0]].copy()
    conflict["label"] = 1
    combined = pd.concat([df, conflict], ignore_index=True)
    deduped, report = deduplicate_texts(combined, tmp_path)
    assert report["conflict_text_values"] == 1
    assert conflict.iloc[0]["text"] not in set(deduped["text"])
    assert (tmp_path / "label_conflicts.csv").exists()
