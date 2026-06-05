import pandas as pd
import pytest

from rednote_aigt.data.prepare import deduplicate_texts
from rednote_aigt.data.split import create_splits, validate_splits


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


def test_deduplicate_texts_removes_conflicting_labels(tmp_path):
    df = make_df(rows_per_label=3)
    conflict = df.iloc[[0]].copy()
    conflict["label"] = 1
    combined = pd.concat([df, conflict], ignore_index=True)
    deduped, report = deduplicate_texts(combined, tmp_path)
    assert report["conflict_text_values"] == 1
    assert conflict.iloc[0]["text"] not in set(deduped["text"])
    assert (tmp_path / "label_conflicts.csv").exists()
