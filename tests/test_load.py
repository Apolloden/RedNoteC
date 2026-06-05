import json

import pytest

from rednote_aigt.data.load import load_rednote_file, load_training_data


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_rednote_file_uses_human_desc_alias(tmp_path):
    path = tmp_path / "human.jsonl"
    write_jsonl(
        path,
        [
            {
                "note_title": "标题",
                "desc": "正文",
                "liked_count": 3,
                "collected_count": 2,
                "comments_count": 1,
                "domain": "美食",
            }
        ],
    )
    df, report = load_rednote_file(path, label=0)
    assert df.loc[0, "label"] == 0
    assert df.loc[0, "note_content"] == "正文"
    assert df.loc[0, "likes"] == 3
    assert report.aliases_used["note_content"] == "desc"


def test_load_training_data_adds_labels(tmp_path):
    human = tmp_path / "human.jsonl"
    aigc = tmp_path / "aigc.jsonl"
    write_jsonl(human, [{"note_title": "人", "desc": "正文"}])
    write_jsonl(aigc, [{"note_title": "AI", "note_content": "正文", "model_family": "gpt"}])
    df, reports = load_training_data(human, aigc)
    assert set(df["label"]) == {0, 1}
    assert len(reports) == 2


def test_load_rednote_file_fails_on_missing_required_content(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"note_title": "标题"}])
    with pytest.raises(ValueError, match="note_content"):
        load_rednote_file(path, label=1)
