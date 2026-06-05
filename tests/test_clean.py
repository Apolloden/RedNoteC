import pandas as pd

from rednote_aigt.data.clean import build_canonical_text, clean_posts, normalize_text


def test_normalize_text_preserves_emojis_hashtags_and_punctuation():
    text = normalize_text("  你好   #种草  😊!!!\n\n\n第二行  ")
    assert text == "你好 #种草 😊!!!\n\n第二行"


def test_normalize_text_handles_none_and_nan():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_build_canonical_text_from_title_and_content():
    text, title_only = build_canonical_text("标题", "正文")
    assert text == "标题：标题\n正文：正文"
    assert title_only is False


def test_build_canonical_text_handles_empty_title_and_title_only():
    content_text, content_title_only = build_canonical_text("", "正文")
    assert content_text == "正文：正文"
    assert content_title_only is False

    title_text, title_title_only = build_canonical_text("标题", "")
    assert title_text == "标题：标题"
    assert title_title_only is True


def test_clean_posts_drops_rows_with_no_text():
    df = pd.DataFrame(
        [
            {"note_title": " 标题 ", "note_content": " 内容 ", "label": 0},
            {"note_title": " ", "note_content": None, "label": 1},
        ]
    )
    cleaned, report = clean_posts(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "标题：标题\n正文：内容"
    assert report["dropped_empty_text_rows"] == 1
