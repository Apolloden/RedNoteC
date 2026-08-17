import pandas as pd
import pytest

from rednote_aigt.evaluation.generator_holdout import (
    attach_in_distribution_baseline,
    build_holdout_fold,
    build_size_matched_control,
    list_generator_families,
)


def make_split(n_human: int, families: dict[str, int]) -> pd.DataFrame:
    rows = [{"text": f"human {i}", "label": 0, "model_family": None} for i in range(n_human)]
    for family, count in families.items():
        rows.extend({"text": f"{family} {i}", "label": 1, "model_family": family} for i in range(count))
    return pd.DataFrame(rows)


def test_lists_families_largest_first():
    df = make_split(5, {"gpt": 3, "glm": 7, "qwen": 5})
    assert list_generator_families(df) == ["glm", "qwen", "gpt"]


def test_held_out_family_is_absent_from_train_and_validation():
    """The point of the experiment: the fold model must never see the family."""
    train = make_split(10, {"gpt": 4, "glm": 3})
    val = make_split(4, {"gpt": 2, "glm": 2})
    test = make_split(6, {"gpt": 3, "glm": 5})

    fold_train, fold_val, fold_test = build_holdout_fold(train, val, test, "glm")

    assert "glm" not in set(fold_train["model_family"].dropna())
    assert "glm" not in set(fold_val["model_family"].dropna())
    # ...and the fold is scored on that family alone, against all human rows.
    assert set(fold_test.loc[fold_test["label"] == 1, "model_family"]) == {"glm"}
    assert int((fold_test["label"] == 1).sum()) == 5
    assert int((fold_test["label"] == 0).sum()) == 6


def test_human_rows_are_untouched_by_the_holdout():
    train = make_split(10, {"gpt": 4, "glm": 3})
    fold_train, _, fold_test = build_holdout_fold(train, train, train, "glm")
    assert int((fold_train["label"] == 0).sum()) == 10
    assert int((fold_test["label"] == 0).sum()) == 10


def test_other_families_remain_available_for_training():
    train = make_split(5, {"gpt": 4, "glm": 3, "qwen": 2})
    fold_train, _, _ = build_holdout_fold(train, train, train, "glm")
    assert set(fold_train["model_family"].dropna()) == {"gpt", "qwen"}


def test_missing_family_column_is_an_explicit_error():
    with pytest.raises(ValueError, match="model_family"):
        list_generator_families(pd.DataFrame({"text": ["a"], "label": [1]}))


def test_baseline_join_computes_the_recall_drop(tmp_path):
    summary = pd.DataFrame({"held_out_family": ["glm", "qwen"], "recall_ai_holdout": [0.50, 0.80]})
    subgroup = tmp_path / "subgroup_metrics_model_family.csv"
    pd.DataFrame({"model_family": ["glm", "qwen"], "recall_ai": [0.90, 1.00]}).to_csv(subgroup, index=False)

    merged = attach_in_distribution_baseline(summary, subgroup).set_index("held_out_family")

    assert merged.loc["glm", "recall_drop"] == pytest.approx(0.40)
    assert merged.loc["qwen", "recall_drop"] == pytest.approx(0.20)


def test_baseline_join_is_optional(tmp_path):
    summary = pd.DataFrame({"held_out_family": ["glm"], "recall_ai_holdout": [0.5]})
    merged = attach_in_distribution_baseline(summary, tmp_path / "absent.csv")
    assert "recall_drop" not in merged.columns


def test_size_matched_control_removes_as_many_rows_but_keeps_the_family():
    """Control isolates data volume: same AI count, family still present."""
    train = make_split(20, {"gpt": 10, "glm": 6, "qwen": 4})
    holdout_train, _, _ = build_holdout_fold(train, train, train, "glm")
    control_train, _, _ = build_size_matched_control(train, train, train, "glm", seed=0)

    assert int((control_train["label"] == 1).sum()) == int((holdout_train["label"] == 1).sum())
    assert "glm" in set(control_train["model_family"].dropna())
    assert int((control_train["label"] == 0).sum()) == 20


def test_size_matched_control_is_deterministic():
    train = make_split(20, {"gpt": 10, "glm": 6, "qwen": 4})
    first, _, _ = build_size_matched_control(train, train, train, "glm", seed=7)
    second, _, _ = build_size_matched_control(train, train, train, "glm", seed=7)
    assert list(first["text"]) == list(second["text"])


def test_control_and_holdout_share_the_same_test_rows():
    train = make_split(8, {"gpt": 5, "glm": 3})
    _, _, holdout_test = build_holdout_fold(train, train, train, "glm")
    _, _, control_test = build_size_matched_control(train, train, train, "glm")
    assert list(holdout_test["text"]) == list(control_test["text"])


def test_baseline_join_works_for_control_folds(tmp_path):
    """Control summaries use a different recall column and must still join."""
    summary = pd.DataFrame({"held_out_family": ["glm"], "recall_ai_control": [0.85]})
    subgroup = tmp_path / "subgroup_metrics_model_family.csv"
    pd.DataFrame({"model_family": ["glm"], "recall_ai": [0.90]}).to_csv(subgroup, index=False)

    merged = attach_in_distribution_baseline(summary, subgroup)

    assert merged.loc[0, "recall_drop"] == pytest.approx(0.05)
