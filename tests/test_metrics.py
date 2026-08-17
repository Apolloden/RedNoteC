import pandas as pd
import pytest

from rednote_aigt.evaluation.metrics import compute_binary_metrics, subgroup_metrics, wilson_interval


def test_compute_binary_metrics_fixed_example():
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 0.5
    assert metrics["recall_ai"] == 0.5
    assert metrics["f1_ai"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["auroc"] == 0.75


def test_macro_averages_use_both_labels_on_single_class_slices():
    """A perfectly classified AI-only slice must not report macro_f1 = 1.0.

    Guards the generator subgroup tables: sklearn's macro average would only
    span the labels it observes, turning "all AI rows found" into a perfect
    two-class score.
    """
    metrics = compute_binary_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert metrics["recall_ai"] == 1.0
    assert metrics["f1_ai"] == 1.0
    assert metrics["f1_human"] == 0.0
    assert metrics["macro_f1"] == 0.5
    assert metrics["macro_precision"] == 0.5
    assert metrics["macro_recall"] == 0.5
    assert metrics["weighted_f1"] == 1.0
    assert metrics["auroc"] is None
    assert metrics["support_human"] == 0


def test_subgroup_metrics_handles_missing_optional_metadata_columns():
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "score_ai": [0.1, 0.9, 0.2, 0.8],
        }
    )
    assert subgroup_metrics(df, "domain").empty


def test_subgroup_metrics_scores_each_group_independently():
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "score_ai": [0.1, 0.9, 0.1, 0.2],
            "domain": ["美食", "美食", "学习", "学习"],
        }
    )
    result = subgroup_metrics(df, "domain").set_index("domain")
    assert result.loc["美食", "recall_ai"] == 1.0
    assert result.loc["学习", "recall_ai"] == 0.0
    assert list(result["rows"]) == [2, 2]


def test_wilson_interval_matches_the_closed_form():
    # Worked example: 10 successes in 100 trials -> (0.0552, 0.1744).
    low, high = wilson_interval(10, 100)
    assert low == pytest.approx(0.05523, abs=1e-4)
    assert high == pytest.approx(0.17437, abs=1e-4)


def test_wilson_interval_stays_inside_zero_one_at_the_boundary():
    """The normal approximation collapses to (1.0, 1.0) here; Wilson must not."""
    low, high = wilson_interval(105, 105)
    assert high == pytest.approx(1.0)
    assert 0.96 < low < 1.0


def test_wilson_interval_widens_on_smaller_samples():
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(90, 100)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_wilson_interval_handles_empty_input():
    assert wilson_interval(0, 0) == (0.0, 0.0)
