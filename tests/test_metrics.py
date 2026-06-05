import pandas as pd

from rednote_aigt.evaluation.metrics import compute_binary_metrics, subgroup_metrics


def test_compute_binary_metrics_fixed_example():
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 0.5
    assert metrics["recall_ai"] == 0.5
    assert metrics["f1_ai"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["auroc"] == 0.75


def test_subgroup_metrics_handles_missing_optional_metadata_columns():
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "score_ai": [0.1, 0.9, 0.2, 0.8],
        }
    )
    assert subgroup_metrics(df, "domain").empty
