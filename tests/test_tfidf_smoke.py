import pandas as pd
import pytest

from rednote_aigt.evaluation.evaluate import evaluate_model
from rednote_aigt.models.tfidf import TfidfLogRegClassifier
from rednote_aigt.training.train import FORBIDDEN_FEATURE_COLUMNS, load_training_frame, train_model


def tiny_tfidf_config():
    return {
        "model_type": "tfidf",
        "text_column": "text",
        "label_column": "label",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_features": 100,
        },
        "classifier": {
            "class_weight": "balanced",
            "max_iter": 200,
            "solver": "liblinear",
            "random_state": 42,
        },
    }


def test_tfidf_can_fit_and_predict_toy_dataset():
    texts = ["人类真实分享", "今天自己做饭", "AI生成模板内容", "智能生成文案"]
    labels = [0, 0, 1, 1]
    model = TfidfLogRegClassifier(tiny_tfidf_config())
    model.fit(texts, labels)
    preds = model.predict(texts)
    scores = model.score_ai(texts)
    assert set(preds).issubset({0, 1})
    assert len(scores) == 4


def test_training_loader_only_returns_text_and_label(tmp_path):
    path = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "text": ["human text", "ai text"],
            "label": [0, 1],
            "domain": ["leak", "leak"],
            "model": ["x", "y"],
            "likes": [999, 0],
        }
    )
    df.to_csv(path, index=False)
    loaded = load_training_frame(path, "text", "label", max_samples=None, seed=42)
    assert list(loaded.columns) == ["text", "label"]
    assert not any(col in loaded.columns for col in FORBIDDEN_FEATURE_COLUMNS)


def test_force_keeps_existing_model_when_the_run_cannot_start(tmp_path):
    """--force must not delete a trained model before the new run is viable."""
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "model.joblib").write_text("previous model")

    with pytest.raises(FileNotFoundError):
        train_model(
            model_name="tfidf_logreg",
            model_config=tiny_tfidf_config(),
            train_path=tmp_path / "missing_train.csv",
            val_path=tmp_path / "missing_val.csv",
            output_dir=output_dir,
            reports_dir=tmp_path / "reports",
            figures_dir=tmp_path / "figures",
            force=True,
        )

    assert (output_dir / "model.joblib").read_text() == "previous model"


def test_evaluation_handles_missing_optional_metadata_columns(tmp_path):
    model_dir = tmp_path / "model"
    reports_dir = tmp_path / "reports"
    figures_dir = tmp_path / "figures"
    test_path = tmp_path / "test.csv"
    config = tiny_tfidf_config()
    model = TfidfLogRegClassifier(config)
    texts = ["human text", "another human", "ai generated", "generated ai"]
    labels = [0, 0, 1, 1]
    model.fit(texts, labels)
    model.save(model_dir, model_name="tfidf_logreg")
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "split": ["test"] * 4,
            "text": texts,
            "label": labels,
        }
    ).to_csv(test_path, index=False)
    metrics = evaluate_model(model_dir, test_path, reports_dir, figures_dir, max_test_samples=None)
    assert "macro_f1" in metrics
    assert (reports_dir / "predictions.csv").exists()
    assert (figures_dir / "confusion_matrix.png").exists()
