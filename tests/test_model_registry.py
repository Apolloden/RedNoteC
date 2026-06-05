from rednote_aigt.models.registry import MODEL_REGISTRY, get_model_class


def test_registry_contains_required_models():
    assert "tfidf_logreg" in MODEL_REGISTRY
    assert "transformer_roberta" in MODEL_REGISTRY


def test_registry_returns_model_classes():
    assert get_model_class("tfidf_logreg").model_type == "tfidf"
    assert get_model_class("transformer_roberta").model_type == "transformer"
