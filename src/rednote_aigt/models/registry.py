"""Simple model registry."""

from __future__ import annotations

from typing import Protocol


class ModelClass(Protocol):
    model_type: str


MODEL_REGISTRY: dict[str, str] = {
    "tfidf_logreg": "rednote_aigt.models.tfidf.TfidfLogRegClassifier",
    "transformer_roberta": "rednote_aigt.models.transformer.TransformerClassifier",
}


def get_model_class(model_name: str) -> type:
    """Return a model class by registry name."""
    if model_name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{model_name}'. Known models: {known}")
    module_name, class_name = MODEL_REGISTRY[model_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def get_model_type(model_name: str) -> str:
    cls = get_model_class(model_name)
    return getattr(cls, "model_type")
