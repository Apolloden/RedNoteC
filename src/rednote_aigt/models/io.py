"""Model loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

from rednote_aigt.models.tfidf import TfidfLogRegClassifier
from rednote_aigt.models.transformer import TransformerClassifier


def load_model(model_dir: Path):
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_type = metadata.get("model_type")
    if model_type == "tfidf":
        return TfidfLogRegClassifier.load(model_dir), metadata
    if model_type == "transformer":
        return TransformerClassifier.load(model_dir), metadata
    raise ValueError(f"Unsupported model_type in {metadata_path}: {model_type}")
