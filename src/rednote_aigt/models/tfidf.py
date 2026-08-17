"""TF-IDF + class-weighted Logistic Regression baseline.

Character n-grams, not words: RedNote text is emoji-heavy, slangy, and
code-switched, so a word segmenter would itself be a confound. Characters need
no segmenter and still capture local surface patterns.

``class_weight="balanced"`` handles the 12% AI prevalence, at the cost of
probability calibration — scores are pushed toward the positive class, so this
model's 0.5 threshold is a more aggressive operating point than the
transformer's. Ranking metrics (AUROC, AUPRC) are unaffected; threshold
metrics should be read with that in mind.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from rednote_aigt.utils.io import ensure_dir, write_json
from rednote_aigt.utils.progress import log_heartbeat

LOGGER = logging.getLogger(__name__)


class TfidfLogRegClassifier:
    """Character n-gram TF-IDF baseline with probabilistic logistic regression."""

    model_type = "tfidf"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.pipeline = self._build_pipeline(self.config)

    def fit(self, texts: list[str], labels: list[int], show_progress: bool = True) -> "TfidfLogRegClassifier":
        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        LOGGER.info("TF-IDF: fitting vectorizer on %s texts", len(texts))
        documents = tqdm(texts, total=len(texts), desc="TF-IDF vectorizing", unit="doc") if show_progress else texts
        with log_heartbeat(LOGGER, "TF-IDF vectorizer fit_transform", interval_seconds=30):
            x_train = vectorizer.fit_transform(documents)
        LOGGER.info(
            "TF-IDF: vectorized train matrix shape=%s nnz=%s",
            x_train.shape,
            x_train.nnz,
        )
        LOGGER.info("TF-IDF: fitting LogisticRegression classifier")
        with tqdm(total=1, desc="LogReg fitting", unit="fit", disable=not show_progress) as progress:
            with log_heartbeat(LOGGER, "LogisticRegression fit", interval_seconds=30):
                classifier.fit(x_train, labels)
                progress.update(1)
        LOGGER.info("TF-IDF: classifier fit complete")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(
        self,
        texts: list[str],
        show_progress: bool = False,
        desc: str = "TF-IDF transforming",
    ) -> np.ndarray:
        if not show_progress:
            return self.pipeline.predict_proba(texts)
        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        documents = tqdm(texts, total=len(texts), desc=desc, unit="doc")
        with log_heartbeat(LOGGER, f"{desc} vectorizer transform", interval_seconds=30):
            x_texts = vectorizer.transform(documents)
        with log_heartbeat(LOGGER, f"{desc} predict_proba", interval_seconds=30):
            return classifier.predict_proba(x_texts)

    def score_ai(
        self,
        texts: list[str],
        show_progress: bool = False,
        desc: str = "TF-IDF scoring",
    ) -> np.ndarray:
        """Return P(AI) per text — column 1, because classes_ is sorted [0, 1]."""
        return self.predict_proba(texts, show_progress=show_progress, desc=desc)[:, 1]

    def save(self, path: Path, model_name: str = "tfidf_logreg") -> None:
        ensure_dir(path)
        LOGGER.info("TF-IDF: saving joblib pipeline to %s", path / "model.joblib")
        joblib.dump(self.pipeline, path / "model.joblib")
        write_json(
            {
                "model_name": model_name,
                "model_type": self.model_type,
                "artifact": "model.joblib",
                "model_input_columns": ["text"],
                "label_column": "label",
                "config": self.config,
            },
            path / "model_metadata.json",
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfLogRegClassifier":
        model = cls(config={})
        model.pipeline = joblib.load(path / "model.joblib")
        return model

    @staticmethod
    def _build_pipeline(config: dict[str, Any]) -> Pipeline:
        vectorizer_cfg = dict(config.get("vectorizer", {}))
        classifier_cfg = dict(config.get("classifier", {}))
        if "ngram_range" in vectorizer_cfg:
            vectorizer_cfg["ngram_range"] = tuple(vectorizer_cfg["ngram_range"])
        classifier_cfg.setdefault("solver", "liblinear")
        classifier_cfg.setdefault("max_iter", 2000)
        classifier_cfg.setdefault("class_weight", "balanced")
        classifier = LogisticRegression(**classifier_cfg)
        vectorizer = TfidfVectorizer(**vectorizer_cfg)
        return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
