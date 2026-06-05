"""Transformer sequence classifier wrapper."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rednote_aigt.utils.io import ensure_dir, write_json
from rednote_aigt.utils.device import empty_mps_cache_if_available, resolve_torch_device

LOGGER = logging.getLogger(__name__)


@dataclass
class TransformerLoadStatus:
    requested_model: str
    actual_model: str
    fallback_used: bool
    error: str | None = None


class TransformerClassifier:
    """Hugging Face AutoModelForSequenceClassification wrapper."""

    model_type = "transformer"
    smoke_fallback_candidates = [
        "hf-internal-testing/tiny-random-bert",
        "bert-base-chinese",
    ]

    def __init__(self, config: dict[str, Any] | None = None, model: Any = None, tokenizer: Any = None) -> None:
        self.config = config or {}
        self.model = model
        self.tokenizer = tokenizer
        self.load_status: TransformerLoadStatus | None = None

    def load_pretrained(self, allow_smoke_fallback: bool = False) -> TransformerLoadStatus:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        requested = self.config.get("pretrained_model_name", "hfl/chinese-roberta-wwm-ext")
        model_load_kwargs = self._model_load_kwargs()
        try:
            with self._safetensors_conversion_guard():
                self.tokenizer = AutoTokenizer.from_pretrained(requested)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    requested,
                    num_labels=2,
                    **model_load_kwargs,
                )
            self.load_status = TransformerLoadStatus(requested, requested, fallback_used=False)
            return self.load_status
        except Exception as exc:
            if not allow_smoke_fallback:
                raise RuntimeError(f"Could not load transformer model '{requested}': {exc}") from exc
            last_error = exc
            for candidate in self.smoke_fallback_candidates:
                try:
                    LOGGER.warning(
                        "Falling back from %s to %s for smoke testing only after load error: %s",
                        requested,
                        candidate,
                        exc,
                    )
                    with self._safetensors_conversion_guard():
                        self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            candidate,
                            num_labels=2,
                            **model_load_kwargs,
                        )
                    self.load_status = TransformerLoadStatus(
                        requested_model=requested,
                        actual_model=candidate,
                        fallback_used=True,
                        error=str(exc),
                    )
                    return self.load_status
                except Exception as fallback_exc:
                    last_error = fallback_exc
            raise RuntimeError(
                f"Could not load requested transformer '{requested}' or smoke-test fallbacks. "
                f"Last error: {last_error}"
            ) from last_error

    def _model_load_kwargs(self) -> dict[str, Any]:
        """Return optional from_pretrained kwargs.

        hfl/chinese-roberta-wwm-ext currently serves PyTorch weights but no
        model.safetensors file. Passing use_safetensors=False avoids a noisy
        Transformers background conversion probe against Hugging Face repo
        discussions, which can emit a 403 even though model loading succeeds.
        """
        kwargs: dict[str, Any] = {}
        if "use_safetensors" in self.config:
            kwargs["use_safetensors"] = bool(self.config["use_safetensors"])
        return kwargs

    @contextmanager
    def _safetensors_conversion_guard(self):
        key = "DISABLE_SAFETENSORS_CONVERSION"
        previous = os.environ.get(key)
        if bool(self.config.get("disable_safetensors_conversion", True)):
            os.environ[key] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def predict_scores(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int | None = None,
        device: str = "auto",
        prefer_mps: bool = True,
        prefer_cuda: bool = False,
    ) -> np.ndarray:
        import torch

        if self.model is None or self.tokenizer is None:
            raise ValueError("Transformer model/tokenizer are not loaded.")
        max_length = int(max_length or self.config.get("max_length", 256))
        torch_device = resolve_torch_device(device, prefer_mps=prefer_mps, prefer_cuda=prefer_cuda)
        LOGGER.info("Running transformer inference on %s", torch_device)
        self.model.to(torch_device)
        self.model.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(torch_device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                scores.append(probs.detach().cpu().numpy())
                if torch_device.type == "mps":
                    empty_mps_cache_if_available()
        return np.concatenate(scores) if scores else np.array([])

    def predict(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int | None = None,
        threshold: float = 0.5,
        device: str = "auto",
        prefer_mps: bool = True,
        prefer_cuda: bool = False,
    ) -> np.ndarray:
        scores = self.predict_scores(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            prefer_mps=prefer_mps,
            prefer_cuda=prefer_cuda,
        )
        return (scores >= threshold).astype(int)

    def save(self, path: Path, model_name: str = "transformer_roberta") -> None:
        ensure_dir(path)
        if self.model is None or self.tokenizer is None:
            raise ValueError("Cannot save unloaded transformer model.")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        write_json(
            {
                "model_name": model_name,
                "model_type": self.model_type,
                "model_input_columns": ["text"],
                "label_column": "label",
                "config": self.config,
                "load_status": self.load_status.__dict__ if self.load_status else None,
            },
            path / "model_metadata.json",
        )

    @classmethod
    def load(cls, path: Path) -> "TransformerClassifier":
        import json

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        metadata_path = path / "model_metadata.json"
        config: dict[str, Any] = {}
        if metadata_path.exists():
            config = json.loads(metadata_path.read_text(encoding="utf-8")).get("config", {})
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(config=config, model=model, tokenizer=tokenizer)
