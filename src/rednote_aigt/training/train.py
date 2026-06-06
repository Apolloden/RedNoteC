"""Training CLI implementation."""

from __future__ import annotations

import logging
import random
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rednote_aigt.evaluation.metrics import compute_binary_metrics
from rednote_aigt.models.registry import get_model_class
from rednote_aigt.training.plots import (
    plot_tfidf_train_val_loss,
    plot_tfidf_train_val_metrics,
    plot_transformer_train_val_loss,
    plot_transformer_train_val_metrics,
)
from rednote_aigt.utils.device import describe_torch_device, resolve_torch_device
from rednote_aigt.utils.io import ensure_dir, write_json
from rednote_aigt.utils.progress import log_heartbeat

LOGGER = logging.getLogger(__name__)

FORBIDDEN_FEATURE_COLUMNS = ["local_time", "likes", "collections", "comments", "domain", "model_family", "model"]

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformer training reports the real import error later.
    TrainerCallback = object


def train_model(
    model_name: str,
    model_config: dict[str, Any],
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
    force: bool = False,
    max_steps: int | None = None,
    allow_smoke_fallback: bool = False,
    device: str | None = None,
    prefer_mps: bool | None = None,
    prefer_cuda: bool | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    ensure_dir(reports_dir)
    ensure_dir(figures_dir)
    _prepare_output_dir(output_dir, force)
    set_seed(seed)

    resolved_config = dict(model_config)
    resolved_config["seed"] = seed
    if max_steps is not None:
        resolved_config["max_steps"] = max_steps
    if device is not None:
        resolved_config["device"] = device
    if prefer_mps is not None:
        resolved_config["prefer_mps"] = prefer_mps
    if prefer_cuda is not None:
        resolved_config["prefer_cuda"] = prefer_cuda
    if batch_size is not None:
        resolved_config["batch_size"] = batch_size
    if max_length is not None:
        resolved_config["max_length"] = max_length

    LOGGER.info("Loading training data from %s", train_path)
    train_df = load_training_frame(train_path, model_config["text_column"], model_config["label_column"], max_train_samples, seed)
    LOGGER.info("Loaded %s training rows", len(train_df))
    LOGGER.info("Loading validation data from %s", val_path)
    val_df = load_training_frame(val_path, model_config["text_column"], model_config["label_column"], max_val_samples, seed)
    LOGGER.info("Loaded %s validation rows", len(val_df))
    LOGGER.info("Saving resolved train config to %s and %s", reports_dir, output_dir)
    write_json(resolved_config, reports_dir / "train_config_resolved.json")
    write_json(resolved_config, output_dir / "train_config_resolved.json")

    if model_config["model_type"] == "tfidf":
        result = _train_tfidf(model_name, resolved_config, train_df, val_df, output_dir, reports_dir, figures_dir)
    elif model_config["model_type"] == "transformer":
        result = _train_transformer(
            model_name,
            resolved_config,
            train_df,
            val_df,
            output_dir,
            reports_dir,
            figures_dir,
            allow_smoke_fallback=allow_smoke_fallback,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_config['model_type']}")
    return result


def load_training_frame(
    path: Path,
    text_column: str,
    label_column: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    df = pd.read_csv(path)
    missing = [col for col in [text_column, label_column] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if max_samples is not None:
        df = stratified_sample(df, max_samples, seed=seed)
    out = df[[text_column, label_column]].copy()
    out[text_column] = out[text_column].fillna("").astype(str)
    out[label_column] = out[label_column].astype(int)
    if set(out[label_column]) != {0, 1}:
        raise ValueError(f"Training/evaluation sample from {path} must contain both labels.")
    return out


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    parts = []
    for _, group in df.groupby("label"):
        take = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(n=min(take, len(group)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    return out.head(n).copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _train_tfidf(
    model_name: str,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    with tqdm(total=7, desc="TF-IDF training stages", unit="stage") as stage_bar:
        cls = get_model_class(model_name)
        model = cls(config=config)
        text_col = config["text_column"]
        label_col = config["label_column"]
        LOGGER.info("Training TF-IDF model using X='%s' and y='%s'", text_col, label_col)
        stage_bar.set_postfix_str("fit vectorizer + classifier")
        model.fit(train_df[text_col].tolist(), train_df[label_col].tolist(), show_progress=True)
        stage_bar.update(1)

        LOGGER.info("TF-IDF: scoring train split")
        stage_bar.set_postfix_str("score train")
        train_scores = model.score_ai(train_df[text_col].tolist(), show_progress=True, desc="TF-IDF scoring train")
        stage_bar.update(1)

        LOGGER.info("TF-IDF: scoring validation split")
        stage_bar.set_postfix_str("score val")
        val_scores = model.score_ai(val_df[text_col].tolist(), show_progress=True, desc="TF-IDF scoring val")
        stage_bar.update(1)

        LOGGER.info("TF-IDF: computing train/validation metrics")
        stage_bar.set_postfix_str("metrics")
        metrics = {
            "train": compute_binary_metrics(train_df[label_col].to_numpy(), train_scores),
            "val": compute_binary_metrics(val_df[label_col].to_numpy(), val_scores),
        }
        stage_bar.update(1)

        LOGGER.info("TF-IDF: saving model artifacts")
        stage_bar.set_postfix_str("save model")
        model.save(output_dir, model_name=model_name)
        stage_bar.update(1)

        LOGGER.info("TF-IDF: saving training metrics")
        stage_bar.set_postfix_str("save metrics")
        write_json(metrics, reports_dir / "train_metrics.json")
        write_json(metrics, output_dir / "train_metrics.json")
        stage_bar.update(1)

        LOGGER.info("TF-IDF: saving training plots")
        stage_bar.set_postfix_str("save plots")
        plot_tfidf_train_val_metrics(metrics, figures_dir / "train_val_metrics.png")
        plot_tfidf_train_val_loss(metrics, figures_dir / "train_val_loss.png")
        stage_bar.update(1)
    LOGGER.info(
        "TF-IDF training complete: val macro_f1=%.4f val recall_ai=%.4f val log_loss=%.4f",
        metrics["val"]["macro_f1"],
        metrics["val"]["recall_ai"],
        metrics["val"]["log_loss"],
    )
    return metrics


def _train_transformer(
    model_name: str,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    allow_smoke_fallback: bool,
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

    from rednote_aigt.models.transformer import TransformerClassifier

    with tqdm(total=10, desc="Transformer training stages", unit="stage") as stage_bar:
        text_col = config["text_column"]
        label_col = config["label_column"]
        max_length = int(config.get("max_length", 256))
        LOGGER.info(
            "Transformer: training using X='%s' and y='%s' with max_length=%s",
            text_col,
            label_col,
            max_length,
        )

        stage_bar.set_postfix_str("select device")
        torch_device = resolve_torch_device(
            config.get("device", "auto"),
            prefer_mps=bool(config.get("prefer_mps", True)),
            prefer_cuda=bool(config.get("prefer_cuda", False)),
        )
        device_info = describe_torch_device()
        device_info["selected_for_training"] = str(torch_device)
        write_json(device_info, reports_dir / "device_info.json")
        LOGGER.info("Selected transformer training device: %s", torch_device)
        stage_bar.update(1)

        stage_bar.set_postfix_str("load model/tokenizer")
        classifier = TransformerClassifier(config=config)
        with log_heartbeat(LOGGER, "Transformer model/tokenizer load", interval_seconds=30):
            load_status = classifier.load_pretrained(allow_smoke_fallback=allow_smoke_fallback)
        classifier.model.to(torch_device)
        write_json(load_status.__dict__, reports_dir / "transformer_load_status.json")
        LOGGER.info(
            "Transformer: loaded model '%s' fallback_used=%s",
            load_status.actual_model,
            load_status.fallback_used,
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("create datasets")
        LOGGER.info("Transformer: creating Hugging Face datasets train=%s val=%s", len(train_df), len(val_df))
        train_ds = Dataset.from_pandas(train_df.rename(columns={label_col: "labels"}), preserve_index=False)
        val_ds = Dataset.from_pandas(val_df.rename(columns={label_col: "labels"}), preserve_index=False)
        stage_bar.update(1)

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return classifier.tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

        stage_bar.set_postfix_str("tokenize train")
        LOGGER.info("Transformer: tokenizing train split")
        train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
        stage_bar.update(1)

        stage_bar.set_postfix_str("tokenize val")
        LOGGER.info("Transformer: tokenizing validation split")
        val_ds = val_ds.map(tokenize, batched=True, desc="Tokenizing val")
        keep_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in keep_cols])
        val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in keep_cols])
        LOGGER.info("Transformer: tokenized columns train=%s val=%s", train_ds.column_names, val_ds.column_names)
        stage_bar.update(1)

        max_steps = int(config.get("max_steps", -1) or -1)
        eval_strategy = "steps" if max_steps > 0 else "epoch"
        eval_steps = 1 if max_steps > 0 else None
        save_steps = 1 if max_steps > 0 else int(config.get("save_steps", 500))
        stage_bar.set_postfix_str("training args")
        training_args_kwargs = build_training_arguments_kwargs(
            config=config,
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            max_steps=max_steps,
            selected_device=str(torch_device),
        )
        write_json(
            {
                "kwargs": training_args_kwargs,
                "selected_device": str(torch_device),
                "transformers_supported_args": sorted(inspect.signature(TrainingArguments.__init__).parameters),
            },
            reports_dir / "training_arguments_resolved.json",
        )
        LOGGER.info(
            "TrainingArguments: batch_size=%s grad_accum=%s epochs=%s max_steps=%s logging_steps=%s eval_strategy=%s",
            training_args_kwargs.get("per_device_train_batch_size"),
            training_args_kwargs.get("gradient_accumulation_steps"),
            training_args_kwargs.get("num_train_epochs"),
            training_args_kwargs.get("max_steps"),
            training_args_kwargs.get("logging_steps"),
            training_args_kwargs.get("eval_strategy"),
        )
        LOGGER.info(
            "TrainingArguments device settings: use_cpu=%s use_mps_device=%s no_cuda=%s fp16=%s bf16=%s",
            training_args_kwargs.get("use_cpu"),
            training_args_kwargs.get("use_mps_device"),
            training_args_kwargs.get("no_cuda"),
            training_args_kwargs.get("fp16"),
            training_args_kwargs.get("bf16"),
        )
        training_args = TrainingArguments(**training_args_kwargs)
        stage_bar.update(1)

        def compute_metrics(eval_pred) -> dict[str, float]:
            logits, labels = eval_pred
            logits = np.asarray(logits)
            probs = _softmax(logits)[:, 1]
            metrics = compute_binary_metrics(labels, probs)
            return {
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "recall_ai": metrics["recall_ai"],
                "average_precision": metrics["average_precision"] or 0.0,
            }

        trainer = Trainer(
            model=classifier.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=classifier.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                ProgressLoggingCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=int(config.get("early_stopping_patience", 2)),
                    early_stopping_threshold=float(config.get("early_stopping_threshold", 0.0)),
                ),
            ],
        )
        stage_bar.set_postfix_str("train")
        LOGGER.info("Transformer: starting Trainer.train(); Hugging Face progress bars will show batch/step progress")
        with log_heartbeat(LOGGER, "Transformer Trainer.train", interval_seconds=60):
            trainer.train()
        stage_bar.update(1)

        stage_bar.set_postfix_str("score train")
        LOGGER.info("Transformer: scoring train split for comparable train/validation metrics")
        train_pred = trainer.predict(train_ds, metric_key_prefix="train")
        train_scores = _softmax(np.asarray(train_pred.predictions))[:, 1]
        train_metrics = compute_binary_metrics(train_pred.label_ids, train_scores)
        stage_bar.update(1)

        stage_bar.set_postfix_str("score val")
        LOGGER.info("Transformer: scoring validation split for comparable train/validation metrics")
        val_pred = trainer.predict(val_ds, metric_key_prefix="val")
        val_scores = _softmax(np.asarray(val_pred.predictions))[:, 1]
        val_metrics = compute_binary_metrics(val_pred.label_ids, val_scores)
        eval_metrics = dict(val_pred.metrics)
        stage_bar.update(1)

        stage_bar.set_postfix_str("save")
        classifier.model = trainer.model
        LOGGER.info("Transformer: saving model/tokenizer to %s", output_dir)
        classifier.save(output_dir, model_name=model_name)
        metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "eval": eval_metrics,
            "load_status": load_status.__dict__,
            "log_history": trainer.state.log_history,
        }
        write_json(metrics, reports_dir / "train_metrics.json")
        write_json(metrics, output_dir / "train_metrics.json")
        plot_transformer_train_val_metrics(metrics, figures_dir / "train_val_metrics.png")
        plot_transformer_train_val_loss(metrics, figures_dir / "train_val_loss.png")
        deprecated_history_plot = figures_dir / "training_history.png"
        if deprecated_history_plot.exists():
            deprecated_history_plot.unlink()
        stage_bar.update(1)
    LOGGER.info(
        "Transformer training complete: val macro_f1=%.4f val recall_ai=%.4f val log_loss=%.4f",
        metrics["val"]["macro_f1"],
        metrics["val"]["recall_ai"],
        metrics["val"]["log_loss"],
    )
    return metrics


def build_training_arguments_kwargs(
    config: dict[str, Any],
    output_dir: Path,
    eval_strategy: str,
    eval_steps: int | None,
    save_steps: int,
    max_steps: int,
    selected_device: str,
) -> dict[str, Any]:
    """Build TrainingArguments kwargs, dropping unsupported keys defensively."""
    from transformers import TrainingArguments

    supported = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_checkpoints"),
        "per_device_train_batch_size": int(config.get("batch_size", 4)),
        "per_device_eval_batch_size": int(config.get("batch_size", 4)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(config.get("num_train_epochs", 1)),
        "max_steps": max_steps,
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.06)),
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_strategy": eval_strategy,
        "save_steps": save_steps,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.get("metric_for_best_model", "eval_macro_f1"),
        "greater_is_better": bool(config.get("greater_is_better", True)),
        "logging_steps": 1 if max_steps > 0 else int(config.get("logging_steps", 50)),
        "logging_first_step": True,
        "report_to": "none",
        "seed": int(config.get("seed", 42)),
        "fp16": bool(config.get("fp16", False)),
        "bf16": bool(config.get("bf16", False)),
        "dataloader_num_workers": int(config.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(config.get("dataloader_pin_memory", False)),
    }
    if selected_device == "cpu":
        kwargs["use_cpu"] = True
        kwargs["no_cuda"] = True
    elif selected_device == "mps":
        kwargs["use_cpu"] = False
        kwargs["no_cuda"] = bool(config.get("no_cuda", True))
        kwargs["use_mps_device"] = True
    elif selected_device == "cuda":
        kwargs["use_cpu"] = False
        kwargs["no_cuda"] = False

    filtered = {key: value for key, value in kwargs.items() if key in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        LOGGER.info("Dropped unsupported TrainingArguments keys for installed transformers: %s", dropped)
    return filtered


def _prepare_output_dir(path: Path, force: bool) -> None:
    ensure_dir(path)
    existing = [item for item in path.iterdir() if item.name != ".gitkeep"]
    if existing and not force:
        raise FileExistsError(f"Model output directory is not empty: {path}. Use --force to overwrite.")
    if force:
        for item in existing:
            if item.is_dir():
                import shutil

                shutil.rmtree(item)
            else:
                item.unlink()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class ProgressLoggingCallback(TrainerCallback):
    """Log concise Trainer progress records through the project logger."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        pieces = []
        for key in ["loss", "learning_rate", "grad_norm", "epoch"]:
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    pieces.append(f"{key}={value:.6g}")
                else:
                    pieces.append(f"{key}={value}")
        if pieces:
            LOGGER.info("Transformer train step %s/%s: %s", state.global_step, state.max_steps, " ".join(pieces))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        pieces = []
        for key in ["eval_loss", "eval_macro_f1", "eval_recall_ai", "eval_average_precision"]:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    pieces.append(f"{key}={value:.6g}")
                else:
                    pieces.append(f"{key}={value}")
        if pieces:
            LOGGER.info("Transformer eval step %s/%s: %s", state.global_step, state.max_steps, " ".join(pieces))
