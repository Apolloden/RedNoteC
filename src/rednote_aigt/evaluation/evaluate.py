"""Evaluation CLI implementation.

Both models run through this one function, which is what makes the numbers in
README.md a like-for-like comparison: same split, same threshold, same metric
code, same subgroup slicing. Only ``score_ai`` is model-specific.

Everything written here is a report about a finished model — nothing in this
module is allowed to influence training or model selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rednote_aigt.evaluation.metrics import (
    add_length_buckets,
    classification_report_dict,
    compute_binary_metrics,
    subgroup_metrics,
)
from rednote_aigt.evaluation.plots import (
    plot_confusion_matrix,
    plot_length_bucket_ai_recall,
    plot_per_class_f1,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_subgroup_domain_macro_f1,
)
from rednote_aigt.models.io import load_model
from rednote_aigt.utils.device import describe_torch_device, resolve_torch_device
from rednote_aigt.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

OPTIONAL_PREDICTION_COLUMNS = ["text_len_chars", "domain", "model_family", "model"]


def evaluate_model(
    model_dir: Path,
    test_path: Path,
    output_dir: Path,
    figures_dir: Path,
    batch_size: int = 8,
    max_test_samples: int | None = None,
    threshold: float = 0.5,
    split_name: str = "test",
    device: str = "auto",
    prefer_mps: bool = True,
    prefer_cuda: bool = False,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Score a saved model on one split and write metrics, tables, and figures.

    Args:
        model_dir: Directory written by ``scripts/train.py``; its
            ``model_metadata.json`` decides how the model is loaded and scored.
        test_path: CSV split to score. Only ``text`` and ``label`` are read;
            metadata columns are used for subgroup tables and error dumps only.
        threshold: Probability cut for the hard label. Reported alongside every
            metric, since precision/recall are meaningless without it.

    Returns:
        The overall metric dict, also written to ``metrics.json``.
    """
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    model, metadata = load_model(model_dir)
    df = pd.read_csv(test_path)
    if max_test_samples is not None:
        df = stratified_sample(df, max_test_samples, seed=42)
    _validate_columns(df)

    texts = df["text"].fillna("").astype(str).tolist()
    if metadata["model_type"] == "tfidf":
        score_ai = model.score_ai(texts)
    elif metadata["model_type"] == "transformer":
        selected_device = resolve_torch_device(device, prefer_mps=prefer_mps, prefer_cuda=prefer_cuda)
        device_info = describe_torch_device()
        device_info["selected_for_evaluation"] = str(selected_device)
        write_json(device_info, output_dir / "device_info.json")
        effective_max_length = max_length or metadata.get("config", {}).get("max_length", 256)
        score_ai = model.predict_scores(
            texts,
            batch_size=batch_size,
            max_length=effective_max_length,
            device=str(selected_device),
            prefer_mps=prefer_mps,
            prefer_cuda=prefer_cuda,
        )
    else:
        raise ValueError(f"Unsupported model type: {metadata['model_type']}")

    result_df = df.copy()
    result_df["score_ai"] = score_ai
    result_df["pred_label"] = (result_df["score_ai"] >= threshold).astype(int)
    result_df["split"] = result_df["split"] if "split" in result_df.columns else split_name
    result_df = add_length_buckets(result_df)

    metrics = compute_binary_metrics(result_df["label"], result_df["score_ai"], threshold)
    report = classification_report_dict(result_df["label"], result_df["pred_label"])
    write_json(metrics, output_dir / "metrics.json")
    write_json(report, output_dir / "classification_report.json")

    _save_predictions(result_df, output_dir / "predictions.csv")
    domain_metrics = _save_subgroup(result_df, "domain", output_dir / "subgroup_metrics_domain.csv", threshold)
    _save_subgroup(result_df[result_df["label"] == 1], "model_family", output_dir / "subgroup_metrics_model_family.csv", threshold)
    _save_subgroup(result_df[result_df["label"] == 1], "model", output_dir / "subgroup_metrics_model.csv", threshold)
    length_metrics = _save_subgroup(result_df, "length_bucket", output_dir / "subgroup_metrics_length_bucket.csv", threshold)
    _save_errors(result_df, output_dir)

    plot_confusion_matrix(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    plot_roc_curve(result_df["label"].to_numpy(), result_df["score_ai"].to_numpy(), figures_dir / "roc_curve.png")
    plot_precision_recall_curve(
        result_df["label"].to_numpy(),
        result_df["score_ai"].to_numpy(),
        figures_dir / "precision_recall_curve.png",
    )
    plot_per_class_f1(metrics, figures_dir / "per_class_f1.png")
    plot_subgroup_domain_macro_f1(domain_metrics, figures_dir / "subgroup_domain_macro_f1.png")
    plot_length_bucket_ai_recall(length_metrics, figures_dir / "length_bucket_ai_recall.png")

    LOGGER.info("Saved evaluation outputs to %s and %s", output_dir, figures_dir)
    return metrics


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    parts = []
    for _, group in df.groupby("label"):
        take = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(n=min(take, len(group)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed)
    return out.head(n).copy()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ["text", "label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Evaluation data missing required columns: {missing}")


def _save_predictions(df: pd.DataFrame, path: Path) -> None:
    columns = ["id", "split", "label", "pred_label", "score_ai"]
    columns.extend(col for col in OPTIONAL_PREDICTION_COLUMNS if col in df.columns)
    if "text" in df.columns:
        out = df.copy()
        out["text"] = out["text"].fillna("").astype(str).str.slice(0, 300)
        columns.append("text")
    else:
        out = df
    out[[col for col in columns if col in out.columns]].to_csv(path, index=False)


def _save_subgroup(df: pd.DataFrame, group_column: str, path: Path, threshold: float) -> pd.DataFrame:
    metrics_df = subgroup_metrics(df, group_column, threshold=threshold)
    if not metrics_df.empty:
        metrics_df.to_csv(path, index=False)
    elif path.exists():
        # Otherwise a table from an earlier run would sit next to this run's
        # metrics and read as current evidence.
        path.unlink()
    return metrics_df


def _save_errors(df: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "id",
        "split",
        "label",
        "pred_label",
        "score_ai",
        "text_len_chars",
        "domain",
        "model_family",
        "model",
        "text",
    ]
    out = df.copy()
    if "text" in out:
        out["text"] = out["text"].fillna("").astype(str).str.slice(0, 300)
    cols = [col for col in cols if col in out.columns]
    out[(out["label"] == 0) & (out["pred_label"] == 1)][cols].to_csv(
        output_dir / "errors_false_positives.csv",
        index=False,
    )
    out[(out["label"] == 1) & (out["pred_label"] == 0)][cols].to_csv(
        output_dir / "errors_false_negatives.csv",
        index=False,
    )
