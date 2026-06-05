"""Metrics for binary AI-generated text detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    score_ai: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    score_arr = np.asarray(score_ai).astype(float)
    y_pred = (score_arr >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y_true_arr, y_pred),
        "precision_human": float(precision[0]),
        "recall_human": float(recall[0]),
        "f1_human": float(f1[0]),
        "precision_ai": float(precision[1]),
        "recall_ai": float(recall[1]),
        "f1_ai": float(f1[1]),
        "macro_precision": float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
        "log_loss": _safe_log_loss(y_true_arr, score_arr),
        "auroc": _safe_roc_auc(y_true_arr, score_arr),
        "average_precision": _safe_average_precision(y_true_arr, score_arr),
        "confusion_matrix": cm.tolist(),
        "support_human": int(support[0]),
        "support_ai": int(support[1]),
        "threshold": float(threshold),
    }
    return metrics


def classification_report_dict(y_true: list[int] | np.ndarray, pred_label: list[int] | np.ndarray) -> dict[str, Any]:
    return classification_report(
        np.asarray(y_true).astype(int),
        np.asarray(pred_label).astype(int),
        labels=[0, 1],
        target_names=["human", "ai"],
        output_dict=True,
        zero_division=0,
    )


def subgroup_metrics(
    df: pd.DataFrame,
    group_column: str,
    threshold: float = 0.5,
    min_rows: int = 1,
) -> pd.DataFrame:
    if group_column not in df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group_df in df.groupby(group_column, dropna=False):
        if len(group_df) < min_rows:
            continue
        metrics = compute_binary_metrics(group_df["label"].to_numpy(), group_df["score_ai"].to_numpy(), threshold)
        rows.append(
            {
                group_column: "<missing>" if pd.isna(group_value) else group_value,
                "rows": int(len(group_df)),
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False) if rows else pd.DataFrame()


def add_length_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text_len_chars" not in out.columns:
        out["text_len_chars"] = out["text"].fillna("").astype(str).str.len()
    try:
        out["length_bucket"] = pd.qcut(
            out["text_len_chars"],
            q=3,
            labels=["short", "medium", "long"],
            duplicates="drop",
        )
    except ValueError:
        out["length_bucket"] = pd.cut(
            out["text_len_chars"],
            bins=3,
            labels=["short", "medium", "long"],
            include_lowest=True,
        )
    out["length_bucket"] = out["length_bucket"].astype(str)
    return out


def _safe_roc_auc(y_true: np.ndarray, score_ai: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, score_ai))
    except ValueError:
        return None


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) >= 2:
        return float(balanced_accuracy_score(y_true, y_pred))
    present_label = int(y_true[0]) if len(y_true) else 0
    return float(recall_score(y_true, y_pred, labels=[present_label], average="macro", zero_division=0))


def _safe_log_loss(y_true: np.ndarray, score_ai: np.ndarray) -> float | None:
    try:
        proba = np.column_stack([1.0 - score_ai, score_ai])
        return float(log_loss(y_true, proba, labels=[0, 1]))
    except ValueError:
        return None


def _safe_average_precision(y_true: np.ndarray, score_ai: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, score_ai))
    except ValueError:
        return None
