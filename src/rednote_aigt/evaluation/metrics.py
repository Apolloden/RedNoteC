"""Metrics for binary AI-generated text detection.

Every metric here is computed over the fixed label set ``[0, 1]`` (0 = human,
1 = AI). That matters for subgroup tables: a subgroup can legitimately contain
only one class (the generator tables hold AI rows only), and scikit-learn's
``average="macro"`` silently averages over the labels it *observes* unless the
label set is pinned. Pinning it keeps a subgroup metric comparable to the same
metric on the full split instead of quietly changing its definition.

Score-based metrics (AUROC, AUPRC) are undefined with one class present and are
returned as ``None`` rather than a misleading number.
"""

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
    recall_score,
    roc_auc_score,
)

LABELS = [0, 1]


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    score_ai: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score AI-probability predictions against binary labels at one threshold."""
    y_true_arr = np.asarray(y_true).astype(int)
    score_arr = np.asarray(score_ai).astype(float)
    y_pred = (score_arr >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred, labels=LABELS)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y_true_arr, y_pred),
        "precision_human": float(precision[0]),
        "recall_human": float(recall[0]),
        "f1_human": float(f1[0]),
        "precision_ai": float(precision[1]),
        "recall_ai": float(recall[1]),
        "f1_ai": float(f1[1]),
        # Macro averages are taken over the pinned per-class arrays above, so a
        # single-class subgroup still averages human and AI instead of one class.
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred, average="weighted", labels=LABELS, zero_division=0)),
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
    """Return scikit-learn's per-class report as a JSON-serializable dict."""
    return classification_report(
        np.asarray(y_true).astype(int),
        np.asarray(pred_label).astype(int),
        labels=LABELS,
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
    """Score each value of ``group_column`` separately (diagnostic slicing).

    Callers may pass a single-class slice on purpose — the generator tables are
    built from AI rows only. Columns that need both classes (``*_human``,
    ``macro_*``, ``auroc``, ``average_precision``) are then structurally
    degenerate; read ``recall_ai`` and ``support_ai`` for those tables.
    """
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
    """Add short/medium/long tertiles of text length.

    Buckets are tertiles *of the frame being evaluated*, not fixed character
    thresholds, so bucket boundaries differ between splits and are only
    comparable across models scored on the same split.
    """
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
        return float(log_loss(y_true, proba, labels=LABELS))
    except ValueError:
        return None


def _safe_average_precision(y_true: np.ndarray, score_ai: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, score_ai))
    except ValueError:
        return None
