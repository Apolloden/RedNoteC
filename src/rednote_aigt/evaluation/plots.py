"""Matplotlib plots for model evaluation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rednote_aigt_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from rednote_aigt.utils.io import ensure_dir


def plot_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=["human", "ai"]).plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true, score_ai, path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(set(y_true)) >= 2:
        RocCurveDisplay.from_predictions(y_true, score_ai, ax=ax)
    else:
        ax.text(0.5, 0.5, "ROC undefined: one class present", ha="center", va="center")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_precision_recall_curve(y_true, score_ai, path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(set(y_true)) >= 2:
        PrecisionRecallDisplay.from_predictions(y_true, score_ai, ax=ax)
    else:
        ax.text(0.5, 0.5, "PR curve undefined: one class present", ha="center", va="center")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_class_f1(metrics: dict, path: Path) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["human", "ai"], [metrics.get("f1_human", 0), metrics.get("f1_ai", 0)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_subgroup_domain_macro_f1(domain_metrics: pd.DataFrame, path: Path) -> None:
    if domain_metrics.empty or "domain" not in domain_metrics or "macro_f1" not in domain_metrics:
        return
    ensure_dir(path.parent)
    data = domain_metrics.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7, max(4, len(data) * 0.35)))
    ax.barh(data["domain"].astype(str), data["macro_f1"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1")
    ax.set_title("Domain Macro F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_length_bucket_ai_recall(length_metrics: pd.DataFrame, path: Path) -> None:
    if length_metrics.empty or "length_bucket" not in length_metrics or "recall_ai" not in length_metrics:
        return
    ensure_dir(path.parent)
    order = ["short", "medium", "long"]
    data = length_metrics.copy()
    data["order"] = data["length_bucket"].map({name: i for i, name in enumerate(order)}).fillna(99)
    data = data.sort_values("order")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(data["length_bucket"].astype(str), data["recall_ai"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AI Recall")
    ax.set_title("AI Recall by Length Bucket")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
