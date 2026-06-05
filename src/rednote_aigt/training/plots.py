"""Training plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rednote_aigt_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rednote_aigt.utils.io import ensure_dir


def plot_train_val_metrics(metrics: dict[str, Any], path: Path, title: str = "Train/Validation Metrics") -> None:
    ensure_dir(path.parent)
    names = ["accuracy", "macro_f1", "recall_ai", "average_precision"]
    rows = []
    for split in ["train", "val"]:
        split_metrics = metrics.get(split, {})
        for name in names:
            value = split_metrics.get(name)
            if value is not None:
                rows.append({"split": split, "metric": name, "value": value})
    if not rows:
        return
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    width = 0.35
    train_vals = [float(data[(data["split"] == "train") & (data["metric"] == name)]["value"].iloc[0]) for name in names]
    val_vals = [float(data[(data["split"] == "val") & (data["metric"] == name)]["value"].iloc[0]) for name in names]
    ax.bar([i - width / 2 for i in x], train_vals, width, label="train")
    ax.bar([i + width / 2 for i in x], val_vals, width, label="val")
    ax.set_xticks(list(x), names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_train_val_loss(metrics: dict[str, Any], path: Path, title: str = "Train/Validation Log Loss") -> None:
    ensure_dir(path.parent)
    train_loss = metrics.get("train", {}).get("log_loss")
    val_loss = metrics.get("val", {}).get("log_loss")
    if train_loss is None or val_loss is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["train", "val"], [train_loss, val_loss])
    ax.set_ylabel("Log loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tfidf_train_val_metrics(metrics: dict[str, Any], path: Path) -> None:
    plot_train_val_metrics(metrics, path, title="TF-IDF Train/Validation Metrics")


def plot_tfidf_train_val_loss(metrics: dict[str, Any], path: Path) -> None:
    plot_train_val_loss(metrics, path, title="TF-IDF Train/Validation Log Loss")


def plot_transformer_train_val_metrics(metrics: dict[str, Any], path: Path) -> None:
    plot_train_val_metrics(metrics, path, title="Transformer Train/Validation Metrics")


def plot_transformer_train_val_loss(metrics: dict[str, Any], path: Path) -> None:
    plot_train_val_loss(metrics, path, title="Transformer Train/Validation Log Loss")


def plot_transformer_history(log_history: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not log_history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    train_rows = [row for row in log_history if "loss" in row and "step" in row]
    eval_rows = [row for row in log_history if "eval_macro_f1" in row and "step" in row]
    if train_rows:
        axes[0].plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    if eval_rows:
        axes[1].plot([row["step"] for row in eval_rows], [row["eval_macro_f1"] for row in eval_rows], marker="o", label="macro_f1")
        if "eval_recall_ai" in eval_rows[0]:
            axes[1].plot([row["step"] for row in eval_rows], [row["eval_recall_ai"] for row in eval_rows], marker="o", label="recall_ai")
        axes[1].legend()
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Step")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
