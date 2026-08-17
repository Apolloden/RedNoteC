"""Leave-one-generator-family-out (LOGO) evaluation.

Answers a question that in-distribution accuracy cannot: is the detector
recognizing AI-generated writing, or recognizing the specific generators it was
trained on?

For each generator family F, a model is trained on every AI row *except* F's
(human rows unchanged) and then scored on the test split restricted to human
rows plus F's AI rows. F is removed from the validation split too, so a model
that selects checkpoints on validation cannot peek at the family it will be
judged on.

The comparison is exact: LOGO recall for family F is measured on the same test
rows that the main in-distribution model was measured on in
``subgroup_metrics_model_family.csv``. The only difference between the two
numbers is whether F appeared in training, so the gap between them is the cost
of meeting an unseen generator.

Read ``recall_ai`` across folds. Precision and AUPRC are *not* comparable
between folds or to the main run: each fold's test set holds all 7,747 human
rows but only one family's AI rows, so positive-class prevalence changes with
the family. AUROC and the human-side false-positive count stay comparable.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from rednote_aigt.evaluation.evaluate import evaluate_model
from rednote_aigt.training.train import train_model
from rednote_aigt.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

FAMILY_COLUMN = "model_family"


def list_generator_families(df: pd.DataFrame, min_rows: int = 1) -> list[str]:
    """Return the generator families present on AI rows, largest first."""
    if FAMILY_COLUMN not in df.columns:
        raise ValueError(f"Column '{FAMILY_COLUMN}' is required for generator holdout, but is absent.")
    counts = df.loc[df["label"] == 1, FAMILY_COLUMN].dropna().value_counts()
    return [str(family) for family, count in counts.items() if count >= min_rows]


def build_holdout_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one fold: train/val without ``family``, test on ``family`` only.

    Human rows are untouched everywhere — only AI rows move — so the human side
    of the problem is held fixed while the generator mix changes.
    """

    def without_family(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["label"] == 0) | (df[FAMILY_COLUMN] != family)].copy()

    fold_test = test_df[(test_df["label"] == 0) | (test_df[FAMILY_COLUMN] == family)].copy()
    return without_family(train_df), without_family(val_df), fold_test


def build_size_matched_control(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    family: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the control that separates "unseen generator" from "less data".

    A holdout fold trains on fewer AI rows than the main model, so part of any
    recall drop could be volume rather than novelty. This control removes the
    *same number* of AI rows, drawn at random across all families, while
    keeping ``family`` in training. Comparing the three gives a decomposition:

    * main model to control: the cost of training on less AI data
    * control to holdout: the cost of the generator being new

    The test set is identical to the holdout fold's, so all three numbers are
    measured on the same rows.
    """

    def drop_random_ai(df: pd.DataFrame) -> pd.DataFrame:
        ai_rows = df[df["label"] == 1]
        n_drop = int((ai_rows[FAMILY_COLUMN] == family).sum())
        dropped = ai_rows.sample(n=min(n_drop, len(ai_rows)), random_state=seed).index
        return df.drop(index=dropped).copy()

    fold_test = test_df[(test_df["label"] == 0) | (test_df[FAMILY_COLUMN] == family)].copy()
    return drop_random_ai(train_df), drop_random_ai(val_df), fold_test


def run_generator_holdout(
    model_name: str,
    model_config: dict[str, Any],
    processed_dir: Path,
    work_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    families: list[str] | None = None,
    seed: int = 42,
    threshold: float = 0.5,
    batch_size: int = 8,
    device: str = "auto",
    max_steps: int | None = None,
    mode: str = "holdout",
) -> pd.DataFrame:
    """Train and score one model per generator family.

    Each fold goes through the same ``train_model`` / ``evaluate_model`` path as
    the headline runs, so a fold's numbers are produced by identical code.

    Args:
        mode: ``"holdout"`` removes the family from training and validation.
            ``"control"`` removes the same *number* of AI rows at random while
            keeping the family in, which is the comparison that separates a
            missing generator from a smaller training set.

    Returns:
        One row per family, also written to ``summary.csv`` in ``reports_dir``.
    """
    if mode not in {"holdout", "control"}:
        raise ValueError(f"mode must be 'holdout' or 'control', got {mode!r}")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    families = families or list_generator_families(train_df)
    if not families:
        raise ValueError("No generator families found to hold out.")

    ensure_dir(reports_dir)
    ensure_dir(figures_dir)
    LOGGER.info("Generator %s for %s over families: %s", mode, model_name, ", ".join(families))

    recall_column = "recall_ai_holdout" if mode == "holdout" else "recall_ai_control"
    rows: list[dict[str, Any]] = []
    for position, family in enumerate(families, start=1):
        LOGGER.info("Fold %s/%s (%s): generator family '%s'", position, len(families), mode, family)
        if mode == "holdout":
            fold_train, fold_val, fold_test = build_holdout_fold(train_df, val_df, test_df, family)
        else:
            fold_train, fold_val, fold_test = build_size_matched_control(train_df, val_df, test_df, family, seed=seed)

        fold_dir = ensure_dir(work_dir / family)
        fold_reports = ensure_dir(reports_dir / family)
        fold_figures = ensure_dir(figures_dir / family)
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        test_path = fold_dir / "test.csv"
        fold_train.to_csv(train_path, index=False)
        fold_val.to_csv(val_path, index=False)
        fold_test.to_csv(test_path, index=False)

        train_model(
            model_name=model_name,
            model_config=model_config,
            train_path=train_path,
            val_path=val_path,
            output_dir=fold_dir / "model",
            reports_dir=fold_reports,
            figures_dir=fold_figures,
            seed=seed,
            force=True,
            max_steps=max_steps,
            device=device,
        )
        metrics = evaluate_model(
            model_dir=fold_dir / "model",
            test_path=test_path,
            output_dir=fold_reports,
            figures_dir=fold_figures,
            batch_size=batch_size,
            threshold=threshold,
            device=device,
        )
        _discard_trainer_checkpoints(fold_dir / "model")

        human_rows, ai_rows = int((fold_test["label"] == 0).sum()), int((fold_test["label"] == 1).sum())
        false_positives = int(metrics["confusion_matrix"][0][1])
        rows.append(
            {
                "held_out_family": family,
                "train_ai_rows": int((fold_train["label"] == 1).sum()),
                "test_human_rows": human_rows,
                "test_ai_rows": ai_rows,
                recall_column: metrics["recall_ai"],
                "auroc": metrics["auroc"],
                "false_positives_human": false_positives,
                "missed_ai_posts": int(metrics["confusion_matrix"][1][0]),
            }
        )
        LOGGER.info(
            "Fold '%s' (%s): AI recall=%.4f on %s rows (AUROC=%.4f)",
            family,
            mode,
            metrics["recall_ai"],
            ai_rows,
            metrics["auroc"] if metrics["auroc"] is not None else float("nan"),
        )

    summary = pd.DataFrame(rows).sort_values("test_ai_rows", ascending=False).reset_index(drop=True)
    summary.to_csv(reports_dir / "summary.csv", index=False)
    designs = {
        "holdout": (
            "Train and validation exclude the held-out family's AI rows; the test set is the "
            "standard test split restricted to human rows plus the held-out family's AI rows."
        ),
        "control": (
            "Train and validation drop the same NUMBER of AI rows as the holdout fold, sampled at "
            "random across all families, with the target family retained; the test set is identical "
            "to the holdout fold's. Isolates training-set size from generator novelty."
        ),
    }
    write_json(
        {
            "model": model_name,
            "mode": mode,
            "families": families,
            "threshold": threshold,
            "seed": seed,
            "design": designs[mode],
        },
        reports_dir / "design.json",
    )
    return summary


def _discard_trainer_checkpoints(model_dir: Path) -> None:
    """Drop mid-training checkpoints once the fold has been scored.

    One transformer fold leaves ~400 MB of Trainer checkpoints behind, and a
    six-family sweep runs the disk down for artifacts nobody reads after the
    final model is saved.
    """
    checkpoints = model_dir / "trainer_checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints, ignore_errors=True)
        LOGGER.info("Removed trainer checkpoints for fold model at %s", model_dir)


def attach_in_distribution_baseline(summary: pd.DataFrame, subgroup_csv: Path) -> pd.DataFrame:
    """Join the in-distribution per-family recall measured on the same test rows."""
    if not subgroup_csv.exists():
        LOGGER.warning("No in-distribution subgroup table at %s; reporting fold recall only.", subgroup_csv)
        return summary
    recall_column = next((col for col in ("recall_ai_holdout", "recall_ai_control") if col in summary.columns), None)
    if recall_column is None:
        return summary
    baseline = pd.read_csv(subgroup_csv)[[FAMILY_COLUMN, "recall_ai"]]
    baseline = baseline.rename(columns={FAMILY_COLUMN: "held_out_family", "recall_ai": "recall_ai_in_distribution"})
    merged = summary.merge(baseline, on="held_out_family", how="left")
    merged["recall_drop"] = merged["recall_ai_in_distribution"] - merged[recall_column]
    return merged
