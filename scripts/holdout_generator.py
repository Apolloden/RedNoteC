#!/usr/bin/env python3
"""Evaluate a model against generators it never saw in training.

Trains one model per generator family, each time excluding that family's AI
posts from training and validation, then scoring on the test split restricted
to human posts plus that family's posts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.evaluation.generator_holdout import attach_in_distribution_baseline, run_generator_holdout
from rednote_aigt.evaluation.plots import plot_generator_holdout_recall
from rednote_aigt.utils.io import read_yaml
from rednote_aigt.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Registered model name, e.g. tfidf_logreg.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch space for per-fold data and models.")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--families", nargs="*", default=None, help="Subset of families to hold out.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=None, help="Transformer-only step cap for smoke runs.")
    parser.add_argument(
        "--mode",
        choices=["holdout", "control"],
        default="holdout",
        help="holdout: drop the family. control: drop as many random AI rows, keeping the family.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    run_name = "generator_holdout" if args.mode == "holdout" else "generator_holdout_control"
    reports_dir = args.reports_dir or Path("outputs/reports") / args.model / run_name
    figures_dir = args.figures_dir or Path("outputs/figures") / args.model / run_name
    # Kept outside models/<model>/ so per-fold artifacts do not make the main
    # model directory look occupied to scripts/train.py.
    work_dir = args.work_dir or Path("models") / run_name / args.model
    try:
        configs = read_yaml(args.config)
        if args.model not in configs:
            raise KeyError(f"Model '{args.model}' not found in {args.config}")
        summary = run_generator_holdout(
            model_name=args.model,
            model_config=dict(configs[args.model]),
            processed_dir=args.processed_dir,
            work_dir=work_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            families=args.families,
            seed=args.seed,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=args.device,
            max_steps=args.max_steps,
            mode=args.mode,
        )
        summary = attach_in_distribution_baseline(
            summary,
            Path("outputs/reports") / args.model / "subgroup_metrics_model_family.csv",
        )
        summary.to_csv(reports_dir / "summary.csv", index=False)
        if args.mode == "holdout":
            plot_generator_holdout_recall(summary, figures_dir.parent / "generator_holdout_recall.png")
    except Exception as exc:
        print(f"Generator {args.mode} failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generator {args.mode} complete")
    print(f"Model: {args.model}")
    print(f"Summary: {reports_dir / 'summary.csv'}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
