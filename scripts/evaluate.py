#!/usr/bin/env python3
"""Evaluate a saved RedNote classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.evaluation.evaluate import evaluate_model
from rednote_aigt.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--prefer-mps", dest="prefer_mps", action="store_true", default=True)
    parser.add_argument("--no-prefer-mps", dest="prefer_mps", action="store_false")
    parser.add_argument("--prefer-cuda", dest="prefer_cuda", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    model_name = args.model_dir.name
    output_dir = args.output_dir or Path("outputs/reports") / model_name
    figures_dir = args.figures_dir or Path("outputs/figures") / model_name
    try:
        metrics = evaluate_model(
            model_dir=args.model_dir,
            test_path=args.test_path,
            output_dir=output_dir,
            figures_dir=figures_dir,
            batch_size=args.batch_size,
            max_test_samples=args.max_test_samples,
            threshold=args.threshold,
            split_name=args.split_name,
            device=args.device,
            prefer_mps=args.prefer_mps,
            prefer_cuda=args.prefer_cuda,
            max_length=args.max_length,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("Evaluation complete")
    print(f"Model dir: {args.model_dir}")
    print(f"Reports: {output_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"AI recall: {metrics['recall_ai']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
