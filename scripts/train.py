#!/usr/bin/env python3
"""Train a registered RedNote classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.training.train import train_model
from rednote_aigt.utils.io import read_yaml
from rednote_aigt.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Registered model name, e.g. tfidf_logreg or transformer_roberta.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val-path", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="Transformer-only cap for smoke tests.")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--prefer-mps", dest="prefer_mps", action="store_true", default=True)
    parser.add_argument("--no-prefer-mps", dest="prefer_mps", action="store_false")
    parser.add_argument("--prefer-cuda", dest="prefer_cuda", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--tfidf-max-features", type=int, default=None, help="Override TF-IDF max_features for quick debug runs.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    try:
        configs = read_yaml(args.config)
        if args.model not in configs:
            raise KeyError(f"Model '{args.model}' not found in {args.config}")
        model_config = dict(configs[args.model])
        if args.model == "transformer_roberta" and args.max_steps is not None:
            model_config["num_train_epochs"] = configs.get("smoke_test", {}).get("transformer_num_train_epochs", 1)
        if args.tfidf_max_features is not None:
            model_config.setdefault("vectorizer", {})["max_features"] = args.tfidf_max_features
        output_dir = args.output_dir or Path("models") / args.model
        reports_dir = args.reports_dir or Path("outputs/reports") / args.model
        figures_dir = args.figures_dir or Path("outputs/figures") / args.model
        metrics = train_model(
            model_name=args.model,
            model_config=model_config,
            train_path=args.train_path,
            val_path=args.val_path,
            output_dir=output_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
            force=args.force,
            max_steps=args.max_steps,
            allow_smoke_fallback=args.max_steps is not None,
            device=args.device,
            prefer_mps=args.prefer_mps,
            prefer_cuda=args.prefer_cuda,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("Training complete")
    print(f"Model: {args.model}")
    print(f"Model artifacts: {args.output_dir or Path('models') / args.model}")
    print(f"Reports: {args.reports_dir or Path('outputs/reports') / args.model}")
    print(f"Figures: {args.figures_dir or Path('outputs/figures') / args.model}")
    if "val" in metrics:
        print(f"Validation macro F1: {metrics['val']['macro_f1']:.4f}")
        print(f"Validation AI recall: {metrics['val']['recall_ai']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
