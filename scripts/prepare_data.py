#!/usr/bin/env python3
"""Prepare cleaned RedNote-Vibe train/validation/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.data.prepare import prepare_data
from rednote_aigt.utils.io import read_yaml
from rednote_aigt.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data config YAML.")
    parser.add_argument("--sample", type=int, default=None, help="Load N human and N AIGC rows for quick testing.")
    parser.add_argument("--force", action="store_true", help="Overwrite processed split outputs if they exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        config = read_yaml(args.config)
        summary = prepare_data(config, sample=args.sample, force=args.force)
    except Exception as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        return 1

    print("Data preparation complete")
    print(f"Raw rows: {summary['raw_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Deduped rows: {summary['deduped_rows']}")
    print(f"Splits: {summary['splits']}")
    print(f"Audit: {summary['audit_json']}")
    print(f"Processed outputs: {summary['processed_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
