#!/usr/bin/env python3
"""Audit RedNote-Vibe raw training data without creating processed splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.data.prepare import audit_data
from rednote_aigt.utils.io import read_yaml
from rednote_aigt.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data config YAML.")
    parser.add_argument("--sample", type=int, default=None, help="Load N human and N AIGC rows for quick testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        config = read_yaml(args.config)
        report = audit_data(config, sample=args.sample)
    except Exception as exc:
        print(f"Data audit failed: {exc}", file=sys.stderr)
        return 1

    print("Data audit complete")
    print(f"Rows audited: {report['total_rows']}")
    print("Outputs: outputs/reports/data_audit.json, outputs/reports/data_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
