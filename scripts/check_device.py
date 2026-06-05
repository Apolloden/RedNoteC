#!/usr/bin/env python3
"""Print Apple Silicon/PyTorch device diagnostics."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rednote_aigt.utils.device import describe_torch_device, get_torch_device


def main() -> int:
    import torch
    import transformers

    selected = get_torch_device(prefer_mps=True, prefer_cuda=False)
    info = describe_torch_device()
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"platform.machine(): {platform.machine()}")
    print(f"platform.platform(): {platform.platform()}")
    print(f"mps built: {info['mps_built']}")
    print(f"mps available: {info['mps_available']}")
    print(f"cuda available: {info['cuda_available']}")
    print(f"selected device: {selected}")
    try:
        a = torch.ones((2, 2), device=selected)
        b = torch.eye(2, device=selected)
        c = a @ b
        print(f"tensor matmul on {selected}: ok, sum={float(c.sum().cpu())}")
    except Exception as exc:
        print(f"tensor matmul on {selected}: failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
