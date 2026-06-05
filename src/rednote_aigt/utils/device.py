"""PyTorch device selection for Apple Silicon and CPU environments."""

from __future__ import annotations

import logging
import platform
from typing import Any

LOGGER = logging.getLogger(__name__)


def get_torch_device(prefer_mps: bool = True, prefer_cuda: bool = False):
    """Return a torch device, preferring MPS by default and CPU otherwise."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and _mps_is_available(torch):
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_device(device: str = "auto", prefer_mps: bool = True, prefer_cuda: bool = False):
    """Resolve an explicit device request, failing clearly if unavailable."""
    import torch

    device = device.lower()
    if device == "auto":
        return get_torch_device(prefer_mps=prefer_mps, prefer_cuda=prefer_cuda)
    if device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        if not _mps_is_built(torch):
            raise RuntimeError("MPS was requested, but this PyTorch build does not include MPS support.")
        if not _mps_is_available(torch):
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{device}'. Use auto, mps, cpu, or cuda.")


def describe_torch_device() -> dict[str, Any]:
    """Return PyTorch platform/device diagnostics."""
    import torch

    selected = get_torch_device()
    return {
        "torch_version": torch.__version__,
        "platform_machine": platform.machine(),
        "platform": platform.platform(),
        "mps_built": _mps_is_built(torch),
        "mps_available": _mps_is_available(torch),
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": str(selected),
    }


def log_device_info() -> dict[str, Any]:
    """Log and return PyTorch platform/device diagnostics."""
    info = describe_torch_device()
    LOGGER.info(
        "Torch %s on %s (%s); MPS built=%s available=%s; CUDA available=%s; selected device=%s",
        info["torch_version"],
        info["platform_machine"],
        info["platform"],
        info["mps_built"],
        info["mps_available"],
        info["cuda_available"],
        info["selected_device"],
    )
    return info


def empty_mps_cache_if_available() -> None:
    """Best-effort MPS cache cleanup after larger inference batches."""
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        return


def _mps_is_built(torch_module) -> bool:
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:
        return False


def _mps_is_available(torch_module) -> bool:
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False
