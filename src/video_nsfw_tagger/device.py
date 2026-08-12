"""Device resolution utilities for PyTorch/ROCm backends."""

import torch


def resolve_device(request: str = "auto") -> str:
    """Resolve a device string to a usable torch device identifier.

    ``auto`` prefers CUDA/ROCm, then MPS, then CPU.
    For gfx1032 (RX 6600) set ``HSA_OVERRIDE_GFX_VERSION=gfx1030`` before
    launching when ROCm complains about the unsupported gfx target.

    Args:
        request: Device identifier such as ``auto``, ``cpu``, ``cuda``,
            ``cuda:0``, or ``mps``.

    Returns:
        A string torch understands (e.g. ``"cpu"``, ``"cuda:0"``, ``"mps"``).

    Raises:
        RuntimeError: If a specific accelerator is requested but unavailable.
        ValueError: If the device string is unknown.
    """
    if request == "auto":
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if request.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA/ROCm requested ({request}) but not available")
        return request

    if request == "mps":
        if not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but not available")
        return request

    if request == "cpu":
        return request

    raise ValueError(f"Unknown device: {request}")
