"""
GPU / Device Detection Utility for RACS2.

Provides helpers to select the compute device for the pipeline and to report
GPU hardware and VRAM usage. Every stage of RACS2 (retrieval, EDICT
steganography, channel simulation, evaluation) should obtain its device from
`get_device()` so behaviour stays consistent and falls back to CPU gracefully
when no CUDA-capable GPU is present.
"""

from typing import Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch() -> None:
    """Raise a clear error if PyTorch is not available in the environment."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the active environment.")


def get_device(prefer_cuda: bool = True) -> "torch.device":
    """
    Selects the compute device for the pipeline.

    Args:
        prefer_cuda: If True (default), returns a CUDA device when one is
            available. If False, always returns the CPU device.

    Returns:
        torch.device: ``torch.device("cuda")`` if a GPU is available and
        requested, otherwise ``torch.device("cpu")``.
    """
    _require_torch()

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info(device_index: int = 0) -> None:
    """
    Prints GPU name and VRAM statistics for the selected device.

    Reports the GPU name, total VRAM, and currently allocated VRAM (in GB).
    If CUDA is unavailable, a short message is printed instead.

    Args:
        device_index: CUDA device ordinal to report on (default: 0).
    """
    _require_torch()

    if not torch.cuda.is_available():
        print("[Device] CUDA is not available. Using CPU.")
        return

    props = torch.cuda.get_device_properties(device_index)
    name = torch.cuda.get_device_name(device_index)

    bytes_per_gb = 1024 ** 3
    total_vram_gb = props.total_memory / bytes_per_gb
    allocated_gb = torch.cuda.memory_allocated(device_index) / bytes_per_gb
    reserved_gb = torch.cuda.memory_reserved(device_index) / bytes_per_gb

    print(f"[Device] GPU name      : {name}")
    print(f"[Device] Total VRAM    : {total_vram_gb:.2f} GB")
    print(f"[Device] Used VRAM     : {allocated_gb:.2f} GB (allocated)")
    print(f"[Device] Reserved VRAM : {reserved_gb:.2f} GB (reserved)")


if __name__ == "__main__":
    print("=== Testing RACS2 Device Utilities ===")
    dev = get_device()
    print(f"Selected device: {dev}")
    print_gpu_info()
