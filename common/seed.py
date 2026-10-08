"""
Deterministic Seeding Utility for RACS2.

Sets random seeds across Python's `random` module, NumPy, PyTorch (CPU and CUDA),
and system environment variables to ensure experiment reproducibility across
all stages of the pipeline (retrieval, latent sampling, EDICT steganography,
communication channel simulation, evaluation).
"""

import os
import random
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = False, verbose: bool = False) -> int:
    """
    Sets seed across Python `random`, `numpy`, and `torch` (CPU + CUDA).

    Args:
        seed: The integer seed value to set (default: 42).
        deterministic: If True, enables PyTorch cuDNN deterministic algorithms.
            Note: Deterministic operations may slightly degrade GPU performance.
        verbose: If True, prints status messages confirming set seed.

    Returns:
        int: The seed value that was set.
    """
    seed = int(seed)

    # 1. Environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Python standard random module
    random.seed(seed)

    # 3. NumPy
    if HAS_NUMPY:
        np.random.seed(seed)

    # 4. PyTorch (CPU & CUDA)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass

    if verbose:
        print(f"[Seed] Set global random seed to {seed} (deterministic={deterministic})")

    return seed


def get_generator(seed: int = 42, device: Union[str, "torch.device"] = "cpu"):
    """
    Creates and seeds a PyTorch Generator instance on the specified device.
    Useful for diffusers pipelines, EDICT sampling, and PyTorch dataloaders.

    Args:
        seed: The seed integer (default: 42).
        device: Device on which generator will reside ('cpu' or 'cuda').

    Returns:
        torch.Generator: Initialized and seeded PyTorch generator.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the active environment.")
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    print("=== Testing RACS2 Deterministic Seeding ===")
    test_seed = 42

    # First run
    set_seed(test_seed, verbose=True)
    py_val1 = random.random()
    np_val1 = np.random.rand(3) if HAS_NUMPY else None
    torch_val1 = torch.randn(3) if HAS_TORCH else None

    # Second run (re-seed)
    set_seed(test_seed, verbose=True)
    py_val2 = random.random()
    np_val2 = np.random.rand(3) if HAS_NUMPY else None
    torch_val2 = torch.randn(3) if HAS_TORCH else None

    # Assert reproducibility
    assert py_val1 == py_val2, "Python random seed failure!"
    if HAS_NUMPY:
        assert np.array_equal(np_val1, np_val2), "NumPy random seed failure!"
    if HAS_TORCH:
        assert torch.equal(torch_val1, torch_val2), "PyTorch random seed failure!"

    print("SUCCESS: Deterministic seeding verified across Python, NumPy, and PyTorch.")
    print(f"Sample Python output: {py_val1:.6f}")
    if HAS_NUMPY:
        print(f"Sample NumPy output: {np_val1}")
    if HAS_TORCH:
        print(f"Sample PyTorch output: {torch_val1}")
