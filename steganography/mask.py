"""Phase 11 — token-derived deterministic perturbation mask."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reference.seed import MASK_DOMAIN, derive_mask_seed  # noqa: E402

DEFAULT_SHAPE = (1, 4, 32, 32)
DEFAULT_FRACTION = 0.05


def token_delta(token: str, shape=DEFAULT_SHAPE, fraction: float = DEFAULT_FRACTION) -> torch.Tensor:
    """Deterministic sparse Gaussian delta derived from the token alone."""
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    shape = tuple(int(v) for v in shape)
    rng = np.random.default_rng(derive_mask_seed(str(token)))
    n = int(np.prod(shape))
    k = max(1, int(round(fraction * n)))
    flat = rng.permutation(n)[:k]
    delta = np.zeros(n, dtype=np.float32)
    delta[flat] = rng.standard_normal(k).astype(np.float32)
    return torch.from_numpy(delta.reshape(shape))


def delta_sha256(delta: torch.Tensor) -> str:
    return hashlib.sha256(delta.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()


def mask_meta(token: str, delta: torch.Tensor) -> dict:
    return {
        "token": str(token),
        "mask_domain": MASK_DOMAIN,
        "mask_seed": int(derive_mask_seed(str(token))),
        "shape": list(delta.shape),
        "nonzero": int((delta != 0).sum()),
        "sha256": delta_sha256(delta),
    }


def token_noise(token: str, shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic noise draw shared by transmitter and receiver."""
    seed = (int(derive_mask_seed(str(token))) ^ 0x5DEECE66) & 0x7FFFFFFF
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(tuple(int(v) for v in shape), generator=generator,
                       dtype=dtype, device="cpu")
