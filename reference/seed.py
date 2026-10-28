"""Deterministic token-derived seeds shared by the reference generator and stego core."""

from __future__ import annotations

import hashlib

SEED_DOMAIN = "racs2-reference-v1"
MASK_DOMAIN = "racs2-mask-v1"


def derive_seed(token: str) -> int:
    """Deterministic 31-bit reference seed from an 8-digit token."""
    digest = hashlib.sha256(f"{SEED_DOMAIN}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def derive_mask_seed(token: str) -> int:
    """Deterministic 31-bit perturbation seed (separate domain)."""
    digest = hashlib.sha256(f"{MASK_DOMAIN}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def derive_delta(token: str, shape, fraction: float = 0.05):
    """Deterministic sparse perturbation tensor from the token alone."""
    import numpy as np

    rng = np.random.default_rng(derive_mask_seed(token))
    n = int(np.prod(shape))
    flat = rng.permutation(n)[: max(1, int(round(fraction * n)))]
    delta = np.zeros(n, dtype=np.float32)
    delta[flat] = rng.standard_normal(len(flat)).astype(np.float32)
    return delta.reshape(tuple(shape))
