"""Phase 12 — additive white Gaussian noise channel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def signal_power(x: torch.Tensor) -> torch.Tensor:
    """Mean signal power (per complex-real dimension convention)."""
    return x.detach().float().pow(2).mean()


def noise_variance_for_snr_db(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """sigma^2 = P_signal / 10^(SNR_dB / 10), derived from the measured power."""
    return signal_power(x) / (10.0 ** (float(snr_db) / 10.0))


def awgn_channel(x: torch.Tensor, *, snr_db: float, seed: int,
                 h: complex = 1 + 0j) -> tuple[torch.Tensor, dict]:
    """received = h * signal + n.

    Real tensors get real Gaussian noise with the same total variance as the
    complex baseband model. Complex tensors split sigma^2 across I and Q.
    """
    if not torch.is_tensor(x):
        raise TypeError("awgn_channel expects a torch tensor")
    is_complex = torch.is_complex(x)
    sigma2 = noise_variance_for_snr_db(x, snr_db)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = tuple(x.shape)
    work = x.detach().clone()
    if is_complex:
        std = (sigma2 / 2.0).sqrt()
        real = torch.randn(shape, generator=generator)
        imag = torch.randn(shape, generator=generator)
        noise = (real + 1j * imag).to(torch.complex64) * std.to(torch.complex64)
        received = complex(h) * work + noise
        measured = received.real.sub((complex(h) * work).real).float().pow(2).mean()
    else:
        std = sigma2.sqrt()
        noise = torch.randn(shape, generator=generator, dtype=torch.float32) * std
        received = work.float() + noise
        measured = (received - work.float()).pow(2).mean()
    meta = {
        "snr_db": float(snr_db),
        "signal_power": float(signal_power(x)),
        "variance_target": float(sigma2),
        "variance_measured": float(measured),
        "complex": bool(is_complex),
        "h": str(h),
        "seed": int(seed),
        "shape": list(shape),
    }
    return received, meta
