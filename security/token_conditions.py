"""Phase 16 — receiver-side token conditions (correct / wrong / no_token)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from steganography.mask import token_delta, token_noise  # noqa: E402

NO_TOKEN_FALLBACK = "00000000"


def mask_hash(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def receive(*, stego, token, token_mode: str, config: dict, context: dict) -> dict:
    """Runs only the receiver half of Phase 15 with the requested token condition."""
    baseline = context["baseline"]
    blob = context["blob"]
    if token_mode == "correct":
        used, delta, noise = token, blob["delta"], blob["noise"]
    elif token_mode == "wrong":
        used = context["wrong_token"]
        delta = token_delta(used, shape=tuple(blob["delta"].shape),
                            fraction=float(config["mask_fraction"]))
        noise = token_noise(used, shape=tuple(blob["noise"].shape))
    elif token_mode == "no_token":
        used = NO_TOKEN_FALLBACK
        delta = torch.zeros_like(blob["delta"])
        noise = token_noise(used, shape=tuple(blob["noise"].shape))
    else:
        raise ValueError(f"unknown token_mode {token_mode!r}")
    recovered = _recover(baseline, stego, used, context["prompt"], config, blob, delta, noise)
    return {"token_mode": token_mode, "token_used": used,
            "mask_sha256": mask_hash(delta), "recovered": recovered}


def _recover(baseline, stego, used, prompt, config, blob, delta, noise):
    from steganography.recover import recover
    return recover(baseline.edict, stego, used, prompt, config, delta=delta,
                   eta=blob["eta"], steps=blob["steps"], mixing=blob["mixing"],
                   guidance=config.get("guidance_scale", 1.0), noise=noise)
