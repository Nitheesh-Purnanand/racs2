"""Phase 11 — recover: stego -> recovered secret (reverse EDICT + perturbation)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from steganography.hide import subtract_delta  # noqa: E402  (single shared equation)
from steganography.mask import token_delta, token_noise  # noqa: E402
from steganography import pipeline  # noqa: E402


def recover(edict, stego_pixels: torch.Tensor, token: str, prompt: str, cfg: dict,
            *, delta=None, eta=None, steps=None, mixing=None,
            guidance=None, noise=None) -> torch.Tensor:
    eta = float(cfg["eta"] if eta is None else eta)
    steps = int(cfg["edict_steps"] if steps is None else steps)
    mixing = float(cfg["edict_mixing"] if mixing is None else mixing)
    guidance = float(cfg.get("guidance_scale", 1.0) if guidance is None else guidance)

    z_stego = pipeline.encode(edict, stego_pixels)
    if delta is None:
        delta = token_delta(token, shape=tuple(z_stego.shape),
                            fraction=float(cfg["mask_fraction"]))
    z_hidden = subtract_delta(z_stego, delta, eta)
    if noise is None:
        noise = token_noise(token, shape=tuple(z_hidden.shape))
    back = edict.inverse_img2img(
        [z_hidden, z_hidden.clone()], prompt=prompt, steps=steps, mixing=mixing,
        guidance_scale=guidance, strength=float(cfg.get("edict_strength", 0.3)),
        noise=noise,
    )
    return pipeline.decode(edict, pipeline.mean_pair(back))
