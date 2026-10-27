"""Phase 11 — hide: secret + reference -> stego (EDICT + token perturbation)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from steganography.mask import mask_meta, token_delta, token_noise  # noqa: E402
from steganography import pipeline  # noqa: E402


def add_delta(pair: List[torch.Tensor], delta: torch.Tensor, eta: float) -> List[torch.Tensor]:
    """The single transmission equation shared by hide and recover."""
    return [(z + float(eta) * delta.to(z.device, z.dtype)) for z in pair]


def subtract_delta(latent: torch.Tensor, delta: torch.Tensor, eta: float) -> torch.Tensor:
    return latent - float(eta) * delta.to(latent.device, latent.dtype)


def hide(edict, secret_pixels: torch.Tensor, reference_pixels: torch.Tensor,
         prompt: str, token: str, cfg: dict) -> Dict[str, object]:
    """Returns the transmitted latent pair, stego pixels and mask metadata."""
    steps = int(cfg["edict_steps"])
    mixing = float(cfg["edict_mixing"])
    guidance = float(cfg.get("guidance_scale", 1.0))
    strength = float(cfg.get("edict_strength", 0.3))
    eta = float(cfg["eta"])
    fraction = float(cfg["mask_fraction"])

    z_secret = pipeline.encode(edict, secret_pixels)
    noise = token_noise(token, shape=tuple(z_secret.shape))
    bounded = edict.forward_img2img(
        z_secret, prompt=prompt, steps=steps, mixing=mixing,
        guidance_scale=guidance, strength=strength, noise=noise,
    )
    pair = bounded["pair"]
    delta = token_delta(token, shape=tuple(pair[0].shape), fraction=fraction)
    z_stego_pair = add_delta(pair, delta, eta)
    stego_pixels = pipeline.decode(edict, pipeline.mean_pair(z_stego_pair))
    return {
        "z_secret": z_secret,
        "pair": pair,
        "noise": noise,
        "strength": strength,
        "delta": delta,
        "z_stego_pair": z_stego_pair,
        "stego_pixels": stego_pixels,
        "prompt": prompt,
        "eta": eta,
        "steps": steps,
        "mixing": mixing,
        "mask": mask_meta(token, delta),
    }


def hide_and_recover(edict, secret_pixels, reference_pixels, prompt, token, cfg,
                     recover_prompt=None):
    from steganography.recover import recover

    hidden = hide(edict, secret_pixels, reference_pixels, prompt, token, cfg)
    recovered = recover(edict, hidden["stego_pixels"], token,
                        recover_prompt or prompt, cfg,
                        delta=hidden["delta"], eta=hidden["eta"],
                        steps=hidden["steps"], mixing=hidden["mixing"])
    hidden["recovered_pixels"] = recovered
    return hidden
