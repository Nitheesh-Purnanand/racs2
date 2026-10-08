"""
Stable Diffusion VAE loader and round-trip helpers for RACS2.

Provides the exact VAE component that RACS2 re-uses downstream: Stable Diffusion
v1.5's `AutoencoderKL`. It compresses a `512x512x3` image to a `64x64x4` latent
(8x spatial downscale, 4 channels) and back. The round-trip is intentionally
lossy; that error is the floor for EDICT / channel / retrieval stages.

Design notes:
- The SD VAE expects pixel values in the range [-1, 1] and a latent `scaling_factor`
  (typically 0.18215) applied on encode and undone on decode.
- `latent_dist.sample()` draws randomness; pass `sample=False` to use
  `latent_dist.mode()` (the deterministic mean) for a reproducible round-trip.

Usage:
    import common.paths  # redirect HF cache to D: before anything else
    from models.vae import load_vae, roundtrip
    vae = load_vae()
    recon, latent = roundtrip(vae, image)   # image: [1,3,512,512] in [-1, 1]
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Allow running this file directly (as a script) as well as importing it as
# `models.vae`: make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the HuggingFace cache is redirected to D: before diffusers/hub import.
import common.paths  # noqa: F401  (import for side effect)

# Read-only checkpoint mappings, installed before diffusers binds
# `safetensors.safe_open`/`load_file` (see common/hf_lowmem.py for why).
import common.hf_lowmem  # noqa: F401,E402

common.hf_lowmem.install()

from diffusers import AutoencoderKL  # noqa: E402

from common.device import get_device

DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


def load_vae(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    model_id: str = DEFAULT_VAE_ID,
) -> AutoencoderKL:
    """
    Loads the Stable Diffusion VAE on the given device in eval mode.

    Args:
        device: Target device. Defaults to ``get_device()`` (CUDA if available).
        dtype: Torch dtype for the weights (fp16 by default to save VRAM).
        model_id: HuggingFace repo id of the VAE checkpoint.

    Returns:
        AutoencoderKL: The loaded VAE, in eval mode with gradients disabled.
    """
    if device is None:
        device = get_device()

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae: AutoencoderKL, image: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """
    Encodes a pixel image into the scaled latent space.

    Args:
        vae: Loaded AutoencoderKL.
        image: Pixel tensor, shape [N, 3, H, W], values in [-1, 1], dtype matching vae.
        sample: If True use `latent_dist.sample()` (random); else `latent_dist.mode()`.

    Returns:
        Latent tensor [N, 4, H/8, W/8], already multiplied by `scaling_factor`.
    """
    latent_dist = vae.encode(image).latent_dist
    latent = latent_dist.sample() if sample else latent_dist.mode()
    return latent * vae.config.scaling_factor


@torch.no_grad()
def decode(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """
    Decodes a scaled latent back to pixel space.

    Args:
        vae: Loaded AutoencoderKL.
        latent: Latent tensor [N, 4, H/8, W/8] in scaled space.

    Returns:
        Reconstructed pixel tensor [N, 3, H, W], values roughly in [-1, 1].
    """
    return vae.decode(latent / vae.config.scaling_factor).sample


@torch.no_grad()
def roundtrip(
    vae: AutoencoderKL, image: torch.Tensor, sample: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full pixel -> latent -> pixel round-trip.

    Args:
        vae: Loaded AutoencoderKL.
        image: Pixel tensor [N, 3, H, W] in [-1, 1].
        sample: Passed to `encode` (False = deterministic mean latent).

    Returns:
        (reconstruction, latent): reconstructed image [N, 3, H, W] and
        scaled latent [N, 4, H/8, W/8].
    """
    latent = encode(vae, image, sample=sample)
    recon = decode(vae, latent)
    return recon, latent


def latent_spatial_shape(height: int, width: int) -> Tuple[int, int]:
    """Returns the expected latent spatial size for a given pixel size (÷8)."""
    return height // 8, width // 8


if __name__ == "__main__":
    import common.paths  # noqa: F401
    from common.seed import set_seed

    set_seed(42)
    dev = get_device()
    print(f"Device: {dev}")
    vae = load_vae(dev)
    print(f"Loaded VAE: {DEFAULT_VAE_ID} (dtype={vae.dtype})")

    dummy = torch.rand(1, 3, 512, 512, device=dev, dtype=vae.dtype) * 2 - 1
    recon, latent = roundtrip(vae, dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Latent: {tuple(latent.shape)}")
    print(f"Output: {tuple(recon.shape)}")
