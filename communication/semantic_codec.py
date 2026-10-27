"""Phase 13 — semantic codec (pretrained SD-VAE substitute).

The paper's SwinJSCC/MDJCM checkpoints are not publicly redistributable here, so
this module uses the already pinned SD VAE as an analog JSCC substitute: encode
stego pixels to latent symbols, normalise power, pass through AWGN, decode.
It is pretrained, frozen, and deterministic; the deviation is registered in
docs/DEVIATIONS.md.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from communication.awgn import awgn_channel  # noqa: E402

SCALING = 0.18215


def state_dict_sha256(vae) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(vae.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_codec(config: dict, edict) -> dict:
    """Binds the frozen VAE and records its identity."""
    vae = edict.vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return {
        "kind": config.get("kind", "sd_vae_substitute"),
        "model_id": config.get("checkpoint", "stabilityai/sd-vae-ft-mse"),
        "compression_ratio": int(config.get("compression_ratio", 12)),
        "trained_snr_db": float(config.get("trained_snr_db", 10)),
        "vae": vae,
        "sha256": state_dict_sha256(vae),
    }


@torch.no_grad()
def codec_roundtrip(x: torch.Tensor, *, snr_db: float, seed: int,
                    codec, config: dict) -> dict:
    """stego [0,1] -> symbols -> power normalise -> AWGN -> symbols -> decode."""
    started = time.time()
    if float(x.min()) < -0.001 or float(x.max()) > 1.001:
        raise ValueError("codec expects [0,1] pixel tensors")
    vae = codec["vae"]
    pixels = x.detach().float().cpu() * 2.0 - 1.0
    encoded = vae.encode(pixels).latent_dist.mode()
    power = encoded.float().pow(2).mean().clamp_min(1e-12)
    symbols = encoded / power.sqrt()
    received, meta = awgn_channel(symbols, snr_db=snr_db, seed=seed, h=1 + 0j)
    decoded = vae.decode(received * power.sqrt()).sample
    out = (decoded / 2.0 + 0.5).clamp(0.0, 1.0).float().cpu()
    symbols_shape = list(symbols.shape)
    meta.update({
        "symbols_shape": symbols_shape,
        "symbol_power": float(power),
        "compression_ratio": int(config.get("compression_ratio", 12)),
    })
    return {"output": out, "symbols_shape": symbols_shape,
            "latency_s": time.time() - started, "meta": meta}
