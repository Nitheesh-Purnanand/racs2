"""Phase 13 — semantic codec (SwinJSCC, with the SD VAE kept as a fallback).

``kind: swinjscc`` loads the official public SwinJSCC release through
``communication.swinjscc``. ``kind: sd_vae_substitute`` keeps the earlier
VAE path available for ablations. Both are pretrained, frozen and
deterministic; the SwinJSCC checkpoint difference (C=64 instead of the
paper's C=48, HRimage instead of CelebA-HQ) is registered in
``docs/DEVIATIONS.md``.
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
    """Binds the frozen codec and records its identity."""
    kind = config.get("kind", "sd_vae_substitute")
    if kind in ("swinjscc", "swin_jscc"):
        from communication.swinjscc import SwinJSCodec

        codec = SwinJSCodec(
            config["checkpoint"],
            channel_number=int(config.get("channel_number", 64)),
            trained_snr_db=float(config.get("trained_snr_db", 10)),
            image_size=int(config.get("input_size", 256)),
            dtype=torch.float16 if str(config.get("dtype", "float32")).lower()
            in ("float16", "fp16") else torch.float32,
        )
        return {
            "kind": "swinjscc",
            "model_id": "semcomm/SwinJSCC",
            "compression_ratio": int(config.get("compression_ratio", 24)),
            "trained_snr_db": float(config.get("trained_snr_db", 10)),
            "channel_number": codec.channel_number,
            "net": codec,
            "sha256": codec.sha256,
        }
    vae = edict.vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return {
        "kind": "sd_vae_substitute",
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
    if codec["kind"] == "swinjscc":
        net = codec["net"]
        symbols = net.encode(x.detach().float())
        received = net.transmit(symbols, snr_db=snr_db, seed=seed)
        out = net.decode(received)
        symbols_shape = list(symbols.shape)
        return {
            "output": out,
            "symbols_shape": symbols_shape,
            "latency_s": time.time() - started,
            "meta": {
                "codec": "swinjscc",
                "checkpoint_sha256": codec["sha256"],
                "snr_db": float(snr_db),
                "seed": int(seed),
                "symbols_shape": symbols_shape,
                "compression_ratio": codec["compression_ratio"],
                "channel_number": codec["channel_number"],
            },
        }
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
