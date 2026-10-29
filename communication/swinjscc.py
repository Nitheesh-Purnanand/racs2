"""Phase 13 — SwinJSCC analog semantic codec (paper backbone).

The paper transmits the stego image through SwinJSCC trained on CelebA-HQ
at 10 dB with ``C = 48``. The official release does not ship a C=48
checkpoint; the available family is C=32/64/96/128/192, so this module
loads the nearest public model (C=64, AWGN, SNR 10, PSNR objective) and
records that deviation. The encoder/decoder, power normalisation and
complex AWGN convention are the official implementation; only the
channel noise generator is replaced with a seeded one so the transmitter
and receiver reproduce the same realisation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_THIRD_PARTY = _ROOT / "third_party" / "SwinJSCC"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

# The vendored model prints layer descriptions at construction; keep the log quiet.
import contextlib  # noqa: E402
import io  # noqa: E402

from net.network import SwinJSCC  # noqa: E402


def _model_config(model: str, model_size: str = "base", image_size: int = 256):
    """Mirror the official ``config`` block for DIV2K/HRimage models."""
    if model_size != "base":
        raise ValueError("only the released base SwinJSCC checkpoints are supported")
    channel_number = None  # set from the checkpoint / args
    image_dims = (3, image_size, image_size)
    downsample = 4
    encoder_kwargs = dict(
        model=model, img_size=(image_dims[1], image_dims[2]), patch_size=2,
        in_chans=3, embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10], C=channel_number, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    decoder_kwargs = dict(
        model=model, img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4], C=channel_number, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True,
    )
    return SimpleNamespace(
        pass_channel=True, CUDA=torch.cuda.is_available(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        norm=False, logger=None, downsample=downsample,
        encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs,
    )


def _strip_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        for prefix in ("module.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out


class SwinJSCodec:
    """Frozen SwinJSCC encoder + seeded complex AWGN + decoder."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        channel_number: int = 64,
        trained_snr_db: float = 10.0,
        image_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.channel_number = int(channel_number)
        self.trained_snr_db = float(trained_snr_db)
        self.image_size = int(image_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_name = "SwinJSCC_w/o_SAandRA"

        # The vendored SwinJSCC module builds a loss and a channel object at
        # construction even when only the encoder/decoder are used; these are
        # the official argument names it reads.
        args = SimpleNamespace(
            model=self.model_name, channel_type="awgn",
            multiple_snr=str(int(trained_snr_db)), C=str(self.channel_number),
            distortion_metric="MSE", trainset="DIV2K", logger=None,
        )
        config = _model_config(self.model_name, "base", self.image_size)
        config.encoder_kwargs["C"] = self.channel_number
        config.decoder_kwargs["C"] = self.channel_number

        with contextlib.redirect_stdout(io.StringIO()):
            self.net = SwinJSCC(args, config)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "params_ema" in state:
            state = state["params_ema"]
        state = _strip_prefix(state)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"SwinJSCC checkpoint is missing {len(missing)} tensors: "
                               f"{missing[:4]}")
        self.net.eval().to(device=self.device, dtype=self.dtype)
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.symbol_shape = None

    # ------------------------------------------------------------------ codec
    @torch.inference_mode()
    def encode(self, images01: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] in [0,1] -> complex symbols stored as a real tensor."""
        x = images01.to(self.device, dtype=self.dtype)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size), mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        symbols = self.net.encoder(
            x, self.trained_snr_db, self.channel_number, self.model_name
        )
        self.symbol_shape = tuple(symbols.shape)
        return symbols.float()

    def _normalise(self, symbols: torch.Tensor):
        power = symbols.pow(2).mean() * 2.0
        tx = symbols / power.sqrt().clamp_min(1e-12)
        return tx, power

    def _complex_noise(self, tx: torch.Tensor, snr_db: float, seed: int):
        """Official complex-AWGN convention with an explicit generator."""
        flat = tx.reshape(-1)
        half = flat.shape[0] // 2
        real = flat[:half]
        imag = flat[half:2 * half]
        sigma = float(np.sqrt(1.0 / (2.0 * 10 ** (float(snr_db) / 10.0))))
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(2 * half, generator=gen, dtype=torch.float32) * sigma
        noise = noise.to(tx.device)
        noisy = torch.cat([real + noise[:half], imag + noise[half:]])
        return noisy.reshape(tx.shape)

    def transmit(self, symbols: torch.Tensor, *, snr_db: float, seed: int) -> torch.Tensor:
        tx, power = self._normalise(symbols.to(self.device))
        noisy = self._complex_noise(tx, snr_db, seed)
        return (noisy * power.sqrt()).to(self.dtype)

    @torch.inference_mode()
    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        out = self.net.decoder(
            symbols.to(self.device, dtype=self.dtype), self.trained_snr_db, self.model_name
        )
        return out.float().clamp(0.0, 1.0)

    @torch.inference_mode()
    def roundtrip(self, images01: torch.Tensor, *, snr_db: float, seed: int) -> torch.Tensor:
        symbols = self.encode(images01)
        received = self.transmit(symbols, snr_db=snr_db, seed=seed)
        return self.decode(received)
