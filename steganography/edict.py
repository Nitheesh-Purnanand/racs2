"""
Phase 10 — EDICT (Exact Diffusion Inversion via Coupled Transformations).

A faithful reimplementation of the Salesforce EDICT DDIM loop for SD v1.5.
Two latent trajectories are coupled with an invertible affine mixing layer;
the sampling and inversion passes are exact algebraic reverses of one another
up to floating-point rounding.

Usage:
    edict = EDICT()
    pair = edict.forward(z, prompt=kpub, steps=50, mixing=0.93)
    pair2 = edict.inverse(pair, prompt=kpub, steps=50, mixing=0.93)
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from PIL import Image  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from common.device import get_device  # noqa: E402

MODEL_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
]


def _get_alpha_beta(t: torch.Tensor, scheduler: DDIMScheduler):
    """Reference EDICT interpolation of alpha_cumprod at a (possibly float) t."""
    if t.dtype == torch.long:
        alpha = scheduler.alphas_cumprod[t]
        return alpha, 1 - alpha
    if float(t) < 0:
        return scheduler.final_alpha_cumprod, 1 - scheduler.final_alpha_cumprod
    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low
    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1 - rem)
    return interpolated_alpha, 1 - interpolated_alpha


def forward_step(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    """One DDIM forward (denoising) step, the exact inverse of reverse_step."""
    prev_timestep = timestep - scheduler.config.num_train_timesteps / num_inference_steps
    alpha_prod_t, beta_prod_t = _get_alpha_beta(timestep, scheduler)
    alpha_prod_t_prev, _ = _get_alpha_beta(prev_timestep, scheduler)
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)
    first = (1.0 / alpha_quotient) * sample
    second = (1.0 / alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third = ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first - second + third


def reverse_step(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    """One DDIM reverse (inversion) step, the exact inverse of forward_step."""
    prev_timestep = timestep - scheduler.config.num_train_timesteps / num_inference_steps
    alpha_prod_t, beta_prod_t = _get_alpha_beta(timestep, scheduler)
    alpha_prod_t_prev, _ = _get_alpha_beta(prev_timestep, scheduler)
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)
    first = alpha_quotient * sample
    second = (beta_prod_t ** 0.5) * model_output
    third = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first + second - third


def _mix(pair: Sequence[torch.Tensor], p: float) -> List[torch.Tensor]:
    """Forward affine mixing layer (contraction)."""
    x, y = pair
    new_x = p * x + (1 - p) * y
    new_y = (1 - p) * new_x + p * y
    return [new_x.clone(), new_y.clone()]


def _inv_mix(pair: Sequence[torch.Tensor], p: float) -> List[torch.Tensor]:
    """Inverse affine mixing layer."""
    x, y = pair
    new_y = (y - (1 - p) * x) / p
    new_x = (x - (1 - p) * new_y) / p
    return [new_x.clone(), new_y.clone()]


class EDICT:
    """SD v1.5 EDICT sampler/inverter (fp16 UNet, float32 latents)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        latent_dtype: torch.dtype = torch.float32,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ) -> None:
        self.device = device or get_device()
        self.dtype = dtype
        self.latent_dtype = latent_dtype

        sources = [model_id] if model_id else list(MODEL_CANDIDATES)
        last_err: Optional[Exception] = None
        self.model_id = ""
        load_kwargs = {}
        if self.device.type == "cuda":
            load_kwargs["device_map"] = str(self.device)
        for source in sources:
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    source, subfolder="unet", torch_dtype=dtype, **load_kwargs
                )
                # VAE and text encoder stay on CPU between calls; moving them to
                # the GPU permanently leaves too little headroom for the 512px
                # VAE activations on a 6 GB card.
                self.vae = AutoencoderKL.from_pretrained(
                    source, subfolder="vae", torch_dtype=torch.float32
                )
                self.tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    source, subfolder="text_encoder", torch_dtype=torch.float32
                )
                self.model_id = source
                break
            except Exception as err:  # noqa: BLE001
                last_err = err
                self._free()
                continue
        if not self.model_id:
            raise RuntimeError(f"Could not load SD v1.5 components: {last_err}")

        self.unet.eval()
        self._install_dtype_guard(self.unet, self.dtype)
        self.vae.eval()
        self.text_encoder.eval()
        for module in (self.unet, self.vae, self.text_encoder):
            for param in module.parameters():
                param.requires_grad_(False)

        self.scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _install_dtype_guard(unet, dtype: torch.dtype) -> None:
        """diffusers' fp16 UNet expects t_emb in the weight dtype; force it."""
        time_proj = unet.time_proj
        if not getattr(time_proj, "_racs2_guarded", False):
            original_proj = time_proj.forward

            def proj(timesteps):
                out = original_proj(timesteps)
                return out.to(device=timesteps.device, dtype=dtype)

            time_proj.forward = proj
            time_proj._racs2_guarded = True

        embedding = unet.time_embedding
        if getattr(embedding, "_racs2_guarded", False):
            return
        original_forward = embedding.forward

        def forward(sample, condition=None):
            return original_forward(sample.to(dtype=dtype), condition)

        embedding.forward = forward
        embedding._racs2_guarded = True

    def _free(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embeddings(self, prompt: str, negative: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        def encode(text: str):
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                return self.text_encoder(tokens.input_ids).last_hidden_state.to(self.device)

        return encode(prompt), encode(negative)

    def _predict(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance with both branches in one UNet batch."""
        latent_in = latent.to(dtype=self.dtype)
        timestep = timestep.to(self.device)
        if float(guidance_scale) == 1.0:
            embeds = cond.to(dtype=self.dtype)
            model_in = latent_in
            with torch.inference_mode():
                pred = self.unet(
                    model_in, timestep, encoder_hidden_states=embeds
                ).sample
            return pred.to(self.latent_dtype)
        embeds = torch.cat([uncond, cond], dim=0).to(dtype=self.dtype)
        model_in = torch.cat([latent_in, latent_in], dim=0)
        with torch.inference_mode():
            pred = self.unet(model_in, timestep, encoder_hidden_states=embeds).sample
        pred_uncond, pred_cond = pred.chunk(2, dim=0)
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        return pred.to(self.latent_dtype)

    def encode_image(self, image: Image.Image, seed: int = 42,
                     size: int = 256) -> torch.Tensor:
        import numpy as np

        arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC)).astype(
            "float32"
        ) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.inference_mode():
            latent = self.vae.encode(tensor).latent_dist.sample(generator=generator)
        return (latent * 0.18215).to(device=self.device, dtype=self.latent_dtype)

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        import numpy as np

        with torch.inference_mode():
            image = self.vae.decode((latent.cpu() / 0.18215).to(torch.float32)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((image * 255).round().astype("uint8"))

    # --------------------------------------------------------------- EDICT
    def forward(
        self,
        latent: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        mixing: float = 0.93,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[torch.Tensor]:
        if not 0.5 < float(mixing) <= 1.0:
            raise ValueError("mixing must satisfy 0.5 < mixing <= 1.0")
        if int(steps) < 1:
            raise ValueError("steps must be >= 1")
        if latent.dim() != 4:
            raise ValueError(f"latent must be 4-D, got {tuple(latent.shape)}")
        pair = [
            latent.clone().to(device=self.device, dtype=self.latent_dtype),
            latent.clone().to(device=self.device, dtype=self.latent_dtype),
        ]
        cond, uncond = self._embeddings(prompt, negative_prompt)
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            for base_i in range(2):
                offset = i % 2
                latent_i = (base_i + offset) % 2
                latent_j = 1 - latent_i
                model_input = pair[latent_j]
                noise_pred = self._predict(
                    model_input, t, cond, uncond, guidance_scale
                )
                pair[latent_i] = forward_step(
                    self.scheduler, noise_pred, t, pair[latent_i], steps
                ).to(self.latent_dtype)
            pair = _mix(pair, mixing)
        return pair

    def inverse(
        self,
        pair: Sequence[torch.Tensor],
        *,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        mixing: float = 0.93,
        guidance_scale: float = 7.5,
    ) -> List[torch.Tensor]:
        if not 0.5 < float(mixing) <= 1.0:
            raise ValueError("mixing must satisfy 0.5 < mixing <= 1.0")
        if int(steps) < 1:
            raise ValueError("steps must be >= 1")
        if len(pair) != 2 or pair[0].shape != pair[1].shape:
            raise ValueError("pair must contain two latents with identical shapes")
        pair = [
            pair[0].clone().to(device=self.device, dtype=self.latent_dtype),
            pair[1].clone().to(device=self.device, dtype=self.latent_dtype),
        ]
        cond, uncond = self._embeddings(prompt, negative_prompt)
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps.flip(0)
        n = len(timesteps)

        for i, t in enumerate(timesteps):
            pair = _inv_mix(pair, mixing)
            for base_i in range(2):
                orig_i = n - (i + 1)
                offset = (orig_i + 1) % 2
                latent_i = (base_i + offset) % 2
                latent_j = 1 - latent_i
                model_input = pair[latent_j]
                noise_pred = self._predict(
                    model_input, t, cond, uncond, guidance_scale
                )
                pair[latent_i] = reverse_step(
                    self.scheduler, noise_pred, t, pair[latent_i], steps
                ).to(self.latent_dtype)
        return pair

    def roundtrip(
        self,
        latent: torch.Tensor,
        *,
        prompt: str,
        steps: int = 50,
        mixing: float = 0.93,
        guidance_scale: float = 7.5,
    ) -> Dict[str, object]:
        pair = self.forward(
            latent, prompt=prompt, steps=steps, mixing=mixing,
            guidance_scale=guidance_scale,
        )
        back = self.inverse(
            pair, prompt=prompt, steps=steps, mixing=mixing,
            guidance_scale=guidance_scale,
        )
        errors = []
        for original, recovered in zip((latent, latent), back):
            diff = (recovered - original).abs()
            errors.append(
                {
                    "max_abs": float(diff.max()),
                    "max_rel": float(diff.max() / original.abs().max().clamp_min(1e-12)),
                }
            )
        return {"pair": pair, "recovered": back, "errors": errors}

    # ------------------------------------------------------- partial strength
    def _partial_timesteps(self, steps: int, strength: float):
        if not 0.0 < float(strength) <= 1.0:
            raise ValueError("strength must satisfy 0 < strength <= 1")
        self.scheduler.set_timesteps(int(steps))
        all_ts = self.scheduler.timesteps
        start = min(max(0, int(round(int(steps) * (1.0 - float(strength))))),
                    int(steps) - 1)
        return all_ts, start

    def forward_img2img(
        self,
        latent: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        mixing: float = 0.93,
        guidance_scale: float = 1.0,
        strength: float = 0.3,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Bounded EDICT chain: noise to strength, then denoise back.

        Unlike ``forward`` this keeps the latent content: the output pair is the
        denoised latent, so it decodes to an image close to the input.
        ``inverse_img2img`` is its exact algebraic reverse.
        """
        if not 0.5 < float(mixing) <= 1.0:
            raise ValueError("mixing must satisfy 0.5 < mixing <= 1.0")
        all_ts, start = self._partial_timesteps(steps, strength)
        ts = all_ts[start:]
        t_start = all_ts[start]
        alpha = self.scheduler.alphas_cumprod[t_start]
        latent = latent.to(device=self.device, dtype=self.latent_dtype)
        if noise is None:
            noise = torch.randn_like(latent)
        noise = noise.to(device=self.device, dtype=self.latent_dtype)
        z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        pair = [z_t.clone(), z_t.clone()]
        cond, uncond = self._embeddings(prompt, negative_prompt)
        for i, t in enumerate(ts):
            g = start + i
            for base_i in range(2):
                latent_i = (base_i + g % 2) % 2
                latent_j = 1 - latent_i
                noise_pred = self._predict(pair[latent_j], t, cond, uncond, guidance_scale)
                pair[latent_i] = forward_step(
                    self.scheduler, noise_pred, t, pair[latent_i], steps
                ).to(self.latent_dtype)
            pair = _mix(pair, mixing)
        return {"pair": pair, "noise": noise, "start_index": start,
                "start_timestep": int(t_start), "strength": float(strength)}

    def inverse_img2img(
        self,
        pair: Sequence[torch.Tensor],
        *,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        mixing: float = 0.93,
        guidance_scale: float = 1.0,
        strength: float = 0.3,
        noise: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Exact reverse of ``forward_img2img`` back to the clean latent."""
        if not 0.5 < float(mixing) <= 1.0:
            raise ValueError("mixing must satisfy 0.5 < mixing <= 1.0")
        all_ts, start = self._partial_timesteps(steps, strength)
        ts = all_ts[start:]
        t_start = all_ts[start]
        alpha = self.scheduler.alphas_cumprod[t_start]
        if len(pair) != 2 or pair[0].shape != pair[1].shape:
            raise ValueError("pair must contain two latents with identical shapes")
        pair = [p.clone().to(device=self.device, dtype=self.latent_dtype) for p in pair]
        if noise is None:
            noise = torch.zeros_like(pair[0])
        noise = noise.to(device=self.device, dtype=self.latent_dtype)
        cond, uncond = self._embeddings(prompt, negative_prompt)
        n = len(ts)
        for i, t in enumerate(reversed(ts)):
            g = start + (n - 1 - i)
            pair = _inv_mix(pair, mixing)
            for base_i in range(2):
                latent_i = (base_i + (g + 1) % 2) % 2
                latent_j = 1 - latent_i
                noise_pred = self._predict(pair[latent_j], t, cond, uncond, guidance_scale)
                pair[latent_i] = reverse_step(
                    self.scheduler, noise_pred, t, pair[latent_i], steps
                ).to(self.latent_dtype)
        z_t = (pair[0] + pair[1]) * 0.5
        clean = (z_t - (1 - alpha).sqrt() * noise) / alpha.sqrt()
        return [clean, clean.clone()]

    def unload(self) -> None:
        for attr in ("unet", "vae", "text_encoder"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._free()
