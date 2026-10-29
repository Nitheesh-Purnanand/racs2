"""Phase 8 — deterministic public-condition reference generation (SD + ControlNet)."""

from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from PIL import Image  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from common.device import get_device  # noqa: E402
from reference.seed import derive_seed  # noqa: E402


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fingerprint(module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        digest.update(f"{name}:{list(param.shape)}:{param.dtype}".encode())
    return digest.hexdigest()


def resolve_model_source(model_id: str) -> str:
    """Accept either a hub id or a repo-relative local model directory."""
    candidate = Path(model_id)
    if not candidate.is_absolute():
        candidate = _ROOT / model_id
    return str(candidate) if candidate.exists() else str(model_id)


def build_pipeline(class_name: str, config: dict):
    """Build the base or ControlNet pipeline for one class.

    ``backend: picx_real`` points ``model_id`` at the local PicX-Real
    checkpoint (the paper's reference generator); ``backend: sd_controlnet``
    keeps plain SD v1.5. Either way the pipeline is SD v1.5-shaped, so the
    SD v1.5 ControlNets apply unchanged. Any component the primary checkpoint
    does not ship is taken from ``fallback_model_id`` and recorded.
    """
    device = get_device()
    dtype = torch.float16 if config.get("dtype", "float16") == "float16" else torch.float32
    device_map = str(device) if device.type == "cuda" else None
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if device_map:
        load_kwargs["device_map"] = device_map
    model_id = config.get("model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    primary = resolve_model_source(model_id)
    fallback = resolve_model_source(
        config.get("fallback_model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    )
    sources: Dict[str, str] = {}

    def component(name: str, loader, **kwargs):
        """Load one sub-module from the primary checkpoint, else the fallback."""
        try:
            module = loader.from_pretrained(primary, subfolder=name, **kwargs)
            sources[name] = primary
            return module
        except Exception as exc:  # noqa: BLE001
            if primary == fallback:
                raise
            print(f"[reference] {name}: primary checkpoint unusable ({type(exc).__name__}); "
                  f"falling back to {fallback}")
            module = loader.from_pretrained(fallback, subfolder=name, **kwargs)
            sources[name] = fallback
            return module

    # Load components one by one. Loading the whole pipeline at once duplicates
    # the ControlNet on CPU and GPU, which exhausts the 6 GB card's commit
    # budget on Windows.
    unet = component("unet", UNet2DConditionModel, **load_kwargs)
    vae = component("vae", AutoencoderKL, **load_kwargs)
    text_encoder = component("text_encoder", CLIPTextModel, **load_kwargs)
    tokenizer = component("tokenizer", CLIPTokenizer)
    scheduler = component("scheduler", DDIMScheduler)

    if class_name == "style":
        controlnet = None
        pipe = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=scheduler, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        )
    else:
        cn_id = (config.get("controlnet_id")
                 if class_name == "common"
                 else config.get("facial_controlnet_id", config.get("controlnet_id")))
        controlnet = ControlNetModel.from_pretrained(cn_id, **load_kwargs)
        pipe = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, scheduler=scheduler, safety_checker=None,
            feature_extractor=None, requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:  # noqa: BLE001
        pass
    pipe.racs_sources = {
        "backend": config.get("backend", "sd_controlnet"),
        "model_id": model_id,
        "resolved": primary,
        "components": sources,
    }
    return pipe, controlnet


def generate_reference(
    pipe,
    *,
    prompt: str,
    control: Optional[Image.Image],
    token: str,
    output_size: int = 256,
    steps: int = 20,
    guidance_scale: float = 7.5,
) -> Dict[str, Any]:
    """Generate one reference with token-seeded initial noise."""
    seed = derive_seed(token)
    device = pipe.device
    latent_size = output_size // 8
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 4, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    latents_sha256 = hashlib.sha256(
        latents.detach().float().cpu().numpy().tobytes()
    ).hexdigest()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "latents": latents,
        "eta": 0.0,
        "output_type": "pil",
    }
    if control is not None:
        kwargs["image"] = control
    result = pipe(**kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0
    return {
        "image": result.images[0],
        "seed": seed,
        "latents_sha256": latents_sha256,
        "peak_vram_gb": float(peak),
        "latency_s": float(elapsed),
    }


def unload(pipe) -> None:
    """Release a pipeline's GPU memory.

    Callers must also drop their own reference (``pipe = None``); verifiers run
    one class per subprocess so a lingering handle cannot block the next load.
    """
    del pipe
    _free()
