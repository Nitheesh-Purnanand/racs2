"""Phase 9 — zero-conv ControlNet fine-tuning (paper Algorithm 2)."""

from __future__ import annotations

import gc
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from diffusers import (  # noqa: E402
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from PIL import Image  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from common.device import get_device  # noqa: E402


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_components(model_id: str, controlnet_id: str, dtype=torch.float16):
    device = get_device()
    kwargs = {"torch_dtype": dtype, "device_map": str(device)} if device.type == "cuda" else {}
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kwargs)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", **kwargs
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    controlnet = ControlNetModel.from_pretrained(controlnet_id, **kwargs)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    for module in (unet, vae, text_encoder):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    controlnet.train()
    for name, param in controlnet.named_parameters():
        param.requires_grad_("controlnet_down_blocks" in name or
                             "controlnet_mid_block" in name)
    return {
        "device": device,
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "controlnet": controlnet,
        "scheduler": scheduler,
    }


def encode_prompt(tokenizer, text_encoder, prompt: str, device) -> torch.Tensor:
    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(tokens.input_ids.to(device)).last_hidden_state


def image_to_latent(vae, image: Image.Image, size: int, device, dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC)).astype(
        "float32"
    ) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae.encode(tensor.to(dtype)).latent_dist.mode()
    return latent * 0.18215


def control_to_tensor(image: Image.Image, size: int, device, dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC)).astype(
        "float32"
    ) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def trainable_state_dict(controlnet) -> Dict[str, torch.Tensor]:
    trainable = {name for name, param in controlnet.named_parameters()
                 if param.requires_grad}
    return {k: v.detach().cpu() for k, v in controlnet.state_dict().items()
            if k in trainable}


def save_checkpoint(controlnet, path: Path) -> str:
    from safetensors.torch import save_file

    state = trainable_state_dict(controlnet)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(state, str(tmp))
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load_finetuned_controlnet(
    base_controlnet_id: str,
    checkpoint_path: Path | str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    return_info: bool = False,
):
    """Load the frozen base ControlNet and apply the zero-conv delta checkpoint."""
    from safetensors.torch import load_file

    kwargs = {"torch_dtype": dtype}
    if device is not None and device.type == "cuda":
        kwargs["device_map"] = str(device)
    controlnet = ControlNetModel.from_pretrained(base_controlnet_id, **kwargs)
    state = load_file(str(checkpoint_path))
    trainable = {name for name, param in controlnet.named_parameters()
                 if "controlnet_down_blocks" in name or "controlnet_mid_block" in name}
    base_state = {name: param.detach().cpu().clone()
                  for name, param in controlnet.named_parameters()
                  if name in trainable}
    missing = trainable - set(state)
    unexpected = set(state) - trainable
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match zero-conv parameters: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    result = controlnet.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {result.unexpected_keys[:5]}")
    changed = sum(
        int(not torch.equal(param.detach().cpu(), base_state[name]))
        for name, param in controlnet.named_parameters() if name in trainable
    )
    info = {
        "checkpoint_tensors": len(state),
        "trainable_tensors": len(trainable),
        "changed_tensors": changed,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "frozen_missing_keys": len(result.missing_keys),
    }
    return (controlnet, info) if return_info else controlnet


def probe_loss(
    controlnet,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    samples: List[Dict],
    *,
    size: int = 256,
    seed: int = 42,
    timestep: int = 500,
) -> float:
    """Fixed-seed noise-prediction MSE on a held-out calibration sample."""
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    losses = []
    was_training = controlnet.training
    controlnet.eval()
    try:
        with torch.no_grad():
            for sample in samples:
                latent = image_to_latent(vae, sample["image"], size, device, dtype)
                noise = torch.randn(latent.shape, generator=gen, device=device, dtype=dtype)
                t = torch.full((1,), timestep, device=device, dtype=torch.long)
                noisy = scheduler.add_noise(latent, noise, t)
                embeds = encode_prompt(tokenizer, text_encoder, sample["prompt"], device)
                control = control_to_tensor(sample["control"], size, device, dtype)
                down, mid = controlnet(
                    noisy, t, encoder_hidden_states=embeds, controlnet_cond=control,
                    return_dict=False,
                )
                pred = unet(
                    noisy, t, encoder_hidden_states=embeds,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                ).sample
                losses.append(float((pred - noise).pow(2).mean()))
    finally:
        if was_training:
            controlnet.train()
    return float(np.mean(losses))


def fingerprint(module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        digest.update(f"{name}:{list(param.shape)}".encode())
    return digest.hexdigest()
