"""Phase 9 — train the ControlNet zero-conv branch on calibration data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths
import common.hf_lowmem

common.hf_lowmem.install()

import numpy as np
import torch
import yaml
from PIL import Image

from common.seed import set_seed
from evaluation.splits import Split
from reference.train_controlnet import (
    control_to_tensor,
    encode_prompt,
    fingerprint,
    image_to_latent,
    load_components,
    probe_loss,
    save_checkpoint,
    trainable_state_dict,
)
from semantic.cache import read_cache_npz, sha256_string


def _samples(split: Split, cache_dir: Path, classes: List[str]) -> List[Dict[str, Any]]:
    out = []
    for cls in classes:
        recs = sorted(split.records(cls), key=lambda r: r["image_id"])
        for rec in recs:
            cond = read_cache_npz(cache_dir / f"{rec['image_id']}.npz")
            preview = cache_dir / f"{rec['image_id']}_preview.png"
            out.append({
                "image_id": rec["image_id"],
                "class": cls,
                "image": Image.open(paths.PROJECT_ROOT / rec["path"]).convert("RGB"),
                "control": Image.open(preview).convert("RGB"),
                "prompt": cond["kpub"],
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: ControlNet zero-conv fine-tune")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    set_seed(42, verbose=True)
    with open(paths.PROJECT_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    cfg = config.get("controlnet_train", {})
    size = args.size or int(cfg.get("image_size", 256))
    max_steps = args.max_steps or int(cfg.get("max_steps", 50))
    lr = float(cfg.get("learning_rate", 1e-5))
    model_id = cfg.get("base_model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    cn_id = cfg.get("controlnet_id", "lllyasviel/control_v11p_sd15_seg")
    classes = list(cfg.get("classes", ["common", "facial"]))

    split = Split.load(role="calibration", allow_calibration=True)
    cache_dir = paths.RESULTS_DIR / "public_conditions"
    all_samples = _samples(split, cache_dir, classes)
    probe_per_class = int(cfg.get("probe_size", 10)) // max(1, len(classes))
    train_samples, probe_samples = [], []
    for cls in classes:
        cls_samples = [s for s in all_samples if s["class"] == cls]
        train_samples.extend(cls_samples[:-probe_per_class])
        probe_samples.extend(cls_samples[-probe_per_class:])
    print(f"[phase9] train={len(train_samples)} probe={len(probe_samples)}", flush=True)

    train_dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    comp = load_components(model_id, cn_id, dtype=train_dtype)
    controlnet, unet, vae = comp["controlnet"], comp["unet"], comp["vae"]
    text_encoder, tokenizer = comp["text_encoder"], comp["tokenizer"]
    scheduler, device = comp["scheduler"], comp["device"]

    if args.resume:
        from safetensors.torch import load_file
        state = load_file(args.resume)
        controlnet.load_state_dict(state, strict=False)
        print(f"[phase9] resumed from {args.resume}", flush=True)

    trainable = [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    initial_state = trainable_state_dict(controlnet)

    zero_probe = probe_loss(
        controlnet, unet, vae, text_encoder, tokenizer, scheduler, probe_samples,
        size=size,
    )
    print(f"[phase9] zero-shot probe loss={zero_probe:.6f}", flush=True)

    losses = []
    t0 = time.time()
    for step in range(max_steps):
        sample = train_samples[step % len(train_samples)]
        latent = image_to_latent(vae, sample["image"], size, device,
                                 next(unet.parameters()).dtype)
        dtype = next(unet.parameters()).dtype
        noise = torch.randn(latent.shape, device=device, dtype=dtype)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = scheduler.add_noise(latent, noise, t)
        embeds = encode_prompt(tokenizer, text_encoder, sample["prompt"], device)
        control = control_to_tensor(sample["control"], size, device, dtype)

        with torch.enable_grad():
            down, mid = controlnet(
                noisy, t, encoder_hidden_states=embeds, controlnet_cond=control,
                return_dict=False,
            )
            pred = unet(
                noisy, t, encoder_hidden_states=embeds,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
            ).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1}/{max_steps} loss={losses[-1]:.6f} "
                  f"vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)

    final_probe = probe_loss(
        controlnet, unet, vae, text_encoder, tokenizer, scheduler, probe_samples,
        size=size,
    )
    print(f"[phase9] fine-tuned probe loss={final_probe:.6f}", flush=True)

    ckpt_dir = paths.PROJECT_ROOT / cfg.get("checkpoint_dir", "experiments/checkpoints")
    config_hash = sha256_string(json.dumps(cfg, sort_keys=True))
    ckpt_path = ckpt_dir / f"controlnet_zero_conv_{max_steps}_{config_hash[:8]}.safetensors"
    ckpt_sha = save_checkpoint(controlnet, ckpt_path)
    trainable_count = sum(p.numel() for p in trainable)
    frozen_count = sum(p.numel() for p in controlnet.parameters() if not p.requires_grad)

    result = {
        "schema_version": 1,
        "approximate": False,
        "config_hash": config_hash,
        "split_role": "calibration",
        "split_version": split.version,
        "model_id": model_id,
        "controlnet_id": cn_id,
        "controlnet_fingerprint": fingerprint(controlnet),
        "image_size": size,
        "trainable_parameters": trainable_count,
        "frozen_parameters": frozen_count,
        "steps": max_steps,
        "batch_size": int(cfg.get("batch_size", 1)),
        "grad_accum_steps": int(cfg.get("grad_accum_steps", 1)),
        "learning_rate": lr,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "probe": {
            "stage": "reference_only",
            "zero_shot_mse": zero_probe,
            "fine_tuned_mse": final_probe,
            "improved": final_probe < zero_probe,
            "n": len(probe_samples),
        },
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
        "elapsed_s": round(time.time() - t0, 2),
        "checkpoint_path": str(ckpt_path.relative_to(paths.PROJECT_ROOT)),
        "checkpoint_sha256": ckpt_sha,
        "decision": "adopt" if final_probe < zero_probe else "pretrained_deviation",
        "initial_state_fingerprint": hashlib.sha256(
            b"".join(v.to(torch.float32).numpy().tobytes()
                     for _, v in sorted(initial_state.items()))
        ).hexdigest(),
    }
    out = paths.RESULTS_DIR / "phase9_training.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"[phase9] wrote {out}", flush=True)
    comp["controlnet"].unload() if hasattr(comp["controlnet"], "unload") else None


if __name__ == "__main__":
    main()
