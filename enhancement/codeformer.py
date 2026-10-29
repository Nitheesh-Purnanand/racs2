"""Phase 14 — optional task-oriented facial enhancement (CodeFormer).

Uses the official CodeFormer architecture and the released ``codeformer.pth``
checkpoint from the vendored ``third_party/CodeFormer`` tree. The network is
frozen, runs at 512 px, and uses the paper's fidelity weight 0.5. Enhancement
is facial-only and off unless enabled in the config.

The earlier pass-through fallback is retained only as an explicit, recorded
degraded mode when the checkpoint is absent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CODEFORMER_ROOT = _ROOT / "third_party" / "CodeFormer"
if str(_CODEFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEFORMER_ROOT))

# basicsr 1.4 imports a module removed in torchvision >= 0.17.
import torchvision.transforms.functional as _tv_functional  # noqa: E402

sys.modules.setdefault("torchvision.transforms.functional_tensor", _tv_functional)


def should_enhance(config: dict, class_name: str) -> bool:
    return bool(config.get("enabled", False)) and class_name in set(config.get("classes", []))


def _weights_dir(config: dict) -> Path:
    candidate = Path(config.get("weights_dir", "models/codeformer"))
    return candidate if candidate.is_absolute() else _ROOT / candidate


def build_enhancer(config: dict) -> Dict[str, Any]:
    """Build the real CodeFormer, or a recorded pass-through if unavailable."""
    weights = _weights_dir(config)
    checkpoint = weights / "codeformer.pth"
    base = {
        "kind": config.get("kind", "codeformer"),
        "model_id": config.get("model_id", "sczhou/CodeFormer"),
        "fidelity_weight": float(config.get("fidelity_weight", 0.5)),
        "frozen": True,
        "checkpoint": str(checkpoint),
    }
    if not checkpoint.exists():
        base.update({
            "backend": "passthrough",
            "model": None,
            "reason": f"CodeFormer checkpoint missing at {checkpoint}",
        })
        return base

    import torch
    from basicsr.archs.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("params_ema", state.get("params", state))
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"CodeFormer checkpoint missing {len(missing)} tensors: {missing[:4]}"
        )
    requested = torch.device(config.get("device", "cuda"))
    device = requested if (requested.type != "cuda" or torch.cuda.is_available())         else torch.device("cpu")
    dtype = torch.float16 if str(config.get("dtype", "float16")).lower() in (
        "float16", "fp16") and device.type == "cuda" else torch.float32
    net = net.to(device=device, dtype=dtype).eval()
    for param in net.parameters():
        param.requires_grad_(False)
    base.update({
        "backend": "codeformer",
        "model": net,
        "device": device,
        "dtype": dtype,
        "input_size": int(config.get("input_size", 512)),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    })
    return base


def checkpoint_sha256(enhancer: Dict[str, Any]) -> str:
    if enhancer.get("sha256"):
        return str(enhancer["sha256"])
    payload = "|".join(str(enhancer.get(k)) for k in
                       ("kind", "model_id", "fidelity_weight", "backend"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enhance(image: Image.Image, *, enhancer: Dict[str, Any], config: dict) -> Image.Image:
    """Apply CodeFormer to a recovered facial image, preserving output size."""
    image = image.convert("RGB")
    original_size = image.size
    model = enhancer.get("model")
    if enhancer.get("backend") != "codeformer" or model is None:
        return image.copy()

    import torch

    size = int(enhancer.get("input_size", 512))
    face = image.resize((size, size), Image.BICUBIC)
    arr = np.asarray(face, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor * 2.0 - 1.0).to(
        device=enhancer["device"], dtype=enhancer.get("dtype", torch.float32)
    )
    fidelity = float(enhancer.get("fidelity_weight", 0.5))
    with torch.inference_mode():
        output = model(tensor, w=fidelity, adain=True)[0]
    out = output[0].permute(1, 2, 0).float().cpu().numpy()
    out = np.clip(out * 0.5 + 0.5, 0.0, 1.0)
    restored = Image.fromarray((out * 255.0).round().astype("uint8"))
    return restored.resize(original_size, Image.BICUBIC)
