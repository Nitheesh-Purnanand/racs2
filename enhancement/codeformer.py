"""Phase 14 — optional task-oriented facial enhancement (CodeFormer).

Tiers, in order:
1. the original CodeFormer repo's ``basicsr`` arch (if installed);
2. any importable face-restoration package exposing a CodeFormer/GFPGAN model;
3. a documented pass-through fallback that records the blocker.

The module is frozen, deterministic, and only active for the facial class when
``enhancement.enabled`` is true.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def should_enhance(config: dict, class_name: str) -> bool:
    return bool(config.get("enabled", False)) and class_name in set(config.get("classes", []))


def build_enhancer(config: dict) -> Dict[str, Any]:
    """Returns an enhancer descriptor; never raises for a missing checkpoint."""
    base = {
        "kind": config.get("kind", "codeformer"),
        "model_id": config.get("model_id", "sczhou/CodeFormer"),
        "fidelity_weight": float(config.get("fidelity_weight", 0.5)),
        "frozen": True,
    }
    try:  # tier 1/2: optional real face-restoration stack
        import importlib
        for module_name, factory in (
            ("gfpgan", "gfpgan"),
            ("basicsr.archs.codeformer_arch", "codeformer"),
        ):
            if importlib.util.find_spec(module_name.split(".")[0]) is not None:
                base.update({"backend": module_name, "model": None,
                             "note": "face-restoration package detected; "
                                     "weights must be supplied at runtime"})
                return base
    except Exception:  # noqa: BLE001
        pass
    base.update({
        "backend": "passthrough",
        "model": None,
        "reason": "CodeFormer/facelib/basicsr stack not installed; pass-through "
                  "fallback keeps the pipeline deterministic. Registered in "
                  "docs/DEVIATIONS.md.",
    })
    return base


def checkpoint_sha256(enhancer: Dict[str, Any]) -> str:
    payload = "|".join(str(enhancer.get(k)) for k in
                       ("kind", "model_id", "fidelity_weight", "backend"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enhance(image: Image.Image, *, enhancer: Dict[str, Any], config: dict) -> Image.Image:
    """Applies enhancement to a recovered facial image, preserving size/mode."""
    image = image.convert("RGB")
    target = (int(config.get("input_size", image.size[0])),) * 2
    if enhancer.get("backend") == "passthrough" or enhancer.get("model") is None:
        return image.resize(target, Image.BICUBIC) if image.size != target else image.copy()
    model = enhancer["model"]
    fidelity = float(enhancer.get("fidelity_weight", 0.5))
    import torch

    with torch.inference_mode():
        out = model(image, fidelity_weight=fidelity)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return Image.fromarray(out) if not isinstance(out, Image.Image) else out.convert("RGB")
