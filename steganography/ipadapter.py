"""IPAdapter-plus reference conditioning (paper Stage 3/5).

The paper injects the reference image ``Xref`` into the diffusion process with
IPAdapter-plus at weight 1.0. This module attaches the official
``h94/IP-Adapter`` plus weights to the EDICT UNet and turns a reference image
into the embeddings that ``UNet2DConditionModel`` expects in
``added_cond_kwargs["image_embeds"]``.

Two properties matter here:

* **Determinism.** The adapter and its image encoder are frozen, so the same
  reference image always produces the same embedding and the transmitter and
  receiver condition the inverse sampler identically.
* **Memory.** The plus adapter's image encoder is CLIP-H, a 2.5 GB checkpoint
  that does not fit beside Stable Diffusion on a 6 GB card, and holding it in
  float32 on the host alongside SwinJSCC and CodeFormer exhausts the Windows
  commit budget. So the encoder is loaded *lazily* and every reference
  embedding is cached on disk, keyed by the reference file's SHA-256. After
  ``scripts/phase8_embeds.py`` has filled the cache, the baseline run never
  loads the encoder at all.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401
import common.hf_lowmem  # noqa: E402

# The image encoder is a 2.5 GB checkpoint; without the read-only mmap shim its
# load blows the Windows paging file (OSError 1455 / exit 0xC0000005).
common.hf_lowmem.install()

from diffusers.loaders import IPAdapterMixin  # noqa: E402
from diffusers.models.embeddings import ImageProjection  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (  # noqa: E402
    StableDiffusionPipeline,
)

DEFAULT_CACHE_DIR = "experiments/results/ipadapter_embeds"


class _IPAdapterHost(IPAdapterMixin):
    """Minimal pipeline-shaped object so ``load_ip_adapter`` can run."""

    def __init__(self, unet, *, device: torch.device, dtype: torch.dtype) -> None:
        self.unet = unet
        self.image_encoder = None
        self.feature_extractor = None
        self.device = device
        self.dtype = dtype

    def register_modules(self, **kwargs) -> None:
        for name, module in kwargs.items():
            setattr(self, name, module)

    def register_to_config(self, **kwargs) -> None:
        return None


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def build_ipadapter(
    unet,
    *,
    root: str | Path,
    subfolder: str = "models",
    weight_name: str = "ip-adapter-plus_sd15.safetensors",
    image_encoder_folder: str = "models/image_encoder",
    scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    encoder_device: str | torch.device = "cpu",
    encoder_dtype: torch.dtype = torch.float32,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    load_encoder: bool = False,
) -> Dict[str, object]:
    """Attach IPAdapter-plus to ``unet`` in place and return a handle.

    The projection layers follow the UNet. The image encoder is only loaded
    when ``load_encoder`` is true or when a reference misses the disk cache.
    """
    root = _resolve(root)
    if not root.exists():
        raise FileNotFoundError(f"IPAdapter root not found: {root}")
    weight_path = root / subfolder / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"IPAdapter weights not found: {weight_path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_device = torch.device(encoder_device)

    # ``load_ip_adapter`` reads host.device/host.dtype for the image encoder
    # only, so the host advertises the encoder's placement.
    host = _IPAdapterHost(unet, device=encoder_device, dtype=encoder_dtype)
    host.load_ip_adapter(
        str(root),
        subfolder=subfolder,
        weight_name=weight_name,
        image_encoder_folder=image_encoder_folder if load_encoder else None,
        low_cpu_mem_usage=True,
    )
    host.set_ip_adapter_scale(float(scale))

    handle: Dict[str, object] = {
        "host": host,
        "scale": float(scale),
        "root": str(root),
        "subfolder": subfolder,
        "weight_name": weight_name,
        "weight_sha256": _file_sha256(weight_path),
        "image_encoder_folder": image_encoder_folder,
        "encoder_device": encoder_device,
        "encoder_dtype": encoder_dtype,
        "unet_device": device,
        "unet_dtype": dtype,
        "cache_dir": str(_resolve(cache_dir)),
        "hits": 0,
        "misses": 0,
    }
    return handle


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_encoder(handle: Dict[str, object]) -> None:
    """Load CLIP-H on first cache miss (and only then)."""
    host = handle["host"]
    if getattr(host, "image_encoder", None) is not None:
        return
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    folder = str(handle["image_encoder_folder"])
    source = _resolve(handle["root"])
    encoder_dir = (source / folder if (source / folder).exists()
                   else source / str(handle["subfolder"]) / folder)
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(encoder_dir), torch_dtype=handle["encoder_dtype"], low_cpu_mem_usage=True,
    ).to(handle["encoder_device"]).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    host.image_encoder = encoder
    if getattr(host, "feature_extractor", None) is None:
        host.feature_extractor = CLIPImageProcessor.from_pretrained(str(encoder_dir)) \
            if (encoder_dir / "preprocessor_config.json").exists() \
            else CLIPImageProcessor(size=224, crop_size=224)


def _is_hidden_state_adapter(unet) -> bool:
    layers = unet.encoder_hid_proj.image_projection_layers
    return not isinstance(layers[0], ImageProjection)


@torch.inference_mode()
def _encode(handle: Dict[str, object], image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the image encoder once; returns the positive and negative embedding."""
    _ensure_encoder(handle)
    host = handle["host"]
    pos, neg = StableDiffusionPipeline.encode_image(
        host, image.convert("RGB"), host.device, 1,
        _is_hidden_state_adapter(host.unet),
    )
    return pos.detach().cpu(), neg.detach().cpu()


def _as_list(handle: Dict[str, object], pos: torch.Tensor, neg: torch.Tensor,
             guidance_scale: float) -> List[torch.Tensor]:
    """Shape the cached embedding the way the UNet wants it, CFG-aware."""
    device = handle.get("unet_device")
    dtype = handle.get("unet_dtype") or torch.float16
    pos = pos[None, :].to(device=device, dtype=dtype)
    neg = neg[None, :].to(device=device, dtype=dtype)
    if float(guidance_scale) == 1.0:
        return [pos]
    return [torch.cat([neg, pos], dim=0)]


def _cache_file(handle: Dict[str, object], key: str) -> Path:
    return Path(str(handle["cache_dir"])) / f"{key}.npz"


def reference_embeds(
    handle: Dict[str, object],
    image: Image.Image,
    *,
    guidance_scale: float,
) -> List[torch.Tensor]:
    """Embed an in-memory ``Xref``. Prefer the path form; this loads CLIP-H."""
    pos, neg = _encode(handle, image)
    handle["misses"] = int(handle["misses"]) + 1
    return _as_list(handle, pos, neg, guidance_scale)


def reference_embeds_for_path(
    handle: Dict[str, object],
    reference_path: str | Path,
    *,
    guidance_scale: float,
    write_cache: bool = True,
) -> List[torch.Tensor]:
    """Embed ``Xref`` from disk, using (and filling) the SHA-256 keyed cache.

    The transmitter and the receiver both come through here, so both ends read
    the identical cached tensor rather than re-deriving it from pixels.
    """
    path = _resolve(reference_path)
    key = _file_sha256(path)
    cache_file = _cache_file(handle, key)
    if cache_file.exists():
        with np.load(cache_file) as blob:
            pos = torch.from_numpy(blob["pos"])
            neg = torch.from_numpy(blob["neg"])
        handle["hits"] = int(handle["hits"]) + 1
        return _as_list(handle, pos, neg, guidance_scale)

    pos, neg = _encode(handle, Image.open(path))
    handle["misses"] = int(handle["misses"]) + 1
    if write_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        # np.savez appends ".npz" to a *path*; pass a handle so the name stands.
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                pos=pos.to(torch.float16).numpy(),
                neg=neg.to(torch.float16).numpy(),
                reference_sha256=np.array(key),
                weight_sha256=np.array(str(handle["weight_sha256"])),
                encoder_folder=np.array(str(handle["image_encoder_folder"])),
            )
        tmp.replace(cache_file)
    return _as_list(handle, pos, neg, guidance_scale)


def cache_stats(handle: Dict[str, object]) -> Dict[str, int]:
    return {"hits": int(handle.get("hits", 0)), "misses": int(handle.get("misses", 0))}


def unload_encoder(handle: Dict[str, object]) -> None:
    """Drop CLIP-H but keep the adapter attached to the UNet."""
    host = handle.get("host")
    if host is None:
        return
    host.image_encoder = None
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_ipadapter(handle: Dict[str, object]) -> None:
    host = handle.get("host")
    if host is None:
        return
    try:
        host.unload_ip_adapter()
    except Exception:  # noqa: BLE001
        pass
    unload_encoder(handle)
