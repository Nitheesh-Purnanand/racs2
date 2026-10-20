"""
Phase 5 — CLIP Encoder for RACS2.

Frozen, off-the-shelf CLIP (default ViT-B/32) used to map images and text into a
shared 512-d embedding space. Both transmitter and receiver must use the *same*
model id, preprocessing, and normalisation, otherwise their vectors differ and
retrieval (Phase 6) selects different references.

Design rules (protocol-critical):
  * image preprocessing is whatever the CLIP processor does -- never hand-roll it;
  * vectors are stored as float32 and L2-normalised, so cosine == inner product;
  * the returned rows always match the input order (no sorting anywhere).

Import order matters on Windows: ``common.paths`` redirects the HF cache to D:
and ``common.hf_lowmem.install()`` must run *before* transformers maps any
safetensors checkpoint (see ``common/hf_lowmem.py`` for the commit-charge bug).

Usage::

    import common.paths  # HF cache on D: before anything else
    from models.clip_encoder import load_clip

    enc = load_clip()
    vecs = enc.encode_texts(["a dog in a park", "a red bus"])
    print(vecs.shape, (vecs ** 2).sum(axis=1))   # (2, 512) [1. 1.]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

# Allow running this file directly as well as importing it as ``models.clip_encoder``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

# Redirect the HuggingFace cache to D: before transformers/hub import.
import common.paths  # noqa: E402,F401  (import for side effect)

# Install the read-only safetensors shim before transformers loads any weights.
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

from common.device import get_device  # noqa: E402

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_MAX_TEXT_TOKENS = 77

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    """Convert a config string (e.g. ``"float16"``) into a ``torch.dtype``."""
    if dtype is None:
        return torch.float16
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation of a 2-D float array."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32, copy=False)


def _extract_features(output) -> torch.Tensor:
    """Return the projection tensor from a transformers CLIP feature output.

    transformers >= 5 returns ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` holds the projected features; older builds return a plain
    tensor. Support both so the module survives an upgrade/downgrade.
    """
    if isinstance(output, torch.Tensor):
        return output
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    if isinstance(output, dict):
        for key in ("pooler_output", "image_embeds", "text_embeds"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"could not extract CLIP features from {type(output)!r}")


class CLIPEncoder:
    """Thin, deterministic wrapper around ``CLIPModel`` + ``CLIPProcessor``."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Union[str, torch.device, None] = None,
        dtype: Union[str, torch.dtype] = torch.float16,
        local_files_only: bool = False,
        model_path: Union[str, Path, None] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.dtype = resolve_dtype(dtype)

        # A self-contained local directory (Phase 5 fetch helper) takes precedence
        # over the Hub id, so a downloaded checkpoint is used offline.
        source = self.model_id
        if model_path:
            candidate = Path(model_path)
            if not candidate.is_absolute():
                candidate = _ROOT / candidate
            if candidate.exists():
                source = str(candidate)
                local_files_only = True
                self.model_path = candidate
            else:
                self.model_path = None
                print(f"[CLIP] model_path not found, falling back to Hub id: {candidate}")
        else:
            self.model_path = None
        self.model_source = source

        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                self.device = get_device()

        try:
            self.model = CLIPModel.from_pretrained(
                source,
                dtype=self.dtype,
                local_files_only=local_files_only,
            )
        except TypeError:
            # Older transformers only knows ``torch_dtype``.
            self.model = CLIPModel.from_pretrained(
                source,
                torch_dtype=self.dtype,
                local_files_only=local_files_only,
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(
            source, local_files_only=local_files_only
        )

        # CLIP preprocessing rule, documented so the manifest can record it.
        self.preprocess = (
            "resize_shortest_side_224_bicubic -> center_crop_224 -> clip_mean_std"
        )

    # -- metadata ----------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        """Projection dimension (512 for ViT-B/32)."""
        config = self.model.config
        dim = getattr(config, "projection_dim", None)
        if dim is None:
            dim = getattr(config, "text_config", None)
            dim = getattr(dim, "projection_dim", None) if dim is not None else None
        if dim is None:
            projection = getattr(self.model, "visual_projection", None)
            dim = getattr(projection, "out_features", None)
        if dim is None:
            raise AttributeError("cannot determine CLIP embedding dimension")
        return int(dim)

    def __repr__(self) -> str:
        return (
            f"CLIPEncoder(model_id={self.model_id!r}, dim={self.embedding_dim}, "
            f"device={self.device}, dtype={self.dtype})"
        )

    # -- text --------------------------------------------------------------
    def text_token_lengths(self, texts: Sequence[str]) -> List[int]:
        """Un-truncated token counts (use to assert no silent 77-token cut)."""
        tokens = self.processor.tokenizer(
            list(texts), padding=False, truncation=False, return_tensors=None
        )
        return [len(ids) for ids in tokens["input_ids"]]

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts -> float32 ``[N, dim]`` in the same order as ``texts``."""
        items = list(texts)
        if not items:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                inputs = self.processor(
                    text=batch,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_MAX_TEXT_TOKENS,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self.model.get_text_features(**inputs)
                feats = _extract_features(output)
                chunks.append(feats.detach().float().cpu().numpy())
                del inputs, output, feats

        return self._finalize(chunks, normalize)

    # -- images ------------------------------------------------------------
    def encode_images(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode PIL images -> float32 ``[N, dim]`` in the same order."""
        items = list(images)
        if not items:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                batch = [im.convert("RGB") for im in items[start : start + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {
                    k: v.to(self.device, dtype=self.dtype)
                    if v.is_floating_point()
                    else v.to(self.device)
                    for k, v in inputs.items()
                }
                output = self.model.get_image_features(**inputs)
                feats = _extract_features(output)
                chunks.append(feats.detach().float().cpu().numpy())
                del inputs, output, feats

        return self._finalize(chunks, normalize)

    def encode_image_paths(
        self,
        paths: Iterable[Union[str, Path]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode image files -> float32 ``[N, dim]`` in the same order."""
        items = [Path(p) for p in paths]
        images: List[Image.Image] = []
        for path in items:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        return self.encode_images(images, batch_size=batch_size, normalize=normalize)

    # -- internals ---------------------------------------------------------
    def _finalize(self, chunks: List[np.ndarray], normalize: bool) -> np.ndarray:
        array = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if array.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"encoder produced dim {array.shape[1]}, expected {self.embedding_dim}"
            )
        if normalize:
            array = l2_normalize(array)
        return array


def load_clip(
    model_id: str = DEFAULT_MODEL_ID,
    device: Union[str, torch.device, None] = None,
    dtype: Union[str, torch.dtype] = torch.float16,
    local_files_only: bool = False,
    model_path: Union[str, Path, None] = None,
) -> CLIPEncoder:
    """Load the CLIP encoder (default: ViT-B/32, fp16, best available device)."""
    return CLIPEncoder(
        model_id=model_id,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        model_path=model_path,
    )


if __name__ == "__main__":  # tiny smoke test (downloads the model on first run)
    encoder = load_clip()
    print(encoder)
    texts = encoder.encode_texts(["a dog in a park", "a red bus on a street"])
    print("text embeddings :", texts.shape, (texts**2).sum(axis=1))
    blank = [Image.new("RGB", (224, 224), (128, 128, 128))]
    images = encoder.encode_images(blank)
    print("image embeddings:", images.shape, (images**2).sum(axis=1))
