"""
Phase 7 — Public Semantic Key (Kpub) Generator for RACS2.

Generates a deterministic text prompt (Kpub) from an image, suitable for
conditioning Stable Diffusion reference generation. Three methods are
supported, selectable via ``configs/base.yaml  semantic.keygen.method``:

  * ``blip_template`` (default): BLIP captioning + a fixed template.
  * ``clip_vocab``: CLIP cosine similarity against a fixed vocabulary.
  * ``local_llm``: (placeholder) local instruction-tuned LLM.

Determinism contract:
  * ``model.eval()``, ``torch.inference_mode()``.
  * ``do_sample=False``, ``num_beams=1`` (greedy only).
  * Fixed dtype and device.
  * Two runs on the same image must return the exact same string.

The Kpub must not contain image_id, token, wrong_token, file path, or SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths  # noqa: E402,F401  (side effect: HF cache redirect)
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from common.device import get_device  # noqa: E402


# ── Kpub validation ──────────────────────────────────────────────────────────

def validate_kpub(
    kpub: str,
    *,
    max_chars: int = 300,
    image_id: str = "",
    token: str = "",
    wrong_token: str = "",
    path: str = "",
    sha256: str = "",
) -> list[str]:
    """Return a list of problems with a Kpub string (empty == valid)."""
    errors: list[str] = []
    if not kpub or not kpub.strip():
        errors.append("Kpub is empty or blank")
        return errors
    if len(kpub) > max_chars:
        errors.append(f"Kpub exceeds {max_chars} chars: {len(kpub)}")
    if "\n" in kpub or "\r" in kpub or "\t" in kpub:
        errors.append("Kpub contains newline or tab")
    # Must not contain identifiers
    for label, value in [
        ("image_id", image_id),
        ("token", token),
        ("wrong_token", wrong_token),
        ("path", path),
        ("sha256", sha256),
    ]:
        if value and value in kpub:
            errors.append(f"Kpub contains {label}: {value!r}")
    # Must read as a generation prompt, not a bare label
    bare_labels = {"common", "facial", "style", "calibration", "evaluation"}
    if kpub.strip().lower() in bare_labels:
        errors.append(f"Kpub is a bare label: {kpub!r}")
    return errors


# ── BLIP Template Keygen ─────────────────────────────────────────────────────

class BLIPKeygen:
    """BLIP image captioning + fixed template for Kpub generation."""

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-base",
        device: Union[str, torch.device, None] = None,
        dtype: str = "float16",
        max_new_tokens: int = 40,
        num_beams: int = 1,
        template: str = (
            "a photorealistic photograph of {caption}, "
            "natural composition, detailed textures, realistic lighting"
        ),
        max_chars: int = 300,
        model_path: Optional[str] = None,
    ) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.method = "blip_template"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.template = template
        self.max_chars = max_chars

        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Resolve dtype
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        self.dtype = dtype_map.get(str(dtype).lower(), torch.float16)

        # Determine source
        source = model_id
        if model_path:
            candidate = Path(model_path)
            if not candidate.is_absolute():
                candidate = _ROOT / candidate
            if candidate.exists():
                source = str(candidate)

        self.model_source = source
        self.processor = BlipProcessor.from_pretrained(source)

        try:
            self.model = BlipForConditionalGeneration.from_pretrained(
                source, torch_dtype=self.dtype
            )
        except TypeError:
            self.model = BlipForConditionalGeneration.from_pretrained(source)

        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Compute weight hash for manifest
        self._weight_sha256 = self._compute_weight_hash()

    def _compute_weight_hash(self) -> str:
        """SHA-256 of sorted parameter names + shapes (lightweight fingerprint)."""
        h = hashlib.sha256()
        for name, param in sorted(self.model.named_parameters()):
            h.update(f"{name}:{list(param.shape)}\n".encode())
        return h.hexdigest()

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def generate_kpub(
        self,
        image: Image.Image,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a Kpub from an image.

        Returns dict with keys: kpub, method, model_id, raw_caption, weight_sha256.
        """
        image_rgb = image.convert("RGB")

        with torch.inference_mode():
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Move pixel_values to model dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )

        raw_caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        raw_caption = raw_caption.strip()

        # Apply template
        kpub = self.template.format(caption=raw_caption)

        # Enforce max_chars
        if len(kpub) > self.max_chars:
            kpub = kpub[: self.max_chars]

        return {
            "kpub": kpub,
            "method": self.method,
            "model_id": self.model_id,
            "raw_caption": raw_caption,
            "weight_sha256": self._weight_sha256,
        }

    def unload(self):
        """Free GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── CLIP Vocabulary Keygen (fallback) ────────────────────────────────────────

class CLIPVocabKeygen:
    """CLIP cosine similarity against a fixed vocabulary."""

    def __init__(
        self,
        vocab: Optional[Dict[str, list]] = None,
        template: str = (
            "a photorealistic photograph of {subject} in {setting}, "
            "{style_desc}, natural composition, detailed textures"
        ),
        max_chars: int = 300,
        **clip_kwargs,
    ) -> None:
        from models.clip_encoder import load_clip

        self.method = "clip_vocab"
        self.template = template
        self.max_chars = max_chars

        self.vocab = vocab or {
            "subject": [
                "a person", "a man", "a woman", "a child",
                "a dog", "a cat", "an animal", "food on a plate",
                "a vehicle", "a building", "a natural landscape",
                "a group of people", "an everyday object",
            ],
            "setting": [
                "an indoor room", "an outdoor park", "a city street",
                "a natural landscape", "a studio background",
            ],
            "style_desc": [
                "sharp focus", "soft focus", "dramatic lighting",
                "natural daylight", "warm tones",
            ],
        }

        self.encoder = load_clip(**clip_kwargs)
        self.model_id = self.encoder.model_id

        # Pre-encode vocabulary
        self._vocab_embeddings = {}
        for category, phrases in sorted(self.vocab.items()):
            self._vocab_embeddings[category] = (
                sorted(phrases),
                self.encoder.encode_texts(sorted(phrases), normalize=True),
            )

        self._weight_sha256 = "clip_vocab_no_separate_weights"

    @property
    def weight_sha256(self) -> str:
        return self._weight_sha256

    def generate_kpub(
        self,
        image: Image.Image,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        img_vec = self.encoder.encode_images([image.convert("RGB")], normalize=True)

        selections = {}
        for category, (phrases, vecs) in self._vocab_embeddings.items():
            sims = (img_vec @ vecs.T).flatten()
            best_idx = int(np.argmax(sims))
            selections[category] = phrases[best_idx]

        kpub = self.template.format(**selections)
        if len(kpub) > self.max_chars:
            kpub = kpub[: self.max_chars]

        return {
            "kpub": kpub,
            "method": self.method,
            "model_id": self.model_id,
            "raw_caption": None,
            "weight_sha256": self._weight_sha256,
        }

    def unload(self):
        if hasattr(self, "encoder"):
            del self.encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Factory ──────────────────────────────────────────────────────────────────

def build_keygen(config: dict) -> Union[BLIPKeygen, CLIPVocabKeygen]:
    """Build a keygen backend from the ``semantic.keygen`` config block."""
    method = config.get("method", "blip_template")

    if method == "blip_template":
        return BLIPKeygen(
            model_id=config.get("model_id", "Salesforce/blip-image-captioning-base"),
            device=config.get("device"),
            dtype=config.get("dtype", "float16"),
            max_new_tokens=config.get("max_new_tokens", 40),
            num_beams=config.get("num_beams", 1),
            template=config.get(
                "template",
                "a photorealistic photograph of {caption}, "
                "natural composition, detailed textures, realistic lighting",
            ),
            max_chars=config.get("max_chars", 300),
            model_path=config.get("model_path"),
        )
    elif method == "clip_vocab":
        return CLIPVocabKeygen(
            template=config.get(
                "template",
                "a photorealistic photograph of {subject} in {setting}, "
                "{style_desc}, natural composition, detailed textures",
            ),
            max_chars=config.get("max_chars", 300),
        )
    else:
        raise ValueError(f"unknown keygen method: {method!r}")
