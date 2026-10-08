"""
Stable Diffusion v1.5 pipeline loader for RACS2 (Phase 3).

Loads the exact generation backbone that AgentSemSteCom uses to synthesise a
reference image (``Xref``) from a semantic key. This is the stage RACS replaces
with retrieval, so we need to be able to run and measure it.

Everything here is tuned for a **6 GB RTX 3050** with tight host RAM:
- fp16 weights (halve VRAM)
- attention + VAE slicing (peak memory friendly)
- ``low_cpu_mem_usage=True`` (avoids an fp32 state-dict blow-up)
- read-only checkpoint mappings via :mod:`common.hf_lowmem` (Windows charges the
  whole file against the commit limit for safetensors' copy-on-write mmap)
- ``device_map="cuda"`` so weights stream to the GPU instead of being
  materialised in host RAM first
- ``safety_checker=None`` (deterministic; skips an extra model)

Usage:
    import common.paths  # HF cache on D: before anything else
    from models.sd_pipeline import load_pipeline, generate
    pipe = load_pipeline()
    image, secs = generate(pipe, "a photograph of an astronaut riding a horse", 30, 7.5, 42)
"""

import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Allow running this file directly as well as importing it as `models.sd_pipeline`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Redirect the HuggingFace cache to D: before any model import.
import common.paths  # noqa: E402,F401

# Patch safetensors to use read-only mappings. This MUST happen before
# transformers/diffusers import `safe_open` / `load_file`, otherwise the large
# checkpoint mmaps exhaust the Windows commit limit (OSError 1455 / MemoryError).
import common.hf_lowmem  # noqa: E402,F401

common.hf_lowmem.install()

from PIL import Image  # noqa: E402

try:
    from diffusers import DDIMScheduler, StableDiffusionPipeline
except ImportError as err:  # pragma: no cover - dependency guidance
    raise ImportError(
        "diffusers/transformers not importable. Install with: "
        "pip install diffusers transformers accelerate"
    ) from err

from common.device import get_device  # noqa: E402

# Tried in order; first that loads wins. SD v1.5 has moved repos over time.
MODEL_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
]


def free_commit_gb() -> Optional[float]:
    """Free commit charge in GB (Windows only; None elsewhere)."""
    if sys.platform != "win32":  # pragma: no cover - non-Windows dev boxes
        return None
    import ctypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPageFile / 1024 ** 3


def _release_memory() -> None:
    """Best-effort cleanup between load attempts."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_out_of_memory(err: BaseException) -> bool:
    text = f"{type(err).__name__}: {err}".lower()
    return any(
        token in text
        for token in ("memory", "paging file", "os error 1455", "cannot allocate", "out of memory")
    )


def load_pipeline(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    model_id: Optional[str] = None,
    use_ddim: bool = True,
    device_map: Optional[Any] = None,
    candidates: Optional[list] = None,
) -> StableDiffusionPipeline:
    """
    Loads Stable Diffusion v1.5 configured for low-VRAM inference.

    Args:
        device: Target device (defaults to ``get_device()``).
        dtype: Weight dtype (fp16 by default).
        model_id: Explicit HuggingFace repo id; if None, tries MODEL_CANDIDATES.
        use_ddim: Replace the scheduler with DDIM (deterministic, fewer steps).
        device_map: Passed to ``from_pretrained``. Defaults to the target device
            when it is a GPU, which streams weights straight to VRAM instead of
            materialising a full fp16 copy in host RAM. Pass ``false``-y
            (e.g. ``"cpu"``) or ``None`` with ``device='cpu'`` to opt out.
        candidates: Override the ordered list of repo ids to try.

    Returns:
        A ready-to-use ``StableDiffusionPipeline`` on ``device``.
    """
    if device is None:
        device = get_device()

    if device_map is None and device.type == "cuda":
        device_map = str(device)

    ordered = [model_id] if model_id else list(candidates or MODEL_CANDIDATES)

    last_err: Optional[Exception] = None
    pipe = None
    for candidate in ordered:
        # Prefer streaming to the GPU; fall back to a plain CPU load for repos
        # or diffusers builds that reject `device_map`.
        attempts = [device_map, None] if device_map else [None]
        for attempt_map in attempts:
            kwargs: Dict[str, Any] = {}
            if attempt_map:
                kwargs["device_map"] = attempt_map
            label = f"device_map={attempt_map!r}" if attempt_map else "no device_map"
            try:
                print(f"[sd_pipeline] loading {candidate} ({label}) ...")
                with warnings.catch_warnings():
                    # diffusers 0.41 renamed `torch_dtype` -> `dtype`; the old name
                    # still works and keeps this compatible with older releases.
                    warnings.filterwarnings(
                        "ignore", message=r".*`torch_dtype` is deprecated.*", category=FutureWarning
                    )
                    pipe = StableDiffusionPipeline.from_pretrained(
                        candidate,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        low_cpu_mem_usage=True,
                        **kwargs,
                    )
                print(f"[sd_pipeline] loaded {candidate} ({label})")
                break
            except Exception as err:  # noqa: BLE001 - we want to fall through
                last_err = err
                pipe = None
                _release_memory()
                print(f"[sd_pipeline] failed {candidate} ({label}): {type(err).__name__}: {err}")
                if attempt_map is not None and not _is_out_of_memory(err):
                    # `device_map` itself was the problem; retry without it.
                    continue
                if _is_out_of_memory(err):
                    avail = free_commit_gb()
                    hint = "" if avail is None else f" (free commit: {avail:.2f} GB)"
                    print(f"[sd_pipeline]   -> looks memory-related{hint}")
                break
        if pipe is not None:
            break

    if pipe is None:
        avail = free_commit_gb()
        detail = "" if avail is None else f" Free commit at failure: {avail:.2f} GB."
        raise RuntimeError(
            f"Could not load any SD v1.5 checkpoint. Tried: {ordered}.{detail}"
        ) from last_err

    if use_ddim:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Memory-saving switches for a 6 GB card.
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:  # noqa: BLE001 - optional
        pass

    if not device_map:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe



@torch.no_grad()
def generate(
    pipe: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> Tuple[Image.Image, float]:
    """
    Generates a single image and returns it with the wall-clock time taken.

    A fresh, seeded ``torch.Generator`` is used each call, so the same
    (prompt, steps, guidance, seed) always yields the same image.

    Returns:
        (PIL.Image, elapsed_seconds)
    """
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipe(
        prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        height=int(height),
        width=int(width),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return result.images[0], elapsed


if __name__ == "__main__":
    from common.seed import set_seed

    set_seed(42)
    dev = get_device()
    print(f"Device: {dev}")
    pipe = load_pipeline(dev)
    print(f"Scheduler: {type(pipe.scheduler).__name__}")
    img, secs = generate(pipe, "a photograph of an astronaut riding a horse", 10, 7.5, 42)
    print(f"Generated {img.size} image in {secs:.2f}s ({secs/10:.3f}s/step)")
    if dev.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
