"""
Centralized Paths & Cache Configuration for RACS2.

The C: drive on this machine is nearly full, so every large download (HuggingFace
models, caches) must live on D:. Importing this module redirects the HuggingFace
and diffusers caches to the project directory *before* any model library is
imported, and exposes the project's canonical paths.

Usage:
    import common.paths  # side effect: HF cache redirected to D:
    from common.paths import RESULTS_DIR, SECRET_IMAGES_DIR
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root = parent of this file's directory (d:\racs2)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data / output locations
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
SECRET_IMAGES_DIR: Path = DATA_DIR / "secret_images"
RETRIEVAL_IMAGES_DIR: Path = DATA_DIR / "retrieval_images"
METADATA_DIR: Path = DATA_DIR / "metadata"
IMAGE_INDEX_PATH: Path = METADATA_DIR / "image_index.json"
RAW_CORPUS_DIR: Path = DATA_DIR / "_raw_coco"

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RESULTS_DIR: Path = EXPERIMENTS_DIR / "results"

# Model + HuggingFace caches (all on D:)
CACHE_DIR: Path = PROJECT_ROOT / "cache"
HF_CACHE_DIR: Path = CACHE_DIR / "huggingface"
HF_HUB_DIR: Path = HF_CACHE_DIR / "hub"
TORCH_CACHE_DIR: Path = CACHE_DIR / "torch"

# ---------------------------------------------------------------------------
# Create directories on import
# ---------------------------------------------------------------------------
for _dir in (
    DATA_DIR,
    SECRET_IMAGES_DIR,
    RETRIEVAL_IMAGES_DIR,
    METADATA_DIR,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    CACHE_DIR,
    HF_CACHE_DIR,
    HF_HUB_DIR,
    TORCH_CACHE_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Redirect caches away from C:  (must happen before importing transformers /
# diffusers / huggingface_hub)
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))


def describe() -> str:
    """Returns a human-readable summary of the resolved paths."""
    lines = [
        f"PROJECT_ROOT       : {PROJECT_ROOT}",
        f"DATA_DIR           : {DATA_DIR}",
        f"SECRET_IMAGES_DIR  : {SECRET_IMAGES_DIR}",
        f"RETRIEVAL_IMAGES_DIR: {RETRIEVAL_IMAGES_DIR}",
        f"RESULTS_DIR        : {RESULTS_DIR}",
        f"HF_HOME            : {os.environ.get('HF_HOME')}",
        f"HF_HUB_CACHE       : {os.environ.get('HUGGINGFACE_HUB_CACHE')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== RACS2 Paths ===")
    print(describe())
