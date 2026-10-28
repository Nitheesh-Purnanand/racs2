"""Read the Phase 7 public conditions and convert Xfeat into ControlNet inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PIL import Image

from semantic.cache import read_cache_npz


def load_conditions(cache_dir: Path, image_id: str) -> Dict[str, Any]:
    """Load one cached Phase 7 record."""
    path = Path(cache_dir) / f"{image_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 7 conditions for {image_id}: {path}")
    record = read_cache_npz(path)
    preview_path = Path(cache_dir) / f"{image_id}_preview.png"
    record["preview_path"] = preview_path
    return record


def control_image(record: Dict[str, Any], size: int) -> Image.Image | None:
    """Return the structural control image, or None for style."""
    kind = record.get("feature_kind", "none")
    if kind == "none":
        return None
    path = record.get("preview_path")
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"missing preview for {record.get('image_id')}: {path}")
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
