"""
Phase 7 — Deterministic Cache and Manifest Helpers for RACS2 Public Conditions.

Handles per-image .npz caching, preview PNG writing, and the manifest that
records provenance, hashes, and model metadata for every cached condition.

Atomic writes (tmp + os.replace) prevent partial files from being mistaken
for valid cache entries.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_CACHE_DIR = RESULTS_DIR / "public_conditions"


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_string(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload) -> str:
    """Canonical JSON hash (sort_keys, no spaces)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_sha256(feature: Optional[np.ndarray]) -> Optional[str]:
    """Canonical hash of a feature array, incorporating dtype and shape.

    Format: sha256(dtype + "|" + str(shape) + "|" + array.tobytes())
    Returns None for absent features (style class).
    """
    if feature is None:
        return None
    canonical = f"{feature.dtype}|{feature.shape}|".encode("utf-8") + feature.tobytes()
    return hashlib.sha256(canonical).hexdigest()


def write_cache_npz(
    path: Path,
    *,
    image_id: str,
    role: str,
    class_name: str,
    source_sha256: str,
    kpub: str,
    kpub_method: str,
    kpub_model_id: str,
    feature_kind: str,
    feature: Optional[np.ndarray],
    config_hash: str,
) -> None:
    """Atomically write a per-image .npz cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    save_kwargs: Dict[str, Any] = {
        "image_id": np.array(image_id),
        "role": np.array(role),
        "class_name": np.array(class_name),
        "source_sha256": np.array(source_sha256),
        "kpub": np.array(kpub),
        "kpub_method": np.array(kpub_method),
        "kpub_model_id": np.array(kpub_model_id),
        "feature_kind": np.array(feature_kind),
        "config_hash": np.array(config_hash),
    }
    if feature is not None:
        save_kwargs["feature"] = feature
        save_kwargs["feature_shape"] = np.array(feature.shape, dtype=np.int32)
        save_kwargs["feature_dtype"] = np.array(str(feature.dtype))

    try:
        # Pass a file handle: np.savez_compressed otherwise appends ".npz" to a
        # path that does not already end in ".npz" and the atomic replace fails.
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **save_kwargs)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_cache_npz(path: Path) -> Dict[str, Any]:
    """Read a per-image .npz cache file and return its contents as a dict."""
    data = np.load(str(path), allow_pickle=False)
    result: Dict[str, Any] = {
        "image_id": str(data["image_id"]),
        "role": str(data["role"]),
        "class_name": str(data["class_name"]),
        "source_sha256": str(data["source_sha256"]),
        "kpub": str(data["kpub"]),
        "kpub_method": str(data["kpub_method"]),
        "kpub_model_id": str(data["kpub_model_id"]),
        "feature_kind": str(data["feature_kind"]),
        "config_hash": str(data["config_hash"]),
    }
    if "feature" in data:
        result["feature"] = data["feature"]
    else:
        result["feature"] = None
    return result


def write_preview(path: Path, preview) -> None:
    """Atomically write a preview PNG."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        preview.save(str(tmp), format="PNG")
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def compute_config_hash(semantic_config: dict) -> str:
    """Hash the resolved semantic config block for manifest provenance."""
    return sha256_json(semantic_config)


def build_manifest_record(
    *,
    image_id: str,
    role: str,
    class_name: str,
    source_sha256: str,
    cache_path: str,
    preview_path: Optional[str],
    kpub_sha256: str,
    feature_sha256_val: Optional[str],
    feature_kind: str,
    status: str,
) -> Dict[str, Any]:
    """Build one record for the manifest."""
    record: Dict[str, Any] = {
        "image_id": image_id,
        "role": role,
        "class": class_name,
        "source_sha256": source_sha256,
        "cache_path": cache_path,
        "kpub_sha256": kpub_sha256,
        "feature_sha256": feature_sha256_val,
        "feature_kind": feature_kind,
        "status": status,
    }
    if preview_path:
        record["preview_path"] = preview_path
    return record


def build_manifest(
    *,
    records: List[Dict[str, Any]],
    split_version: str,
    corpus_version: str,
    config_hash: str,
    models: Dict[str, Dict[str, str]],
    counts: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Build the full manifest document."""
    # Sort records by (role, class, image_id)
    records_sorted = sorted(records, key=lambda r: (r["role"], r["class"], r["image_id"]))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_version": split_version,
        "corpus_version": corpus_version,
        "config_hash": config_hash,
        "models": models,
        "counts": counts,
        "records": records_sorted,
    }


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Atomically write the manifest JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with io.open(str(tmp), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_manifest(path: Path) -> Dict[str, Any]:
    """Read the manifest JSON."""
    with io.open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)
