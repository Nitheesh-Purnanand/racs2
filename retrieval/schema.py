"""
Phase 4 (Step 6) — Retrieval Index Schema & Integrity Helpers for RACS2.

`data/metadata/image_index.json` is the single source of truth for the retrieval
corpus. It is a **shared protocol resource**: the transmitter (Tx) and receiver
(Rx) must load byte-identical image bytes, in the same order, or they retrieve
different references and `Xref_tx != Xref_rx` and recovery fails.

This module defines the schema, the `corpus_version` (a content hash over the
ordered `image_id` list) and small helpers used by both the builder
(`scripts/phase4_build_corpus.py`) and the loader (`retrieval/corpus.py`) so the
format is defined in exactly one place.

Schema (schema_version == 1)::

    {
      "schema_version": 1,
      "corpus_version": "<sha256 of the ordered image_id list>",
      "source": "coco_val2017",
      "canonical_size": 512,
      "resize_mode": "shortest_side_then_center_crop",
      "count": 1000,
      "created_utc": "2026-10-08T00:00:00Z",
      "images": [
        {"image_id": ..., "image_path": ..., "source_id": ...,
         "caption": ..., "width": 512, "height": 512, "sha256": ...},
        ...
      ]
    }
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1

#: top-level keys every index must contain
INDEX_FIELDS: tuple = (
    "schema_version",
    "corpus_version",
    "source",
    "canonical_size",
    "resize_mode",
    "count",
    "created_utc",
    "images",
)

#: keys every record in ``images`` must contain
IMAGE_FIELDS: tuple = (
    "image_id",
    "image_path",
    "source_id",
    "caption",
    "width",
    "height",
    "sha256",
)

_HASH_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path: "str | os.PathLike") -> str:
    """SHA-256 hex digest of a file's bytes (streamed, so large files are fine)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_corpus_version(image_ids: Sequence[str]) -> str:
    """
    Content hash that *identifies the corpus*.

    Hashes the ``image_id`` list **in canonical order**, one per line, so that a
    reorder, an added/removed image, or a renamed id changes the version. Tx and
    Rx log this value and must refuse to exchange if the two differ.
    """
    payload = "\n".join(str(i) for i in image_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Index assembly / IO
# ---------------------------------------------------------------------------
def build_index(
    records: Iterable[Dict[str, Any]],
    *,
    source: str,
    canonical_size: int,
    resize_mode: str,
    created_utc: str,
) -> Dict[str, Any]:
    """Assemble a full index document from per-image records (order preserved)."""
    recs: List[Dict[str, Any]] = [dict(r) for r in records]
    ids = [r["image_id"] for r in recs]
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_version": compute_corpus_version(ids),
        "source": source,
        "canonical_size": int(canonical_size),
        "resize_mode": resize_mode,
        "count": len(recs),
        "created_utc": created_utc,
        "images": recs,
    }


def write_index_atomic(index: Dict[str, Any], path: "str | os.PathLike") -> None:
    """Write the index as pretty JSON, atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_index(path: "str | os.PathLike") -> Dict[str, Any]:
    """Load an index JSON document (no validation; see ``validate_index``)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_index(
    index: Dict[str, Any],
    *,
    require_files: bool = False,
    expected_count: int | None = None,
    root: "str | os.PathLike | None" = None,
) -> List[str]:
    """
    Return a list of human-readable problems with ``index`` (empty list == valid).

    Checks shape and internal consistency only. With ``require_files=True`` it
    also verifies that every ``image_path`` exists (resolved against ``root``
    when the stored path is relative), which the verifier uses.
    """
    errors: List[str] = []
    root_path: Path | None = Path(root) if root is not None else None
    if require_files and root_path is None:
        from common.paths import PROJECT_ROOT  # local import: keep module light

        root_path = PROJECT_ROOT

    if not isinstance(index, dict):
        return ["index is not a JSON object"]

    for key in INDEX_FIELDS:
        if key not in index:
            errors.append(f"missing top-level key: {key!r}")
    if errors:
        return errors

    if index["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"schema_version={index['schema_version']!r}, expected {SCHEMA_VERSION}"
        )

    images = index.get("images")
    if not isinstance(images, list):
        errors.append("'images' is not a list")
        return errors

    if index["count"] != len(images):
        errors.append(f"count={index['count']} but len(images)={len(images)}")

    if expected_count is not None and len(images) != int(expected_count):
        errors.append(f"count={len(images)}, expected {expected_count}")

    seen: set = set()
    for i, rec in enumerate(images):
        if not isinstance(rec, dict):
            errors.append(f"images[{i}] is not an object")
            continue
        for key in IMAGE_FIELDS:
            if key not in rec:
                errors.append(f"images[{i}] missing field {key!r}")
        iid = rec.get("image_id")
        if iid in seen:
            errors.append(f"duplicate image_id: {iid!r}")
        seen.add(iid)
        if require_files:
            p = Path(rec["image_path"]) if rec.get("image_path") else None
            if p is None:
                errors.append(f"images[{i}] has no image_path")
            else:
                resolved = p if p.is_absolute() else root_path / p
                if not resolved.exists():
                    errors.append(f"missing file for {iid!r}: {resolved}")

    expected_version = compute_corpus_version([r.get("image_id") for r in images])
    if index.get("corpus_version") != expected_version:
        errors.append(
            "corpus_version mismatch: stored "
            f"{index.get('corpus_version')!r} != recomputed {expected_version!r}"
        )

    return errors


if __name__ == "__main__":  # tiny self-test
    ids = ["coco_val2017_000000000139", "coco_val2017_000000000285"]
    doc = build_index(
        [
            {
                "image_id": i,
                "image_path": f"data/retrieval_images/{i}.png",
                "source_id": i.split("_")[-1],
                "caption": "",
                "width": 512,
                "height": 512,
                "sha256": "0" * 64,
            }
            for i in ids
        ],
        source="coco_val2017",
        canonical_size=512,
        resize_mode="shortest_side_then_center_crop",
        created_utc="2026-10-08T00:00:00Z",
    )
    assert doc["count"] == 2
    assert validate_index(doc) == [], validate_index(doc)
    assert doc["corpus_version"] == compute_corpus_version(ids)
    print("schema self-test: OK")
    print("corpus_version:", doc["corpus_version"])
