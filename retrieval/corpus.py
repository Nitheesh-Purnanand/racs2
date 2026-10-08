"""
Phase 4 (Step 7) — Retrieval Corpus Loader for RACS2.

Small, dependency-light reader for the frozen image corpus described by
`data/metadata/image_index.json` (schema in `retrieval/schema.py`).

The loader is deliberately deterministic: it always returns records in the
index's canonical order and never re-sorts, because Phase 5 embeddings, Phase 6
FAISS rows and Phase 7 token selection all address images by position in this
list. Re-ordering the index would silently break the whole stack.

Only the standard library and Pillow are required (no torch), so Tx/Rx can use
this loader on any machine.

Usage::

    from retrieval.corpus import Corpus
    corpus = Corpus()
    print(len(corpus), corpus.version())
    for rec, img in corpus.iter_images():
        ...  # img is an RGB PIL.Image, canonical_size x canonical_size
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

# Allow running this module directly (python retrieval/corpus.py) for a smoke test.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from common.paths import IMAGE_INDEX_PATH, PROJECT_ROOT
from retrieval.schema import read_index, validate_index


class Corpus:
    """Read-only view over a frozen retrieval corpus + its metadata index."""

    def __init__(
        self,
        index_path: "str | Path | None" = None,
        root: "str | Path | None" = None,
        validate: bool = True,
    ) -> None:
        self.index_path: Path = (
            Path(index_path) if index_path is not None else IMAGE_INDEX_PATH
        )
        self.root: Path = Path(root) if root is not None else PROJECT_ROOT

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"retrieval index not found: {self.index_path}\n"
                "Build it first with: .venv\\Scripts\\python.exe scripts\\phase4_build_corpus.py"
            )

        self.index: Dict[str, Any] = read_index(self.index_path)
        if validate:
            problems = validate_index(self.index)
            if problems:
                raise ValueError(
                    "invalid retrieval index:\n  - " + "\n  - ".join(problems)
                )

        self._records: List[Dict[str, Any]] = list(self.index["images"])
        self._by_id: Dict[str, Dict[str, Any]] = {
            r["image_id"]: r for r in self._records
        }

    # -- metadata ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def version(self) -> str:
        """`corpus_version` — the shared protocol identifier for this corpus."""
        return self.index["corpus_version"]

    @property
    def source(self) -> str:
        return self.index["source"]

    @property
    def canonical_size(self) -> int:
        return int(self.index["canonical_size"])

    @property
    def resize_mode(self) -> str:
        return self.index["resize_mode"]

    def summary(self) -> str:
        return (
            f"Corpus(source={self.source}, count={len(self)}, "
            f"size={self.canonical_size}, resize={self.resize_mode}, "
            f"version={self.version()[:12]}...)"
        )

    # -- record access -----------------------------------------------------
    def ids(self) -> List[str]:
        """Ordered `image_id` list (canonical order)."""
        return [r["image_id"] for r in self._records]

    def records(self) -> Tuple[Dict[str, Any], ...]:
        """Tuple of index records in canonical order (immutable snapshot)."""
        return tuple(self._records)

    def get(self, image_id: str) -> Dict[str, Any]:
        """Return the index record for `image_id` (raises KeyError if absent)."""
        return self._by_id[image_id]

    def resolve_path(self, image_id: str) -> Path:
        """Absolute path to an image; stored paths are project-relative."""
        p = Path(self.get(image_id)["image_path"])
        return p if p.is_absolute() else (self.root / p)

    # -- image access ------------------------------------------------------
    def load(self, image_id: str) -> Image.Image:
        """Load one image as an RGB `PIL.Image`."""
        path = self.resolve_path(image_id)
        with Image.open(path) as im:
            img = im.convert("RGB")
        return img

    def iter_images(self) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """Yield `(record, PIL.Image)` in canonical order."""
        for rec in self._records:
            yield rec, self.load(rec["image_id"])


if __name__ == "__main__":  # tiny smoke test
    corpus = Corpus()
    print(corpus.summary())
    for rec, img in corpus.iter_images():
        print(f"  {rec['image_id']}  {img.size}  {rec['caption'][:60]!r}")
        break
