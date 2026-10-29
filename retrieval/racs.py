"""Phases 17-20 — RACS retrieval stack (index, token selection, structure, rerank).

Every function here is shared by the transmitter and the receiver; there is no
second implementation. The query is built only from public material (``Kpub``
text and the ``Xfeat`` structure map), so Rx recomputes the same reference
without ever seeing the secret.

    Kpub --CLIP text--> exact inner-product Top-K   (Phase 17)
    Xfeat --edges-->    structural score per row    (Phase 19)
    alpha*sem + beta*struct, stable tie-break       (Phase 20)
    SHA-256(domain|token)[:8] mod K -> one row      (Phase 18)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

_ROOT = Path(__file__).resolve().parent.parent

# faiss and torch ship clashing OpenMP runtimes on Windows, so the pipeline
# searches with numpy (same exact inner product as IndexFlatIP) and Phase 17
# builds the faiss index in its own process and checks the rankings agree.
faiss = None
ENGINE = "numpy_exact"


# ---------------------------------------------------------------------------
# Phase 17 — exact retrieval
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus(cfg_all: dict) -> dict:
    """Phase 5 embeddings + Phase 4 index, row order asserted identical."""
    index = json.loads((_ROOT / cfg_all["retrieval"]["index_path"]).read_text("utf-8"))
    with np.load(_ROOT / cfg_all["embedding"]["embeddings_path"], allow_pickle=False) as z:
        ids = [str(i) for i in z["image_ids"]]
        emb = np.ascontiguousarray(z["embeddings"], dtype=np.float32)
    index_ids = [r["image_id"] for r in index["images"]]
    if ids != index_ids:
        raise RuntimeError("embedding row order != image_index.json order")
    return {
        "ids": ids,
        "paths": [str(_ROOT / r["image_path"]) for r in index["images"]],
        "sha256": [r["sha256"] for r in index["images"]],
        "embeddings": emb,
        "corpus_version": index["corpus_version"],
        "embeddings_sha256": sha256_bytes(emb.tobytes()),
    }


class ExactIndex:
    """IndexFlatIP over L2-normalised float32 rows; ties broken by row index."""

    def __init__(self, embeddings: np.ndarray):
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.engine = ENGINE

    def scores(self, query: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        return (self.emb @ q).astype(np.float32)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        order = np.lexsort((np.arange(len(s)), -s))  # (-score, row) stable
        return [(int(r), float(s[r])) for r in order[:top_k]]


class ClipText:
    """The Phase 5 CLIP encoder (float32, CPU), so queries match the corpus rows."""

    def __init__(self, cfg_all: dict, device: str = "cpu"):
        import common.paths  # noqa: F401  (HF cache on D: before model libs)
        from models.clip_encoder import load_clip

        e = cfg_all["embedding"]
        self.enc = load_clip(e["model_id"], device=device, dtype="float32",
                             local_files_only=True, model_path=_ROOT / e["model_path"])

    def image(self, image: Image.Image) -> np.ndarray:
        return self.enc.encode_images([image.convert("RGB")])[0].astype(np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self.enc.encode_texts([text])[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 18 — token selection (the only implementation; never Python hash())
# ---------------------------------------------------------------------------
def select_index(token: str, k: int, domain: str = "racs2-token-selection-v1") -> int:
    if k < 1:
        raise ValueError("k must be >= 1")
    digest = hashlib.sha256(f"{domain}|{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % k


# ---------------------------------------------------------------------------
# Phase 19 — structural representation (binary edges, dilated IoU)
# ---------------------------------------------------------------------------
def edge_mask(image: Image.Image, size: int = 128, threshold: float = 0.10,
              dilation_px: int = 2) -> np.ndarray:
    gray = image.convert("L").resize((size, size), Image.BICUBIC)
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    m = Image.fromarray(((edges > threshold) * 255).astype(np.uint8))
    if dilation_px:
        m = m.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    return np.asarray(m) > 0


def feature_mask(feature: np.ndarray | None, size: int = 128,
                 dilation_px: int = 2) -> np.ndarray | None:
    """Edge mask of the public Xfeat: segmentation boundaries or keypoint dots."""
    if feature is None or feature.size == 0:
        return None
    if feature.ndim == 2 and feature.shape[0] > 8 and feature.shape[1] > 8:  # label map
        seg = np.asarray(Image.fromarray(feature.astype(np.uint8)).resize(
            (size, size), Image.NEAREST)).astype(np.int32)
        b = np.zeros_like(seg, dtype=bool)
        b[:, 1:] |= seg[:, 1:] != seg[:, :-1]
        b[1:, :] |= seg[1:, :] != seg[:-1, :]
    else:  # (N, 3) keypoints in 512-px coordinates
        b = np.zeros((size, size), dtype=bool)
        for x, y, c in np.asarray(feature, dtype=np.float32).reshape(-1, 3):
            if c > 0:
                xi, yi = int(x * size / 512), int(y * size / 512)
                if 0 <= xi < size and 0 <= yi < size:
                    b[yi, xi] = True
    m = Image.fromarray((b * 255).astype(np.uint8))
    if dilation_px:
        m = m.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    return np.asarray(m) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def coverage(query: np.ndarray, cand: np.ndarray) -> float:
    """Fraction of query structure found in the candidate (for sparse keypoints)."""
    n = query.sum()
    return float(np.logical_and(query, cand).sum() / n) if n else 0.0


def build_structural_cache(paths: Sequence[str], size: int = 128) -> np.ndarray:
    return np.stack([edge_mask(Image.open(p), size=size) for p in paths]).astype(np.uint8)


def structural_score(query_mask: np.ndarray | None, cand_mask: np.ndarray,
                     sparse: bool = False) -> float:
    if query_mask is None:
        return 0.0
    return coverage(query_mask, cand_mask) if sparse else iou(query_mask, cand_mask)


# ---------------------------------------------------------------------------
# Phase 20 — rerank (fixed transforms, stable tie-break)
# ---------------------------------------------------------------------------
def rerank(cands: List[Tuple[int, float]], struct: Dict[int, float],
           alpha: float, beta: float) -> List[Tuple[int, float, float, float]]:
    rows = []
    for row, sem in cands:
        s_unit = (sem + 1.0) / 2.0  # cosine -> [0, 1]
        st = float(struct.get(row, 0.0))
        rows.append((row, alpha * s_unit + beta * st, sem, st))
    rows.sort(key=lambda r: (-round(r[1], 12), -round(r[2], 12), r[0]))
    return rows


class Retriever:
    """The full RACS reference-acquisition step, identical at Tx and Rx."""

    def __init__(self, cfg_all: dict, structural_cache: np.ndarray | None = None):
        r = cfg_all["retrieval"]
        self.cfg = r
        self.corpus = load_corpus(cfg_all)
        self.index = ExactIndex(self.corpus["embeddings"])
        self.clip = ClipText(cfg_all)
        self.top_k = int(r["ranking"]["top_k"])
        self.pool_k = int(r["ranking"].get("pool_k", self.top_k))
        self.alpha = float(r["ranking"]["alpha"])
        self.beta = float(r["ranking"]["beta"])
        self.domain = r["token_selection"]["domain"]
        self.size = int(r["structural"]["size"])
        if structural_cache is None:
            cache_path = _ROOT / r["structural"]["cache_path"]
            with np.load(cache_path) as z:
                structural_cache = z["edges"].astype(bool)
        self.struct = structural_cache

    def acquire(self, kpub: str, feature: np.ndarray | None, token: str,
                alpha: float | None = None, beta: float | None = None,
                top_k: int | None = None) -> dict:
        k = int(top_k or self.top_k)
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        q = self.clip(kpub)
        cands = self.index.search(q, max(k, self.pool_k))
        qm = feature_mask(feature, size=self.size)
        sparse = feature is not None and feature.ndim == 2 and feature.shape[1] == 3
        st = {row: structural_score(qm, self.struct[row], sparse=sparse) for row, _ in cands}
        ranked = rerank(cands, st, a, b)[:k]
        pick = select_index(token, k, self.domain)
        row = ranked[pick][0]
        return {
            "row": row,
            "image_id": self.corpus["ids"][row],
            "path": self.corpus["paths"][row],
            "sha256": self.corpus["sha256"][row],
            "pick": pick,
            "clip_order": [c[0] for c in cands],
            "ranked": [{"row": r, "score": s, "semantic": se, "structural": sx}
                       for r, s, se, sx in ranked],
            "query_sha256": sha256_bytes(q.tobytes()),
        }
