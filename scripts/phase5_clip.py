"""
Phase 5 — CLIP Embeddings for RACS2.

Runs the three plan experiments and embeds the frozen 1,000-image corpus:

  1. text -> text similarity (related paraphrases vs unrelated prompts)
  2. image -> image nearest neighbours
  3. text -> most similar corpus images
  4. gate demonstration: high CLIP similarity with low structural (edge) overlap
  5. full corpus pass -> float32, L2-normalised [1000, 512] vectors

Artifacts (all under ``experiments/results/``):

  phase5_corpus_embeddings.npz    image_ids + embeddings (+ caption_embeddings) + provenance
  phase5_corpus_embeddings.json   manifest describing exactly how vectors were produced
  phase5_experiments.json         numbers for experiments 1-3 and the gate demo
  phase5_exp2_neighbors.png       image -> top-5 nearest grid
  phase5_exp3_text_queries.png    text -> top-5 images grid
  phase5_semantic_vs_structure.png

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\phase5_clip.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401  (HF cache on D: before model libs)

import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from common.paths import PROJECT_ROOT as _ROOT, RESULTS_DIR  # noqa: E402
from common.seed import set_seed  # noqa: E402
from models.clip_encoder import load_clip, resolve_dtype  # noqa: E402
from retrieval.corpus import Corpus  # noqa: E402
from retrieval.schema import sha256_file  # noqa: E402

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_py(value):
    """Convert numpy scalars/arrays to plain Python for JSON serialisation."""
    if isinstance(value, np.ndarray):
        return [to_py(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_py(v) for v in value]
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_py(payload), fh, indent=2, sort_keys=False)
        fh.write("\n")
    tmp.replace(path)


def make_grid(
    rows: Sequence[Sequence[Tuple[Image.Image, str]]],
    path: Path,
    cell: int = 112,
    label_h: int = 16,
    max_label: int = 22,
) -> None:
    """Compose labelled rows of images (row = one query + its neighbours)."""
    if not rows:
        return
    cols = max(len(row) for row in rows)
    width = cols * cell
    height = len(rows) * (cell + label_h)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (img, label) in enumerate(row):
            thumb = img.convert("RGB").resize((cell, cell), Image.BICUBIC)
            x, y = c * cell, r * (cell + label_h)
            sheet.paste(thumb, (x, y))
            text = label if len(label) <= max_label else label[: max_label - 1] + "~"
            draw.text((x + 3, y + cell + 2), text, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def edge_map(image: Image.Image, size: int = 64) -> np.ndarray:
    """Cheap structural proxy: grayscale edge magnitude at low resolution."""
    gray = image.convert("L").resize((size, size), Image.BICUBIC)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return np.asarray(edges, dtype=np.float32) / 255.0


def edge_similarity(a: Image.Image, b: Image.Image, threshold: float = 0.10) -> Dict[str, float]:
    """Cosine + IoU between two edge maps (a structural, not semantic, score)."""
    ea, eb = edge_map(a).ravel(), edge_map(b).ravel()
    denom = float(np.linalg.norm(ea) * np.linalg.norm(eb)) or 1e-12
    cosine = float(np.dot(ea, eb) / denom)
    ma, mb = ea > threshold, eb > threshold
    union = float(np.logical_or(ma, mb).sum())
    inter = float(np.logical_and(ma, mb).sum())
    iou = inter / union if union else 0.0
    return {"edge_cosine": cosine, "edge_iou": iou}


def top_k(scores: np.ndarray, k: int, exclude: int | None = None) -> List[Tuple[int, float]]:
    """Indices of the k highest scores, optionally excluding one index."""
    order = np.argsort(-scores, kind="stable")
    out: List[Tuple[int, float]] = []
    for idx in order:
        if exclude is not None and int(idx) == exclude:
            continue
        out.append((int(idx), float(scores[idx])))
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------
def load_embedding_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = dict(raw.get("embedding") or {})
    if not cfg:
        raise SystemExit(f"no `embedding:` block in {config_path}")
    cfg["_seed"] = int(raw.get("seed", DEFAULT_SEED))
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5: CLIP embedding experiments + corpus pass.")
    p.add_argument("--config", default=str(_ROOT / "configs" / "base.yaml"))
    p.add_argument("--device", default=None, help="cuda / cpu (default: config)")
    p.add_argument("--dtype", default=None, help="float16 / float32 (default: config)")
    p.add_argument("--batch-size", type=int, default=None, help="image batch size override")
    p.add_argument("--limit", type=int, default=None,
                   help="embed only the first N corpus images (smoke test)")
    p.add_argument("--skip-captions", action="store_true",
                   help="do not embed the 1,000 captions")
    p.add_argument("--embeddings-path", default=None,
                   help="override config embeddings_path")
    p.add_argument("--manifest-path", default=None, help="override config manifest_path")
    return p.parse_args(argv)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path)


def display_path(value) -> str | None:
    """Project-relative POSIX path when possible, for portable provenance."""
    if not value:
        return None
    try:
        path = Path(str(value))
        if path.is_absolute():
            return path.relative_to(_ROOT).as_posix()
    except Exception:  # noqa: BLE001
        pass
    return str(value).replace("\\", "/")


# ---------------------------------------------------------------------------
# Experiment 1: text -> text similarity
# ---------------------------------------------------------------------------
def experiment1_text_similarity(
    encoder, prompts: Sequence[str], groups: Sequence[int]
) -> dict:
    section("Experiment 1 - text similarity (related vs unrelated)")
    if sum(groups) != len(prompts):
        raise SystemExit(
            f"text_prompt_groups {groups} do not sum to {len(prompts)} prompts"
        )

    lengths = encoder.text_token_lengths(prompts)
    truncated = [i for i, n in enumerate(lengths) if n > 77]
    print(f"  prompts: {len(prompts)}  groups: {list(groups)}")
    print(f"  token lengths: min={min(lengths)} max={max(lengths)}")
    if truncated:
        print(f"  [warn] prompts truncated at 77 tokens: {truncated}")

    vecs = encoder.encode_texts(prompts, batch_size=max(1, len(prompts)))
    sim = vecs @ vecs.T

    within, across = [], []
    start = 0
    bounds = []
    for size in groups:
        bounds.append((start, start + size))
        start += size
    for gi, (lo, hi) in enumerate(bounds):
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                within.append(float(sim[i, j]))
    for gi, (lo_i, hi_i) in enumerate(bounds):
        for gj, (lo_j, hi_j) in enumerate(bounds):
            if gj <= gi:
                continue
            for i in range(lo_i, hi_i):
                for j in range(lo_j, hi_j):
                    across.append(float(sim[i, j]))

    within_mean = float(np.mean(within)) if within else 0.0
    across_mean = float(np.mean(across)) if across else 0.0
    ok = within_mean > across_mean
    print(f"  within-topic mean similarity : {within_mean:.4f}")
    print(f"  across-topic mean similarity : {across_mean:.4f}")
    print(f"  gap                          : {within_mean - across_mean:+.4f}")

    print("\n  cosine matrix (rows = prompts):")
    print("      " + " ".join(f"{i:6d}" for i in range(len(prompts))))
    for i in range(len(prompts)):
        print(f"  {i:2d}  " + " ".join(f"{sim[i, j]:6.3f}" for j in range(len(prompts))))

    return {
        "prompts": list(prompts),
        "groups": list(groups),
        "token_lengths": lengths,
        "truncated_prompt_indices": truncated,
        "similarity_matrix": sim,
        "within_topic_mean": within_mean,
        "across_topic_mean": across_mean,
        "within_gt_across": bool(ok),
    }


# ---------------------------------------------------------------------------
# Full corpus pass
# ---------------------------------------------------------------------------
def embed_corpus(
    encoder, corpus: Corpus, ids: Sequence[str], batch_size: int
) -> Tuple[np.ndarray, float]:
    section(f"Corpus pass - embedding {len(ids)} images")
    chunks: List[np.ndarray] = []
    t0 = time.perf_counter()
    done = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        images = [corpus.load(image_id) for image_id in batch_ids]
        chunks.append(encoder.encode_images(images, batch_size=batch_size))
        done += len(batch_ids)
        if done == len(ids) or done % (batch_size * 8) == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0.0
            print(f"  {done:4d}/{len(ids)}  {elapsed:6.1f}s  {rate:5.1f} img/s", flush=True)

    elapsed = time.perf_counter() - t0
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    print(f"  done: {embeddings.shape} in {elapsed:.2f}s ({len(ids)/elapsed:.1f} img/s)")
    return embeddings, float(elapsed)


# ---------------------------------------------------------------------------
# Experiment 2: image -> nearest images
# ---------------------------------------------------------------------------
def experiment2_image_neighbors(
    corpus: Corpus,
    ids: Sequence[str],
    embeddings: np.ndarray,
    out_path: Path,
    subset: int = 64,
    queries: int = 4,
    k: int = 5,
) -> dict:
    section("Experiment 2 - image nearest neighbours")
    n = min(subset, len(ids))
    sub = embeddings[:n]
    sim = sub @ sub.T
    query_indices = [int(round(i * (n - 1) / max(queries - 1, 1))) for i in range(queries)]

    rows = []
    records = []
    for qi in query_indices:
        neighbours = top_k(sim[qi], k=k, exclude=qi)
        row = [(corpus.load(ids[qi]), f"Q{qi}")]
        entry = {"query_index": qi, "query_id": ids[qi], "neighbors": []}
        for ni, score in neighbours:
            row.append((corpus.load(ids[ni]), f"{score:.3f}"))
            entry["neighbors"].append({"image_id": ids[ni], "score": score})
        rows.append(row)
        records.append(entry)
        print(f"  query {qi:2d} ({ids[qi]}): " + ", ".join(
            f"{nid.split('_')[-1]}={s:.3f}" for nid, s in [(e['image_id'], e['score']) for e in entry['neighbors']]
        ))

    make_grid(rows, out_path, cell=112, label_h=16)
    print(f"  saved {out_path}")
    return {
        "subset_size": n,
        "queries": records,
        "grid": str(out_path.relative_to(_ROOT)),
        "_sim": sim,
    }


# ---------------------------------------------------------------------------
# Experiment 3: text -> images
# ---------------------------------------------------------------------------
def experiment3_text_queries(
    encoder,
    corpus: Corpus,
    ids: Sequence[str],
    embeddings: np.ndarray,
    text_queries: Sequence[str],
    out_path: Path,
    k: int = 5,
) -> dict:
    section("Experiment 3 - text query -> most similar corpus images")
    query_vecs = encoder.encode_texts(text_queries)
    sims = query_vecs @ embeddings.T

    rows = []
    records = []
    for query, scores in zip(text_queries, sims):
        neighbours = top_k(scores, k=k)
        row = [(corpus.load(ids[ni]), f"{score:.3f}") for ni, score in neighbours]
        rows.append(row)
        entry = {
            "query": query,
            "top_k": [
                {"image_id": ids[ni], "score": score} for ni, score in neighbours
            ],
        }
        records.append(entry)
        print(f"  {query!r}:")
        for ni, score in neighbours:
            print(f"      {score:.4f}  {ids[ni]}")

    make_grid(rows, out_path, cell=112, label_h=16)
    print(f"  saved {out_path}")
    return {"k": k, "queries": records, "grid": str(out_path.relative_to(_ROOT))}


# ---------------------------------------------------------------------------
# Gate demonstration: semantic != structural
# ---------------------------------------------------------------------------
def gate_demonstration(
    corpus: Corpus,
    ids: Sequence[str],
    embeddings: np.ndarray,
    exp2: dict,
    out_path: Path,
) -> dict:
    section("Gate demonstration - CLIP similarity vs structural similarity")

    candidates: List[Tuple[float, int, int]] = []
    for entry in exp2["queries"]:
        qi = entry["query_index"]
        for neighbour in entry["neighbors"][:3]:
            j = ids.index(neighbour["image_id"])
            candidates.append((float(neighbour["score"]), qi, int(j)))

    # de-duplicate unordered pairs
    seen = set()
    unique = []
    for clip_score, i, j in candidates:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        unique.append((clip_score, i, j))

    scored = []
    for clip_score, i, j in unique:
        img_i, img_j = corpus.load(ids[i]), corpus.load(ids[j])
        metrics = edge_similarity(img_i, img_j)
        scored.append(
            {
                "i": i,
                "j": j,
                "image_id_i": ids[i],
                "image_id_j": ids[j],
                "clip_cosine": clip_score,
                "edge_cosine": metrics["edge_cosine"],
                "edge_iou": metrics["edge_iou"],
            }
        )

    # Prefer a semantically close pair with weak structural overlap.
    strong = [s for s in scored if s["clip_cosine"] >= 0.80]
    pool = strong or scored
    chosen = min(pool, key=lambda s: (s["edge_iou"], -s["clip_cosine"]))

    img_a, img_b = corpus.load(chosen["image_id_i"]), corpus.load(chosen["image_id_j"])
    make_grid(
        [[
            (img_a, f"CLIP {chosen['clip_cosine']:.3f}"),
            (img_b, f"edge IoU {chosen['edge_iou']:.3f}"),
        ]],
        out_path,
        cell=224,
        label_h=20,
        max_label=24,
    )
    print(f"  pair : {chosen['image_id_i']}  vs  {chosen['image_id_j']}")
    print(f"  CLIP cosine  : {chosen['clip_cosine']:.4f}  (semantic)")
    print(f"  edge cosine  : {chosen['edge_cosine']:.4f}")
    print(f"  edge IoU     : {chosen['edge_iou']:.4f}  (structural)")
    print(f"  saved {out_path}")

    return {
        "chosen": chosen,
        "candidates": scored,
        "grid": str(out_path.relative_to(_ROOT)),
        "interpretation": (
            "CLIP keeps semantically related images close even when their edge "
            "layout differs; structural compatibility is a separate signal."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_embedding_config(Path(args.config))

    set_seed(int(cfg.get("_seed", DEFAULT_SEED)), verbose=True)

    model_id = str(cfg.get("model_id", "openai/clip-vit-base-patch32"))
    model_path = cfg.get("model_path")
    device = args.device or cfg.get("device") or None
    dtype_cfg = args.dtype or cfg.get("dtype", "float16")
    dtype = resolve_dtype(dtype_cfg)
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    text_batch_size = int(cfg.get("text_batch_size", 64))
    normalize = bool(cfg.get("normalize", True))

    embeddings_path = resolve_path(
        args.embeddings_path or cfg.get(
            "embeddings_path", "experiments/results/phase5_corpus_embeddings.npz"
        )
    )
    manifest_path = resolve_path(
        args.manifest_path or cfg.get(
            "manifest_path", "experiments/results/phase5_corpus_embeddings.json"
        )
    )

    print("Phase 5 - CLIP embeddings")
    print(f"  model_id   : {model_id}")
    print(f"  dtype      : {dtype}")
    print(f"  batch_size : {batch_size}")
    print(f"  embeddings : {embeddings_path}")

    section("1. Load corpus")
    corpus = Corpus()
    ids_all = corpus.ids()
    if args.limit is not None:
        ids_all = ids_all[: max(1, int(args.limit))]
        print(f"  [smoke test] limiting to first {len(ids_all)} images")
    print(f"  {corpus.summary()}")
    print(f"  embedding {len(ids_all)} of {len(corpus)} indexed images")

    section("2. Load CLIP encoder")
    encoder = load_clip(model_id=model_id, device=device, dtype=dtype, model_path=model_path)
    print(f"  {encoder}")
    print(f"  source    : {encoder.model_source}")
    print(f"  preprocess: {encoder.preprocess}")

    # Warm-up: first forward compiles/caches kernels; keep it out of timings.
    warm_image = corpus.load(ids_all[0])
    encoder.encode_images([warm_image], batch_size=1)
    encoder.encode_texts(["warm up"], batch_size=1)
    del warm_image
    print("  warm-up done")

    failures: List[str] = []

    # -- Experiment 1 ------------------------------------------------------
    prompts = list(cfg.get("text_prompts") or [])
    groups = list(cfg.get("text_prompt_groups") or [])
    if not prompts:
        failures.append("no text_prompts in config")
        exp1 = {}
    else:
        exp1 = experiment1_text_similarity(encoder, prompts, groups)
        if not exp1["within_gt_across"]:
            failures.append("experiment 1: within-topic similarity <= across-topic")

    # -- Corpus pass -------------------------------------------------------
    embeddings, embed_seconds = embed_corpus(encoder, corpus, ids_all, batch_size)
    if embeddings.shape != (len(ids_all), encoder.embedding_dim):
        failures.append(
            f"embedding shape {embeddings.shape} != ({len(ids_all)}, {encoder.embedding_dim})"
        )
    if not np.isfinite(embeddings).all():
        failures.append("embeddings contain NaN/Inf")

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        max_dev = float(np.max(np.abs(norms - 1.0))) if len(norms) else 0.0
        print(f"  unit-norm check: max |1 - ||v||| = {max_dev:.2e}")
        if max_dev > 1e-4:
            failures.append(f"vectors not unit norm (max dev {max_dev:.2e})")

    # -- Experiment 2 ------------------------------------------------------
    if len(ids_all) >= 4 and not args.limit:
        exp2 = experiment2_image_neighbors(
            corpus, ids_all, embeddings, RESULTS_DIR / "phase5_exp2_neighbors.png"
        )
    else:
        print("\n[Experiment 2] skipped (smoke mode or corpus too small)")
        exp2 = {"skipped": True, "queries": []}

    # -- Experiment 3 ------------------------------------------------------
    text_queries = list(cfg.get("text_queries") or [])
    if text_queries and not args.limit:
        exp3 = experiment3_text_queries(
            encoder,
            corpus,
            ids_all,
            embeddings,
            text_queries,
            RESULTS_DIR / "phase5_exp3_text_queries.png",
        )
    else:
        print("\n[Experiment 3] skipped (smoke mode, no queries, or too small)")
        exp3 = {"skipped": True, "queries": []}

    # -- Gate demonstration ------------------------------------------------
    if exp2.get("queries"):
        gate = gate_demonstration(
            corpus,
            ids_all,
            embeddings,
            exp2,
            RESULTS_DIR / "phase5_semantic_vs_structure.png",
        )
    else:
        print("\n[Gate demonstration] skipped (needs experiment 2 neighbours)")
        gate = {"skipped": True}

    # -- Caption embeddings + sanity --------------------------------------
    caption_embeddings = None
    caption_top1 = None
    if not args.skip_captions:
        section("Caption embeddings + cross-modal sanity")
        captions = [corpus.get(image_id).get("caption", "") for image_id in ids_all]
        caption_embeddings = encoder.encode_texts(captions, batch_size=text_batch_size)
        if caption_embeddings.shape != embeddings.shape:
            failures.append("caption embeddings shape mismatch")
        else:
            sims = caption_embeddings @ embeddings.T
            predicted = np.argmax(sims, axis=1)
            truth = np.arange(len(ids_all))
            caption_top1 = float((predicted == truth).mean())
            print(f"  caption -> image top-1 accuracy: {caption_top1:.3f} "
                  f"({int((predicted == truth).sum())}/{len(ids_all)})")
            if caption_top1 < 0.05:
                failures.append(f"caption->image top-1 too low ({caption_top1:.3f})")

    # -- Save artifacts ----------------------------------------------------
    section("Save artifacts")
    index_sha = sha256_file(corpus.index_path)
    weight_sha = None
    if getattr(encoder, "model_path", None):
        weight_path = Path(encoder.model_path) / "pytorch_model.bin"
        if weight_path.exists():
            print("  hashing model weights ...", flush=True)
            weight_sha = sha256_file(weight_path)
    created = utc_now()

    npz_payload = {
        "image_ids": np.array(ids_all),
        "embeddings": embeddings.astype(np.float32, copy=False),
        "model_id": np.array(model_id),
        "corpus_version": np.array(corpus.version()),
        "index_sha256": np.array(index_sha),
        "created_utc": np.array(created),
    }
    if caption_embeddings is not None:
        npz_payload["caption_embeddings"] = caption_embeddings.astype(np.float32)
    if caption_top1 is not None:
        npz_payload["caption_top1_accuracy"] = np.array(caption_top1)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, **npz_payload)
    print(f"  saved {embeddings_path} ({embeddings_path.stat().st_size/1e6:.2f} MB)")

    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_source": display_path(encoder.model_source),
        "model_path": display_path(encoder.model_path),
        "weight_sha256": weight_sha,
        "embedding_dim": int(encoder.embedding_dim),
        "count": int(len(ids_all)),
        "corpus_version": corpus.version(),
        "index_sha256": index_sha,
        "preprocess": encoder.preprocess,
        "image_size": int(cfg.get("image_size", 224)),
        "dtype_stored": "float32",
        "normalized": bool(normalize),
        "device_used": str(encoder.device),
        "dtype_compute": str(dtype).replace("torch.", ""),
        "seconds_total": round(embed_seconds, 3),
        "images_per_second": round(len(ids_all) / embed_seconds, 3) if embed_seconds else None,
        "created_utc": created,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"  saved {manifest_path}")

    experiments_payload = {
        "model_id": model_id,
        "corpus_version": corpus.version(),
        "count": int(len(ids_all)),
        "created_utc": created,
        "experiment1_text_similarity": exp1,
        "experiment2_image_neighbors": exp2,
        "experiment3_text_queries": exp3,
        "gate_semantic_vs_structure": gate,
        "caption_top1_accuracy": caption_top1,
    }
    write_json_atomic(RESULTS_DIR / "phase5_experiments.json", experiments_payload)
    print(f"  saved {RESULTS_DIR / 'phase5_experiments.json'}")

    # -- Summary -----------------------------------------------------------
    section("SUMMARY")
    print(f"  vectors      : {embeddings.shape}")
    print(f"  model        : {model_id}")
    print(f"  corpus       : {corpus.version()[:16]}...")
    if exp1:
        print(f"  exp1 within  : {exp1['within_topic_mean']:.4f}")
        print(f"  exp1 across  : {exp1['across_topic_mean']:.4f}")
    if caption_top1 is not None:
        print(f"  caption top-1: {caption_top1:.3f}")
    if gate.get("chosen"):
        c = gate["chosen"]
        print(f"  gate pair    : CLIP {c['clip_cosine']:.3f} vs edge IoU {c['edge_iou']:.3f}")

    section("RESULT")
    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("PASS - CLIP embeddings written and aligned with the corpus index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
