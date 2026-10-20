"""
Phase 5 (Step 7) — Verify CLIP Embeddings for RACS2.

Offline gate for ``experiments/results/phase5_corpus_embeddings.npz`` produced by
``scripts/phase5_clip.py``. It never hits the network: model weights come from the
local HF cache (``local_files_only=True``) and all corpus data is already on disk.

Checks:
  1. npz loads; image_ids length == embeddings.shape[0] == manifest count
  2. embedding dim matches the manifest and the model projection dim
  3. stored vectors are float32; caption embeddings have the same shape
  4. all values finite
  5. row order matches ``image_index.json`` exactly
  6. unit-norm vectors (max deviation < 1e-4)
  7. provenance: corpus_version + index sha256 match the live index
  8. determinism: re-encode 8 sampled images and compare with the stored rows
  9. cross-modal sanity: caption -> image top-1 accuracy far above chance
 10. retrieval sanity: nearest non-self neighbour scores are < 1.0 and sensible
 11. reports embedding throughput from the manifest

Exits 0 on PASS, 1 on FAIL.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase5_verify.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common.paths  # noqa: E402,F401  (HF cache on D: before model libs)

import yaml  # noqa: E402

from common.paths import PROJECT_ROOT as _ROOT, RESULTS_DIR  # noqa: E402
from models.clip_encoder import load_clip, resolve_dtype  # noqa: E402
from retrieval.corpus import Corpus  # noqa: E402
from retrieval.schema import sha256_file  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the RACS2 Phase 5 CLIP embeddings.")
    p.add_argument("--config", default=str(_ROOT / "configs" / "base.yaml"))
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--sample", type=int, default=8,
                   help="how many images to re-encode for the determinism check")
    p.add_argument("--skip-determinism", action="store_true",
                   help="skip the re-encode check (still needs no network)")
    return p.parse_args(argv)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path)


def main(argv=None) -> int:
    args = parse_args(argv)

    section("1. Load config, manifest and embeddings")
    with open(Path(args.config), "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}
    cfg = dict(raw_cfg.get("embedding") or {})
    corpus_cfg = dict(raw_cfg.get("retrieval") or {})

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

    failures: List[str] = []
    checks: List[Dict[str, object]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'ok' if ok else 'x'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name + (f" ({detail})" if detail else ""))

    if not embeddings_path.exists():
        print(f"  [x] missing embeddings: {embeddings_path}")
        print("\nFAIL - run scripts/phase5_clip.py first.")
        return 1
    if not manifest_path.exists():
        print(f"  [x] missing manifest: {manifest_path}")
        print("\nFAIL - run scripts/phase5_clip.py first.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    data = np.load(embeddings_path, allow_pickle=False)
    image_ids = np.array([str(x) for x in data["image_ids"]])
    embeddings = np.asarray(data["embeddings"])
    caption_embeddings = (
        np.asarray(data["caption_embeddings"]) if "caption_embeddings" in data else None
    )

    print(f"  embeddings : {embeddings_path}")
    print(f"  manifest   : {manifest_path}")
    print(f"  vectors    : {embeddings.shape}  ids: {image_ids.shape}")

    section("2. Shape and dtype")
    count = int(manifest.get("count", -1))
    check("image_ids length == manifest count", len(image_ids) == count,
          f"{len(image_ids)} vs {count}")
    check("embeddings rows == image_ids length", embeddings.shape[0] == len(image_ids),
          f"{embeddings.shape[0]} vs {len(image_ids)}")
    check("embedding dim matches manifest",
          embeddings.shape[1] == int(manifest.get("embedding_dim", -1)),
          f"{embeddings.shape[1]} vs {manifest.get('embedding_dim')}")
    check("stored dtype is float32", embeddings.dtype == np.float32, str(embeddings.dtype))
    if caption_embeddings is not None:
        check("caption embeddings shape matches",
              caption_embeddings.shape == embeddings.shape, str(caption_embeddings.shape))
    else:
        print("  [--] caption_embeddings absent (skipping caption sanity)")

    section("3. Values and normalisation")
    check("all embeddings finite", bool(np.isfinite(embeddings).all()))
    norms = np.linalg.norm(embeddings, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0))) if len(norms) else 1.0
    check("unit-norm vectors (max dev < 1e-4)", max_dev < 1e-4, f"max dev {max_dev:.2e}")

    section("4. Row order vs corpus index")
    corpus = Corpus()
    corpus_ids = corpus.ids()
    check("vector count == corpus count", len(image_ids) == len(corpus_ids),
          f"{len(image_ids)} vs {len(corpus_ids)}")
    if len(image_ids) == len(corpus_ids):
        mismatch = int(np.sum(image_ids != np.array(corpus_ids)))
        check("row order matches image_index.json", mismatch == 0, f"{mismatch} mismatches")
    else:
        prefix = min(len(image_ids), len(corpus_ids))
        mismatch = int(np.sum(image_ids[:prefix] != np.array(corpus_ids[:prefix])))
        check("row order matches (prefix)", mismatch == 0,
              f"compared {prefix}, {mismatch} mismatches")
    check("image_ids are unique", len(set(image_ids.tolist())) == len(image_ids))

    section("5. Provenance")
    live_version = corpus.version()
    check("manifest corpus_version == live corpus",
          manifest.get("corpus_version") == live_version,
          f"{manifest.get('corpus_version')} vs {live_version}")
    live_sha = sha256_file(corpus.index_path)
    check("manifest index_sha256 == live index",
          manifest.get("index_sha256") == live_sha)
    npz_version = str(data["corpus_version"]) if "corpus_version" in data else None
    check("npz corpus_version == manifest", npz_version == manifest.get("corpus_version"))
    model_id = str(manifest.get("model_id", cfg.get("model_id", "")))
    check("manifest records a model_id", bool(model_id), model_id)
    check("manifest records preprocessing", bool(manifest.get("preprocess")),
          str(manifest.get("preprocess")))

    section("6. Determinism (re-encode sample)")
    if args.skip_determinism:
        print("  skipped (--skip-determinism)")
    else:
        dtype_name = str(manifest.get("dtype_compute", cfg.get("dtype", "float16")))
        dtype = resolve_dtype(dtype_name)
        device = cfg.get("device") or None
        encoder = load_clip(
            model_id=model_id or None,
            device=device,
            dtype=dtype,
            local_files_only=True,
            model_path=cfg.get("model_path"),
        )
        print(f"  encoder: {encoder}")

        row_of = {image_id: i for i, image_id in enumerate(image_ids.tolist())}
        sample_size = min(int(args.sample), len(image_ids))
        rng = random.Random(1234)
        sample_ids = rng.sample(image_ids.tolist(), sample_size)

        images = [corpus.load(image_id) for image_id in sample_ids]
        reencoded = encoder.encode_images(images, batch_size=max(1, sample_size))
        stored = np.stack([embeddings[row_of[image_id]] for image_id in sample_ids])
        diff = float(np.max(np.abs(reencoded - stored)))
        check("stored vectors reproducible (max|diff| <= 1e-4)", diff <= 1e-4,
              f"max|diff| {diff:.2e} using {dtype_name}")

        if diff > 1e-4 and dtype != "float32":
            # Diagnose whether fp16 itself is run-to-run unstable.
            again = encoder.encode_images(images, batch_size=max(1, sample_size))
            run_diff = float(np.max(np.abs(reencoded - again)))
            print(f"  [info] fp16 run-to-run max|diff| = {run_diff:.2e}")
            print("  [hint] rerun: python scripts/phase5_clip.py --dtype float32")

        check("embedding dim matches model", embeddings.shape[1] == encoder.embedding_dim,
              f"{embeddings.shape[1]} vs {encoder.embedding_dim}")
        del encoder

    section("7. Caption -> image sanity")
    if caption_embeddings is None:
        print("  skipped (caption_embeddings not stored)")
    else:
        sims = caption_embeddings @ embeddings.T
        predicted = np.argmax(sims, axis=1)
        top1 = float((predicted == np.arange(len(predicted))).mean())
        check("caption->image top-1 far above chance (>= 0.05)", top1 >= 0.05,
              f"{top1:.3f}")

    section("8. Retrieval sanity")
    rng = random.Random(4321)
    n_queries = min(20, len(embeddings))
    query_rows = rng.sample(range(len(embeddings)), n_queries)
    first_scores = []
    for row in query_rows:
        scores = embeddings @ embeddings[row]
        order = np.argsort(-scores, kind="stable")
        neighbour = next(int(i) for i in order if int(i) != row)
        first_scores.append(float(scores[neighbour]))
    mean_top1 = float(np.mean(first_scores)) if first_scores else 0.0
    check("nearest non-self similarity < 1.0",
          all(s < 1.0 for s in first_scores), f"max {max(first_scores):.4f}")
    check("nearest non-self similarity sensible (> 0.3)", mean_top1 > 0.3,
          f"mean {mean_top1:.4f}")

    section("9. Throughput")
    print(f"  seconds_total    : {manifest.get('seconds_total')}")
    print(f"  images_per_second: {manifest.get('images_per_second')}")

    payload = {
        "pass": not failures,
        "checks": checks,
        "count": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "model_id": model_id,
        "corpus_version": live_version,
        "mean_nearest_non_self_similarity": mean_top1,
        "failures": failures,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase5_verify.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  saved {out}")

    section("RESULT")
    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("PASS - embeddings are aligned, normalised, reproducible and semantically sane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
