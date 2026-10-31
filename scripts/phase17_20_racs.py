"""Phases 17-20 — build and verify the RACS retrieval stack in one pass.

Writes phase17..phase20 artifacts and a ``phaseNN_verify.json`` per phase.
Tuning (Phase 20 alpha sweep) reads the calibration split only.
CPU only; a few minutes on this machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import common.paths as paths  # noqa: E402
from retrieval import racs  # noqa: E402
from steganography.pipeline import load_config, write_json  # noqa: E402

R = paths.RESULTS_DIR


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def verdict(phase: int, checks: dict, extra: dict | None = None) -> bool:
    ok = all(bool(v) for v in checks.values())
    write_json(R / f"phase{phase}_verify.json",
               {"phase": phase, "pass": ok, "checks": checks, "created_utc": now(),
                **(extra or {})})
    print(f"[phase{phase}] {'PASS' if ok else 'FAIL'} {checks}", flush=True)
    return ok


def subprocess_select(token: str, k: int) -> int:
    code = ("import sys; sys.path.insert(0, r'%s'); from retrieval.racs import select_index;"
            "print(select_index('%s', %d))" % (_ROOT, token, k))
    return int(subprocess.check_output([sys.executable, "-c", code], text=True).strip())


def conditions(image_id: str):
    with np.load(R / "public_conditions" / f"{image_id}.npz", allow_pickle=True) as z:
        feat = z["feature"] if "feature" in z.files else None
        return str(z["kpub"]), feat


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    rc = cfg["retrieval"]
    split = json.loads(paths.METADATA_DIR.joinpath("test_split.json").read_text("utf-8"))
    tokens = {r["image_id"]: r for r in json.loads(paths.TOKENS_PATH.read_text())["records"]}
    results = {}

    # ---------------- Phase 17 ----------------
    corpus = racs.load_corpus(cfg)
    index = racs.ExactIndex(corpus["embeddings"])
    clip = racs.ClipText(cfg)
    self_hits = [index.search(corpus["embeddings"][i], 1)[0] for i in range(len(corpus["ids"]))]
    self_top1 = float(np.mean([h[0] == i for i, h in enumerate(self_hits)]))
    self_score_min = float(min(h[1] for h in self_hits))
    q = clip("a dog playing fetch in a park")
    run1 = index.search(q, 50)
    run2 = racs.ExactIndex(corpus["embeddings"]).search(clip("a dog playing fetch in a park"), 50)
    sweep = {str(k): [(corpus["ids"][r], round(s, 6)) for r, s in index.search(q, k)]
             for k in rc["faiss"]["k_sweep"]}
    # faiss IndexFlatIP in a separate process (OpenMP clash with torch)
    ip = _ROOT / rc["faiss"]["index_path"]
    ip.parent.mkdir(parents=True, exist_ok=True)
    qpath = ip.with_name("query.npy")
    np.save(qpath, q)
    code = (
        "import faiss, numpy as np, json; faiss.omp_set_num_threads(1);"
        f"z=np.load(r'{_ROOT / cfg['embedding']['embeddings_path']}');"
        "e=np.ascontiguousarray(z['embeddings'],dtype=np.float32);"
        "ix=faiss.IndexFlatIP(e.shape[1]); ix.add(e);"
        f"faiss.write_index(ix, r'{ip}');"
        f"d,i=ix.search(np.load(r'{qpath}').reshape(1,-1).astype(np.float32),50);"
        "print(json.dumps([[int(a),float(b)] for a,b in zip(i[0],d[0])]))")
    faiss_top = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    faiss_agrees = [r for r, _ in faiss_top] == [r for r, _ in run1] and all(
        abs(a[1] - b[1]) < 1e-5 for a, b in zip(faiss_top, run1))
    index_sha = racs.sha256_bytes(ip.read_bytes())
    (R / "phase17").mkdir(parents=True, exist_ok=True)
    write_json(R / "phase17" / "image_ids.json", corpus["ids"])
    manifest17 = {"engine": index.engine, "index_type": rc["faiss"]["index_type"],
                  "rows": len(corpus["ids"]), "corpus_version": corpus["corpus_version"],
                  "embeddings_sha256": corpus["embeddings_sha256"], "index_sha256": index_sha,
                  "self_query_top1": self_top1, "self_query_min_score": self_score_min,
                  "k_sweep_text_query": sweep, "created_utc": now()}
    write_json(R / "phase17_index_manifest.json", manifest17)
    results[17] = verdict(17, {
        "row_order_matches_image_index": True,  # load_corpus raises otherwise
        "self_query_top1_is_1": self_top1 == 1.0,
        "self_query_score_is_1": abs(self_score_min - 1.0) < 1e-4,
        "rebuild_deterministic": run1 == run2,
        "faiss_index_flat_ip_agrees": faiss_agrees,
        "k_sweep_logged": len(sweep) == len(rc["faiss"]["k_sweep"]),
    }, {"engine": index.engine})

    # ---------------- Phase 18 ----------------
    dom = rc["token_selection"]["domain"]
    toks = [r["token"] for r in tokens.values()]
    dist = {}
    for k in (1, 5, 10, 20, 50):
        c = Counter(racs.select_index(t, k, dom) for t in toks + [f"{i:08d}" for i in range(2000)])
        dist[str(k)] = {"buckets_used": len(c), "max_share": max(c.values()) / sum(c.values())}
    xproc = all(subprocess_select(t, 10) == racs.select_index(t, 10) for t in toks[:5])
    import ast
    tree = ast.parse((_ROOT / "retrieval" / "racs.py").read_text())
    builtin_hash_calls = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == "hash" for n in ast.walk(tree))
    write_json(R / "phase18_token_selection.json",
               {"domain": dom, "hash": "sha256", "index_bytes": 8, "distribution": dist,
                "guess_probability": {k: 1 / int(k) for k in dist}, "created_utc": now()})
    results[18] = verdict(18, {
        "cross_process_identical": xproc,
        "k1_always_zero": dist["1"]["buckets_used"] == 1,
        "k10_spreads_all_buckets": dist["10"]["buckets_used"] == 10,
        "k10_near_uniform": dist["10"]["max_share"] < 0.15,
        "no_builtin_hash": builtin_hash_calls == 0,
    })

    # ---------------- Phase 19 ----------------
    size = int(rc["structural"]["size"])
    cache = racs.build_structural_cache(corpus["paths"], size=size)
    cache2 = racs.build_structural_cache(corpus["paths"][:20], size=size)
    cp = _ROOT / rc["structural"]["cache_path"]
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, edges=cache, image_ids=np.array(corpus["ids"]))
    edges = cache.astype(bool)
    # CLIP-similar but structurally different corpus pairs
    rng = np.random.default_rng(42)
    probe = sorted(rng.choice(len(corpus["ids"]), 50, replace=False).tolist())
    pairs = []
    for i in probe:
        for j, s in index.search(corpus["embeddings"][i], 6)[1:]:
            pairs.append({"a": corpus["ids"][i], "b": corpus["ids"][j], "clip_cos": s,
                          "edge_iou": racs.iou(edges[i], edges[j])})
    disagree = sorted([p for p in pairs if p["clip_cos"] > 0.85], key=lambda p: p["edge_iou"])[:5]
    write_json(R / "phase19_structural.json",
               {"kind": "edges", "size": size, "rows": int(cache.shape[0]),
                "structural_maps_sha256": racs.sha256_bytes(cache.tobytes()),
                "probe_size": 50, "clip_vs_structure_disagreements": disagree,
                "created_utc": now()})
    results[19] = verdict(19, {
        "cache_covers_corpus": cache.shape[0] == len(corpus["ids"]),
        "cache_deterministic": bool(np.array_equal(cache[:20], cache2)),
        "at_least_3_disagreements": len(disagree) >= 3,
    })

    # ---------------- Phase 20 (calibration only) ----------------
    retr = racs.Retriever(cfg, structural_cache=edges)
    retr.clip = clip
    cal = [r for r in split["records"] if r["role"] == "calibration"]
    secret = {}
    for r in cal:
        img = Image.open(_ROOT / r["path"])
        secret[r["image_id"]] = (clip.image(img), racs.edge_mask(img, size=size))
    sweep20 = []
    for a in rc["ranking"]["sweep_alpha"]:
        sem_q, st_q, changed = [], [], 0
        for r in cal:
            kpub, feat = conditions(r["image_id"])
            out = retr.acquire(kpub, feat, tokens[r["image_id"]]["token"], alpha=a, beta=1 - a)
            pool = [x["row"] for x in out["ranked"]]
            changed += pool != out["clip_order"][: len(pool)]
            se, sm = secret[r["image_id"]]
            sem_q.append(np.mean([float(corpus["embeddings"][p] @ se) for p in pool]))
            st_q.append(np.mean([racs.iou(sm, edges[p]) for p in pool]))
        sweep20.append({"alpha": a, "beta": 1 - a, "pool_semantic_to_secret": float(np.mean(sem_q)),
                        "pool_edge_iou_to_secret": float(np.mean(st_q)),
                        "pool_changed_fraction": changed / len(cal)})
        print(f"[phase20] alpha={a} {sweep20[-1]}", flush=True)
    # objective: equal-weight unit semantic + structure; ties -> larger alpha
    for s in sweep20:
        s["objective"] = 0.5 * (s["pool_semantic_to_secret"] + 1) / 2 + 0.5 * s["pool_edge_iou_to_secret"]
    best = max(sweep20, key=lambda s: (round(s["objective"], 6), s["alpha"]))
    clip_only = next(s for s in sweep20 if s["alpha"] == 1.0)
    # determinism: Tx and Rx (fresh Retriever) select the same reference
    rx = racs.Retriever(cfg, structural_cache=edges)
    rx.clip = racs.ClipText(cfg)
    same = all(
        retr.acquire(*conditions(r["image_id"]), tokens[r["image_id"]]["token"],
                     alpha=best["alpha"], beta=best["beta"])["sha256"]
        == rx.acquire(*conditions(r["image_id"]), tokens[r["image_id"]]["token"],
                      alpha=best["alpha"], beta=best["beta"])["sha256"]
        for r in cal[::6])
    write_json(R / "phase20_ranking_sweep.json",
               {"tune_split": "calibration", "sweep": sweep20,
                "frozen": {"alpha": best["alpha"], "beta": best["beta"],
                           "top_k": int(rc["ranking"]["top_k"]),
                           "pool_k": int(rc["ranking"]["pool_k"])},
                "ab": {"clip_only": clip_only, "chosen": best,
                       "helps": best["objective"] > clip_only["objective"]},
                "reference_sensitivity": "IPAdapter-plus weight 1 conditions EDICT on Xref "
                                         "pixels at both ends (see docs/handoff_file.md)",
                "created_utc": now()})
    results[20] = verdict(20, {
        "sweep_on_calibration_only": all(r["role"] == "calibration" for r in cal),
        "rerank_changes_pool": max(s["pool_changed_fraction"] for s in sweep20) > 0,
        "tx_rx_same_reference": same,
    }, {"frozen_alpha": best["alpha"], "frozen_beta": best["beta"],
        "helps_vs_clip_only": best["objective"] > clip_only["objective"]})

    print(f"[phase17-20] done in {time.time() - t0:.0f}s; "
          f"frozen alpha={best['alpha']} beta={best['beta']} -> set in configs/base.yaml")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
