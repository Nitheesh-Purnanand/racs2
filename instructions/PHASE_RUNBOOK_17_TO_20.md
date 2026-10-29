# Phase Runbook 17-20: RACS Retrieval Stack

> **Scope**: Phases 17-20 of `racs2_implementation_plan.md` (Milestone M6). Phase numbering follows
> the implementation plan, not the older `docs/RACS_ROADMAP.md` numbering.
>
> **Entry condition**: Phases 15 and 16 are frozen. `phase15_verify.json` and
> `phase16_verify.json` pass against the artifacts you intend to compare with. No RACS artifact
> may retroactively change a Phase 15/16 number.
>
> **Exit condition**: retrieval, token selection, structural representation and reranking are
> frozen, with `phase20_verify.json` PASS. Phase 21 may then assemble RACS-Base.

---

## 1. Order and dependencies

```text
Phase 17  FAISS retrieval          <- banked Phase 4 corpus + Phase 5 embeddings
   |
Phase 18  token selection          <- Phase 17 ordered Top-K
   |
Phase 19  structural representation<- Phase 17 index (probe), Phase 7 extractors (optional)
   |
Phase 20  structure-aware reranking<- Phases 17, 18, 19 + Phase 15 downstream
   |
Phase 21  RACS-Base (next)         <- frozen Phases 17-20 + Phase 15/16 downstream
```

Do not start a phase before the previous verify script passes. Each phase ends with a commit that
contains its modules, scripts, config delta and artifacts manifest. Keep `configs/base.yaml`
changes in that same commit so the config hash is traceable.

Estimated effort: 6-8 working days total, dominated by Phase 19's corpus pass and Phase 20's
paired A/B.

---

## 2. One-time pre-flight

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase5_verify.py
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py
.venv\Scripts\python.exe scripts\phase15_verify.py
.venv\Scripts\python.exe scripts\phase16_verify.py
.venv\Scripts\python.exe -c "import numpy, cv2, skimage; print(numpy.__version__, cv2.__version__, skimage.__version__)"
.venv\Scripts\python.exe -m pip install "faiss-cpu==1.8.0.post1"
.venv\Scripts\python.exe -c "import faiss; print(faiss.__version__)"
```

**Baseline state: resolved on 2026-10-29 (State A).** Every paper asset is fetched and
pinned in `models/PAPER_ASSETS.json`, `configs/base.yaml` selects the paper components, and
Phases 7-16 were rerun and re-frozen on them. The superseded substitute baseline is kept
under `experiments/results/archive/`. So the pre-flight above simply has to pass:

```powershell
.venv\Scripts\python.exe scripts\run_baseline.py --verify-only --yes
```

Phases 17-20 do not depend on the codec or enhancer weights, but Phase 20's A/B and Phase 21
read the frozen Phase 15/16 artifacts, which now come from the paper stack.

---

## 3. Config additions (combined)

Paste these nested blocks under the existing `retrieval:` key, preserving every existing key.

```yaml
retrieval:
  # ... existing corpus keys stay unchanged ...
  faiss:
    engine: "faiss"                      # faiss | numpy_exact (fallback + DEVIATIONS)
    index_type: "IndexFlatIP"
    metric: "cosine_via_inner_product"
    index_path: "experiments/results/phase17/corpus.faiss"
    image_ids_path: "experiments/results/phase17/image_ids.json"
    manifest_path: "experiments/results/phase17_index_manifest.json"
    sweep_path: "experiments/results/phase17_k_sweep.json"
    grid_path: "experiments/results/phase17_k_sweep.png"
    default_top_k: 10
    k_sweep: [1, 5, 10, 20, 50]
    tie_break: "row_index_ascending"
    omp_threads: 1

  token_selection:
    domain: "racs2-token-selection-v1"
    hash: "sha256"
    index_bytes: 8
    mode: "hash_mod_k"
    top_k: 10

  structural:
    kind: "edges"
    size: 128
    grayscale: true
    threshold: 0.10
    dilation_px: 2
    primary_score: "iou_dilated"
    diagnostic_score: "edge_cosine"
    cache_path: "experiments/results/phase19/structural_edges.npz"
    manifest_path: "experiments/results/phase19_structural.json"
    probe_size: 50
    probe_seed: 42

  ranking:
    alpha: 0.5
    beta: 0.5
    top_k: 10
    semantic_transform: "cosine_to_unit"
    structural_transform: "identity"
    tie_break: "semantic_then_row"
    tune_split: "calibration"
    sweep_alpha: [0.0, 0.25, 0.5, 0.75, 1.0]
    freeze: false
```

After Phase 20, set `ranking.alpha`, `ranking.beta`, `ranking.top_k` to the frozen values and
`ranking.freeze: true`.

---

## 4. Phase-by-phase gates and artifacts

| Phase | Build | Gate | Artifacts |
| :-- | :-- | :-- | :-- |
| 17 | `retrieval/build_index.py`, `retrieval/retrieve.py`, `scripts/phase17_faiss.py` | row order exact; self-query top-1 = 1.0; cross-process identical; K sweep logged | `phase17/corpus.faiss`, `phase17/image_ids.json`, `phase17_index_manifest.json`, `phase17_k_sweep.json`, `phase17_k_sweep.png`, `phase17_verify.json` |
| 18 | `retrieval/token_selection.py`, `scripts/phase18_token_selection.py` | SHA-256 only; cross-process stable; K=1 stable; 1/K bound; no `hash()` | `phase18_token_selection.json`, `phase18_distribution.png`, `phase18_verify.json` |
| 19 | `retrieval/structural.py`, `scripts/phase19_structural.py` | deterministic cache over 1000 rows; >= 3 CLIP-vs-structure disagreement examples | `phase19/structural_edges.npz`, `phase19_structural.json`, `phase19_semantic_vs_structural.png`, `phase19_verify.json` |
| 20 | `retrieval/rerank.py`, `scripts/phase20_rerank.py` | pre-registered calibration sweep; reference-sensitivity probe; frozen alpha/beta/K; positive or negative result reported honestly | `phase20_ranking_sweep.json`, `phase20_ranking_ab.json`, `phase20_rerank_grid.png`, `phase20_verify.json` |

---

## 5. Cross-phase invariants

1. **Row order is the protocol.** `image_index.json` order == Phase 5 npz order == Phase 17 index
   rows == Phase 19 cache rows. Every builder asserts it.
2. **Float32 everywhere in CLIP and in stored embeddings.** fp16 moved vectors by ~7e-4 in
   Phase 5 and would break Tx/Rx agreement.
3. **Exact retrieval only.** `IndexFlatIP`, single thread, stable `(-score, row)` tie-break.
   No IVF/PQ/HNSW, no score rounding.
4. **One selection implementation.** `H(domain|token)[:8] mod K` in exactly one module, used by
   both ends.
5. **One ranking implementation.** `alpha * semantic + beta * structural` with fixed transforms
   and a stable tie-break, used by both ends.
6. **K is frozen.** `faiss.default_top_k == token_selection.top_k == ranking.top_k`.
7. **Evaluation split is untouched until Phase 21.** Design, thresholds, alpha/beta and K are
   chosen on calibration or corpus data only.
8. **The evaluation set is disjoint from the retrieval corpus.** Phase 6 enforces this by id and
   SHA-256; Phase 17's self-query check uses a corpus image, never a secret.
9. **Every artifact carries hashes.** Index, image ids, embeddings bytes, structural maps,
   config hash, code commit, corpus version, split version.
10. **Baseline artifacts are read-only.** Phases 17-20 do not modify `baseline_table.json`,
    `baseline_manifest.json`, `baseline_security.json` or their verify files.

---

## 6. The Phase 20 decision gate (read before writing code)

**Resolved on 2026-10-29.** IPAdapter-plus is now enabled at weight 1 and injects `Xref`
into the EDICT sampler at *both* ends (`steganography/ipadapter.py`, cached per reference by
SHA-256). The stego therefore depends on the reference pixels, so reranking has a mechanism and
the Phase 20 A/B is meaningful. Re-run the probe below anyway to confirm it on your tree.

Before running the Phase 20 A/B, hide the same secret with two different reference images:

```text
if stego hashes differ -> reference-sensitive; the A/B is meaningful
if stego hashes equal  -> reference-insensitive; choose one:
    A. enable IP-Adapter, verify sensitivity, and re-freeze baseline v2 (both systems)
    B. keep the frozen baseline, adopt alpha = 1.0, report an offline ranking study,
       and state in Phase 22 that no pixel-level reference-quality claim is made
```

Do not compare an image-conditioned RACS against a text-only baseline. Record the probe result in
`phase20_ranking_sweep.json` and, if relevant, a DEVIATIONS row.

---

## 7. Phase 21 handoff contract

Phase 21 will build:

```text
Secret -> CLIP(query) -> Phase 17 retrieve Top-K -> Phase 20 rerank
       -> Phase 18 select(token) -> Xref
       -> the exact Phase 15 downstream modules
```

It must prove:

- `Xref_tx == Xref_rx` bit-exactly for 100 percent of evaluation images, all classes, all SNRs.
- The downstream is imported, not copied.
- The retrieved `Xref` is reference-sensitive in the chosen protocol (or the limitation is
  stated).
- `index_sha256`, `structural_maps_sha256`, `alpha`, `beta`, `K`, `token domain` and the token
  map are recorded in `racs_manifest.json`.

If any of those cannot be stated, Phase 21 is not ready.

---

## 8. Suggested commits

```text
Phase 17: freeze FAISS index and retrieval protocol
Phase 18: freeze SHA-256 token selection
Phase 19: freeze structural representation and corpus cache
Phase 20: freeze reranking weights and reference-sensitivity decision
```

Each commit includes its config block, modules, scripts, artifacts and a one-line status update
in `project_status.md`.

---

## 9. Done definition

```text
[ ] phase17_verify.json pass=true
[ ] phase18_verify.json pass=true
[ ] phase19_verify.json pass=true
[ ] phase20_verify.json pass=true
[ ] all four config blocks frozen and committed
[ ] alpha/beta/K and token domain recorded with hashes
[ ] reference-sensitivity decision recorded
[ ] DEVIATIONS updated for engine fallback, structural kind, conditioning choice
[ ] project_status.md updated: Phases 17-20 complete, Phase 21 next
```
