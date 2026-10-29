# Phase 17: FAISS Retrieval

> **Goal**: Build an exact inner-product index over the banked Phase 5 corpus embeddings and
> expose a deterministic `retrieve(query, top_k)` API. This is the retrieval engine that will
> replace diffusion generation in RACS-Base.
> **Why**: RACS needs a fast, exact semantic neighbour lookup whose row order, scores and
> tie-breaking are identical on the transmitter and the receiver. Exactness is a protocol
> requirement, not a preference.

> **Precondition**: Phases 15 and 16 are frozen. Do not read or write any Phase 15/16 artifact
> from this phase. The corpus and embeddings are already built and verified:
> `data/metadata/image_index.json` (1000 images), `experiments/results/phase5_corpus_embeddings.npz`
> (1000 x 512 float32, L2-normalised).

> **Leakage rule**: the evaluation split is disjoint from the retrieval corpus by construction
> (Phase 6 excludes corpus ids and SHA-256s). Never use an evaluation secret as a retrieval
> query in this phase. The self-query check uses a corpus image.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase15_verify.py
.venv\Scripts\python.exe scripts\phase5_verify.py
.venv\Scripts\python.exe -c "import numpy as np; z=np.load('experiments/results/phase5_corpus_embeddings.npz'); print(z['image_ids'].shape, z['embeddings'].shape, z['embeddings'].dtype); n=np.linalg.norm(z['embeddings'],axis=1); print(float(n.min()), float(n.max()))"
```

Expected: Phase 15 verify PASS, Phase 5 verify PASS, shape `(1000,) (1000, 512) float32`,
norms within `1 +/- 1e-5`.

> As of 2026-10-29 the paper assets are in place and Phases 7-16 were re-frozen on them, so
> `phase15_verify.py` passes as-is (Gate A misses stay documented in `docs/DEVIATIONS.md`).
> This phase does not depend on those components.

Install FAISS:

```powershell
.venv\Scripts\python.exe -m pip install "faiss-cpu==1.8.0.post1"
.venv\Scripts\python.exe -c "import faiss; print(faiss.__version__)"
```

If no wheel is available for Python 3.10 on Windows, use the documented NumPy fallback described
in Step 7 and add a DEVIATIONS row. Do not silently switch engines.

---

## Step 1: Config

Append under the existing `retrieval:` block (keep every existing key):

```yaml
retrieval:
  # ... existing corpus keys stay unchanged ...
  faiss:
    engine: "faiss"                     # faiss | numpy_exact (fallback, deviations row required)
    index_type: "IndexFlatIP"
    metric: "cosine_via_inner_product"  # embeddings are L2-normalised
    index_path: "experiments/results/phase17/corpus.faiss"
    image_ids_path: "experiments/results/phase17/image_ids.json"
    manifest_path: "experiments/results/phase17_index_manifest.json"
    sweep_path: "experiments/results/phase17_k_sweep.json"
    grid_path: "experiments/results/phase17_k_sweep.png"
    default_top_k: 10
    k_sweep: [1, 5, 10, 20, 50]
    tie_break: "row_index_ascending"
    omp_threads: 1
```

`omp_threads: 1` removes any doubt about parallel reduction order. Flat inner product is exact
either way, but one thread makes the determinism claim trivial to defend.

---

## Step 2: Structure

```text
retrieval/
  build_index.py       # npz + image_index -> IndexFlatIP + image_ids.json
  retrieve.py          # query -> ordered (image_id, score) with stable tie-break
scripts/
  phase17_faiss.py     # build, K sweep, grids, manifest
  phase17_verify.py    # offline gate
experiments/results/phase17/
  corpus.faiss
  image_ids.json
  phase17_k_sweep.json
  phase17_k_sweep.png
experiments/results/
  phase17_index_manifest.json
  phase17_verify.json
```

---

## Step 3: Implement `retrieval/build_index.py`

```python
def build_index(
    *,
    embeddings_path: str | Path,
    image_index_path: str | Path,
    index_path: str | Path,
    image_ids_path: str | Path,
    engine: str = "faiss",
) -> dict:
    """Build and persist the retrieval index; returns the manifest payload."""


def load_index(index_path: str | Path, image_ids_path: str | Path) -> dict:
    """Return {"index": obj, "image_ids": list[str], "engine": str}."""
```

Rules:

- Load the npz with `allow_pickle=False`.
- Assert `list(npz["image_ids"]) == [r["image_id"] for r in image_index["images"]]`. A single
  mismatch is a hard failure. Row `i` in the embedding matrix and the index must be exactly
  `image_index["images"][i]`.
- Assert `npz["corpus_version"] == image_index["corpus_version"]` and
  `npz["index_sha256"] == sha256(image_index file)`.
- Assert shape `(N, 512)`, dtype float32, all finite, `abs(norm - 1) <= 1e-5`. Do not
  renormalise silently; if it fails, fix the source, do not hide it.
- `index = faiss.IndexFlatIP(dim)` then `index.add(np.ascontiguousarray(embeddings))`.
- Persist with `faiss.write_index(index, path)`. Do not rely on `index.reconstruct` for ids.
- Write `image_ids.json` as
  `{"schema_version": 1, "corpus_version": "...", "image_ids": [...], "count": N}`.
- Manifest records at minimum:
  `schema_version, phase, created_utc, engine, faiss_version, index_type, metric,
  embedding_path, embeddings_array_sha256, npz_file_sha256, index_path, index_sha256,
  image_ids_path, image_ids_sha256, image_index_sha256, corpus_version, count, dim, dtype,
  normalized, tie_break, omp_threads, code_commit`.

`embeddings_array_sha256` is SHA-256 over `embeddings.tobytes()` (stable). The `.npz` file hash
is recorded too but is not the protocol identifier, because zip metadata can change.

---

## Step 4: Implement `retrieval/retrieve.py`

```python
def retrieve(
    query: np.ndarray,
    *,
    index: dict,
    top_k: int = 10,
    exclude_image_ids: Sequence[str] = (),
) -> list[dict]:
    """Return [{"rank", "image_id", "score", "row"}, ...] best-first."""
```

Rules:

- Accept a `(512,)` or `(1, 512)` query. Flatten. Reject non-finite values.
- Reject a query whose L2 norm is outside `1 +/- 1e-4`; do not normalise implicitly on the
  receiver side. The caller normalises once, before retrieval, with the frozen CLIP preprocessing.
- `faiss.omp_set_num_threads(cfg["omp_threads"])` before search.
- Search `top_k + len(exclude_image_ids)` rows, then drop excluded ids, then truncate.
- Sort candidates by `(-score, row)` using a stable sort. This is the frozen tie-break: equal
  scores are ordered by canonical corpus row, never by hash-map order.
- Do not round scores. Return float64 values produced by FAISS.
- `exclude_image_ids` exists only for the self-query sanity check; RACS-Base calls it empty
  because test secrets are disjoint from the corpus.

Add a second helper:

```python
def retrieve_text(text: str, *, index: dict, encoder, top_k: int = 10) -> list[dict]:
    """Text -> CLIP text embedding -> retrieve. Phase 5 sanity path only."""
```

Use `CLIPEncoder.encode_texts` with the same float32 bytes-level model. Text retrieval is a
sanity check, not part of the Tx/Rx RACS path.

---

## Step 5: Implement `scripts/phase17_faiss.py`

Algorithm:

```text
1. set_seed(42)
2. build_index(...) and load it back
3. self-query check: for 10 corpus rows, query with embeddings[i]
   -> top-1 must be image_ids[i] with score >= 1 - 1e-5
4. text-query check: 4 queries from configs embedding.text_queries -> top-5 ids/scores
5. K sweep: for K in [1,5,10,20,50], 10 corpus image queries + 4 text queries
   -> log ordered ids and scores for every K
6. save phase17_k_sweep.json, phase17_k_sweep.png, phase17_index_manifest.json
```

Output schema (`phase17_k_sweep.json`):

```json
{
  "schema_version": 1,
  "phase": 17,
  "corpus_version": "...",
  "index_sha256": "...",
  "image_ids_sha256": "...",
  "k_sweep": [1, 5, 10, 20, 50],
  "queries": [
    {"kind": "image", "query_id": "coco_val2017_000000000724", "results": {
      "1": [{"rank": 1, "image_id": "...", "score": 1.0}],
      "10": [{"rank": 1, "image_id": "...", "score": 1.0}]
    }, "top1_is_self": true}
  ],
  "self_query_top1_rate": 1.0,
  "deterministic_across_processes": true
}
```

The PNG is a simple K x score curve per query family (mean top-1 score and mean top-10 score),
not a per-image contact sheet.

---

## Step 6: Implement `scripts/phase17_verify.py`

Checks:

```text
[ ] manifest exists, parses, and its index_sha256 matches the index file on disk
[ ] index row count == image_ids count == image_index count == 1000
[ ] image_ids order is byte-identical to image_index order
[ ] corpus_version matches across npz, image_index, image_ids.json, manifest
[ ] embeddings_array_sha256 recomputes exactly
[ ] L2 norms of the npz embeddings are 1 +/- 1e-5 and no value is NaN/Inf
[ ] self-query top-1 rate == 1.0 and every self score >= 1 - 1e-5
[ ] a second process load + query returns identical ordered ids and scores
[ ] ties: a synthetic tie case returns row-index ascending order
[ ] text query returns a finite ranked list of length K
[ ] K sweep contains every K and each result list length == K
[ ] default_top_k <= max(k_sweep)
[ ] no evaluation-split image_id appears in any query recorded in the sweep
[ ] phase17_k_sweep.png exists
```

Exit non-zero on failure. A single row-order mismatch is a stop condition.

---

## Step 7: NumPy fallback (only if FAISS cannot be installed)

- `engine: "numpy_exact"` in config.
- Compute `scores = embeddings @ query` in float64 by upcasting both operands, then apply the
  same `(-score, row)` stable sort. This is exact for 1000 x 512 and deterministic.
- Record a DEVIATIONS row: paper uses FAISS; we use an exact NumPy inner-product matmul because
  no FAISS wheel was available; measured impact: ranking identical to FAISS within float32
  tolerance, latency differs.
- The manifest records `engine`, so Phase 21 can prove which path ran.
- Do not claim FAISS was used if it was not.

---

## Step 8: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase17_faiss.py
.venv\Scripts\python.exe scripts\phase17_verify.py
```

Expected summary:

```text
corpus_version : <64 hex>
count          : 1000, dim 512, dtype float32
self-query     : top-1 rate 1.0, min score 1.000000
process replay : identical ids and scores
K sweep        : 1, 5, 10, 20, 50 logged
verify         : PASS
```

---

## Checklist

```text
[ ] faiss-cpu installed (or numpy_exact fallback + DEVIATIONS row)
[ ] IndexFlatIP over the frozen 1000 x 512 float32 embeddings
[ ] row order asserted identical to image_index.json
[ ] corpus_version and both hashes recorded
[ ] stable tie-break by row index is implemented and tested
[ ] retrieve() rejects non-normalised and non-finite queries
[ ] image self-query top-1 rate 1.0
[ ] text query works through CLIP
[ ] K sweep in {1, 5, 10, 20, 50} logged
[ ] cross-process determinism verified
[ ] no evaluation secret used as a query
[ ] phase17_index_manifest.json and phase17_verify.json written
[ ] no Phase 15/16 artifact modified
```

**When this passes, the retrieval engine is frozen.** Phase 18 may not change the index, the
embedding model, the preprocessing or the tie-break.

---

## The Gate Question (answer in your own words)

> *Why does RACS require an exact index and a frozen tie-break, when an approximate index would
> be faster and "close enough" for ordinary search?*

Answer with your artifacts:

- The transmitter and receiver must derive the same reference independently. A retrieval result
  that differs by one row, or by score rounding, changes the Top-K list.
- Top-K feeds token selection (`H(token) mod K`). A different K-th candidate changes the selected
  reference and breaks `Xref_tx == Xref_rx`, which is the Phase 21 hard invariant.
- Approximate indexes can return different neighbours across builds, thread counts or library
  versions. Exact flat inner product over a 1000-vector database is already fast enough, so
  there is no accuracy/cost trade-off worth taking.
- Ties must be resolved by a protocol value (canonical row order), not by memory layout or a
  hash map, or the same scores produce different lists on different runs.

---

## Report back

- FAISS version and engine actually used
- corpus_version, index sha256, image_ids sha256, embeddings array sha256
- Self-query top-1 rate and minimum self score
- Cross-process replay result
- K sweep table (mean top-1 / top-10 scores per query family)
- Any deviations (engine fallback, threads, version)
- Your answer to the gate question

---

## Notes and risks

- **Do not rebuild Phase 5 embeddings here.** If they are missing, stop and rerun Phase 5.
- **Do not re-sort the corpus by id.** `retrieval/corpus.py` deliberately preserves canonical
  order; everything addresses images by row.
- **Float32 is the protocol.** A float16 query or embedding will move neighbours and break Tx/Rx
  agreement.
- **Self-query is not a RACS result.** It only proves the index maps rows correctly. Evaluation
  secrets are disjoint from the corpus by Phase 6 design.
- **FAISS `IndexFlatIP` is exact, but `IndexIVF*`/`HNSW` are not.** Do not "upgrade" to an
  approximate index in this project.
- **Windows wheel pin**: if `faiss-cpu==1.8.0.post1` fails to resolve, try the newest wheel for
  Python 3.10 before falling back to NumPy.
