# Phase 19: Structural Representation

> **Goal**: Build a deterministic structural descriptor that measures arrangement (edges, layout,
> segmentation, landmarks) rather than semantics, cache it for the 1000-image corpus, and prove
> with concrete examples that CLIP can rank a semantically similar but structurally wrong image
> above a structurally right one.
> **Why**: CLIP answers "what is in the image". RACS also needs "is it arranged the same way".
> Structural compatibility is the reranking signal in Phase 20.

> **Precondition**: Phase 17 is frozen. Phase 18 is frozen. The corpus row order is the protocol
> order and must be reused unchanged.

> **Leakage rule**: build the probe and choose thresholds on corpus images only. Do not inspect
> evaluation secrets to design the structural metric. The evaluation split stays untouched until
> Phase 21.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase17_verify.py
.venv\Scripts\python.exe scripts\phase18_verify.py
.venv\Scripts\python.exe -c "import cv2, skimage, numpy; print(cv2.__version__, skimage.__version__, numpy.__version__)"
```

Expected: Phases 17 and 18 PASS. OpenCV and scikit-image are installed; either is acceptable for
edges. Prefer one and freeze it.

Decision to make explicitly before coding:

```text
Default structural kind : edges (Canny / FIND_EDGES)
Frozen size             : 128 x 128, grayscale
Frozen binarisation     : threshold 0.10 on [0,1] edge magnitude
Frozen tolerance        : dilate each edge map by 2 px before IoU
Primary score           : dilated IoU in [0, 1]
Diagnostic score        : edge-magnitude cosine
```

`scripts/phase5_clip.py` already contains a validated `edge_map()` / `edge_similarity()` pair
(`FIND_EDGES` at 64 px, threshold 0.10). Reuse the idea, but implement the frozen 128 px version
in `retrieval/structural.py`; do not import a script module from a library module.

Optional class-specific representations, pre-registered before any evaluation run:

```text
common : segmentation IoU, reusing semantic/features.py (SegFormer/B0 or OneFormer)
facial : landmark PCK / skeleton IoU, reusing the Phase 7 face extractor
style  : edges only (Phase 7 uses none for style)
```

If a class-specific extractor is too slow or unavailable, stay on `edges` for every class and
record the choice. Do not mix representations between Tx and Rx, and do not switch per candidate.

---

## Step 1: Config

Append under `retrieval:`:

```yaml
retrieval:
  # ... existing keys stay unchanged ...
  structural:
    kind: "edges"                 # edges | segmentation | landmarks
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
```

The cache path encodes the kind. If you add `segmentation`, use a separate cache file; never
overwrite the edges cache.

---

## Step 2: Structure

```text
retrieval/
  structural.py          # extract, similarity, cache build/load
scripts/
  phase19_structural.py  # corpus cache + probe experiment + grid
  phase19_verify.py
experiments/results/phase19/
  structural_edges.npz   # image_ids + uint8 maps + provenance
experiments/results/
  phase19_structural.json
  phase19_semantic_vs_structural.png
  phase19_verify.json
```

---

## Step 3: Implement `retrieval/structural.py`

```python
def extract_structural(image: PIL.Image, *, kind: str = "edges", size: int = 128,
                       threshold: float = 0.10) -> np.ndarray:
    """Return a uint8 array. edges: (size, size) binary; segmentation: (size, size) labels."""

def structural_similarity(a: np.ndarray, b: np.ndarray, *, kind: str = "edges",
                          dilation_px: int = 2) -> dict:
    """Return {"primary": float in [0,1], "iou": float, "cosine": float}."""

def build_structural_cache(corpus, *, kind: str, config: dict,
                           out_path: Path) -> dict: ...

def load_structural_cache(manifest_path: Path) -> dict:
    """Return {"maps": np.ndarray, "image_ids": list[str], "similarity_kind": str, ...}."""
```

Rules:

- Deterministic only. No random seeds, no sampling, no GPU nondeterminism. Canny/FIND_EDGES and
  resizing are deterministic.
- `kind == "edges"`:
  - load RGB, convert to grayscale, resize to `(size, size)` BICUBIC.
  - compute edge magnitude (`cv2.Canny` on uint8, or `PIL.ImageFilter.FIND_EDGES` scaled to
    [0,1]; pick one and freeze).
  - store a uint8 binary map (`magnitude >= threshold`). Store the magnitude too only if the
    cache stays under ~50 MB; otherwise recompute the diagnostic cosine from the binary map.
- `kind == "segmentation"`:
  - reuse the Phase 7 extractor (`semantic.features.build_feature_extractor`) at its frozen
    output size, then nearest-neighbour resize of the label map to `size`.
  - store uint8/uint16 labels. Do not remap class ids.
- `kind == "landmarks"`:
  - reuse the Phase 7 facial extractor. Store a fixed-length keypoint vector padded with a
    visibility mask; define PCK with a fixed normaliser (inter-ocular or image diagonal).
- Similarity:
  - `edges`: dilate both binary maps by `dilation_px` (default 2), compute
    `IoU = |A and B| / |A or B|`, with `0.0` when the union is empty. Report magnitude cosine as
    a diagnostic. Return `primary = iou`.
  - `segmentation`: mean IoU over the union of classes present in either map, ignoring the
    background only if that is frozen in config.
  - `landmarks`: PCK at a frozen threshold, mapped to [0,1].
- Cache builder:
  - iterate `Corpus().records()` in canonical order.
  - assert the resulting `image_ids` list matches `Corpus().ids()` exactly.
  - save with `np.savez_compressed` (or `np.savez` if compression is too slow) and record
    `maps_sha256` over `np.ascontiguousarray(maps).tobytes()`, plus `corpus_version`,
    `image_index_sha256`, `kind`, `size`, `threshold`, `dilation_px`, `count`, `dtype`,
    `created_utc`, `code_commit`.
- Loading validates row count, order and hash. A mismatch is a hard failure.

Do not compute structural features for evaluation secrets in this phase except in the final
Phase 21 run. The probe uses corpus images.

---

## Step 4: Implement `scripts/phase19_structural.py`

Algorithm:

```text
1. set_seed(42)
2. build the corpus structural cache (1000 images, canonical order)
3. load it back and verify row order + hash
4. probe: choose probe_size=50 corpus images with seed 42 (fixed permutation)
   for each query:
     a. CLIP embed the query (reuse the frozen Phase 5 encoder, float32)
     b. retrieve Top-10 from the frozen Phase 17 index (exclude self for this analysis)
     c. compute structural similarity(query_map, candidate_map) for each candidate
     d. record CLIP score, structural score, rank
5. find disagreement cases: CLIP rank <= 3 but structural similarity <= 0.40,
   or the structurally best candidate is not the CLIP top-1
6. select at least 3 concrete examples, save their numbers
7. build phase19_semantic_vs_structural.png:
   rows = example; cols = query | CLIP top-1 (score) | structural best (score)
8. save phase19_structural.json
```

Output schema:

```json
{
  "schema_version": 1,
  "phase": 19,
  "kind": "edges",
  "size": 128,
  "threshold": 0.10,
  "dilation_px": 2,
  "primary_score": "iou_dilated",
  "cache_path": "...",
  "maps_sha256": "...",
  "corpus_version": "...",
  "count": 1000,
  "probe": {
    "size": 50,
    "seed": 42,
    "mean_clip_top1": 0.0,
    "mean_structural_top1": 0.0,
    "mean_structural_of_clip_top1": 0.0,
    "correlation_clip_vs_structural": 0.0,
    "disagreement_count_at_0.40": 0
  },
  "examples": [
    {
      "query_id": "...",
      "clip_top1_id": "...", "clip_top1_score": 0.0,
      "clip_top1_structural": 0.0,
      "structural_best_id": "...", "structural_best_score": 0.0,
      "structural_best_clip": 0.0,
      "note": "CLIP ranks a different layout first"
    }
  ],
  "grid_path": "experiments/results/phase19_semantic_vs_structural.png"
}
```

The correlation is descriptive only. Do not report a p-value without stating the test and n.

---

## Step 5: Implement `scripts/phase19_verify.py`

Checks:

```text
[ ] cache count == 1000 and image_ids order == image_index order
[ ] maps_sha256 recomputes exactly
[ ] corpus_version and image_index_sha256 match the Phase 17 manifest
[ ] all maps have the frozen shape and dtype
[ ] every map is finite and within the declared value range
[ ] extraction is deterministic: recompute 10 maps and compare bytes
[ ] similarity is symmetric: sim(a,b) == sim(b,a) within 1e-12
[ ] similarity of a map with itself is 1.0
[ ] empty-vs-nonempty edge maps return 0.0, not NaN
[ ] at least 3 disagreement examples recorded with numbers
[ ] phase19_semantic_vs_structural.png exists
[ ] no evaluation secret appears in the probe
[ ] phases 17 and 18 artifacts are unchanged
```

Exit non-zero on failure.

---

## Step 6: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase19_structural.py
.venv\Scripts\python.exe scripts\phase19_verify.py
```

Expected summary:

```text
kind            : edges, 128x128, threshold 0.10, dilation 2
cache           : 1000 maps, canonical order, hash verified
probe           : 50 corpus queries
disagreements   : >= 3 with CLIP high / structure low
grid            : phase19_semantic_vs_structural.png
verify          : PASS
```

Inspect the grid. Each row must show a query, a CLIP-favoured candidate that looks semantically
right but structurally different, and a structurally closer candidate.

---

## Checklist

```text
[ ] one frozen structural kind and parameter set
[ ] deterministic extraction, no randomness
[ ] corpus cache in canonical order with maps_sha256
[ ] similarity metric in [0,1] with documented degenerate behaviour
[ ] self-similarity 1.0, symmetry tested
[ ] probe uses 50 corpus images with a frozen seed
[ ] at least 3 concrete CLIP-vs-structure disagreement examples with numbers
[ ] class-specific extractors are opt-in and pre-registered, not ad hoc
[ ] no evaluation secret used for design or threshold choice
[ ] no Phase 17/18 artifact modified
[ ] phase19_structural.json and phase19_verify.json written
```

**When this passes, the structural representation is frozen.** Phase 20 may change how it is
weighted, but not how it is extracted or compared.

---

## The Gate Question (answer in your own words)

> *What is the difference between semantic similarity and structural compatibility, and why can
> CLIP alone pick the wrong reference for steganography?*

Answer with your artifacts:

- CLIP embeds "what is in the image" and is largely insensitive to pose, layout, viewpoint and
  fine geometry. Two images of the same object in different arrangements can have high cosine.
- Structural similarity compares arrangement: edges, segmentation layout, landmarks. It can be
  low even when CLIP is high.
- A reference that is semantically right but structurally wrong gives the hiding/conditioning
  stage a mismatched layout. That mismatch is what the reranking stage is meant to reduce.
- The three examples are the evidence: report CLIP and structural scores side by side, and show
  the structurally better candidate.
- This is a representation test, not a quality result. Phase 20 measures whether reranking helps.

---

## Report back

- Kind, size, threshold, dilation and similarity definition
- Cache count, order, maps sha256, corpus_version
- Probe size, mean CLIP and structural scores, correlation
- The three disagreement examples with exact numbers and the grid path
- Whether a class-specific extractor was used or rejected, and why
- Your answer to the gate question

---

## Notes and risks

- **Do not tune the structural metric on evaluation secrets.** Use the corpus probe; Phase 21
  measures the evaluation effect.
- **Edges are translation-sensitive.** A small shift collapses IoU. The 2 px dilation is the
  frozen tolerance; changing it changes Phase 20 rankings.
- **Segmentation class imbalance can inflate IoU.** Report the per-class IoU and the background
  policy if segmentation is used.
- **Landmark models can be nondeterministic across CPU/GPU or library versions.** Pin the
  extractor and run it on one device; record the version.
- **Cache provenance matters.** If `image_index.json` changes, the cache is invalid; rebuild,
  do not patch.
- **Do not add a learned structural model here.** That is a different project and would weaken
  the reproducibility story.
