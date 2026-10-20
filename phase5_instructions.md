# Phase 5 — CLIP Embeddings

> **Goal**: Turn every corpus image and every text query into a CLIP vector, prove that
> "semantically similar" means "nearby in embedding space", and persist aligned
> embeddings for all 1,000 images.
> **Why**: CLIP is how RACS measures semantic similarity. Phase 6 builds a FAISS index
> over these vectors, and Phase 7 picks one candidate with a token. If the embedding
> rows are not aligned with `image_index.json`, every later phase silently retrieves the
> wrong reference and `Xref_tx != Xref_rx`.

---

## Step 0: Pre-flight checks (do these first)

Phase 5 downloads one model (~600 MB fp32 for ViT-B/32). Confirm disk, connectivity,
and free commit charge:

```powershell
cd d:\racs2

# space on D: (C: is full; every cache must stay on D:)
Get-PSDrive D | Select-Object Used,Free

# HuggingFace reachable? (expect HTTP 200)
try { (Invoke-WebRequest -UseBasicParsing https://huggingface.co -TimeoutSec 20).StatusCode } catch { "UNREACHABLE: $($_.Exception.Message)" }

# model load still needs ~2 GB of free commit on this 16 GB box
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
```

> **If FreeCommitGB is under ~2 GB**, close memory-heavy apps (an IDE language server can
> hold 10 GB) before loading CLIP. `common/hf_lowmem.py` removes the checkpoint-mmap
> commit charge, but the weights still have to be materialised once.

Current state: Phase 4 is complete. `data/retrieval_images/` holds 1,000 PNGs,
`data/metadata/image_index.json` is valid, `corpus_version =
cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff`. **There are no
embeddings yet** — that is what this phase produces.

---

## Step 1: Dependencies

The recommended path needs **no new packages**. `transformers` is already installed and
ships `CLIPModel`, `CLIPProcessor`, and `CLIPTokenizer`. `torchvision` provides the image
transform.

```
transformers 5.19.0 · torch 2.6.0+cu124 · torchvision 0.21.0 · numpy 2.2.6 · Pillow
```

Confirm:

```powershell
.venv\Scripts\python.exe -c "import transformers, torch, torchvision, numpy; print('ok', transformers.__version__)"
```

> **Alternative**: `open_clip_torch` is **not** installed. Only install it (pinned) if you
> deliberately prefer OpenAI's reference implementation. Either choice is fine, but the
> model id, preprocessing, and embedding dimension must then be frozen in the manifest.
> Do **not** switch implementations later — changing the encoder changes every vector.

### Step 0b (only on a flaky network): fetch the checkpoint with the helper

`huggingface_hub`'s resumable downloader can reset mid-transfer and discard the
partial bytes, which makes a 605 MB checkpoint impossible on a poor link. If that
happens, use the chunked downloader:

```powershell
.venv\Scripts\python.exe scripts\phase5_fetch_model.py --threads 12 --chunk-mb 2
```

It downloads 2 MB HTTP ranges in parallel, retries each chunk, resumes from
`pytorch_model.bin.parts.json`, verifies the LFS SHA-256, and fetches the config
and tokenizer files into `models/clip-vit-base-patch32/`. The encoder prefers
`embedding.model_path` when that directory exists, so the run is fully offline.

---

## Step 2: Update `configs/base.yaml`

Add an `embedding` block so the encoder parameters live in one place:

```yaml
embedding:
  model_id: "openai/clip-vit-base-patch32"   # ViT-B/32: 512-d, ~600 MB, fast on 6 GB
  model_path: "models/clip-vit-base-patch32" # local self-contained download (see Step 0b)
  embedding_dim: 512
  image_size: 224                            # CLIP's native input (corpus is 512, resized)
  batch_size: 32
  text_batch_size: 64
  dtype: "float16"                           # fall back to "float32" if the determinism check fails
  normalize: true                            # L2-normalise -> cosine == inner product
  device: "cuda"
  embeddings_path: "experiments/results/phase5_corpus_embeddings.npz"
  manifest_path: "experiments/results/phase5_corpus_embeddings.json"
  # Experiment 1: two topics x three paraphrases
  text_prompts:
    - "a dog playing fetch in a park"
    - "a puppy running through grass"
    - "a golden retriever catching a ball"
    - "a red city bus on a street"
    - "a bus driving through traffic"
    - "a public transit bus at a stop"
    - "a plate of pasta with tomato sauce"
    - "a bowl of spaghetti on a table"
  # Experiment 3: free-text queries for cross-modal retrieval
  text_queries:
    - "a person riding a motorcycle"
    - "a plate of food on a table"
    - "a stop sign"
    - "people playing a sport"
```

**Protocol decision**: `model_id`, `image_size`, `normalize`, and the preprocessing rule
are frozen. Record them in the manifest and never change them mid-experiment.

---

## Step 3: Create the Phase 5 structure

```
d:\racs2\
├── models/
│   └── clip_encoder.py                 # CLIP load + image/text encoding
├── scripts/
│   ├── phase5_fetch_model.py           # resilient chunked CLIP download (flaky networks)
│   ├── phase5_clip.py                  # 3 experiments + embed the whole corpus
│   └── phase5_verify.py                # alignment/determinism/sanity gate
└── experiments/
    └── results/
        ├── phase5_corpus_embeddings.npz    # vectors + ordered image_ids
        ├── phase5_corpus_embeddings.json   # manifest (what produced the vectors)
        ├── phase5_experiments.json         # exp 1-3 numbers
        ├── phase5_exp2_neighbors.png       # image -> nearest images grid
        ├── phase5_exp3_text_queries.png    # text -> top-K images grid
        └── phase5_semantic_vs_structure.png# the gate demonstration
```

No new top-level folders. Do **not** add `faiss` and do **not** build an index here.

---

## Step 4: Write `models/clip_encoder.py`

**Import order matters** (the same trap as Phases 2-3):

```python
import common.paths                            # redirect HF cache to D:
import common.hf_lowmem as hf_lowmem
hf_lowmem.install()                            # BEFORE transformers
import torch
from transformers import CLIPModel, CLIPProcessor
```

Otherwise `transformers` may mmap safetensors with the default copy-on-write behaviour
and trip the Windows commit-charge limit (`OSError 1455`).

**API:**

```python
class CLIPEncoder:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 device: str | None = None, dtype: torch.dtype = torch.float16):
        ...

    @property
    def embedding_dim(self) -> int: ...        # 512 for ViT-B/32

    def encode_images(self, images: list[PIL.Image], batch_size: int = 32,
                      normalize: bool = True) -> np.ndarray:
        """-> float32 [N, 512], rows in the same order as `images`."""

    def encode_image_paths(self, paths: list[Path], batch_size: int = 32,
                           normalize: bool = True) -> np.ndarray: ...

    def encode_texts(self, texts: list[str], batch_size: int = 64,
                     normalize: bool = True) -> np.ndarray:
        """-> float32 [N, 512], rows in the same order as `texts`."""


def load_clip(model_id=..., device=None, dtype=torch.float16) -> CLIPEncoder: ...
```

**Implementation requirements:**
- Use `CLIPModel.from_pretrained(model_id)` + `CLIPProcessor.from_pretrained(model_id)`.
  Pass `dtype=torch.float16` if this `transformers` build accepts it; otherwise load
  fp32 and call `model.to(dtype)`. Keep `model.eval()` and wrap inference in
  `torch.inference_mode()`.
- Use `processor(images=..., return_tensors="pt")` from **PIL RGB** images. Never pass
  tensors that skipped the processor: the resize/crop/mean/std are part of the model.
- Use `model.get_image_features(**inputs)` and
  `model.get_text_features(**inputs)`. Do not use `logits_per_image` (that is a
  contrastive head and would change the geometry).
- Convert features to **float32 numpy** (`.float().cpu().numpy()`), then L2-normalise
  along axis 1. Never store fp16 vectors: Phase 6's inner-product index should not
  inherit fp16 rounding.
- `processor(text=..., padding=True, truncation=True, max_length=77,
  return_tensors="pt")` for text. CLIP truncates at 77 tokens silently — keep prompts
  short and assert the tokenizer did not truncate in the experiment script.
- Batch both modalities; free GPU memory between batches with no retained graph
  (`inference_mode` is enough).
- Document the preprocessing rule in the docstring:
  `resize_shortest_side_224_bicubic -> center_crop_224 -> CLIP mean/std`.

**It should work like this:**

```python
from models.clip_encoder import load_clip
enc = load_clip()
v = enc.encode_texts(["a dog in a park", "a red bus"])
print(v.shape, (v ** 2).sum(axis=1))   # (2, 512) [1. 1.]
```

---

## Step 5: Write `scripts/phase5_clip.py`

This script does the three plan experiments **and** the full-corpus embedding pass.

```
1. import common.paths; import common.hf_lowmem + install(); set_seed(42, verbose=True)
2. Load configs/base.yaml -> embedding block; print device + model id
3. enc = load_clip(model_id, device, dtype)
4. corpus = Corpus(); assert len(corpus) == 1000; print corpus.summary()
5. WARM-UP: encode one image and one text, discard (first call compiles kernels)
6. EXPERIMENT 1 — text similarity
   a. encode text_prompts; assert tokenizer did not truncate
   b. compute the full cosine matrix (vectors are unit norm -> dot product)
   c. report within-topic mean vs across-topic mean; assert within > across
   d. save the matrix + means into phase5_experiments.json
7. EXPERIMENT 2 — image -> nearest images
   a. encode a deterministic subset (first 64 images in canonical order)
   b. cosine similarity to the rest of the subset; exclude self
   c. for 4 query images, save a grid: query on the left, top-5 nearest on the right
      -> experiments/results/phase5_exp2_neighbors.png
8. EXPERIMENT 3 — text -> most similar corpus images
   a. encode all 1,000 corpus images (this also produces the deliverable vectors)
   b. for each free-text query, compute cosine vs all 1,000, take top-5
   c. save a grid -> experiments/results/phase5_exp3_text_queries.png
   d. record the top-5 ids + scores in phase5_experiments.json
9. GATE DEMONSTRATION — semantic != structural
   a. take the highest-scoring cross-modal pair (or a known same-topic pair)
   b. compute a cheap structural proxy: convert both to grayscale, apply
      ImageFilter.FIND_EDGES (or a Sobel filter), downsample to 64x64, and compare
      the edge maps (cosine or normalised cross-correlation)
   c. save the pair side by side with both scores
      -> experiments/results/phase5_semantic_vs_structure.png
   d. record "high CLIP score + low edge correlation" cases in the JSON
10. FULL CORPUS PASS
   a. iterate corpus.records() in canonical order (never iter_images() with a re-sort)
   b. encode in batches of batch_size, appending in order
   c. assert rows == 1000 == len(corpus) and ids match corpus.ids() exactly
   d. time the pass; report images/sec and total seconds
11. OPTIONAL caption embeddings
   a. encode the 1,000 `caption` fields from the index (same row order)
   b. useful for Phase 6 checks and for the report; save as `caption_embeddings`
12. SAVE
   a. np.savez_compressed(embeddings_path,
        image_ids=np.array(corpus.ids()),
        embeddings=image_embeddings.astype(np.float32),
        caption_embeddings=... (optional),
        model_id=..., corpus_version=..., index_sha256=..., created_utc=...)
   b. write the manifest JSON (Step 6) atomically (tmp + os.replace)
13. PRINT a summary: dim, count, norms, timing, within/across text similarity,
    caption->image top-1 accuracy
```

**Experiment 1 detail**: use several paraphrases per topic so the result is not a
single-pair coincidence. Report the full 8x8 matrix in the JSON and one line like
`within=0.xx across=0.xx`.

**Experiment 3 detail**: prefer queries whose answers exist in COCO (`a stop sign`,
`a plate of food`, `a person riding a motorcycle`) so the visual grid is checkable.

---

## Step 6: Define the embedding artifact schema

`experiments/results/phase5_corpus_embeddings.npz`:

| key | dtype / shape | meaning |
| :-- | :-- | :-- |
| `image_ids` | unicode `[1000]` | canonical order, **identical** to `corpus.ids()` |
| `embeddings` | float32 `[1000, 512]` | L2-normalised image vectors, row i ↔ `image_ids[i]` |
| `caption_embeddings` | float32 `[1000, 512]` | optional, same rows, from the index captions |
| `model_id`, `corpus_version`, `index_sha256`, `created_utc` | scalar | provenance |

`experiments/results/phase5_corpus_embeddings.json`:

```json
{
  "schema_version": 1,
  "model_id": "openai/clip-vit-base-patch32",
  "embedding_dim": 512,
  "count": 1000,
  "corpus_version": "cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff",
  "index_sha256": "<sha256 of data/metadata/image_index.json>",
  "preprocess": "resize_shortest_side_224_bicubic -> center_crop_224 -> clip_mean_std",
  "dtype_stored": "float32",
  "normalized": true,
  "device_used": "cuda",
  "dtype_compute": "float16",
  "seconds_total": 41.2,
  "images_per_second": 24.3,
  "created_utc": "2026-10-20T00:00:00Z"
}
```

**Rules that matter:**
- Row order is the contract. Phase 6 will build FAISS row i from `embeddings[i]` and
  map it back through `image_ids[i]`. Never sort, filter, or dedupe this file.
- Store float32 even if inference ran in fp16.
- Store `corpus_version` and `index_sha256`. If either differs from the live index, the
  embeddings are stale and must be regenerated.
- Keep the manifest next to the `.npz`; it is what makes the numbers reproducible.

---

## Step 7: Write `scripts/phase5_verify.py`

The gate script. It must run **offline** (no model download) and check:

```
1. .npz loads; image_ids length == 1000 == embeddings.shape[0]
2. embeddings.shape[1] == manifest embedding_dim == model.config.projection_dim
3. embeddings dtype is float32; caption_embeddings (if present) has the same shape
4. every value is finite (no NaN/Inf)
5. row order: image_ids[i] == corpus.ids()[i] for all i
6. normalisation: ||embeddings[i]||_2 ≈ 1 (max deviation < 1e-4)
7. provenance: manifest corpus_version == corpus.version(); index_sha256 matches
8. determinism: re-encode 8 sampled images with the same encoder and compare to the
   stored rows. PASS if max|diff| <= 1e-4. If it fails only in fp16, rerun those 8 in
   float32 and report which mode is stable.
9. cross-modal sanity: encode the 1,000 captions (or reuse caption_embeddings),
   compute cosine against the image matrix, and report caption->image top-1 accuracy.
   Assert it is well above chance (chance = 0.001); expect a large number.
10. retrieval sanity: for 20 random corpus images, the top-1 image neighbour is
    usually themselves at similarity 1.0 and the next neighbours score < 1.0.
11. report timing and vectors/second from the manifest.
```

Exit non-zero on any failed check. Save a small JSON summary next to the artifacts.

---

## Step 8: Run and Check

```powershell
cd d:\racs2
# optional, only if the normal HF download keeps resetting:
.venv\Scripts\python.exe scripts\phase5_fetch_model.py --threads 12 --chunk-mb 2
.venv\Scripts\python.exe scripts\phase5_clip.py      # downloads CLIP, runs 3 experiments, embeds corpus
.venv\Scripts\python.exe scripts\phase5_verify.py    # offline gate
```

The first run downloads ViT-B/32 (~600 MB) into `D:\racs2\cache\huggingface`. Expect a
few minutes: model load, up to ~80 experiment encodings, then 1,000 images (roughly
20-30 images/s on the RTX 3050 in fp16).

### ✅ Checklist (all must pass):

```
[ ] CLIP ViT-B/32 loads (fp16) and both modalities encode
[ ] Experiment 1: within-topic text similarity > across-topic
[ ] Experiment 2: image nearest-neighbour grid saved and sensible
[ ] Experiment 3: text query -> top-5 corpus images saved and sensible
[ ] 1,000 image embeddings saved as float32 [1000, 512]
[ ] Row order exactly matches image_index.json order
[ ] Vectors are unit norm (max deviation < 1e-4), all finite
[ ] Manifest records model id, preprocessing, corpus_version, index_sha256
[ ] Re-encoding a sample reproduces the stored vectors (<= 1e-4)
[ ] Caption -> image top-1 accuracy far above chance
[ ] Gate demonstration saved: high CLIP similarity but low structural overlap
[ ] No FAISS index built (that is Phase 6)
```

**When all checks pass, Phase 5 is done.** Then we move to **Phase 6 — FAISS
Retrieval** (`IndexFlatIP` over these exact embeddings, `Top-K` sweep).

---

## The Gate Question (answer in your own words)

> *What does CLIP similarity actually measure, and why is it not the same as structural
> compatibility — and why does that matter for steganography?*

Be ready to show it with your own numbers:

- CLIP places images and text that describe the **same content** near each other. Two
  photos of "a dog in a park" score high even if one dog is left-facing and the other is
  right-facing, one is wide and the other is portrait, and the backgrounds have
  completely different layouts.
- Your gate demo should show at least one pair with **high CLIP cosine** but a **low edge
  map correlation** — same semantics, different geometry.
- Why it matters: EDICT hides the secret in the latent space of the **retrieved
  reference**. A semantically right but structurally wrong reference can still produce a
  valid stego image, but the secret's latent structure no longer matches the reference's
  structure, so recovery quality degrades. That gap is precisely what structural
  reranking (Phases 8-9) exists to close.
- Also be able to explain cosine similarity, why normalising makes inner product equal
  cosine, and why `IndexFlatIP` (Phase 6) is the deterministic choice.

---

## Report back

- What I built / files added
- Command(s) I ran, plus `model_id`, `corpus_version`, and `index_sha256`
- Observed numbers: embedding time (total, images/s), within vs across text similarity,
  caption->image top-1 accuracy, samples of text queries -> top-5 ids
- The gate demo pair and its two scores
- What surprised me / what is unclear
- My answer to the gate question

---

## Notes / risks

- **Import order**: `common.paths` -> `common.hf_lowmem.install()` -> `transformers`.
  Skipping this can reintroduce the Windows `OSError 1455` commit-charge failure.
- **fp16 determinism**: GPU fp16 reductions are usually reproducible but not guaranteed.
  If the sample re-encode differs by more than 1e-4, switch `embedding.dtype` to
  `float32` and regenerate. Protocol correctness beats a few seconds of speed.
- **Row alignment is the whole phase**: always iterate `corpus.records()` (or `ids()`),
  never a re-sorted path glob, when building the matrix.
- **Storage**: 1,000 x 512 float32 is ~2 MB. Keep the `.npz` in git (small); the 429 MB
  corpus stays gitignored.
- **Do not** add a projection head, a fine-tuned CLIP, or prompt engineering. This phase
  is a frozen, off-the-shelf encoder so Tx and Rx can reproduce the same vectors.
- **Scope boundary**: no FAISS, no token selection, no structural features beyond the
  cheap gate demo. Those are Phases 6-9.
