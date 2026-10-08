# Phase 4 — Build the Retrieval Image Database

> **Goal**: Build the shared image corpus that RACS retrieves from — a folder of
> images plus a versioned metadata index and a loader that can iterate it
> deterministically.
> **Why**: The database *is* the RACS core. No database means no retrieval. Both the
> transmitter (Tx) and the receiver (Rx) must use the **exact same** corpus, ordering,
> and preprocessing, or they retrieve different references and recovery fails.

---

## Step 0: Pre-flight checks (do these first)

Phase 4 needs **no new model weights** — only disk and a corpus download. Confirm space
and connectivity:

```powershell
cd d:\racs2
# free space on D: (C: is nearly full; keep everything on D:)
Get-PSDrive D | Select-Object Used,Free

# are the COCO hosts reachable? (expect HTTP 200)
try { (Invoke-WebRequest -UseBasicParsing http://images.cocodataset.org/zips/val2017.zip -Method Head -TimeoutSec 20).StatusCode } catch { "UNREACHABLE: $($_.Exception.Message)" }
```

> **Target size**: 1,000 images. As 512×512 PNGs that is roughly **0.4–0.8 GB** on
> disk. The raw COCO download is ~1 GB more (delete the zip + raw copies once the
> canonical corpus is built, per Step 4).

Current state: `data/retrieval_images/` exists but is **empty**, and there is no
`data/metadata/` yet. That is what this phase fills in.

---

## Step 1: Dependencies

Nothing new is required for the corpus build or the loader. These are already installed:

```
Pillow 12.3.0 · torchvision 0.21.0+cu124 · numpy 2.2.6 · PyYAML 6.0.3
huggingface_hub 1.33.0 · requests 2.34.2 · tqdm 4.70.1
```

Confirm quickly:

```powershell
.venv\Scripts\python.exe -c "import PIL, torchvision, numpy, yaml; print('ok')"
```

> `datasets` is **not** installed. Prefer the direct COCO download below (no extra
> dependency, fully reproducible). If you would rather pull a HF dataset, install
> `datasets` first — but then pin the dataset revision so the corpus is reproducible.

---

## Step 2: Pick and document the corpus source

Start small and **frozen**. Recommended source: **COCO val2017** — 5,000 images with
5 captions each, so Phase 5 gets free text for the cross-modal CLIP experiment.

```powershell
# into a scratch dir on D: (gitignored later / deleted in Step 4)
$raw = "D:\racs2\data\_raw_coco"
New-Item -ItemType Directory -Force -Path $raw | Out-Null

# images: ~778 MB zip, 5000 jpg
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "$raw\val2017.zip"

# annotations: ~241 MB zip (needed for captions)
Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "$raw\annotations.zip"
```

Unzip both into `$raw`, giving `$raw\val2017\*.jpg` and
`$raw\annotations\captions_val2017.json`.

**Alternatives** (only if COCO is unreachable): an ImageNet-1k val subset, an Unsplash
download, or a pinned HF dataset. Whatever you choose, record the source, license, and a
revision/date in the index header so the corpus is reproducible.

> **Do not** fetch live at Tx/Rx runtime. Acquisition happens **once**, here; the frozen
> corpus ships with the repo.

---

## Step 3: Update `configs/base.yaml`

Add a `retrieval` block so the corpus parameters live in one place:

```yaml
retrieval:
  corpus_dir: "data/retrieval_images"
  metadata_dir: "data/metadata"
  index_path: "data/metadata/image_index.json"
  corpus_size: 1000          # frozen target; grow 1k -> 5k -> 10k later
  subset_seed: 42            # deterministic which images are picked
  canonical_size: 512        # stored image edge, matches SD/EDICT later
  resize_mode: "shortest_side_then_center_crop"   # documented, not implicit
  source: "coco_val2017"
```

The resize rule is a **protocol decision**: pick one, write it down, and never change it,
because Tx and Rx must produce byte-identical canonical images.

---

## Step 4: Create the Phase 4 structure

```
d:\racs2\
├── data/
│   ├── retrieval_images/            # canonical corpus: <image_id>.png (512x512 RGB)
│   └── metadata/
│       └── image_index.json         # the versioned index (see Step 6)
├── retrieval/
│   ├── __init__.py
│   └── corpus.py                    # loader: iterate + lookup
└── scripts/
    ├── phase4_build_corpus.py       # acquire/subset/preprocess -> corpus + index
    └── phase4_verify.py             # integrity + determinism checks
```

```powershell
New-Item -ItemType Directory -Force -Path "d:\racs2\data\retrieval_images" | Out-Null
New-Item -ItemType Directory -Force -Path "d:\racs2\data\metadata" | Out-Null
New-Item -ItemType Directory -Force -Path "d:\racs2\retrieval" | Out-Null
New-Item -ItemType File -Force -Path "d:\racs2\retrieval\__init__.py" | Out-Null
```

Extend `common/paths.py` (do not duplicate path logic elsewhere) with:

- `RETRIEVAL_IMAGES_DIR`, `METADATA_DIR`, `IMAGE_INDEX_PATH`.

---

## Step 5: Write `scripts/phase4_build_corpus.py`

The one-time corpus builder. It should do these things **in order**:

```
1. import common.paths; set_seed(42, verbose=True)
2. Load configs/base.yaml -> retrieval block
3. List raw source images, sort by id (stable, locale-independent)
4. Deterministically pick corpus_size ids:
       rng = random.Random(subset_seed); ids = sorted(rng.sample(all_ids, corpus_size))
   (sort the sample so the canonical order does NOT depend on dict/FS order)
5. For EACH selected id:
   a. Open with PIL, convert("RGB")
   b. Apply the documented resize rule -> exactly 512x512
   c. Save as data/retrieval_images/<image_id>.png  (image_id = "coco_val2017_<12-digit>")
   d. Collect: image_id, image_path, source_id, caption (first COCO caption), width,
      height, sha256 (of the saved PNG bytes), added_utc
6. Copy the COCO captions into the index (or a sidecar captions file)
7. Write data/metadata/image_index.json (schema in Step 6) atomically
   (write to *.tmp then os.replace)
8. Print: count written, bytes on disk, unique-image check, elapsed seconds
```

**Failure handling**: skip and log any image that fails to decode, but keep the count
honest. If more than ~1% fail, stop and investigate.

Clean up afterwards (`Remove-Item -Recurse $raw`) once you have verified the corpus, so
the ~1.8 GB of raw downloads does not sit in the repo.

---

## Step 6: Define the index schema

`data/metadata/image_index.json` — the single source of truth for the corpus:

```json
{
  "schema_version": 1,
  "corpus_version": "<sha256 of the sorted image_id list>",
  "source": "coco_val2017",
  "canonical_size": 512,
  "resize_mode": "shortest_side_then_center_crop",
  "count": 1000,
  "created_utc": "2026-10-08T00:00:00Z",
  "images": [
    {
      "image_id": "coco_val2017_000000000139",
      "image_path": "data/retrieval_images/coco_val2017_000000000139.png",
      "source_id": "139",
      "caption": "A dining table with a bowl of fruit.",
      "width": 512,
      "height": 512,
      "sha256": "<hex digest of the PNG bytes>"
    }
  ]
}
```

**Rules that matter:**
- `corpus_version` (hash of the ordered id list) is the protocol version. Log it on
  every Tx/Rx run; if it differs, abort rather than recover garbage.
- `sha256` per image lets both ends prove they hold identical bytes.
- Keep `image_id` stable forever — Phase 5/6 embeddings and Phase 7 token selection
  index into this list, so re-ordering or re-numbering breaks the whole stack.

---

## Step 7: Write `retrieval/corpus.py`

A small, dependency-light loader (stdlib + PIL only):

**Functions / API:**
- `load_index(path=None) -> dict` — reads and validates the JSON (`schema_version`,
  `count == len(images)`).
- `class Corpus`:
  - `__len__()` -> count
  - `ids()` -> ordered list of `image_id`
  - `records()` -> iterator over index records in canonical order
  - `get(image_id) -> record`
  - `load(image_id) -> PIL.Image` (RGB)
  - `version()` -> `corpus_version`
- `iter_images(corpus)` — yields `(record, PIL.Image)` in canonical order.

**It should work like this:**
```python
from retrieval.corpus import Corpus
c = Corpus()
print(len(c), c.version())
for rec, img in c.iter_images():
    ...  # img is RGB 512x512, deterministic order
```

---

## Step 8: Write `scripts/phase4_verify.py`

The integrity + determinism gate. It should check:

```
1. index loads; schema_version == 1; count == len(images) == corpus_size
2. every image_path exists on disk
3. every image opens as RGB and is exactly 512x512
4. count unique image_ids == count  (no duplicates)
5. re-load the corpus twice and confirm ids() order is identical (determinism)
6. recompute corpus_version from the id list and confirm it matches the stored value
7. re-hash a random sample (say 20) images and confirm sha256 matches the index
8. report total bytes on disk and mean image size
9. save a contact sheet (e.g. 8x8 of the first 64) to
   experiments/results/phase4_corpus_grid.png
```

---

## Step 9: Run and Check

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase4_build_corpus.py     # download/subset/build (once)
.venv\Scripts\python.exe scripts\phase4_verify.py           # integrity + determinism
```

The build is the only network step; `verify` must run **offline**.

### ✅ Checklist (all must pass):

```
[ ] Corpus built deterministically (same subset_seed -> same ids)
[ ] data/retrieval_images/ contains exactly corpus_size PNGs (1000)
[ ] data/metadata/image_index.json written, schema_version=1, count=1000
[ ] Every indexed path exists; every image is RGB 512x512
[ ] No duplicate image_ids
[ ] corpus_version recomputed == stored corpus_version
[ ] Sampled sha256 hashes match the index
[ ] Loader iterates in a stable order across runs
[ ] Contact sheet saved to experiments/results/phase4_corpus_grid.png
[ ] Raw COCO downloads removed (repo stays small)
```

**When all checks pass, Phase 4 is done.** Then we move to **Phase 5 — CLIP embeddings**
(image + text embeddings over this exact corpus; no retrieval index yet).

---

## The Gate Question (answer in your own words)

> *Why must the retrieval database be a fixed, versioned, shared resource — and what
> exactly breaks at the receiver if Tx and Rx disagree?*

Be ready to explain:
- Why a hash/version (`corpus_version`) plus per-image `sha256` is stronger than trusting
  "we both downloaded COCO".
- Why preprocessing (resize rule, RGB, 512×512) is part of the protocol: a 1-pixel shift
  or a different resize filter changes the retrieved image and thus `Xref`.
- How the corpus will grow (1k → 5k → 10k) without invalidating earlier embeddings or
  breaking `image_id` stability.

---

## Report back

- What I built / files added
- Command(s) I ran (and the corpus source + `corpus_version`)
- Observed numbers (image count, disk size, build time)
- What surprised me / what is unclear
- My answer to the gate question

---

## Notes / risks

- **Disk**: keep the raw download in a scratch dir on D: and delete it after the build.
  C: is full; if a tool writes to `%TEMP%`, point it at `D:\racs2\temp` first.
- **Reproducibility**: sort before sampling, store the ordered id list hash, and never
  re-order the index. This is the same determinism discipline used in Phases 2–3.
- **Scope boundary**: Phase 4 stores **no embeddings** and builds **no FAISS index** —
  that is Phases 5–6. Keep this phase to corpus + metadata + loader.
