# Phase 7: Public Conditions (`Kpub` and `Xfeat`)

> **Goal**: Build the two public conditioning inputs that AgentSemSteCom uses to generate a
> reference image: the public semantic key `Kpub` and the implicit structural feature `Xfeat`.
> Cache them for every image in both frozen splits, prove they are deterministic, and inspect
> them visually.
> **Why**: These are the inputs to the reference generator in Phase 8. They are public in the
> paper's threat model, but transmitter and receiver must derive byte-identical values or they
> generate different references and `Xref_tx != Xref_rx`. The same artifacts become the RACS
> retrieval query in Phase 17, so this work is shared by both systems.

> **Ordering rule**: Phase 7 is the first AgentSemSteCom component. It may use the Phase 6
> evaluation and calibration splits as read-only inputs. It must not use the retrieval corpus
> to build conditions, must not train anything, and must not touch EDICT or FAISS.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2

# Phase 6 must still be valid
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py

# Split must be present and complete
.venv\Scripts\python.exe -c "import json; m=json.load(open('data/metadata/test_split.json')); print(m['split_version']); print(m['counts'])"

# Disk and HF reachability
.venv\Scripts\python.exe -c "import shutil; print(shutil.disk_usage('D:/').free/1024**3, 'GB free')"
.venv\Scripts\python.exe -c "import requests; print(requests.get('https://huggingface.co', timeout=20).status_code)"

# GPU
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3, 'GB')"
```

Expected:

```text
83 passed, 0 failed
split_version  a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3
counts         calibration 22/22/22, evaluation 22/22/22
```

If Phase 6 fails, stop and repair it. Phase 7 must not paper over a broken split.

---

## Step 1: Dependencies

Already present and used by earlier phases:

```text
torch, transformers, accelerate, Pillow, numpy, PyYAML, requests
```

New dependencies depend on the chosen feature backend:

```powershell
# Preferred facial backend (paper-faithful OpenPose family)
.venv\Scripts\python.exe -m pip install --no-cache-dir controlnet-aux opencv-python-headless

# Lighter fallback if controlnet-aux is unavailable or too heavy
.venv\Scripts\python.exe -m pip install --no-cache-dir mediapipe
```

Rules:

- Do not install a new dependency without recording it in `requirements` or the phase notes.
- Do not train or fine-tune any model in this phase.
- All weights are pretrained and frozen.
- Pin exact model ids in the manifest. Do not use a moving `main` revision.

---

## Step 2: Update `configs/base.yaml`

Add a `semantic` block. Keep model ids and methods explicit so the manifest can hash them.

```yaml
semantic:
  cache_dir: "experiments/results/public_conditions"
  manifest_path: "experiments/results/public_conditions/manifest.json"
  keygen:
    method: "blip_template"          # blip_template | clip_vocab | local_llm
    model_id: "Salesforce/blip-image-captioning-base"
    model_path: null
    device: "cuda"
    dtype: "float16"
    max_new_tokens: 40
    num_beams: 1                     # greedy only, no sampling
    template: "a photorealistic photograph of {caption}, natural composition, detailed textures, realistic lighting"
    max_chars: 300
  features:
    common:
      method: "segformer"            # segformer | oneformer
      model_id: "nvidia/segformer-b0-finetuned-ade-512-512"
      model_path: null
      device: "cuda"
      dtype: "float16"
      output_size: 512
      num_classes: 150
    facial:
      method: "openpose"             # openpose | mediapipe
      model_path: null
      output_size: 512
      min_keypoints: 5
    style:
      method: "none"                 # paper: style is Kpub-only
```

Protocol decisions:

- `method`, `model_id`, `template`, `output_size`, and model weights are frozen once the cache
  is written. Changing any of them invalidates the manifest.
- `Kpub` is a prompt, not a verbatim caption. Keep the raw caption in an audit field if useful,
  but store the templated prompt as `Kpub`.
- `style` has no `Xfeat`. Represent this by an explicit `feature_kind = "none"`, not by an empty
  or blank image.

---

## Step 3: Create the Phase 7 structure

```text
d:\racs2\
├── semantic\
│   ├── __init__.py
│   ├── keygen.py                  # Kpub generation
│   ├── features.py                # Xfeat extraction and preview rendering
│   └── cache.py                   # deterministic cache + manifest helpers
├── scripts\
│   ├── phase7_build_conditions.py # build/cache both splits
│   └── phase7_verify.py           # offline determinism + content gate
├── experiments\
│   └── results\
│       ├── public_conditions\
│       │   ├── <image_id>.npz
│       │   ├── <image_id>_preview.png
│       │   └── manifest.json
│       ├── phase7_conditions_grid.png
│       └── phase7_verify.json
└── docs\
    └── DEVIATIONS.md              # append only if the paper's models are replaced
```

Add the cache arrays to `.gitignore` but keep the manifest and gate evidence:

```gitignore
experiments/results/public_conditions/*
!experiments/results/public_conditions/manifest.json
```

The cached arrays and previews are derived from split images. The conditions are public in the
paper's model, but the raw pixels should still stay out of git.

---

## Step 4: Implement `semantic/keygen.py`

### API

```python
def build_keygen(config: dict): ...
def generate_kpub(image: PIL.Image, *, config: dict, context: dict) -> dict:
    """Returns {"kpub": str, "method": str, "model_id": str, "raw_caption": str|None}."""
```

`context` carries `image_id`, `role`, and `class` for logging and leakage checks, not for the
prompt content.

### Method A: `blip_template` (default)

```text
image
  -> BLIP processor (RGB, model-specific resize)
  -> BLIP image-captioning model, fp16, eval, no gradients
  -> greedy caption (num_beams=1, do_sample=False, max_new_tokens=40)
  -> strip whitespace and newlines
  -> Kpub = template.format(caption=caption)
  -> enforce max_chars, no image_id, no token, no path
```

Determinism requirements:

- `model.eval()`, `torch.inference_mode()`.
- `do_sample=False`, `num_beams=1`.
- Fixed dtype and device.
- Fixed `max_new_tokens`.
- No global RNG dependence. If a backend still uses RNG, seed it explicitly.
- Two runs on the same image must return exactly the same string.

### Method B: `clip_vocab` (fallback, no captioning model)

Use the frozen Phase 5 CLIP encoder and a small, versioned vocabulary:

```text
image
  -> CLIP image embedding
  -> cosine similarity against fixed subject / setting / style phrases
  -> top-1 subject, top-1 setting, top-1 style
  -> Kpub = template.format(subject=..., setting=..., style=...)
```

Rules:

- The vocabulary lives in `configs/semantic_vocab.yaml` and is hashed into the manifest.
- Sort candidates before scoring so ties resolve deterministically.
- This is a documented deviation from an LLM-generated key.

### Method C: `local_llm` (optional)

Only if a local instruction-tuned model fits the 6 GB budget. Requirements:

- Fixed model id and revision.
- Greedy decoding only.
- No cloud API in the Tx/Rx path.
- A deterministic template around the model output.
- Record the method and weights hash in the manifest.

### Kpub validation

- Non-empty string.
- At most `max_chars` characters.
- Single line, no tabs, no newlines.
- Does not contain `image_id`, `role`, `class`, token, wrong token, file path, or SHA-256.
- Must read as a generation prompt, not a label such as `"facial"` or `"common"`.
- The raw caption, if any, is stored only as audit metadata and is not the `Kpub`.

---

## Step 5: Implement `semantic/features.py`

### API

```python
def build_feature_extractor(config: dict, class_name: str): ...
def extract_feature(image: PIL.Image, *, class_name: str, config: dict) -> dict:
    """Returns feature kind, array, preview image, and model metadata."""
```

`class_name` is one of `common`, `facial`, `style`.

### Common: semantic segmentation

Preferred paper model is OneFormer. A 6 GB-safe default is SegFormer-B0 on ADE20K.

```text
image 512x512 RGB
  -> SegFormer image processor
  -> SegFormer model, fp16, eval, no gradients
  -> logits [1, C, h, w]
  -> bilinear upsample to 512x512
  -> argmax over C
  -> class-index map [512, 512], uint8
  -> colorized ADE20K RGB preview [512, 512, 3]
```

Store:

```text
feature_kind = "segmentation"
feature      = uint8 [512, 512] class indices
preview      = RGB PNG using the ADE20K palette
num_classes  = 150
```

Validation:

- shape exactly `(512, 512)`.
- dtype `uint8`.
- every value `< num_classes`.
- at least two distinct classes and a non-trivial foreground fraction.
- no all-zero map.

### Facial: skeleton graph

Preferred paper model is OpenPose. Two implementations are acceptable:

1. `openpose`: `controlnet_aux.OpenposeDetector` or an equivalent OpenPose model. Include face
   keypoints if the backend exposes them. Render the standard OpenPose skeleton colors.
2. `mediapipe`: MediaPipe Face Mesh or Pose. Render a deterministic graph from the landmark
   edges.

Data flow:

```text
image 512x512 RGB
  -> detector
  -> keypoints [K, 3] as (x, y, confidence)
  -> keep keypoints above the confidence threshold
  -> draw edges and joints on a black 512x512 canvas
  -> skeleton RGB preview
```

Store:

```text
feature_kind = "skeleton"
feature      = float32 [K, 3] keypoints
preview      = RGB PNG [512, 512, 3]
min_keypoints = 5
```

Validation:

- at least `min_keypoints` valid keypoints.
- keypoints inside the image bounds after clamping.
- skeleton preview is not blank.
- the source image has a detectable face or pose. If not, record `status = "no_face"` and fail the
  gate for that image rather than silently writing an empty skeleton.

### Style: no structural feature

```text
feature_kind = "none"
feature      = absent
preview      = none
reason       = "paper conditions style on Kpub only"
```

Do not create a blank image. Do not fake a segmentation map. Absence is the contract.

---

## Step 6: Cache and manifest schema

### Per-image cache

Write one `.npz` per image, atomically. Suggested fields:

```text
image_id          str
role              str
class             str
source_sha256     str
kpub              str
kpub_method       str
kpub_model_id     str
feature_kind      str
feature           array, or absent for style
feature_shape     int32[2] or int32[3]
feature_dtype     str
config_hash       str
```

Write `<image_id>_preview.png` for `common` and `facial`. Do not embed the preview in the `.npz`.

### Manifest

`experiments/results/public_conditions/manifest.json`:

```json
{
  "schema_version": 1,
  "created_utc": "2026-10-23T00:00:00Z",
  "split_version": "a57d9c8c...",
  "corpus_version": "cc20af21...",
  "config_hash": "<sha256 of the resolved semantic config>",
  "models": {
    "keygen": {"method": "blip_template", "model_id": "...", "weight_sha256": "..."},
    "common": {"method": "segformer", "model_id": "...", "weight_sha256": "..."},
    "facial": {"method": "openpose", "model_id": "...", "weight_sha256": "..."}
  },
  "counts": {
    "calibration": {"common": 22, "facial": 22, "style": 22},
    "evaluation": {"common": 22, "facial": 22, "style": 22}
  },
  "records": [
    {
      "image_id": "eva_common_0000",
      "role": "evaluation",
      "class": "common",
      "source_sha256": "...",
      "cache_path": "experiments/results/public_conditions/eva_common_0000.npz",
      "preview_path": "experiments/results/public_conditions/eva_common_0000_preview.png",
      "kpub_sha256": "<sha256 of the Kpub string bytes>",
      "feature_sha256": "<sha256 of canonical feature bytes>",
      "feature_kind": "segmentation",
      "status": "ok"
    }
  ]
}
```

`feature_sha256` must include shape and dtype, not just raw bytes. A simple canonical form:

```text
sha256(dtype + "|" + str(shape) + "|" + array.tobytes())
```

Write the manifest atomically with `.tmp` then `os.replace`. Sort records by
`(role, class, image_id)`.

---

## Step 7: Implement `scripts/phase7_build_conditions.py`

### CLI

```powershell
python scripts\phase7_build_conditions.py
python scripts\phase7_build_conditions.py --split evaluation
python scripts\phase7_build_conditions.py --limit 2 --force
python scripts\phase7_build_conditions.py --method clip_vocab
```

### Algorithm

```text
1. set_seed(42, verbose=True)
2. load configs/base.yaml -> semantic block
3. load both split roles through evaluation.splits.Split
   - role="calibration" requires allow_calibration=True
4. load keygen and feature backends once, reuse across images
5. for each record in deterministic order (role, class, image_id):
     a. resolve source path and verify source sha256 against the split record
     b. skip if cache exists, hashes match, and not --force
     c. load image as RGB 512x512
     d. generate Kpub
     e. extract Xfeat for the class
     f. validate Kpub and feature
     g. write .npz atomically
     h. write preview PNG for common and facial
     i. append manifest record
6. write manifest atomically
7. write a contact sheet
8. print counts, cache size, config hash, and any failures
```

### Contact sheet

`experiments/results/phase7_conditions_grid.png`:

```text
rows:    common, facial, style
columns: source | condition preview | one or two examples per class
labels:  image_id, role, class, feature kind
```

For style, show the source and a text label `Kpub only` in the preview column.

### Error policy

- If an image fails, record `status = "failed"` with the reason and continue. Do not write a
  partial cache file.
- At the end, if any image in either split failed, exit non-zero.
- Never silently replace a failed feature with a blank image.

---

## Step 8: Implement `scripts/phase7_verify.py`

The verifier runs offline and writes `experiments/results/phase7_verify.json`.

### Checks

```text
1. MANIFEST
   a. manifest exists, schema_version == 1
   b. split_version matches data/metadata/test_split.json
   c. config_hash recomputes
   d. counts are 22/22/22 for both roles
   e. every record has a unique image_id and is in the split

2. CACHE INTEGRITY
   a. every cache file exists
   b. source_sha256 matches the split record
   c. kpub_sha256 recomputes from the stored Kpub
   d. feature_sha256 recomputes from the stored array
   e. preview files exist for common and facial, absent for style

3. KPUB CONTENT
   a. non-empty, <= max_chars, single line
   b. no image_id, token, path, or sha256 substring
   c. method and model_id recorded
   d. no duplicate Kpub across different images above a small controlled threshold
      (duplicates are allowed but reported)

4. FEATURES
   a. common: shape (512,512), uint8, values < num_classes, >=2 distinct classes,
      foreground fraction in a sane range
   b. facial: keypoints >= min_keypoints, coordinates in bounds, preview non-blank
   c. style: feature_kind == "none" and no feature array
   d. every class has the expected feature kind

5. DETERMINISM
   a. rebuild a fixed sample (at least 2 images per class per role) with the same config
   b. Kpub strings are exactly equal
   c. feature arrays are exactly equal, shape and dtype included
   d. preview hashes are equal
   e. report max absolute difference for float keypoints, expected exactly 0

6. VISUAL
   a. phase7_conditions_grid.png exists and has the expected row count
   b. the grid is inspected manually before the gate is signed off
```

### Output

```json
{
  "pass": true,
  "split_version": "a57d9c8c...",
  "config_hash": "...",
  "checks": [{"check": "...", "pass": true, "detail": "..."}],
  "counts": {"calibration": {...}, "evaluation": {...}},
  "failures": []
}
```

Exit non-zero on any failure.

---

## Step 9: Run and check

```powershell
cd D:\racs2

# Build both roles
.venv\Scripts\python.exe scripts\phase7_build_conditions.py

# Verify
.venv\Scripts\python.exe scripts\phase7_verify.py
```

Expected summary:

```text
calibration  common=22 facial=22 style=22
evaluation   common=22 facial=22 style=22
cache records: 132
failed: 0
split_version: a57d9c8c...
verify: PASS
```

Then open:

```text
experiments/results/phase7_conditions_grid.png
```

Confirm:

- common rows show a coherent segmentation map over the main object and background.
- facial rows show a face or pose skeleton aligned with the source.
- style rows show the source with an explicit `Kpub only` preview.
- no row is blank, misaligned, or from the wrong class.

---

## Checklist

```text
[ ] Phase 6 selftest still passes 83/83
[ ] semantic/keygen.py exists and records method, model id, and weights hash
[ ] semantic/features.py implements common segmentation, facial skeleton, style none
[ ] Kpub is deterministic, single-line, bounded, and contains no identifier or token
[ ] common feature is a valid class-index map, not a blank or all-zero map
[ ] facial feature has at least 5 keypoints and a non-blank skeleton preview
[ ] style explicitly has feature_kind = "none" and no fake image
[ ] both roles cached: 22 per class, 132 records total
[ ] manifest split_version matches test_split.json
[ ] every source_sha256 matches the Phase 6 split
[ ] cache and preview files exist and are atomically written
[ ] two runs produce exactly equal Kpub and feature arrays on the determinism sample
[ ] phase7_verify.json exists with pass=true
[ ] phase7_conditions_grid.png inspected and coherent
[ ] no model trained, no FAISS, no EDICT, no retrieval integration
[ ] any paper deviation (OneFormer to SegFormer, OpenPose to MediaPipe, LLM to BLIP) is
    appended to docs/DEVIATIONS.md
```

**When all checks pass, Phase 7 is done.** Then move to Phase 8: the reference generator.

---

## The Gate Question (answer in your own words)

> *Why are `Kpub` and `Xfeat` called public conditions, and why must transmitter and receiver
> still derive exactly the same values if they are public?*

Answer with your artifacts:

- Public means they carry no reversible information about the secret. An eavesdropper learning
  them does not learn the secret.
- They are still conditioning inputs. If Tx and Rx produce different `Kpub` or `Xfeat`, they
  generate different `Xref`, and EDICT inversion fails even with the correct token.
- "Public" removes the private-key exchange problem. It does not remove the determinism
  requirement.
- The cache manifest is the evidence: same split, same model ids, same weights, same config,
  same Kpub, same feature array.

---

## Report back

- What was built and which files were added
- Kpub method and model, common segmentation model, facial skeleton backend
- Whether any paper model was replaced, and the `docs/DEVIATIONS.md` entry
- `config_hash`, split version, record counts, cache size
- Kpub examples for one image per class
- Determinism result: exact equality for Kpub and feature arrays
- Contact sheet path and what was visually confirmed
- What surprised you or remains unclear
- Your answer to the gate question

---

## Notes and risks

- **Determinism is the product.** Do not use temperature sampling, beam search with unstable
  tie-breaking, or cloud APIs. If a model cannot be made deterministic, do not put it in the
  Tx/Rx path.
- **Public does not mean safe to leak identifiers.** `Kpub` must not contain the image id,
  token, path, or hash. Those are metadata, not semantics.
- **Class leakage is still a trap.** Common segmentation must come from a common image, facial
  skeleton from a facial image, style must have no feature. Do not let the facial detector run on
  a cat and call the result a skeleton.
- **SegFormer is a deviation from OneFormer.** The plan allows it for the 6 GB budget. Record it
  if used.
- **OpenPose vs MediaPipe is a deviation.** Prefer the paper's family. If MediaPipe is used,
  record it and state that the skeleton topology differs.
- **BLIP template is a deviation from an LLM-generated key.** It is deterministic and local.
  Record it. Do not claim it is the paper's exact LLM.
- **Do not train anything.** No fine-tuning, no LoRA, no adapters. All models are frozen.
- **Do not integrate retrieval.** Phase 7 may use CLIP for the fallback keygen, but it must not
  build a FAISS index or select references.
- **Cache size:** 132 segmentation maps and skeleton previews are small, but keep the arrays out
  of git. Commit the manifest, verifier output, and contact sheet only.
- **Changing a model invalidates the cache.** Bump the config hash, rebuild all records, and
  re-run Phase 8. Do not patch one class.
