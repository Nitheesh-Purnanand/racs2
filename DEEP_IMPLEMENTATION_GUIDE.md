# RACS2 Deep Implementation Guide (Phases 1-6)

This document is the implementation-level companion to `PHASE_RUNBOOK_1_TO_6.md`.
The runbook says what to run and what output to expect. This document explains what
the code actually does, which data enters and leaves each phase, how tensors and
files change, which models are loaded, whether anything is trained, and what every
artifact in `experiments/results/` contains.

Scope: Phases 1 through 6. Phases 4 and 5 are included because their artifacts exist
and are consumed later, even though they are banked until Phase 17.

---

## 1. One-page system view

RACS2 is being built in two layers:

```text
Layer A: the AgentSemSteCom baseline (Phases 7-16)
  semantic key -> reference image generator -> EDICT hiding -> channel -> recovery

Layer B: the RACS contribution (Phases 17-22)
  semantic key -> CLIP + FAISS retrieval -> semantic/structural reranking
               -> token-controlled selection -> Xref
               -> reuse the exact same EDICT/channel/recovery stack
```

Phases 1-6 do not build either layer. They build the foundation and the measurement
rig:

```text
Phase 1  environment, seeding, device, image/tensor contract
Phase 2  Stable Diffusion VAE round-trip, latent-space error floor
Phase 3  Stable Diffusion v1.5 generation cost and output verification
Phase 4  frozen 1,000-image retrieval corpus + index (banked)
Phase 5  1,000 CLIP embeddings + retrieval experiments (banked)
Phase 6  frozen calibration/evaluation splits + metric/report harness
```

The most important architectural fact for Phases 1-6:

> No model is trained or fine-tuned. Every neural network is loaded from pretrained
> weights, frozen, and used for inference only.

Training enters much later, mainly at Phase 9 (ControlNet fine-tuning, if the paper's
zero-conv branch is reproduced). No training data is accumulated in Phases 1-6, and
the evaluation split is never used for any optimization.

---

## 2. Repository map

```text
D:\racs2\
├── common\
│   ├── paths.py        centralized paths + HF/diffusers/torch cache redirection
│   ├── seed.py         Python/NumPy/PyTorch seeding, torch.Generator helper
│   ├── device.py       CUDA/CPU selection and VRAM reporting
│   └── hf_lowmem.py    read-only safetensors mmap shim for Windows commit limits
├── models\
│   ├── vae.py          AutoencoderKL load/encode/decode/roundtrip
│   ├── sd_pipeline.py  SD v1.5 low-VRAM loader + deterministic generate()
│   └── clip_encoder.py frozen CLIP image/text encoder
├── retrieval\
│   ├── schema.py       image_index.json schema, hashing, validation
│   └── corpus.py       deterministic read-only corpus loader
├── evaluation\
│   ├── metrics.py      MSE, PSNR, SSIM, LPIPS, distribution, all_metrics
│   ├── splits.py       role-tagged split loader, token loader, manifests
│   └── report.py       frozen report schema + atomic writer
├── configs\base.yaml   all protocol constants
├── data\
│   ├── secret_images\  5 source images used by Phases 1-3
│   ├── retrieval_images\  1,000 canonical PNGs (Phase 4, gitignored)
│   ├── calibration\unistega\  66 calibration PNGs (Phase 6, gitignored)
│   ├── test_set\  66 evaluation PNGs (Phase 6, gitignored)
│   ├── metadata\  image_index.json, test_split.json, tokens.json
│   └── _raw_sources\  downloads and extraction caches (gitignored)
├── scripts\            phase drivers and verifiers
├── experiments\results\  all measured outputs
└── docs\DEVIATIONS.md  append-only deviation register
```

Everything under `data/retrieval_images/`, `data/calibration/`, and `data/test_set/`
is gitignored. The manifests in `data/metadata/` are committed.

---

## 3. Environment and hardware

Measured in the project virtual environment:

```text
Python       3.10.11
PyTorch      2.6.0+cu124      CUDA 12.4, available=True
GPU          NVIDIA GeForce RTX 3050 6GB Laptop GPU
NumPy        2.2.6
Pillow       12.3.0
diffusers    0.41.0
transformers 5.19.0
accelerate   1.15.0
scipy        1.15.3
scikit-image 0.25.2
PyYAML       6.0.3
requests     2.34.2
LPIPS        installed (AlexNet backbone used)
```

Storage footprint at the time of writing:

| Location | Files | Size |
| :-- | --: | --: |
| `data/secret_images` | 5 | 4.0 MiB |
| `data/retrieval_images` | 1,000 | 409.4 MiB |
| `data/calibration/unistega` | 66 | 24.0 MiB |
| `data/test_set` | 66 | 24.9 MiB |
| `data/metadata` | 3 | 0.4 MiB |
| `data/_raw_sources` | 406 | 129.1 MiB |
| `models/clip-vit-base-patch32` | 18 | 580.7 MiB |
| `cache/huggingface` | 117 | 33.8 GiB |
| `cache/torch` | 1 | 233.1 MiB |
| `experiments/results` | 33 | 17.75 MiB |

`cache/huggingface/hub` contains SD v1.5, the SD VAE, and CLIP. The local CLIP
directory is a self-contained copy used offline.

---

## 4. Global data contracts

These conventions are shared by every phase. Breaking them silently changes results.

### 4.1 Image representation

| Context | Representation |
| :-- | :-- |
| On disk | PNG, RGB, uint8, `[0, 255]` |
| Phase 1 tensor | `[1, 3, 512, 512]`, float32, `[0, 1]` |
| VAE input | `[1, 3, 512, 512]`, float16 on GPU, `[-1, 1]` |
| VAE latent | `[1, 4, 64, 64]`, float16, scaled by the VAE scaling factor |
| CLIP input | processor output, 224x224, CLIP mean/std normalization |
| Metric input | `[N, 3, H, W]`, float, `[0, 1]` |
| LPIPS internal | `[-1, 1]` after explicit `x * 2 - 1` |

The project stores images in `[0, 1]` for metrics and converts to `[-1, 1]` only
inside the VAE/LPIPS boundaries. `evaluation/metrics.py` asserts the `[0, 1]`
range before converting for LPIPS.

### 4.2 Canonicalization rule

The frozen rule, used by Phase 4 and Phase 6:

```text
open image
-> convert to RGB
-> scale so the shortest side becomes 512, using bicubic
-> center crop to 512 x 512
-> save PNG
```

This is recorded in manifests as:

```text
shortest_side_512_bicubic -> center_crop_512
```

No EXIF rotation, auto-contrast, augmentation, or timestamp metadata is applied.

### 4.3 Determinism and hashing

| Object | Hash definition |
| :-- | :-- |
| Retrieval corpus | SHA-256 of the newline-joined ordered `image_id` list |
| Image file | SHA-256 of the exact PNG bytes |
| Split | `sha256_json` over ordered `[role, class, image_id, sha256]` tuples |
| Tokens | `sha256_json` over ordered `[image_id, token, wrong_token]` triples |
| Report config | `sha256_json` over the resolved config dict |

`sha256_json` uses `json.dumps(..., sort_keys=True, separators=(",", ":"))`, so
dictionary key order is irrelevant but list order matters.

### 4.4 Seeding

`common.seed.set_seed(seed, deterministic=False)` sets:

- `PYTHONHASHSEED`
- Python `random`
- NumPy global RNG
- `torch.manual_seed`, `torch.cuda.manual_seed`, `torch.cuda.manual_seed_all`

If `deterministic=True`, it also enables cuDNN deterministic mode and
`torch.use_deterministic_algorithms(True, warn_only=True)`.

The diffusion pipeline does not rely on global RNG. Each `generate()` call builds a
fresh `torch.Generator` on the pipeline device and seeds it explicitly.

### 4.5 Cache redirection

`common.paths` is imported before any model library. It sets:

```text
HF_HOME                  -> D:\racs2\cache\huggingface
HUGGINGFACE_HUB_CACHE    -> D:\racs2\cache\huggingface\hub
TRANSFORMERS_CACHE       -> D:\racs2\cache\huggingface\hub
DIFFUSERS_CACHE          -> D:\racs2\cache\huggingface\hub
TORCH_HOME               -> D:\racs2\cache\torch
```

This exists because `C:` is nearly full. `common.hf_lowmem.install()` then replaces
safetensors loading with a read-only `mmap.ACCESS_READ` mapping. On Windows, the
normal copy-on-write mmap charges the entire checkpoint against the system commit
limit, which caused `OSError 1455` and `MemoryError` while loading SD.

---

## 5. Phase 1: Environment and tensor contract

### 5.1 Files

| File | Role |
| :-- | :-- |
| `common/paths.py` | path constants, cache redirection, directory creation |
| `common/seed.py` | deterministic seeding, `get_generator()` |
| `common/device.py` | `get_device()`, `print_gpu_info()` |
| `scripts/phase1_verify.py` | 9-section verification script |
| `configs/base.yaml` | global seed, image size, device, dtype |

### 5.2 Implementation

`phase1_verify.py`:

1. Calls `set_seed(42, verbose=True)`.
2. Calls `get_device()`, which returns CUDA when available.
3. Prints GPU name, total VRAM, allocated VRAM, reserved VRAM.
4. Loads `configs/base.yaml`, asserts `seed == 42`.
5. Opens `data/secret_images/test_001.png`, converts to RGB.
6. Applies `torchvision.transforms.Resize((512, 512))` then `ToTensor()`.
7. Adds a batch dimension, producing `[1, 3, 512, 512]`, float32, `[0, 1]`.
8. Moves the tensor to `cuda:0` and reports VRAM.
9. Draws two `torch.randn(3, 3)` tensors after reseeding and asserts `torch.equal`.

### 5.3 Data change

```text
data/secret_images/test_001.png
  original size 1600 x 902, RGB
  -> Resize((512, 512))
  -> ToTensor()
  -> [1, 3, 512, 512], float32
  -> min 0.035294, max 1.000000
  -> cuda:0
```

No file is written. The value of this phase is the fixed contract, not an artifact.

### 5.4 Expected result

```text
Shape : (1, 3, 512, 512)
Dtype : torch.float32
Tensor device after move: cuda:0
Reproducibility test: PASS
```

---

## 6. Phase 2: VAE round-trip and latent-space error floor

### 6.1 Files

| File | Role |
| :-- | :-- |
| `models/vae.py` | `load_vae`, `encode`, `decode`, `roundtrip` |
| `scripts/phase2_vae.py` | run all secret images, measure, save comparisons |
| `evaluation/metrics.py` | MSE, PSNR, SSIM |
| `data/secret_images/test_001.png` ... `test_005.png` | input images |

### 6.2 Model

Loaded model: `stabilityai/sd-vae-ft-mse`, an `AutoencoderKL`.

Architecture from the checkpoint config:

```text
input channels       3
output channels      3
latent channels      4
block channels       [128, 256, 512, 512]
layers per block     2
spatial downscale    8x
runtime scaling      vae.config.scaling_factor (diffusers AutoencoderKL default 0.18215)
```

The model is loaded with `torch_dtype=torch.float16`, moved to CUDA, set to
`eval()`, and every parameter gets `requires_grad_(False)`. It is not trained.

### 6.3 Encode/decode math

```python
latent_dist = vae.encode(image).latent_dist
latent = latent_dist.mode() if deterministic else latent_dist.sample()
latent = latent * vae.config.scaling_factor

recon = vae.decode(latent / vae.config.scaling_factor).sample
```

The project uses `sample=False`, so the deterministic latent mean is used. This
makes the round-trip reproducible.

### 6.4 Script data flow

```text
for each PNG in data/secret_images:
  RGB
  -> Resize((512, 512))
  -> ToTensor() * 2 - 1          # [1, 3, 512, 512], [-1, 1]
  -> VAE encode
  -> [1, 4, 64, 64], scaled latent
  -> VAE decode
  -> [1, 3, 512, 512], [-1, 1]
  -> (x + 1) / 2, clamp          # back to [0, 1]
  -> mse, psnr, ssim against original
  -> horizontal concatenation original | reconstruction
  -> experiments/results/phase2_<name>_compare.png
```

### 6.5 Measured result

| Image | MSE | PSNR (dB) | SSIM |
| :-- | --: | --: | --: |
| test_001 | 0.000561 | 32.508 | 0.936399 |
| test_002 | 0.000484 | 33.150 | 0.934338 |
| test_003 | 0.000384 | 34.153 | 0.936783 |
| test_004 | 0.000569 | 32.450 | 0.936245 |
| test_005 | 0.000663 | 31.784 | 0.933472 |
| mean | 0.000532 | 32.809 | 0.935447 |

Interpretation: the VAE alone loses about 0.5 percent of pixel energy. Any later
EDICT/channel/recovery stack adds to this floor. The paper's reported numbers cannot
be reproduced below this reconstruction ceiling.

### 6.6 Artifacts

```text
experiments/results/phase2_test_001_compare.png  ...  phase2_test_005_compare.png
```

Each PNG is `1024 x 512`, original on the left, reconstruction on the right.

---

## 7. Phase 3: Stable Diffusion generation cost

### 7.1 Files

| File | Role |
| :-- | :-- |
| `models/sd_pipeline.py` | low-VRAM SD v1.5 loader and `generate()` |
| `scripts/phase3_diffusion.py` | timing sweep + five verification checks |
| `scripts/phase3_img2img.py` | image-to-image strength sweep |
| `configs/base.yaml` `generation` block | prompt, steps, guidance, sweeps |

### 7.2 Model

Pipeline: `StableDiffusionPipeline`, Stable Diffusion v1.5.

Candidate repositories, tried in order:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
sd-legacy/stable-diffusion-v1-5
runwayml/stable-diffusion-v1-5
```

Components from `model_index.json`:

```text
text_encoder  transformers.CLIPTextModel
tokenizer     transformers.CLIPTokenizer
unet          diffusers.UNet2DConditionModel
vae           diffusers.AutoencoderKL
scheduler     diffusers.PNDMScheduler, replaced by DDIMScheduler
```

UNet config:

```text
in_channels          4
out_channels         4
sample_size          64
block_out_channels   [320, 640, 1280, 1280]
layers_per_block     2
cross_attention_dim  768
```

The model is loaded in fp16 with `safety_checker=None`,
`requires_safety_checker=False`, `low_cpu_mem_usage=True`, and, on CUDA,
`device_map="cuda"`. It enables attention slicing and VAE slicing.

The model is not trained. Weights are pretrained SD v1.5 weights. The only change
is the scheduler replacement with DDIM.

### 7.3 Generation data flow

```text
prompt (config default: "a photograph of an astronaut riding a horse")
  -> CLIP text encoder
  -> text embeddings [1, 77, 768]

random latent noise, generated by a fresh seeded torch.Generator
  -> [1, 4, 64, 64]

DDIM denoising loop, N steps
  - with CFG > 1, each step runs the UNet twice:
    conditional prediction + unconditional prediction
  -> denoised latent [1, 4, 64, 64]

VAE decoder
  -> image [1, 3, 512, 512], [-1, 1]
  -> PIL RGB image
  -> PNG
```

### 7.4 Timing results (authoritative JSON)

From `experiments/results/phase3_timings.json`:

| Steps | Seed | Seconds | s/step | Peak VRAM |
| --: | --: | --: | --: | --: |
| 20 | 42 | 5.5195 | 0.2760 | 2.604 GB |
| 20 | 123 | 5.5340 | 0.2767 | 2.604 GB |
| 30 | 42 | 8.1569 | 0.2719 | 2.604 GB |
| 30 | 123 | 8.1713 | 0.2724 | 2.604 GB |
| 50 | 42 | 13.4369 | 0.2687 | 2.604 GB |
| 50 | 123 | 13.4732 | 0.2695 | 2.604 GB |

Mean: `0.2725 s/step`. A 50-step reference costs about 13.4 s. Peak VRAM is
2.604 GB, leaving room under the 6 GB budget.

These are one run from the artifact, not hardware constants. Re-running changes
`seconds` and `seconds_per_step` with machine load. The image hashes stay identical
for the same prompt, steps, guidance, and seed.

CFG cost at 30 steps, seed 42:

```text
CFG off (guidance 1.0)  4.653 s  0.1551 s/step
CFG on  (guidance 7.5)  8.157 s  0.2719 s/step
speedup without CFG     1.75x
```

### 7.5 Verification checks

All true in the JSON:

```text
determinism_same_seed_identical  True
seed_changes_output              True
steps_change_output              True
guidance_changes_output          True
output_non_trivial               True
pass                             True
```

The checks compare SHA-256 of raw RGB bytes plus mean absolute pixel difference.
Determinism means identical output bytes for identical prompt, steps, guidance, and
seed. The other checks prove that each input actually changes the image.

### 7.6 Image-to-image

`scripts/phase3_img2img.py` reuses the loaded SD components to build a
`StableDiffusionImg2ImgPipeline` without a second weight load.

```text
data/secret_images/test_001.png
  -> prepare_input (512x512 RGB)
  -> VAE encode
  -> add noise according to strength
  -> partial DDIM denoising
  -> VAE decode
  -> output PNG
```

Measured:

| Strength | Seconds | Peak VRAM | Mean abs diff vs input |
| --: | --: | --: | --: |
| 0.25 | 3.401 | 2.605 GB | 10.45 |
| 0.50 | 4.451 | 2.605 GB | 19.864 |
| 0.75 | 6.322 | 2.605 GB | 38.066 |

Higher strength means more of the original image is replaced by the prompt. This is
a diagnostic experiment, not part of the baseline pipeline.

### 7.7 Artifacts

```text
phase3_seed42_steps20.png ... phase3_seed123_steps50.png   six sweep images
phase3_verify_base_seed42_steps20.png
phase3_verify_nocfg_seed42_steps20.png
phase3_verify_nocfg_seed42_steps30.png
phase3_timings.json
phase3_run.log
phase3_img2img_test_001_s0.25.png
phase3_img2img_test_001_s0.5.png
phase3_img2img_test_001_s0.75.png
phase3_img2img_test_001_grid.png
phase3_img2img.json
```

---

## 8. Phase 4 (banked): retrieval corpus

Phase 4 is built and verified but consumed only at Phase 17. It is documented here
because its output is a shared protocol resource.

### 8.1 Files

| File | Role |
| :-- | :-- |
| `retrieval/schema.py` | schema constants, hashing, index build/validate/IO |
| `retrieval/corpus.py` | deterministic read-only loader |
| `scripts/phase4_build_corpus.py` | one-time corpus builder |
| `scripts/phase4_verify.py` | offline integrity gate |
| `data/metadata/image_index.json` | frozen index |
| `data/retrieval_images/*.png` | 1,000 canonical images |

### 8.2 Builder data flow

```text
COCO val2017 raw images in data/_raw_coco
  -> sorted file stems
  -> random.Random(42).sample(stems, 1000)
  -> sorted chosen ids
  -> Image.open -> convert RGB
  -> shortest side 512 bicubic -> center crop 512
  -> data/retrieval_images/<image_id>.png
  -> sha256 of PNG bytes
  -> record
```

`image_id` format: `coco_val2017_<12-digit COCO id>`.

Each index record has:

```json
{
  "image_id": "coco_val2017_000000000724",
  "image_path": "data/retrieval_images/coco_val2017_000000000724.png",
  "source_id": "000000000724",
  "caption": "A stop sign is mounted upside-down on it's post. ",
  "width": 512,
  "height": 512,
  "sha256": "ca81e685..."
}
```

Index-level fields:

```text
schema_version  1
corpus_version  cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff
source          coco_val2017
canonical_size  512
resize_mode     shortest_side_then_center_crop
count           1000
created_utc     2026-10-08T17:59:42Z
index_sha256    7b02b77cbbfb4a649d32465c8d0fcafaa71770d76d799eeec4c270331cebc19b
```

`corpus_version` is the SHA-256 of the ordered `image_id` list joined by newlines.
`index_sha256` is the SHA-256 of the whole JSON file, recorded later by Phase 5.

### 8.3 Why order is protocol-critical

Phase 5 embeddings, the future FAISS index, and token selection all address images
by row position. Tx and Rx must load the same bytes in the same order. If either
side re-sorts the index, `Xref_tx != Xref_rx` and recovery fails. `Corpus` never
re-sorts.

### 8.4 Verify gate

`phase4_verify.py` checks:

```text
index exists
schema valid, expected count, every file present
every image decodes as RGB 512x512
no duplicate image_ids
canonical order stable across two loads
corpus_version recomputes
sampled sha256 digests match
contact sheet written
```

Measured result: PASS, 1,000 images, 409.4 MiB on disk, mean about 429 kB/image.
Contact sheet: `experiments/results/phase4_corpus_grid.png`.

---

## 9. Phase 5 (banked): CLIP embeddings

Phase 5 is built and verified but consumed only at Phase 17. It is also used by the
Phase 6 facial-candidate filter.

### 9.1 Files

| File | Role |
| :-- | :-- |
| `models/clip_encoder.py` | frozen CLIP wrapper |
| `scripts/phase5_clip.py` | experiments 1-3, full corpus pass, artifact writer |
| `scripts/phase5_verify.py` | offline gate |
| `scripts/phase5_fetch_model.py` | resumable parallel checkpoint downloader |
| `experiments/results/phase5_corpus_embeddings.npz` | embeddings |
| `experiments/results/phase5_corpus_embeddings.json` | manifest |

### 9.2 Model

Loaded model: `openai/clip-vit-base-patch32`.

Architecture from the checkpoint config:

```text
vision tower:
  hidden size        768
  image size         224
  patch size         32
  layers             12
  attention heads    12
  intermediate size  3072

text tower:
  hidden size        512
  layers             12
  attention heads    8
  vocab size         49408
  max positions      77
  intermediate size  2048

projection dim       512
```

The local checkpoint is `models/clip-vit-base-patch32`. Weight SHA-256:

```text
a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
```

The model is loaded with `CLIPModel.from_pretrained`, moved to the device, and set
to `eval()`. It is not trained. Compute dtype is float32 by protocol.

### 9.3 Preprocessing and normalization

The encoder never hand-rolls preprocessing. It uses `CLIPProcessor`:

```text
resize shortest side to 224, bicubic
center crop 224
CLIP mean/std normalization
```

Text is tokenized with padding, truncation, and `max_length=77`. Image and text
features are projected to 512-d, converted to float32, then row-wise L2 normalized
with `l2_normalize`. Cosine similarity is therefore a dot product.

### 9.4 Corpus embedding data flow

```text
data/retrieval_images/ 1,000 PNGs
  -> Corpus loader, canonical order
  -> batches of 32
  -> PIL RGB -> CLIP processor -> 224x224 tensor
  -> CLIP image encoder -> 512-d float32
  -> L2 normalize
  -> concatenate
  -> [1000, 512] float32
  -> np.savez_compressed
```

Measured: 18.709 s for 1,000 images (`53.451 img/s`), file size 3.82 MB.

NPZ keys:

```text
image_ids              (1000,) strings
embeddings             (1000, 512) float32
caption_embeddings     (1000, 512) float32
model_id               string
corpus_version         string
index_sha256           string
created_utc            string
caption_top1_accuracy  scalar
```

### 9.5 Experiments

Experiment 1, text similarity:

```text
9 prompts, 3 topics x 3 paraphrases
within-topic mean cosine   0.779148
across-topic mean cosine   0.542280
gap                       +0.236868
```

This proves the text encoder groups paraphrases by topic.

Experiment 2, image nearest neighbours:

```text
subset 64 images
4 queries
top-5 neighbours each
```

Artifact: `phase5_exp2_neighbors.png`.

Experiment 3, text query to images:

```text
4 free-text queries
top-5 images each
```

Artifact: `phase5_exp3_text_queries.png`.

Gate demonstration, semantic versus structural:

```text
pair: coco_val2017_000000024243 vs coco_val2017_000000011511
CLIP cosine  0.642682
edge cosine  0.269649
edge IoU     0.198600
```

This is the core motivation for RACS. CLIP says the pair is semantically related
while their edge layout is structurally different. A semantic-only retrieval
pipeline misses that difference. RACS later adds a structural reranker.

Caption to image sanity:

```text
top-1 accuracy 0.476 (chance 0.001)
```

### 9.6 Verify gate

`phase5_verify.py` checks:

```text
shape, dtype, finite values, unit norms
row order matches image_index.json exactly
image_ids unique
manifest corpus_version and index_sha256 match live corpus
npz corpus_version matches manifest
model id and preprocessing recorded
re-encoded sample matches stored vectors (max|diff| <= 1e-4)
caption top-1 above chance
nearest non-self similarity sensible
```

Measured result: PASS.

```text
re-encode max|diff|          2.53e-07
caption top-1               0.476
nearest non-self mean       0.7707
nearest non-self max        0.8794
```

### 9.7 Why float32, not float16

fp16 CLIP changed embeddings by about `7e-4` between batch sizes. That is larger
than the reproducibility budget for Tx/Rx identity. float32 is batch-size invariant
at the `2.5e-07` level, so it is the protocol default.

---

## 10. Phase 6: evaluation harness and frozen splits

### 10.1 Files and responsibilities

| File | Responsibility |
| :-- | :-- |
| `evaluation/metrics.py` | MSE, PSNR, SSIM, LPIPS, `distribution`, `all_metrics` |
| `evaluation/splits.py` | manifest load, role enforcement, token load, atomic writes |
| `evaluation/report.py` | aggregate per-case results into the frozen report schema |
| `scripts/phase6_build_testset.py` | source, filter, canonicalize, freeze, contact sheet |
| `scripts/phase6_eval_selftest.py` | 83-check offline gate |
| `scripts/eval_report.py` | reusable system x class x SNR driver |
| `scripts/dummy_runner.py` | deterministic fake system for the driver |
| `data/metadata/test_split.json` | frozen split manifest |
| `data/metadata/tokens.json` | fixed token and wrong token per image |

### 10.2 Split sources

| Role | Class | Source | Count |
| :-- | :-- | :-- | --: |
| calibration | common | UniStega `content_prompts` | 22 |
| calibration | facial | UniStega `similar_prompts`, FFHQ/CelebA-HQ only | 22 |
| calibration | style | UniStega `style_prompts` | 22 |
| evaluation | common | COCO val2017, disjoint from retrieval corpus | 22 |
| evaluation | facial | Wikimedia Commons portraits, title + CLIP filtered | 22 |
| evaluation | style | Wikimedia Commons cat photographs | 22 |

Total: 132 records, 66 per role, 22 per class.

Calibration is the paper's release, linked through paper reference [46]. It is
read-only and only used for approximate reproduction at Phase 15A.

Evaluation is authoritative. Every claim and every baseline-versus-RACS comparison
uses it.

### 10.3 Calibration facial filtering

The released `similar_prompts` folder mixes 22 face images with 8 COCO non-face
images such as a stop sign, motorcycle, juice, building, jockey, donuts, bird, and
eagle. `unistega_allowed_names()` reads the subset's `config.yaml` and keeps only
entries whose `original_dataset` is `FFHQ` or `CelebA-HQ`. This prevents class
leakage before canonicalization.

Calibration counts from the builder log:

```text
content_prompts:  42 images -> common
similar_prompts:  22 images -> facial
style_prompts:    28 images -> style
```

### 10.4 Evaluation facial filtering

Commons search is lexical. A query for portraits returns paintings, sculptures,
group photographs, and synthetic faces. The builder applies two filters:

```text
title filter:
  reject painting, drawing, engraving, statue, sculpture, bust, collage,
  family/group/three/two-men/two-women, museum/archive identifiers,
  synthetic-face identifiers, animals, masked or covered faces,
  and non-visible-face titles

near-duplicate filter:
  collapse titles that differ only by "(cropped ...)"

CLIP filter:
  keep images whose best match is "a close-up photograph of a human face"
  among four prompts: human face, animal, scene/object, painting/artwork
```

On the frozen build:

```text
Commons candidates fetched: 49
title filter kept:          29
CLIP filter kept:           28
selected:                   22
```

The contact sheet `phase6_splits_grid.png` was inspected. Evaluation facial rows
contain single human faces. Evaluation style rows contain cats. Common rows contain
natural scenes and objects.

### 10.5 Canonicalization and materialisation

`canonicalise()`:

```python
image = Image.open(source).convert("RGB")
if width <= height:
    new_w, new_h = size, max(size, round(height * size / width))
else:
    new_w, new_h = max(size, round(width * size / height)), size
image = image.resize((new_w, new_h), Image.BICUBIC)
left = (new_w - size) // 2
top = (new_h - size) // 2
image = image.crop((left, top, left + size, top + size))
image.save(destination, format="PNG", compress_level=6, optimize=False)
```

`materialise()` writes `eva_<class>_0000.png` or `cal_<class>_0000.png`, computes
the SHA-256 of the saved PNG, drops exact duplicates by hash, and records:

```json
{
  "image_id": "eva_facial_0000",
  "role": "evaluation",
  "class": "facial",
  "path": "data/test_set/facial/eva_facial_0000.png",
  "sha256": "52434c54...",
  "width": 512,
  "height": 512,
  "mode": "RGB",
  "source": "commons:filetype:bitmap portrait photograph person face ...",
  "license": "CC BY-SA 2.0",
  "source_ref": "File:Elderly Gambian woman face portrait.jpg",
  "original_id": "faces_eval_b14afc18_0000"
}
```

### 10.6 Split manifest contract

Top-level fields:

```json
{
  "schema_version": 1,
  "split_version": "a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3",
  "created_utc": "2026-10-23T09:50:08Z",
  "canonical": "shortest_side_512_bicubic -> center_crop_512",
  "classes": ["common", "facial", "style"],
  "counts": {
    "calibration": {"common": 22, "facial": 22, "style": 22},
    "evaluation": {"common": 22, "facial": 22, "style": 22}
  },
  "corpus_version": "cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff",
  "records": [...]
}
```

`split_version` is recomputed at load time over ordered
`[role, class, image_id, sha256]` tuples. Any pixel change, rename, reorder, or role
change invalidates it.

### 10.7 Role enforcement

`Split.load`:

- rejects an unknown role
- requires `allow_calibration=True` for the calibration role
- recomputes `split_version` and refuses a mismatch
- validates required fields, unique ids, unique sha256, unique paths
- verifies every file exists and matches its sha256
- asserts every record has a valid role and class

`Split.get(image_id)` raises `RoleError` if a record's role does not match the
requested split. `assert_disjoint()` compares both paths and sha256 sets against
another split. `assert_disjoint_from_index()` compares against the retrieval corpus.

Measured invariants:

```text
calibration x evaluation overlap        0
calibration x retrieval corpus overlap  0
evaluation x retrieval corpus overlap   0
```

### 10.8 Tokens

One token record per image in both roles.

Implementation:

```python
digest = sha256(image_id).hexdigest()
value = int(digest[:16], 16)
token = value % 100_000_000
wrong = (token + 1 + (value >> 17) % 9_999_999) % 100_000_000
if wrong == token:
    wrong = (token + 7) % 100_000_000
```

Both are zero-padded to 8 decimal digits. `wrong_token` must differ from `token`.
The "no token" condition is represented by absence, not a sentinel.

Measured:

```text
tokens_version           201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
records                  132
token uniqueness         true
wrong-token uniqueness   true
token != wrong_token     true
token/wrong-token overlap 0
```

### 10.9 Metrics implementation

MSE:

```text
mean((a - b)^2) over all pixels and channels
```

PSNR:

```text
10 * log10(data_range^2 / MSE)
data_range = 1.0
returns inf when MSE == 0
```

SSIM:

```text
Gaussian window 11x11, sigma 1.5
C1 = (0.01 * data_range)^2
C2 = (0.03 * data_range)^2
per-channel SSIM map, then mean
```

LPIPS:

```text
backbone: alex, frozen, cached per (net, device)
input: [0, 1] asserted
convert: x * 2 - 1
batch size: 1 by default
lower is better
```

`distribution(values, percentiles=(5, 25, 50, 75, 95))`:

- counts `n`, `n_inf`, `n_missing`, `n_finite`
- computes mean/median/std/min/max over finite values only
- emits `p05`, `p25`, `p50`, `p75`, `p95`
- never lets `inf` PSNR poison the mean

### 10.10 Report schema

`build_report()` enforces:

- role is `calibration` or `evaluation`
- split is a `Split`
- split role matches report role
- `approximate=True` exactly when role is `calibration`
- every result's `image_id` exists in the split
- result class matches the record class
- no duplicate `(image_id, snr_db)`
- PSNR may be infinite, other metrics must be finite
- missing metrics are omitted, not zero-filled
- cells are sorted by class order `common, facial, style` then SNR

Output shape:

```json
{
  "schema_version": 1,
  "system": "dummy",
  "role": "evaluation",
  "approximate": false,
  "split_version": "a57d9c8c...",
  "split_counts": {...},
  "corpus_version": "cc20af21...",
  "config_hash": "edff9ca5...",
  "lpips_net": "alex",
  "percentiles": [5, 25, 50, 75, 95],
  "created_utc": "2026-10-23T10:14:25Z",
  "meta": {
    "lpips_net": "alex",
    "runner": "scripts.dummy_runner:run_case",
    "tokens_version": "20166234...",
    "device": "cuda",
    "seconds_total": 34.623
  },
  "n_results": 264,
  "n_expected": 66,
  "cells": [
    {
      "class": "common",
      "snr_db": 5,
      "n": 22,
      "metrics": {
        "mse":  {"n": 22, "n_inf": 0, "mean": ..., "p05": ..., ...},
        "psnr": {"n": 22, "n_inf": 0, "mean": ..., "p05": ..., ...},
        "ssim": {"n": 22, "n_inf": 0, "mean": ..., "p05": ..., ...},
        "lpips":{"n": 22, "n_inf": 0, "mean": ..., "p05": ..., ...}
      },
      "extra": {
        "reference_latency_s": {...},
        "total_latency_s": {...}
      }
    }
  ]
}
```

### 10.11 Evaluation driver flow

`scripts/eval_report.py`:

```text
parse args: --system, --split, --runner, --out, --config, --snr, --limit,
            --approximate, --device
refuse calibration without --approximate
load configs/base.yaml
load Split with role enforcement
load tokens and tokens_version
import runner from module:function or path.py:function
for each class:
  for each record:
    merge token and wrong_token into the record
    for each SNR:
      call runner(record, snr_db, role)
      require "secret" and "recovered" PIL images of equal size
      convert both to [0, 1] tensors
      compute MSE/PSNR/SSIM/LPIPS
      collect latencies and VRAM if returned
build_report(...)
write_report atomically
print a compact class x SNR table
```

### 10.12 Dummy runner

`scripts/dummy_runner.py` proves the harness without a real pipeline.

```text
secret = the split image
snr_linear = 10^(snr_db / 10)
sigma = sqrt(1 / (snr_linear * 255))
rng = default_rng(sha256(image_id | snr_db) first 16 hex digits)
noisy = clip(secret/255 + rng.normal(0, sigma), 0, 1)
recovered = uint8(noisy * 255 + 0.5)
```

It returns `secret`, `recovered`, `stego=secret`, and fixed latencies. The metrics
are plausible but are not a system result.

Measured dummy report: 264 cases, 12 cells, `approximate=False`.

### 10.13 Selftest coverage

`phase6_eval_selftest.py` runs 83 checks offline and exits non-zero on failure. It
covers:

```text
metrics:
  identical image -> mse 0, psnr inf, ssim 1, lpips 0
  mild Gaussian noise -> psnr in (20, 45), lpips in (0, 0.5)
  structural change -> ssim < 0.6, lpips > 0.15
  project SSIM vs scikit-image SSIM within 0.02
  distribution handles finite values, inf, NaN, and empty input
  all_metrics default keys unchanged
  lpips rejects out-of-range input, accepts grayscale, is batch-invariant

splits:
  both roles load
  counts match config
  every file exists and sha256 matches
  split_version recomputes
  calibration/evaluation disjoint
  both disjoint from the retrieval corpus
  >= 20 images per class
  record schema, 512x512 RGB, provenance, relative paths, no blank images
  manifest metadata and corpus_version consistent
  no duplicate ids or hashes

corruption and role paths:
  corrupted split_version rejected
  duplicate manifest record rejected
  sha256 mismatch rejected
  unknown role rejected
  missing image_id raises KeyError
  other-role id absent from split
  strict token load rejects other-role extras
  calibration load requires allow_calibration

tokens:
  tokens_version recomputes
  one token per image
  uniqueness and token != wrong_token
  token and wrong-token sets do not collide

report:
  evaluation report approximate=False
  calibration report approximate=True
  mixed-role report raises
  report rejects mismatched split role, class mismatch, missing SNR,
  duplicate image+SNR, non-finite non-PSNR metrics
  missing metrics omitted
  write_report writes valid JSON and cleans temp files
  report_digest ignores created_utc
  config_hash stable and sensitive
  cell ordering and counts correct

canonicalisation and cross-checks:
  canonicalise deterministic and handles RGB/gray/RGBA/palette/CMYK
  SSIM and LPIPS symmetric
  records() and manifest return defensive copies
  retrieval-corpus overlap detected
  tokens reproducible from image_id
  config has no duplicate YAML keys
  builder rejects unknown classes and handles HF image URL shapes
```

Measured: `83 passed, 0 failed`.

### 10.14 Phase 6 artifacts

```text
data/metadata/test_split.json
data/metadata/tokens.json
experiments/results/phase6_build_summary.json
experiments/results/phase6_selftest.json
experiments/results/phase6_dummy_eval.json
experiments/results/phase6_splits_grid.png
```

`phase6_splits_grid.png` has six rows, one per role and class, with five samples
each. It is the visual class-coherence check.

---

## 11. Model training register

This is the direct answer to "which model is trained on which data".

### 11.1 What is actually trained in Phases 1-6

Nothing. There is no optimizer, no loss.backward(), no training dataloader, and no
weight update in any phase driver or module.

| Model | Status in this project | Weights | Frozen? |
| :-- | :-- | :-- | :-- |
| SD VAE `stabilityai/sd-vae-ft-mse` | inference only | pretrained | yes |
| SD v1.5 UNet, text encoder, VAE | inference only | pretrained | yes |
| CLIP ViT-B/32 | inference only | pretrained | yes |
| LPIPS AlexNet | inference only | pretrained | yes |

### 11.2 Pretrained provenance

The following descriptions come from the public model cards and the CLIP paper.
They describe the pretrained weights. They are not training runs performed by this
project.

| Model | Public training description |
| :-- | :-- |
| CLIP ViT-B/32 | OpenAI CLIP, contrastive image-text training on about 400 million web image-text pairs (WIT), as described in the CLIP paper |
| SD v1.5 | Latent diffusion model resumed from v1-2, trained for 595k steps at 512x512 on `laion-aesthetics v2 5+`, with 10 percent text-conditioning dropout for classifier-free guidance |
| SD VAE `sd-vae-ft-mse` | Resumed from the `ft-EMA` autoencoder, trained for another 280k steps on a 1:1 ratio of LAION-Aesthetics and LAION-Humans, loss `MSE + 0.1 * LPIPS`, batch size 192. Only the decoder was fine-tuned |
| LPIPS AlexNet | Pretrained AlexNet features plus learned linear weights, used as a perceptual distance metric |

The exact pretrained corpora are not redistributed with this project. Only the
weights are cached locally. The CLIP card itself states the architecture and the
contrastive objective. The 400 million pair figure is from the CLIP paper.

### 11.3 What data would be used if training were added

Not part of Phases 1-6. The plan's future training point is Phase 9, the ControlNet
zero-conv branch. A training set there would be pairs of (condition image, target
image) built from the training split, with the same canonicalization and no
evaluation leakage.

Rules that must hold if training is added:

- calibration data is never used for training
- evaluation data is never used for training or model selection
- the split version and corpus version are recorded in the training manifest
- any fine-tuned weights are hashed and reported
- a training run is a deviation from the paper if the data or objective differs,
  so it goes into `docs/DEVIATIONS.md`

---

## 12. `experiments/results` catalog

All 33 files, grouped by producing phase.

### Phase 2

| File | What it is |
| :-- | :-- |
| `phase2_test_001_compare.png` | original left, VAE reconstruction right |
| `phase2_test_002_compare.png` | same for test_002 |
| `phase2_test_003_compare.png` | same for test_003 |
| `phase2_test_004_compare.png` | same for test_004 |
| `phase2_test_005_compare.png` | same for test_005 |

### Phase 3

| File | What it is |
| :-- | :-- |
| `phase3_seed42_steps20.png` | 20-step sample, seed 42 |
| `phase3_seed42_steps30.png` | 30-step sample, seed 42 |
| `phase3_seed42_steps50.png` | 50-step sample, seed 42 |
| `phase3_seed123_steps20.png` | 20-step sample, seed 123 |
| `phase3_seed123_steps30.png` | 30-step sample, seed 123 |
| `phase3_seed123_steps50.png` | 50-step sample, seed 123 |
| `phase3_verify_base_seed42_steps20.png` | base verification image, CFG on |
| `phase3_verify_nocfg_seed42_steps20.png` | same prompt/seed, CFG off |
| `phase3_verify_nocfg_seed42_steps30.png` | 30-step CFG-off image for cost comparison |
| `phase3_timings.json` | six timing records, CFG cost, verification flags |
| `phase3_run.log` | console log from the timing run |
| `phase3_img2img_test_001_s0.25.png` | img2img output, strength 0.25 |
| `phase3_img2img_test_001_s0.5.png` | img2img output, strength 0.5 |
| `phase3_img2img_test_001_s0.75.png` | img2img output, strength 0.75 |
| `phase3_img2img_test_001_grid.png` | input plus three strengths |
| `phase3_img2img.json` | img2img timings and mean absolute differences |

### Phase 4

| File | What it is |
| :-- | :-- |
| `phase4_corpus_grid.png` | 64-image contact sheet from the 1,000-image corpus |

### Phase 5

| File | What it is |
| :-- | :-- |
| `phase5_corpus_embeddings.npz` | `(1000, 512)` image and caption embeddings plus provenance |
| `phase5_corpus_embeddings.json` | manifest, weight SHA, corpus version, throughput |
| `phase5_exp2_neighbors.png` | 4 image queries with top-5 neighbours |
| `phase5_exp3_text_queries.png` | 4 text queries with top-5 images |
| `phase5_semantic_vs_structure.png` | chosen pair with CLIP and edge scores |
| `phase5_experiments.json` | full experiment 1-3 results and gate pair |
| `phase5_verify.json` | offline verifier checks |

### Phase 6

| File | What it is |
| :-- | :-- |
| `phase6_build_summary.json` | split and token versions, counts, bytes, canonical rule |
| `phase6_selftest.json` | 83 checks, pass/fail, hashes, counts |
| `phase6_dummy_eval.json` | 264-case dummy report, 12 cells |
| `phase6_splits_grid.png` | six-row role x class visual check |

No artifact in this folder is a trained model. The only learned objects are the
pretrained weights in `cache/` and `models/clip-vit-base-patch32/`.

---

## 13. `data` catalog

| Path | Contents | Git tracked |
| :-- | :-- | :-- |
| `data/secret_images/` | 5 PNGs used by Phases 1-3 | yes |
| `data/retrieval_images/` | 1,000 canonical COCO PNGs | no |
| `data/calibration/unistega/` | 66 UniStega PNGs | no |
| `data/test_set/` | 66 evaluation PNGs | no |
| `data/metadata/image_index.json` | retrieval corpus index | yes |
| `data/metadata/test_split.json` | frozen role-tagged split | yes |
| `data/metadata/tokens.json` | 132 token records | yes |
| `data/_raw_sources/` | UniStega zip/extraction, COCO annotations, Commons downloads, HF cache | no |

Raw source directories:

```text
data/_raw_sources/unistega/
data/_raw_sources/coco_eval/
data/_raw_sources/faces_eval_<query-hash>/
data/_raw_sources/style_eval_<query-hash>/
```

The query hash in the name prevents a changed Commons query from silently reusing
old files.

---

## 14. Config reference

`configs/base.yaml` keys that matter:

| Key | Value | Meaning |
| :-- | :-- | :-- |
| `seed` | 42 | global seed |
| `image_size` | 512 | project image edge |
| `device` | cuda | preferred device |
| `dtype` | float16 | model inference dtype |
| `generation.prompt` | astronaut prompt | placeholder until the paper's Kpub example is known |
| `generation.steps_sweep` | [20, 30, 50] | Phase 3 sweep |
| `generation.seeds_sweep` | [42, 123] | Phase 3 sweep |
| `retrieval.corpus_size` | 1000 | frozen corpus target |
| `retrieval.subset_seed` | 42 | deterministic COCO subset |
| `retrieval.canonical_size` | 512 | stored image edge |
| `embedding.model_id` | openai/clip-vit-base-patch32 | CLIP model |
| `embedding.dtype` | float32 | batch-invariant protocol choice |
| `embedding.embedding_dim` | 512 | projection size |
| `evaluation.per_class` | 22 | images per class per role |
| `evaluation.snr_db` | [5, 10, 15, 20] | evaluation SNRs |
| `evaluation.metrics` | mse, psnr, ssim, lpips | reported metrics |
| `evaluation.lpips_net` | alex | frozen metric backbone |
| `evaluation.lpips_batch_size` | 1 | deterministic and memory-safe |
| `evaluation.report_percentiles` | [5, 25, 50, 75, 95] | distribution output |
| `evaluation.sources.facial.commons` | portrait query | Commons facial source |
| `evaluation.sources.style.commons` | cat query | Commons style source |

Changing `classes`, `snr_db`, `metrics`, `lpips_net`, or the split version after
artifacts are written invalidates every stored result. That is why these values are
frozen and recorded in report metadata.

---

## 15. End-to-end data lineage

```text
COCO val2017
  -> Phase 4: deterministic 1,000-image subset
  -> canonical 512x512 PNG corpus + image_index.json
  -> Phase 5: CLIP image embeddings (1000, 512) + caption embeddings
  -> Phase 17: FAISS index (future)

UniStega release
  -> Phase 6 calibration split (66 images, 22/class)
  -> Phase 15A: approximate reproduction only

COCO val2017 (disjoint ids)
  -> Phase 6 evaluation common (22)

Wikimedia Commons portraits
  -> title filter -> CLIP face filter -> Phase 6 evaluation facial (22)

Wikimedia Commons cats
  -> Phase 6 evaluation style (22)

evaluation split (66) + tokens
  -> eval_report.py
  -> runner(record, snr_db, role)
  -> metrics on [0, 1] tensors
  -> build_report
  -> phase6_dummy_eval.json or future system reports
```

---

## 16. Verification status

```text
Phase 1  PASS, 9-section script
Phase 2  PASS, 5 images, PSNR 32.809 dB mean
Phase 3  PASS, all five verification flags true
Phase 4  PASS, 1,000 images, deterministic, index valid
Phase 5  PASS, (1000, 512) float32, row order exact, re-encode 2.53e-07
Phase 6  PASS, 83/83 selftest checks, dummy run 264 cases / 12 cells
```

Frozen identifiers:

```text
corpus_version  cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff
index_sha256    7b02b77cbbfb4a649d32465c8d0fcafaa71770d76d799eeec4c270331cebc19b
CLIP weight     a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
split_version   a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3
tokens_version  201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
```

---

## 17. Known deviations and open items

Recorded in `docs/DEVIATIONS.md`:

- UniStega calibration release is used as-is for approximate reproduction.
- Evaluation facial source is Wikimedia Commons portraits, not the original
  `Norod78/Vintage-Faces-FFHQAligned` dataset, because that dataset contains vintage
  illustrations rather than natural photographs.
- LPIPS backbone is AlexNet. The paper does not state a backbone.
- UniStega calibration uses 22 images per class, the number of clean faces in the
  release. The gate requires at least 20.

Open items that affect later phases:

- `generation.prompt` is still the placeholder astronaut prompt. Phase 8/15 must
  replace it with the paper's Kpub example if one is specified.
- The retrieval corpus is 1,000 images. The plan allows growth to 5k or 10k later,
  which would change `corpus_version` and require re-embedding.
- Phase 9 training (ControlNet) is not implemented. It is the first point where a
  model would actually be trained.
- No FAISS index exists yet. Phase 17 consumes the Phase 5 `.npz`.
- No EDICT, channel model, semantic codec, or security matrix exists yet.

---

## 18. Minimum mental model to continue

If you remember only five things before starting Phase 7:

1. Images are RGB, 512x512, float `[0, 1]` for metrics, `[-1, 1]` only at VAE and
   LPIPS boundaries.
2. CLIP is float32, L2-normalized, and row order must match `image_index.json`.
3. The retrieval corpus, split, and tokens are frozen by SHA-256. Do not re-sample
   or patch one record. If a source changes, rebuild everything and invalidate old
   results.
4. Calibration is the paper's data and is approximate-only. Evaluation is our data
   and is the only basis for claims.
5. Nothing in Phases 1-6 is trained. The first real training point is Phase 9.
