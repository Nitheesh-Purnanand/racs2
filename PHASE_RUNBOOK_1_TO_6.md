# RACS2 Runbook: Phases 1-6

Scope: the non-banked phases on the current critical path. Phases 4 (retrieval corpus)
and 5 (CLIP embeddings) are intentionally excluded because they are banked and consumed
later at Phase 17.

All commands assume the project root and the project virtual environment:

```powershell
cd D:\racs2
.venv\Scripts\python.exe --version
```

Do not double-click the `.py` files and do not use the system `python`. Use the venv
interpreter shown above. Every script bootstraps the project root, so it can be run from
any working directory, but the commands below use `D:\racs2` for clarity.

---

## Quick reference

| Phase | Command | Expected result |
| :-- | :-- | :-- |
| 1 | `.venv\Scripts\python.exe scripts\phase1_verify.py` | Ends with `Phase 1 verification complete`; reproducibility `PASS` |
| 2 | `.venv\Scripts\python.exe scripts\phase2_vae.py` | 5 images round-trip; PSNR mean `32.809 dB`; SSIM mean `0.935447` |
| 3 | `.venv\Scripts\python.exe scripts\phase3_diffusion.py` | 6 timing points; `VERIFICATION RESULT: PASS` |
| 3 | `.venv\Scripts\python.exe scripts\phase3_img2img.py` | Strength sweep `0.25/0.5/0.75`; output grid written |
| 6 | `.venv\Scripts\python.exe scripts\phase6_build_testset.py` | 22 images per class per role; prints `split_version` |
| 6 | `.venv\Scripts\python.exe scripts\phase6_eval_selftest.py` | `83 passed, 0 failed` |
| 6 | `.venv\Scripts\python.exe scripts\eval_report.py --system dummy --split evaluation --runner scripts.dummy_runner:run_case --out experiments/results/phase6_dummy_eval.json` | `264` cases, `12` cells, `approximate=False` |

---

## Phase 1: Environment and tensor contract

Command:

```powershell
.venv\Scripts\python.exe scripts\phase1_verify.py
```

Expected output, in order:

```text
[Seed] Set global random seed to 42
Selected device: cuda
GPU name      : NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total VRAM    : 6.00 GB
Seed from config: 42
Loaded image: D:\racs2\data\secret_images\test_001.png
Shape : (1, 3, 512, 512)
Dtype : torch.float32
Min   : 0.035294
Max   : 1.000000
Tensor device after move: cuda:0
Reproducibility test: PASS
Phase 1 verification complete
```

Data processing:

```text
data/secret_images/test_001.png
  -> RGB
  -> torchvision Resize((512, 512))
  -> ToTensor()  (float32, range [0, 1], shape [3, 512, 512])
  -> unsqueeze(0) (shape [1, 3, 512, 512])
  -> .to("cuda")
  -> two seeded torch.randn draws must be bit-identical
```

This phase fixes the project-wide image contract: RGB, float32, `[0, 1]`, batch-first
`[N, C, H, W]`, on CUDA when available.

---

## Phase 2: VAE round-trip and the latent space

Command:

```powershell
.venv\Scripts\python.exe scripts\phase2_vae.py
```

Expected output:

```text
input  shape (1, 3, 512, 512)
latent shape (1, 4, 64, 64)
output shape (1, 3, 512, 512)
PSNR : mean=32.809 dB  (min=31.784, max=34.153)
SSIM : mean=0.935447   (min=0.933472, max=0.936783)
```

Artifacts: `experiments/results/phase2_test_00{1..5}_compare.png`.

Data processing:

```text
secret PNG
  -> RGB tensor [1, 3, 512, 512], range [0, 1]
  -> VAE encoder
  -> latent [1, 4, 64, 64]   (8x spatial downscale, 4 channels)
  -> VAE decoder
  -> reconstructed image [1, 3, 512, 512], range [0, 1]
  -> MSE / PSNR / SSIM against the input
```

Why it matters: every diffusion and EDICT stage later runs in this latent space, so the
VAE round-trip error is a hard floor on reconstruction quality. Phase 2 measures that floor.

---

## Phase 3: Diffusion generation cost

Commands:

```powershell
.venv\Scripts\python.exe scripts\phase3_diffusion.py
.venv\Scripts\python.exe scripts\phase3_img2img.py
```

Expected output from `phase3_diffusion.py`:

```text
steps  seed      sec   s/step   peakGB
   20    42     8.60    0.430     2.60
   20   123     8.64    0.432     2.60
   30    42    12.89    0.430     2.60
   30   123    12.62    0.420     2.60
   50    42    20.52    0.410     2.60
   50   123    21.00    0.420     2.60

VERIFICATION RESULT: PASS
CFG cost: 1.76x  (classifier-free guidance = 2 U-Net passes per step)
```

Numbers are machine-load dependent. Treat `s/step` as a range, not a constant. Artifacts:
`experiments/results/phase3_timings.json` and `phase3_*.png`.

Data processing, text-to-image:

```text
text prompt
  -> CLIP text encoder
  -> random latent noise (seeded)
  -> DDIM denoising loop, 20/30/50 steps
     each step runs the U-Net twice when CFG > 1
  -> VAE decoder
  -> PNG reference image
  -> wall-clock, seconds/step, peak VRAM, output hashes
```

Data processing, image-to-image (`phase3_img2img.py`):

```text
input image + prompt + strength (0.25, 0.5, 0.75)
  -> VAE encoder
  -> add noise according to strength
  -> partial DDIM denoising
  -> VAE decoder
  -> output image
  -> mean absolute difference vs input
```

Why it matters: a 50-step 512x512 reference costs roughly 13-21 s and 2.61 GB peak VRAM
on this machine. That generation cost is the baseline that RACS retrieval later replaces
with a fast similarity lookup over a prebuilt index.

---

## Phase 6: Evaluation harness and frozen splits

Phase 6 has three runnable commands. Run them in this order after any split change.

### 1. Build the splits

```powershell
.venv\Scripts\python.exe scripts\phase6_build_testset.py
```

Use `--rebuild` only when you intentionally want to delete and recreate both split
folders. A plain rerun is idempotent: if the records are unchanged, the manifests are
left untouched.

Expected output:

```text
calibration  common= 22 facial= 22 style= 22
evaluation   common= 22 facial= 22 style= 22
split_version  : a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3
tokens_version : 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
bytes on disk  : 48.9 MB
```

Artifacts:

```text
data/metadata/test_split.json          frozen manifest, role-tagged
data/metadata/tokens.json              one token + one wrong token per image
experiments/results/phase6_build_summary.json
experiments/results/phase6_splits_grid.png
```

### 2. Run the gate

```powershell
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py
```

Expected output ends with:

```text
=== 83 passed, 0 failed ===
```

The selftest is offline and covers metrics, split integrity, role enforcement, tokens,
report shape, canonicalisation and cross-checks. Artifact:
`experiments/results/phase6_selftest.json`.

### 3. Run the dummy evaluation end to end

```powershell
.venv\Scripts\python.exe scripts\eval_report.py --system dummy --split evaluation `
    --runner scripts.dummy_runner:run_case `
    --out experiments/results/phase6_dummy_eval.json
```

Expected output:

```text
=== dummy on evaluation (approximate=False) ===
class     SNR    n     PSNR    SSIM   LPIPS
common      5   22   29.245  0.6933  0.1844
common     10   22   34.164  0.8556  0.0782
common     15   22   39.093  0.9447  0.0241
common     20   22   43.976  0.9808  0.0058
facial      5   22   29.171  0.6376  0.2544
facial     10   22   34.116  0.8297  0.1202
facial     15   22   39.065  0.9351  0.0443
facial     20   22   43.954  0.9775  0.0127
style       5   22   29.154  0.7425  0.1820
style      10   22   34.115  0.8830  0.0789
style      15   22   39.065  0.9559  0.0265
style      20   22   43.960  0.9848  0.0073
cases: 264
wrote: experiments\results\phase6_dummy_eval.json
```

The numbers above come from the dummy runner, which adds mild noise. They prove the
driver and report schema work. They are not a system result and must never be quoted as
a baseline.

### Phase 6 data processing

Sources:

```text
calibration/common  UniStega content_prompts (official release)
calibration/facial  UniStega similar_prompts, filtered to FFHQ/CelebA-HQ
calibration/style   UniStega style_prompts
evaluation/common   COCO val2017, excluded if sha256 exists in the retrieval corpus
evaluation/facial   Wikimedia Commons portraits, title filter + CLIP human-face filter
evaluation/style    Wikimedia Commons cat photographs
```

Canonicalisation applied to every source image:

```text
open image
  -> convert to RGB
  -> resize shortest side to 512 with bicubic
  -> center crop to 512 x 512
  -> save PNG without ancillary chunks
  -> sha256 of the final PNG bytes
```

Manifest construction:

```text
ordered (role, class, image_id, sha256) tuples
  -> split_version = sha256 over the ordered tuples
tokens:
  seed = int(sha256(image_id)[:16], 16)
  token = seed mod 10^8, zero-padded to 8 digits
  wrong_token = deterministic neighbour, asserted different
  -> tokens_version = sha256 over ordered (image_id, token, wrong_token)
```

Invariants enforced by the loader and selftest:

- calibration and evaluation are disjoint by sha256
- both are disjoint from the banked retrieval corpus by sha256
- calibration requires `allow_calibration=True`
- an evaluation request cannot return a calibration record
- every file exists and its sha256 matches the manifest at load time
- calibration reports are always `approximate=True`; evaluation reports are always
  `approximate=False`
- missing metrics are omitted, never zero-filled

Report flow:

```text
runner(record, snr_db, role)
  -> {"secret": image, "recovered": image, latencies, VRAM}
  -> MSE / PSNR / SSIM / LPIPS on [0, 1] tensors
  -> aggregate per (class, snr_db) with distribution()
  -> evaluation/report.py build_report()
  -> atomic JSON write
```

---

## Troubleshooting: scripts do not run

1. Use the venv interpreter explicitly:
   `.venv\Scripts\python.exe scripts\<file>.py`.
   Using `python` may pick the system interpreter, which does not have `torch`,
   `diffusers`, `lpips` or the other project dependencies.
2. Run from `D:\racs2`. The scripts can bootstrap the project root, but relative data
   paths and artifact paths are clearest from the root.
3. Check which interpreter is actually used:
   `.venv\Scripts\python.exe -c "import sys; print(sys.executable)"`.
4. If you see `ModuleNotFoundError`, the wrong interpreter is being used. Re-run with
   the venv path above.
5. If a model script fails on memory, check free commit charge before loading SD or CLIP.
   Close memory-heavy applications, then retry.
6. `phase4_*` and `phase5_*` are banked. Their artifacts are already built
   (`data/retrieval_images/`, `experiments/results/phase5_corpus_embeddings.npz`) and are
   intentionally not part of the Phase 1-6 critical path here.

---

## Current verified state

```text
Phase 1: PASS
Phase 2: PASS
Phase 3: PASS
Phase 6: 83/83 selftest checks PASS
split_version  : a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3
tokens_version : 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
dummy run      : 264 cases, 12 cells, approximate=False
```

Next phase: Phase 7, public conditions (`Kpub` + `Xfeat`).
