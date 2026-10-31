# Phase 8: Reference Generator (PicX-Real / SD + ControlNet + IPAdapter-plus)

> **Goal**: Build the deterministic reference generator that turns the public conditions from
> Phase 7 into a reference image `Xref`. The noise is seeded from the digital token, the prompt
> comes from `Kpub`, and the structure comes from `Xfeat`.
> **Why**: This is the one stage RACS replaces. It has to exist, be reproducible bit-for-bit,
> and fit the 6 GB GPU before any baseline or comparison work can start.

> **Ordering rule**: Phase 8 consumes the Phase 7 cache read-only. It must not train anything,
> must not use retrieval, and must not touch EDICT. Phase 9 may fine-tune the ControlNet branch;
> Phase 10 is independent.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2

# Phase 7 must be complete and verified
.venv\Scripts\python.exe scripts\phase7_verify.py

# Phase 6 loader must still work
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py

# GPU, disk, network
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3, 'GB')"
.venv\Scripts\python.exe -c "import shutil; print(round(shutil.disk_usage('D:/').free/1024**3, 1), 'GB free')"
.venv\Scripts\python.exe -c "import requests; print('HF', requests.get('https://huggingface.co', timeout=20).status_code)"
```

Expected: `phase7_verify.json` has `"pass": true`, 132 cache records, and the Phase 6
selftest still reports 83/83.

If Phase 7 fails, stop and repair it. Do not generate references from a partial condition
cache.

---

## Step 1: Dependencies and weights

Already present: `torch`, `transformers`, `diffusers`, `accelerate`, `Pillow`, `numpy`,
`PyYAML`, `requests`.

New for the preferred stack:

```powershell
.venv\Scripts\python.exe -m pip install --no-cache-dir diffusers controlnet-aux
```

IPAdapter-plus and the segmentation ControlNet are loaded through `diffusers`; use the
resumable downloader already built in Phase 5. Do not rely on a moving `main` revision: pin a
commit or a local directory and record the resolved path.

```powershell
.venv\Scripts\python.exe scripts\phase5_fetch_model.py --help
```

Rules:

- All weights are pretrained and frozen in this phase. No training, no LoRA, no adapters.
- PicX-Real is preferred if a usable checkpoint exists on this network. Otherwise use
  SD v1.5 + a segmentation-conditioned ControlNet + IPAdapter-plus. Record the choice in
  `docs/DEVIATIONS.md` with the measured visual and metric impact.
- Keep `common/hf_lowmem.py` installed before importing `diffusers`/`transformers`.

---

## Step 2: Update `configs/base.yaml`

Add a `reference:` block. Keep every value explicit so the manifest can hash it.

```yaml
reference:
  cache_dir: "experiments/results/references"
  manifest_path: "experiments/results/phase8_references.json"
  backend: "sd_controlnet_ipadapter"   # picx_real | sd_controlnet_ipadapter
  model_id: "runwayml/stable-diffusion-v1-5"
  controlnet_id: "lllyasviel/control_v11p_sd15_seg"
  facial_controlnet_id: "lllyasviel/control_v11p_sd15_openpose"
  ipadapter_id: "h94/IP-Adapter"
  ipadapter_subfolder: "models"
  ipadapter_weight: "ip-adapter-plus_sd15.bin"
  device: "cuda"
  dtype: "float16"
  scheduler: "ddim"
  steps: 50
  guidance_scale: 7.5
  eta: 0.0
  output_size: 512
  seed_scheme: "sha256_token_v1"
  max_peak_vram_gb: 5.5
```

Protocol decisions:

- `steps`, `guidance_scale`, `eta`, scheduler, model ids and seed scheme are frozen once the
  baseline is generated. Bumping any of them invalidates every downstream baseline artifact.
- `eta` must stay `0.0`. DDIM with `eta > 0` injects noise and breaks the determinism
  contract.
- The prompt is exactly the stored `kpub`; do not re-template it here.

---

## Step 3: Create the Phase 8 structure

```text
reference/
  __init__.py
  seed.py                 # canonical seed derivation from the token
  generate.py             # pipeline loading and reference generation
  conditions.py           # convert Xfeat to ControlNet / IPAdapter inputs
scripts/
  phase8_reference.py     # CLI: build references, determinism check, grid
  phase8_verify.py        # offline hash + manifest gate
experiments/results/
  references/
    <image_id>__<token>.png
  phase8_references.json
  phase8_reference_grid.png
  phase8_verify.json
```

Cache arrays and reference PNGs stay out of git. Commit the manifest, the verifier output and
the grid.

---

## Step 4: Implement `reference/seed.py`

This contract is used again by Phase 11 for the token-perturbation mask. Implement it once.

```python
import hashlib

SEED_DOMAIN = "racs2-reference-v1"

def derive_seed(token: str) -> int:
    """Deterministic 31-bit seed from an 8-digit token."""
    digest = hashlib.sha256(f"{SEED_DOMAIN}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF

def derive_mask_seed(token: str) -> int:
    """Separate domain so the noise seed and the mask seed never coincide."""
    digest = hashlib.sha256(f"racs2-mask-v1:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
```

Rules:

- No Python `hash()`, no `random`, no time, no PID, no path in the derivation.
- The token is a string of exactly eight decimal digits, as enforced by
  `evaluation.splits.load_tokens`.
- The same token must give the same seed on Windows and Linux.

---

## Step 5: Implement `reference/generate.py`

### API

```python
def build_reference_pipeline(config: dict): ...

def generate_reference(
    *,
    image: PIL.Image | None,
    kpub: str,
    feature: np.ndarray | None,
    feature_kind: str,
    class_name: str,
    token: str,
    config: dict,
) -> dict:
    """Returns {"image": PIL.Image, "seed": int, "latents_sha256": str,
                "peak_vram_gb": float, "backend": str, "model_id": str}."""
```

### Generation order

```text
kpub + token + class
  -> seed = derive_seed(token)
  -> torch.Generator(device="cuda").manual_seed(seed)
  -> z_T ~ N(0, I) with that generator
  -> encode kpub with CLIP text encoder (frozen)
  -> build control image from Xfeat:
       segmentation -> ADE20K colour map (Phase 7 preview)
       skeleton     -> skeleton preview (Phase 7 preview)
       none         -> no ControlNet call at all
  -> route to controlnet_id for common and facial_controlnet_id for facial,
     or use a multi-ControlNet union if only one branch is loaded; record the choice
  -> IPAdapter image prompt: the source image if the backend requires one;
     for the paper-faithful path `image` is None and IPAdapter is skipped
  -> DDIM(z_T, prompt=kpub, control=control, steps=50, guidance=7.5, eta=0)
  -> VAE.decode / 0.18215
  -> Xref [512, 512, 3] uint8
```

Important: the reference is generated from **public conditions**, not from the secret. The
source image may only be used for the IPAdapter image prompt if you deliberately choose that
ablation; record it in `docs/DEVIATIONS.md`. The default paper path does not pass the secret to
the generator.

### Determinism requirements

- `pipeline.eval()`, text encoder frozen, `torch.inference_mode()`.
- `scheduler.set_timesteps(steps)` before the call, `eta=0.0`.
- Fresh generator created per call from the token seed; never rely on global RNG.
- No random dropout, no random augmentation, no `torch.backends.cudnn.benchmark` dependence.
- Pass the same `latents` tensor derived from the generator, not `generator=` plus a pipeline
  internal `randn`, unless the diffusers version is pinned and tested.
- Compute `latents_sha256` over the float32 canonical bytes for the determinism check.

### 6 GB budget

```python
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()   # or model_cpu_offload if it fits
pipe = pipe.to("cuda")                  # only after offload setup
```

Measure and record `torch.cuda.max_memory_allocated() / 1024**3`. If peak exceeds the configured
budget, try `enable_model_cpu_offload()` first, then reduce the ControlNet resolution, and
record the change as a deviation. Do not silently switch to CPU.

---

## Step 6: Implement `scripts/phase8_reference.py`

### CLI

```powershell
python scripts\phase8_reference.py
python scripts\phase8_reference.py --split evaluation --class facial --limit 2
python scripts\phase8_reference.py --token-mode wrong --limit 2
python scripts\phase8_reference.py --force
```

### Algorithm

```text
1. set_seed(42, verbose=True)
2. load configs/base.yaml -> reference block
3. load the Phase 7 manifest and cache through semantic.cache
4. load the split through evaluation.splits.Split with role enforcement
5. load the pipeline once, reuse across images
6. for each record in (role, class, image_id) order:
     a. read cached kpub and feature
     b. derive the seed from the correct or wrong token
     c. generate Xref
     d. write PNG atomically
     e. append a manifest record with hashes, seed, VRAM, latency, backend, model ids
7. write the manifest atomically
8. write a 3-class grid
9. print counts, peak VRAM, mean latency, and failures
```

### Manifest schema

```json
{
  "schema_version": 1,
  "created_utc": "2026-10-27T00:00:00Z",
  "split_version": "a57d9c8c...",
  "conditions_manifest_sha256": "...",
  "config_hash": "...",
  "backend": "sd_controlnet_ipadapter",
  "models": {"base": "...", "controlnet": "...", "ipadapter": "...", "vae": "..."},
  "records": [
    {
      "image_id": "eva_facial_0000",
      "role": "evaluation",
      "class": "facial",
      "token_mode": "correct",
      "seed": 123456789,
      "kpub_sha256": "...",
      "feature_sha256": "...",
      "reference_path": "experiments/results/references/eva_facial_0000__12345678.png",
      "reference_sha256": "...",
      "latents_sha256": "...",
      "peak_vram_gb": 4.8,
      "latency_s": 21.3,
      "status": "ok"
    }
  ]
}
```

Sort records by `(role, class, image_id, token_mode)`. Write with `.tmp` then `os.replace`.

---

## Step 7: Implement `scripts/phase8_verify.py`

Offline checks, writing `experiments/results/phase8_verify.json`:

```text
1. MANIFEST
   a. schema_version == 1 and config_hash recomputes
   b. split_version matches data/metadata/test_split.json
   c. every split record has one reference per requested token mode
   d. reference files exist and their sha256 matches the manifest

2. SEED
   a. derive_seed(token) matches the recorded seed for every record
   b. the same token appears once per image, and wrong tokens differ

3. DETERMINISM
   a. regenerate a fixed sample (>= 2 per class) twice
   b. reference_sha256 and latents_sha256 are exactly equal
   c. report max|diff| of the decoded arrays, expected exactly 0

4. BUDGET
   a. peak_vram_gb <= reference.max_peak_vram_gb
   b. latency is recorded and finite

5. VISUAL
   a. the grid exists
   b. each row is coherent for its class (segmentation for common, skeleton for facial,
      prompt-only for style)
```

Exit non-zero on any failure.

---

## Step 8: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase8_reference.py
.venv\Scripts\python.exe scripts\phase8_verify.py
```

Expected summary:

```text
calibration  common=22 facial=22 style=22
evaluation   common=22 facial=22 style=22
references: 132
failed: 0
peak VRAM: <= 5.5 GB
determinism: PASS
verify: PASS
```

Open `experiments/results/phase8_reference_grid.png` and confirm:

- facial references look like plausible portraits aligned with the source skeleton,
- common references follow the segmentation layout,
- style references follow `Kpub` with no structural control,
- no row is blank, duplicated, or from the wrong class.

---

## Checklist

```text
[ ] Phase 7 verify still passes and its manifest is unchanged
[ ] reference/seed.py derives the same seed from the same token across two processes
[ ] reference/generate.py generates from Kpub + Xfeat, not from the secret payload
[ ] DDIM eta = 0.0 and the scheduler/steps/guidance are frozen in config
[ ] two runs of the same image and token give bit-identical references
[ ] peak VRAM is measured and inside the 6 GB budget
[ ] references exist for every split record, with hashes in the manifest
[ ] phase8_verify.json has pass=true
[ ] the reference grid was inspected and is coherent per class
[ ] PicX-Real or the substitute is recorded in docs/DEVIATIONS.md
[ ] no training, no retrieval, no EDICT in this phase
```

**When these pass, Phase 8 is done.** Phase 9 may fine-tune the ControlNet branch. Phase 10 can
run in parallel because it does not import the reference generator.

---

## The Gate Question (answer in your own words)

> *Why must the reference noise be derived from the token rather than from a random seed
> stored in the message, and why is `eta = 0` necessary?*

Answer with your artifacts:

- The token is the only shared secret. Deriving the seed from it means the receiver can
  regenerate the exact same noise without transmitting a seed, so the reference costs zero
  payload bits.
- A transmitted seed would be a second secret and would leak or consume capacity. A wall-clock
  or global RNG seed would make Tx and Rx diverge.
- `eta > 0` injects fresh stochastic noise at every DDIM step. Even with the same starting
  latent, the output would differ between runs and EDICT's inverse would not recover the
  payload.
- The manifest is the evidence: same split, same conditions, same token, same seed, same
  reference hash.

---

## Report back

- Backend used (PicX-Real or SD + ControlNet + IPAdapter-plus) and why
- Exact model ids and weight hashes
- Seed scheme and one worked example (`token -> seed`)
- Record counts, peak VRAM, mean latency per class
- Determinism result (reference and latent hashes, max|diff|)
- Grid path and what was visually confirmed
- Any deviation appended to `docs/DEVIATIONS.md`
- Your answer to the gate question

---

## Notes and risks

- **Determinism is the product.** A visually good reference that changes between runs is
  worthless for this system.
- **Do not condition on the secret.** The generator is public-conditioned. If you pass the
  secret image to IPAdapter, you have changed the threat model and must say so.
- **`enable_sequential_cpu_offload` is slow but safe.** Measure both offload modes; record
  which one produced the reported latency.
- **Scheduler state is mutable.** Always call `set_timesteps` per generation, and never reuse a
  scheduler whose internal counters were advanced by a previous call.
- **A model change invalidates everything downstream.** Bump `config_hash`, regenerate all
  references, and re-run Phase 15.
