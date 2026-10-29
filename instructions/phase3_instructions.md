# Phase 3 — Diffusion Models & Generation Cost

> **Goal**: Run Stable Diffusion to generate reference images, and **measure** how
> expensive that generation is (wall-clock time + peak VRAM) on this machine.
> **Why**: You cannot argue RACS is cheaper until you have measured the cost it
> replaces. This phase produces the baseline number that retrieval must beat.

---

## Step 0: Pre-flight checks (do these first)

Phase 3 downloads a full model (~2.5 GB fp16). Confirm there is room and that
HuggingFace is reachable before starting:

```powershell
cd d:\racs2
# disk free on D:
Get-PSDrive D | Select-Object Used,Free

# HF reachable? (should return HTTP 200, not a timeout)
try { (Invoke-WebRequest -UseBasicParsing https://huggingface.co -TimeoutSec 15).StatusCode } catch { "UNREACHABLE: $($_.Exception.Message)" }
```

> **Blocker note (history)**: an earlier attempt failed because `huggingface.co`
> was unreachable. We already downloaded the VAE from HF in Phase 2, so it is
> reachable now. If it ever fails again, retry later; the script needs no changes.

---

## Step 1: Dependencies

`diffusers` + `accelerate` are already installed (Phase 2). Nothing new is
required for the base pipeline. Confirm:

```powershell
.venv\Scripts\python.exe -c "import diffusers, accelerate; print(diffusers.__version__, accelerate.__version__)"
```

---

## Step 2: Update `configs/base.yaml`

Add a `generation` section so the sweep parameters live in one place instead of
being hard-coded. Keep the existing keys:

```yaml
seed: 42
image_size: 512
device: "cuda"
dtype: "float16"

generation:
  prompt: "a photograph of an astronaut riding a horse"   # TODO: replace with the paper's example Kpub
  num_inference_steps: 30
  guidance_scale: 7.5
  scheduler: "ddim"
  # sweep values used by the timing experiment:
  steps_sweep: [20, 30, 50]
  seeds_sweep: [42, 123]
```

> The plan refers to "the paper's example prompt". If you know the exact public
> key used in the paper, put it here. Otherwise the placeholder above is a
> standard SD example and the mechanics are identical.

---

## Step 3: Create the Phase 3 Structure

```
d:\racs2\
├── models/
│   └── sd_pipeline.py          # loads StableDiffusionPipeline (SD v1.5)
├── scripts/
│   └── phase3_diffusion.py     # generation + timing experiment
└── experiments/
    └── results/
        ├── phase3_timings.json         # measured results
        └── phase3_<prompt>_seed<N>.png # generated reference images
```

```powershell
New-Item -ItemType Directory -Force -Path "d:\racs2\experiments\results"
```

---

## Step 4: Write `models/sd_pipeline.py`

Loads Stable Diffusion v1.5 configured for a **6 GB** GPU.

**Key requirements (all matter on this hardware):**
- `StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None, low_cpu_mem_usage=True)`
  - `low_cpu_mem_usage=True` avoids an fp32 state-dict blowup on your tight host RAM.
  - `safety_checker=None` keeps the run deterministic and skips an extra model.
- Use the **DDIM** scheduler (`DDIMScheduler.from_config(...)`) — deterministic, fewer steps.
- `pipe.enable_attention_slicing()` and `pipe.enable_vae_slicing()` to fit VRAM.
- Move to GPU, `pipe.set_progress_bar_config(disable=True)` for clean timing.
- If a repo id 404s, try these alternatives in order:
  `stable-diffusion-v1-5/stable-diffusion-v1-5`, `runwayml/stable-diffusion-v1-5`.
  (The original `CompVis/...` and `runwayml/...` repos have moved; the first id is
  the current canonical SD v1.5 location.)

**Functions:**
- `load_pipeline(device=None, dtype=torch.float16, model_id=...)` → the pipeline.
- `generate(pipe, prompt, steps, guidance_scale, seed)` → `(PIL.Image, elapsed_seconds)`.
  Seed with a fresh `torch.Generator` per call so seeds are independent and reproducible.

**It should work like this:**
```python
from models.sd_pipeline import load_pipeline, generate
pipe = load_pipeline()
image, secs = generate(pipe, "a photograph of an astronaut riding a horse", 30, 7.5, 42)
```

---

## Step 5: Write `scripts/phase3_diffusion.py`

The measurement script. It should do these things **in order**:

```
1. import common.paths (HF cache on D:); set_seed(42, verbose=True)
2. device = get_device(); print it
3. pipe = load_pipeline(device)
4. Load configs/base.yaml (prompt, steps, guidance, sweeps)
5. WARM-UP: run one generation and DISCARD it
   (first call includes CUDA/cuDNN kernels compilation; timing it would be wrong)
6. For each (steps in steps_sweep) x (seed in seeds_sweep):
   a. torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
   b. t0 = time.perf_counter()
   c. image = generate(pipe, prompt, steps, guidance, seed)
   d. torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
   e. peak_vram = torch.cuda.max_memory_allocated() / 1024**3   (GB)
   f. print: steps, seed, total sec, sec/step (= total/steps), peak VRAM
   g. save the image: experiments/results/phase3_seed<seed>_steps<steps>.png
   h. append a record dict
7. Save ALL records to experiments/results/phase3_timings.json
8. Print a SUMMARY table and, if it was measured, the cost equation:
   e.g. "50 steps x 0.7 s/step = 35 s per generated reference"
```

**Record schema** (for the JSON):
```json
{
  "prompt": "...",
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "seed": 42,
  "seconds": 21.4,
  "seconds_per_step": 0.713,
  "peak_vram_gb": 3.21,
  "image_path": "experiments/results/phase3_seed42_steps30.png"
}
```

**VRAM / timing tips:**
- Use `torch.cuda.max_memory_allocated()` (peak) not `memory_allocated()` (current)
  — the peak is what tells you whether 6 GB is enough.
- Always `torch.cuda.synchronize()` around timing, or you will measure only the
  kernel-launch time, not the actual compute.
- Keep `guidance_scale` fixed for the main sweep; add one extra run at guidance
  `1.0` (no CFG) and `7.5` to see the guidance cost difference.

---

## Step 6: Run and Check

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase3_diffusion.py
```

The first run downloads SD v1.5 (~2.5 GB) into the D: HF cache, then runs a
warm-up plus the sweep. Expect several minutes total.

### ✅ Checklist (all must pass):

```
[ ] SD v1.5 pipeline loads fp16 on the 6 GB GPU without OOM
[ ] DDIM scheduler active, attention slicing enabled
[ ] Cold start (download + warm-up) completes
[ ] Wall-clock time + peak VRAM measured for >= 2 step counts and >= 2 seeds
[ ] sec/step is roughly constant across step counts (validates the cost model)
[ ] Reference image saved for the example prompt
[ ] experiments/results/phase3_timings.json written and well-formed
[ ] Peak VRAM stays under ~6 GB
```

**When all checks pass, Phase 3 is done.** Paste the output and the JSON and we
move to Phase 4 (Building the Retrieval Image Database).

---

## The Gate Question (answer in your own words)

> *Why is diffusion generation expensive?*

Be ready to state it with **your** numbers, e.g.:

> "Generation runs the U-Net `N` times, once per denoising step. At 30 steps I
> measured `X` s (≈ `X/30` s per step) and `Y` GB peak VRAM. So one reference
> image costs ≈ `X` seconds, versus a retrieval query which should cost a fraction
> of a second. That gap is the efficiency argument for RACS."

Also be able to explain:
- Forward vs reverse process, and why the reverse process needs many steps.
- Why DDIM needs fewer steps than DDPM and is deterministic.
- What `guidance_scale` does (classifier-free guidance: prompt adherence vs diversity, and it doubles U-Net passes per step).
- Why denoising happens in latent space (Phase 2), not pixels.
