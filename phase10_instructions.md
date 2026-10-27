# Phase 10: EDICT as an Isolated Mini-Project

> **Goal**: Implement Exact Diffusion Inversion via Coupled Transformations and prove that
> `latent -> forward -> inverse -> latent` closes to float32 machine precision.
> **Why**: EDICT is the reversible hiding and recovery mechanism. It is the highest-risk
> unknown in the project, so it is proven alone before retrieval, reranking or comparison.

> **Caution**: Do NOT integrate retrieval, Kpub, Xfeat, AWGN or the semantic codec here. This
> phase is EDICT alone.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase2_vae.py --help
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: the Phase 2 VAE module loads and can encode/decode a 512x512 image.

---

## Step 1: Config

Add an `edict:` block.

```yaml
edict:
  scheduler: "ddim"
  steps: 50
  mixing: 0.93
  eta: 0.0
  dtype: "float32"          # inversion precision is float32 even if the VAE is fp16
  sweep_steps: [10, 20, 50]
  sweep_mixing: [0.5, 0.7, 0.93, 0.99]
  num_random_latents: 20
  num_image_latents: 20
  results_path: "experiments/results/phase10_edict.json"
```

Protocol decisions:

- `steps=50`, `mixing=0.93` are the paper's operating point unless the sweep says otherwise.
- The mixing coefficient is applied symmetrically in forward and inverse; any asymmetry breaks
  exactness.
- EDICT operates on the latent, not on decoded pixels.

---

## Step 2: Structure

```text
steganography/
  __init__.py
  edict.py              # EDICT forward and inverse
scripts/
  phase10_edict.py      # sweep, round-trip tests, report
  phase10_verify.py     # offline gate
experiments/results/
  phase10_edict.json
  phase10_edict_sweep.png
  phase10_verify.json
```

---

## Step 3: Implement `steganography/edict.py`

### API

```python
def edict_forward(
    z: torch.Tensor,
    *,
    prompt_embeds: torch.Tensor,
    unet,
    scheduler,
    steps: int = 50,
    mixing: float = 0.93,
) -> torch.Tensor: ...

def edict_inverse(
    z: torch.Tensor,
    *,
    prompt_embeds: torch.Tensor,
    unet,
    scheduler,
    steps: int = 50,
    mixing: float = 0.93,
) -> torch.Tensor: ...
```

Maintain two trajectories `x` and `y` initialised from the same latent (the paper uses one
latent plus a copy). At each step:

1. replace the current noisy latent with the affine coupling of `x` and `y` using `mixing`,
2. run one scheduler step / UNet noise prediction,
3. interleave the update back into the opposite trajectory,
4. swap the roles on the next step.

The inverse runs the same sequence in reverse order with the coupling inverted. Both functions
must share one step implementation so the operations cannot drift apart.

### Exactness requirements

- `torch.float32`, `torch.inference_mode()` for the test harness, no AMP inside EDICT.
- No in-place operation that changes a tensor used later by the other trajectory.
- Run scheduler steps in exact reverse order in the inverse.
- The UNet must be called with the same prompt embeddings, timestep and conditioning in both
  directions.
- No dropout, no randomness, no `eta > 0`.

---

## Step 4: Implement `scripts/phase10_edict.py`

### Tests

```text
A. 20 random latents N(0, I) of shape [1, 4, 64, 64]
B. 20 real image latents: Phase 6 calibration or evaluation images -> VAE.encode
   (use the same fixed sample for every sweep point)
C. for steps in {10, 20, 50} and mixing in {0.5, 0.7, 0.93, 0.99}:
     error = max|inverse(forward(z)) - z|
     rel   = error / max|z|
     record both, plus the VAE-decode error after the round-trip
```

Report per-cell aggregates, not a single best number. The grid is
`phase10_edict_sweep.png`: x = mixing, y = steps, cell = log10 max absolute error.

### Output schema

```json
{
  "schema_version": 1,
  "config_hash": "...",
  "operating_point": {"steps": 50, "mixing": 0.93},
  "cells": [
    {"kind": "random", "steps": 50, "mixing": 0.93,
     "max_abs_error": 1.2e-7, "max_rel_error": 3.1e-8,
     "n": 20, "finite": true}
  ],
  "decision": "50 steps with 0.93 adopted because ...",
  "seed": 42
}
```

---

## Step 5: Implement `scripts/phase10_verify.py`

Checks:

```text
[ ] operating point matches the config
[ ] every cell has finite errors and n == 20
[ ] float32 relative error at the operating point <= 1e-6
[ ] error does not increase when steps increase from 20 to 50
[ ] re-running the operating point reproduces the same error to within 1e-9
[ ] no retrieval, AWGN, codec or reference-generator import appears in the EDICT module
[ ] the sweep grid exists
```

Exit non-zero on failure.

---

## Step 6: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase10_edict.py
.venv\Scripts\python.exe scripts\phase10_verify.py
```

Expected operating point: 50 steps, mixing 0.93, relative error at or below `1e-6` for float32
latents, and a monotone improvement from 10 to 50 steps. If 50 steps is not the best cell, adopt
the measured best and justify it in `phase10_edict.json`; do not blindly copy the paper.

---

## Checklist

```text
[ ] steganography/edict.py has one shared step implementation
[ ] forward and inverse are exact reverses
[ ] 20 random + 20 real latents round-tripped
[ ] steps and mixing swept as configured
[ ] operating point chosen from measured data
[ ] relative error <= 1e-6 at the operating point
[ ] phase10_verify.json has pass=true
[ ] no integration with retrieval, codec, channel or reference generator
```

**When this passes, Phase 10 is done.** Phase 11 may consume EDICT; nothing in EDICT may consume
Phase 11.

---

## The Gate Question (answer in your own words)

> *Why can EDICT invert exactly when plain DDIM inversion drifts, and why does that exactness
> matter for steganography?*

Answer with your artifacts:

- DDIM inversion approximates the reverse trajectory and accumulates truncation error at every
  step, so the reconstructed latent is close but not equal.
- EDICT couples two latent trajectories with an invertible affine transform. Every operation in
  the forward pass has an exact algebraic inverse, so the composition is invertible up to
  floating-point rounding, not up to model error.
- Hiding requires that the receiver recover the same latent the sender embedded. Any drift
  becomes a secret-recovery error, and it compounds through the VAE decoder. Machine-precision
  inversion puts the error floor at the VAE and channel, where it is measured, not at the
  inversion algorithm.
- The sweep and the `max_rel_error` table are the evidence.

---

## Report back

- EDICT implementation shape and the shared-step design
- Sweep table (steps x mixing, random and image latents)
- Operating point and the measured max absolute/relative error
- Decode error after the latent round-trip
- What surprised you or remains unclear
- Your answer to the gate question

---

## Notes and risks

- **Float32 is not optional for the exactness claim.** fp16 will hit a different, much larger
  floor and must be reported as such.
- **Do not hide a real secret yet.** This phase proves the mechanism on latents.
- **Two trajectories, one scheduler.** Reusing a scheduler whose counters moved causes silent
  drift. Reset the timesteps for every forward and inverse call.
- **If exactness fails, stop.** Do not proceed to Phase 11 with a lossy inverse and call the
  loss "channel noise".
