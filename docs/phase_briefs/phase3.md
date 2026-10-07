# Phase 3 brief — Diffusion Fundamentals + Generation Cost

> Template from `docs/RACS_ROADMAP.md` §5.
> Status: **BLOCKED** on Hugging Face reachability (see `project_status.md`).
> The runnable script is already in place: `scripts/phase3_diffusion.py`.

## Goal

Understand diffusion well enough to explain *why* reference generation is
expensive, with your own measured numbers. Then quantify the cost that RACS
replaces.

## Why RACS needs it

`AgentSemSteCom` generates `Xref` by running a diffusion/DDIM sampler. This is
the single most expensive stage and the one RACS swaps for retrieval. You cannot
argue RACS is cheaper until you have measured the generation cost on this exact
machine. Phase 3 produces that baseline number.

## Theory to learn

- Forward process: clean image -> add noise -> pure noise over `T` steps.
- Reverse process: noise -> denoise -> ... -> image (repeated model evaluations).
- Timesteps, noise schedule, U-Net backbone, conditioning.
- Latent diffusion: denoising happens in the VAE latent space (Phase 2), not pixels.
- DDIM vs DDPM: deterministic fewer-step sampling.
- Classifier-free guidance: how `guidance_scale` trades fidelity for prompt adherence.
- Cost model: generation time ≈ (num_steps) x (per-step U-Net forward cost).

## Build tasks

1. Run: `.venv\Scripts\python.exe scripts\phase3_diffusion.py --steps 20`.
   The script loads SD v1.5 (`StableDiffusionPipeline`, fp16, DDIM, attention
   slicing) and generates the paper's example prompt.
2. Exercise: seed, prompt, number of steps, guidance.
3. Measure: wall-clock generation time and peak VRAM per setting.
4. Record: `experiments/results/phase3_timings.json` + sample images.

## Acceptance gate

- You can explain why generation is expensive using your own measured numbers
  (seconds per step x steps).
- Observed generation time + peak VRAM for at least 2 step counts and 2 seeds.
- A reference image is saved for the paper's example public key.

## Report back

- What I built / files added
- Command(s) I ran
- Observed numbers (time, VRAM per setting)
- What surprised me / what is unclear
- My answer to the "explain" question

## Blocker (2026-10-06)

`huggingface.co` is currently unreachable from this machine
(`Invoke-WebRequest` -> timeout; `pypi.org` -> 200). Only config/tokenizer files
of SD v1.5 are cached; the ~2 GiB fp16 weight files are not. Retry when HF is
reachable; the script needs no changes.

## Notes / risks

- **Host RAM is tight** (~3.4 GiB free of 15.3 GiB). The loader uses
  `low_cpu_mem_usage=True` (accelerate meta-device sharding) to avoid an fp32
  state-dict blowup. An earlier attempt failed with a 3.6 MB allocation error
  before this fix, while the Rust downloader was still running.
- **VRAM:** SD v1.5 fp16 (~2.5 GiB) + attention slicing fits in 6 GiB.
- Use `safety_checker=None` so the run is deterministic and avoids an extra model.
