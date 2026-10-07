# Phase 2 brief — Image Tensors, VAE and Latent Space

> Template from `docs/RACS_ROADMAP.md` §5. Mentor deliverable: theory, tasks,
> and the acceptance gate. Code authorship is pending the user's call on
> working agreement (see `project_status.md`).

## Goal

Understand image tensors and latent space, and run + measure a VAE round trip
(encode -> latent -> decode -> image).

## Why RACS needs it

EDICT and the diffusion stages operate in **latent space**, not pixel space.
Before touching EDICT (Phase 10) you must be fluent in: what a tensor is
(`B×C×H×W`), what a VAE encoder/decoder do, and how much information the
reconstruction loses. Stable Diffusion's VAE is the exact component RACS will
re-use, so its compression factor and reconstruction error are part of the
system's error floor.

## Theory to learn

- `B × C × H × W`: batch, channels, height, width. Understand indexing/slicing.
- Pixel space vs latent space; the VAE spatial compression factor (SD VAE:
  `512×512×3` -> `64×64×4`, i.e. 8× downscale, 4 latent channels).
- Encoder `E(·)`, decoder `D(·)`, and what "reconstruction" means.
- Normalization (pixel range `[-1, 1]` for SD VAE) and dtype (fp16 on 6 GB VRAM).
- Metrics: MSE, PSNR (`10·log10(1/MSE)` for unit-range images), SSIM.
- Why we measure round-trip error: it is a floor for everything downstream.

Answer for yourself: *Why does Stable Diffusion work in latent space instead of
directly in pixel space?* (Hint: cost scales with spatial area; the VAE trades a
small reconstruction error for a large compute reduction.)

## Build tasks

1. Add missing dependency for this phase: `diffusers` (+ `accelerate` likely).
   Diffusers is needed here (VAE) and again in Phase 3 (SD pipeline).
2. Load a VAE checkpoint. Prefer the SD v1.5 VAE (`AutoencoderKL`,
   `stabilityai/sd-vae-ft-mse` or the SD v1.5 VAE) — same checkpoint the
   baseline paper uses. Keep it on GPU as fp16.
3. Take one image (any real image; resize to `512×512`), normalize to `[-1,1]`.
4. Run `image -> VAE encode -> latent -> VAE decode -> reconstructed image`.
   Print every tensor shape (input, latent, output).
5. Measure round-trip **MSE / PSNR / SSIM** between original and reconstruction.
6. Repeat on a handful of images (e.g. 5) and report the spread, not one value.
7. Save a side-by-side image to `experiments/results/`.
8. Seed with `common.seed.set_seed` so the run is reproducible.

Suggested files: `steganography/vae/roundtrip.py`, shared metrics later in
`evaluation/{mse,psnr,ssim}.py`.

## Acceptance gate

- VAE round-trip runs on GPU and is **measured** (MSE/PSNR/SSIM reported for >1 image).
- Tensor shapes are printed and understood (`512×512×3` -> `64×64×4` -> `512×512×3`).
- You can explain: latent vs pixel space, the compression factor, why the
  reconstruction is not identical, and why SD works in latent space.

## Report back

- What I built / files added
- Command(s) I ran
- Observed numbers (MSE/PSNR/SSIM per image)
- What surprised me / what is unclear
- My answer to the "explain" question

## Notes / risks

- **Dependency:** install `diffusers` (+ `accelerate`). Put the HF cache on D:
  (already handled by `common/paths.py`). Watch C: free space.
- **VRAM:** a single SD VAE round trip at 512×512 in fp16 is small (<1 GiB);
  fine on the 6 GB 3050.
- **Determinism:** the VAE is not the random part — EDICT/diffusion later are.
  Still seed everything for reproducibility.
