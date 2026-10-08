# RACS2 — Session Handoff

> **Last updated**: 2026-10-08 (evening)
> **Repo**: `D:\racs2` · branch `main` · HEAD `174204f`
> **Purpose**: one-page state transfer so a fresh agent/session can continue without re-reading the whole repo.
> **Companion doc**: `project_status.md` (long-lived status + change log). Read it for full history.

---

## 1. What this project is

**RACS2** (Retrieval-Augmented Coverless Steganography v2) is an Agentic-AI system for
secure, coverless, semantic steganography over wireless channels. Instead of *generating*
a reference image with diffusion, it **retrieves** a semantically-matching image from a
shared database (`Xref_tx == Xref_rx`), then uses EDICT on SD v1.5 to embed the secret
payload bit-exactly in latent space. The efficiency claim is retrieval (sub-second) vs
diffusion (tens of seconds per reference).

---

## 2. Current status

| Phase | Scope | Status |
| :-- | :-- | :-- |
| 1 | Environment & skeleton (`configs/base.yaml`, seed, device) | ✅ Complete |
| 2 | VAE + latent space round-trip | ✅ Complete |
| 3 | Diffusion & generation cost (SD v1.5 baseline) | ✅ Complete |
| 4 | **Build the retrieval image database** | ⏳ **Next** |
| 5 | CLIP embeddings | pending |
| 6 | FAISS retrieval | pending |

Full checklists and gate answers live in `project_status.md` §2 and the phase briefs in
`docs/phase_briefs/`.

### Headline numbers (quiet-machine run, authoritative)
- VAE round-trip: PSNR **32.809 dB** (31.78–34.15), SSIM **0.9354** on 5 images.
- SD v1.5, 512×512, fp16: **~0.26 s/step**, 50 steps = **12.98 s**, peak VRAM **2.60 GB**.
- CFG doubles U-Net passes: **0.262 s/step (on) vs 0.150 (off) = 1.75x**.
- img2img on our own image: `mean|diff|` grows 10.5 → 19.9 → 38.1 at strength 0.25/0.5/0.75.
- Gate answer: one diffusion reference costs ~13–21 s vs a retrieval query ≈ a fraction
  of a second; that gap is the RACS efficiency argument.

---

## 3. Environment (this machine) — read before running anything

- Windows, **RTX 3050 Laptop 6 GB**, 16 GB RAM. venv at `.venv` (PyTorch 2.6.0+cu124).
- HF/diffusers/torch cache and pip temp are redirected to **`D:\racs2\cache`** and
  `D:\racs2\temp` (C: is full) — see `common/paths.py`.
- **Commit-charge trap**: safetensors mmaps used to blow the Windows paging file
  (`OSError 1455`). Fixed by `common/hf_lowmem.py` (read-only mmap shim) which must be
  imported **before** `diffusers`/`transformers`. Do not remove that import.
- Loading a diffusion model still needs ~2 GB free *commit*. Check:
  ```powershell
  Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
  ```
  An IDE language server can silently hold 10 GB; restart it if headroom is low.

---

## 4. How things run

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase1_verify.py        # 7/7 checks
.venv\Scripts\python.exe scripts\phase2_vae.py           # VAE round-trip + metrics
.venv\Scripts\python.exe scripts\phase3_diffusion.py     # timing/VRAM sweep (~2 min)
.venv\Scripts\python.exe scripts\phase3_img2img.py       # img2img strength sweep
```

Key modules: `common/{paths,seed,device,hf_lowmem}.py`,
`models/{vae,sd_pipeline}.py`, `evaluation/metrics.py`.
Config: `configs/base.yaml` (seed 42, 512 px, fp16, DDIM, guidance 7.5).
Artifacts land in `experiments/results/`.

---

## 5. Repo layout

```
common/      paths, seed, device, hf_lowmem
configs/     base.yaml
data/        secret_images/ (test_001..005) · retrieval_images/ (EMPTY → Phase 4)
evaluation/  metrics.py (mse/psnr/ssim)
models/      vae.py, sd_pipeline.py
scripts/     phase1_verify.py, phase2_vae.py, phase3_diffusion.py, phase3_img2img.py
experiments/results/   phase2_*_compare.png, phase3_*.png, phase3_timings.json, phase3_img2img.json
docs/        RACS_ROADMAP.md, PROJECT_UNDERSTANDING.md, CONCURRENCY.md, phase_briefs/
cache/ temp/ pip_cache/   (gitignored)
```

---

## 6. Git state

- Branch had a **corrupted `.git/refs/heads/main`** (held whitespace, so every git
  command failed with "current branch appears to be broken"). **Repaired on 2026-10-08**
  by restoring the ref to HEAD `174204f`; history intact back to `e9e0b78`.
- Remote: `https://github.com/Nitheesh-Purnanand/racs2.git`. Local `main` is ahead of
  `origin/main` (origin last saw `147b359`), so a `git push` is pending whenever desired.
- `.gitignore` currently ignores `.venv temp pip_cache cache`. `__pycache__/` and
  `experiments/results/phase3_run.log` are NOT ignored and show as untracked noise.

---

## 7. Immediate next steps — Phase 4 (Retrieval Image Database)

> **Full build guide**: `phase4_instructions.md` (written 2026-10-08). Follow that
> end-to-end; the summary below is just the shape of it.

Goal: a corpus RACS retrieves from, shared verbatim by Tx and Rx.

1. Curate ~**1,000 images** into `data/retrieval_images/` (COCO / ImageNet / Unsplash subset).
2. Build `data/metadata/image_index.json`: `image_id`, `image_path`, optional caption.
3. Write a corpus loader that iterates the index.
4. Plan growth 1k → 5k → 10k.
5. **Gate**: working corpus + metadata + a loader that iterates it.

Invariant to preserve: identical database, preprocessing, embedding model, and index on
both ends, or `Xref_tx ≠ Xref_rx` and recovery fails.

---

## 8. Open items / known noise

- `scripts/phase3_img2img.py`: `strength=0.0` is an invalid edge case in this diffusers
  build (empty timestep slice) and was dropped; keep strength > 0.
- `configs/base.yaml` `generation.prompt` is a placeholder SD prompt — replace with the
  paper's example `Kpub` if known (mechanics unchanged).
- `imp_code/` and several `temp/` pip dirs are leftover junk; safe to prune.
- Prefer adding `__pycache__/` and `*.log` to `.gitignore` before the next commit.
