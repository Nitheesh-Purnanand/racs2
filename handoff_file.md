# RACS2 — Session Handoff

> **Last updated**: 2026-10-20
> **Repo**: `D:\racs2` · branch `main` · HEAD `08d50ff`
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
| 4 | Build the retrieval image database | ✅ Complete |
| 5 | CLIP embeddings | 🟡 **Code complete, run blocked on CLIP download** |
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
- Retrieval corpus: **1,000** COCO val2017 images, canonical **512×512 RGB** PNGs,
  **429.3 MB**; `corpus_version = cc20af21...89aff`. Build+verify PASS.

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
data/        secret_images/ (test_001..005) · retrieval_images/ (1000 PNGs, gitignored) · metadata/image_index.json
retrieval/   schema.py (index schema), corpus.py (loader)
evaluation/  metrics.py (mse/psnr/ssim)
models/      vae.py, sd_pipeline.py
scripts/     phase1_verify.py, phase2_vae.py, phase3_diffusion.py, phase3_img2img.py, phase4_build_corpus.py, phase4_verify.py
experiments/results/   phase2_*_compare.png, phase3_*.png, phase3_timings.json, phase3_img2img.json, phase4_corpus_grid.png
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
- `.gitignore` currently ignores `.venv temp pip_cache cache`, plus the large corpus
  dirs and build noise (`data/_raw_coco/`, `data/retrieval_images/`, `__pycache__/`,
  `*.pyc`, `*.log`).

---

## 7. Immediate next steps — Phase 5 (CLIP embeddings)

> Phase 5 is **implemented and synthetically validated**. The only missing step is the
> 605 MB CLIP checkpoint download, which the current network cannot sustain (~10–75 KB/s,
> Wi-Fi signal 33%). No CLIP weights exist elsewhere on this machine.

Files already built:
- `models/clip_encoder.py` — frozen CLIP ViT-B/32 wrapper (`CLIPModel` + `CLIPProcessor`),
  fp16 default, float32 L2-normalised outputs, batched, `hf_lowmem` before `transformers`.
- `scripts/phase5_clip.py` — experiments 1–3, semantic-vs-structural gate demo, full
  1,000-image pass, caption embeddings, atomic `.npz` + manifest + experiments JSON.
- `scripts/phase5_verify.py` — offline gate (shape, dtype, finite, norms, row order,
  provenance, re-encode determinism, caption→image top-1, retrieval sanity).
- `configs/base.yaml` — `embedding:` block.
- `phase5_instructions.md` — full build guide.

To finish (when bandwidth allows):

```powershell
cd d:\racs2
$env:HF_HUB_DISABLE_XET = "1"          # Xet stalled at 0 bytes on this network
.venv\Scripts\python.exe scripts\phase5_clip.py      # downloads CLIP, runs experiments, embeds corpus
.venv\Scripts\python.exe scripts\phase5_verify.py    # offline gate; expect PASS
```

Expected artifacts: `experiments/results/phase5_corpus_embeddings.npz` plus manifest,
`phase5_experiments.json`, three PNG grids, and a PASS from the verifier.

Invariant to preserve: identical database, preprocessing, embedding model, and index on
both ends, or `Xref_tx ≠ Xref_rx` and recovery fails. Row order in the `.npz` must stay
identical to `image_index.json`.

---

## 8. Open items / known noise

- `scripts/phase3_img2img.py`: `strength=0.0` is an invalid edge case in this diffusers
  build (empty timestep slice) and was dropped; keep strength > 0.
- `configs/base.yaml` `generation.prompt` is a placeholder SD prompt — replace with the
  paper's example `Kpub` if known (mechanics unchanged).
- `.gitignore` now ignores `.venv temp pip_cache cache` plus `data/_raw_coco/`,
  `data/retrieval_images/`, `__pycache__/`, `*.pyc`, `*.log`.
- `imp_code/` and several `temp/` pip dirs are leftover junk; safe to prune.
