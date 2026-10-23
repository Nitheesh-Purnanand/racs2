# RACS2 — Session Handoff

> **Last updated**: 2026-10-23
> **Repo**: `D:\racs2` · branch `main` · HEAD `4af625a`
> **Plan**: `racs2_implementation_plan.md` **v2 (baseline-first)**. Phases after 5 were
> renumbered on 2026-10-23. Read the plan's Section 5 for the old-to-new map.
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
| 3 | Diffusion & generation cost (SD v1.5) | ✅ Complete (now read as a lower bound) |
| 4 | Build the retrieval image database | ✅ Complete, **banked** |
| 5 | CLIP embeddings | ✅ Complete, **banked** |
| 6 | Evaluation harness + 3-class test set | ⏳ **Next** |
| 7-16 | **AgentSemSteCom reproduction** (baseline gate at 15) | ⏳ Not started |
| 17-22 | RACS retrieval, RACS-Base, comparison | ⏳ Not started |
| 23-33 | Variants, evaluation, write-up | ⏳ Not started |

**Ordering rule**: the baseline is reproduced and frozen before any RACS component is wired
into the pipeline. Phases 4-5 are not wasted work; they are consumed at Phase 17.

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
- CLIP ViT-B/32: **1,000 × 512** float32 vectors, 18.2 s (~55 img/s), caption→image
  top-1 **0.476**, within-topic **0.779** vs across **0.542**, verify PASS.

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
.venv\Scripts\python.exe scripts\phase5_clip.py          # CLIP experiments + 1k embeddings
.venv\Scripts\python.exe scripts\phase5_verify.py        # offline embedding gate
```

Key modules: `common/{paths,seed,device,hf_lowmem}.py`,
`models/{vae,sd_pipeline}.py`, `evaluation/metrics.py`.
Config: `configs/base.yaml` (seed 42, 512 px; generation fp16, CLIP embeddings float32;
DDIM, guidance 7.5).
Artifacts land in `experiments/results/`.

---

## 5. Repo layout

```
common/      paths, seed, device, hf_lowmem
configs/     base.yaml
data/        secret_images/ (test_001..005) · retrieval_images/ (1000 PNGs, gitignored) · metadata/image_index.json
retrieval/   schema.py (index schema), corpus.py (loader)
evaluation/  metrics.py (mse/psnr/ssim)
models/      vae.py, sd_pipeline.py, clip_encoder.py, clip-vit-base-patch32/ (weights, gitignored)
scripts/     phase1_verify.py, phase2_vae.py, phase3_diffusion.py, phase3_img2img.py,
             phase4_build_corpus.py, phase4_verify.py, phase5_fetch_model.py,
             phase5_clip.py, phase5_verify.py
experiments/results/   phase2_*_compare.png, phase3_*.png, phase3_timings.json,
             phase3_img2img.json, phase4_corpus_grid.png,
             phase5_corpus_embeddings.npz/.json, phase5_experiments.json,
             phase5_verify.json, phase5_*.png
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
- `.gitignore` ignores `.venv temp pip_cache cache`, the large corpus dirs, the local
  CLIP model (`models/clip-vit-base-patch32/`), and build noise
  (`data/_raw_coco/`, `data/retrieval_images/`, `__pycache__/`, `*.pyc`, `*.log`).

---

## 7. Where we are and what is next (plan v2)

> Phases 1-5 are **complete and verified**. Phase 4 (corpus) and Phase 5 (CLIP) are now
> **banked**: built and verified, but consumed later at Phase 17 (FAISS). The critical path
> now runs through the AgentSemSteCom baseline, not through retrieval.

Phase 5 gate results:

```
model          : openai/clip-vit-base-patch32 (local: models/clip-vit-base-patch32)
weight sha256  : a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
vectors        : (1000, 512) float32, L2-normalised
throughput     : 18.2 s (~55 img/s)
exp1           : within-topic 0.7791 vs across-topic 0.5423
caption->image : top-1 0.476 (chance 0.001)
determinism    : re-encode max|diff| 2.53e-07 (float32)
retrieval      : nearest non-self mean 0.7707, max 0.8794
gate demo      : CLIP cosine 0.643 vs edge IoU 0.199
verification   : PASS
```

Artifacts: `experiments/results/phase5_corpus_embeddings.npz` (3.82 MB), its manifest,
`phase5_experiments.json`, `phase5_verify.json`, and three PNG grids.

**Protocol note**: float32 is required. fp16 differed by ~7e-4 between batch sizes, which
would make Tx/Rx vectors non-identical. Do not switch back to fp16.

**Model fetch**: `scripts/phase5_fetch_model.py` is the reliable path on this network
(`huggingface_hub`'s single-stream download resets). It resumes from
`pytorch_model.bin.parts.json` and verifies the SHA-256.

**Next: Phase 6, evaluation harness and 3-class test set** (full detail in `phase6_instructions.md`)
1. Install `lpips`, `scipy`, `scikit-image`.
2. Extend `evaluation/metrics.py` with LPIPS and a distribution `report()` helper.
3. Build two role-tagged splits, never mixed:
   - `data/calibration/unistega/`: the paper's dataset, **approximate check only**.
   - `data/test_set/`: our authoritative evaluation set, openly licensed sources only,
     no UniStega images.
   Both 100 images, 20+ per class. Freeze in `data/metadata/test_split.json` with `role`,
   per-file sha256 and `split_version`. Both disjoint from `data/retrieval_images/` and from
   each other.
4. Write `data/metadata/tokens.json` (one fixed token per image in both splits).
5. Write `scripts/eval_report.py`, the single driver used by the baseline, RACS and the
   comparison, with loader enforcement that rejects calibration images for `role=evaluation`.
6. Create `docs/DEVIATIONS.md` (the deviation register).

**Then Phases 7-16: reproduce AgentSemSteCom**
1. Phase 7: `Kpub` + `Xfeat` (LLM key, segmentation for common, pose skeleton for facial).
2. Phase 8: reference generator (PicX-Real or SD v1.5 + ControlNet + IPAdapter-plus) with
   token-seeded noise; fp16 + sequential offload on the 6 GB card.
3. Phase 9: ControlNet zero-conv fine-tune, or a documented deviation.
4. Phase 10: EDICT in isolation; inversion to machine precision.
5. Phase 11: local hide/recover with eta = 0.05 and a manual reference.
6. Phases 12-14: AWGN, semantic codec (SwinJSCC-style C=48 / MDJCM mod 64), CodeFormer.
7. **Phase 15 gate**: 15A approximate reproduction on the UniStega calibration split
   (facial 5 dB PSNR 22.73, facial 20 dB LPIPS 0.2127, style 5 dB LPIPS 0.3850, common
   5 dB PSNR 19.71, wide tolerance). 15B authoritative baseline on our evaluation split.
   Calibration numbers never enter any claim table.
8. Phase 16: baseline security and steganalysis, then freeze the baseline.

**Only then Phases 17-22**: FAISS (loads the banked `.npz`), token selection, structural
reranking, RACS-Base with `Xref_tx == Xref_rx`, and the controlled comparison.

Invariant to preserve: identical database, preprocessing, embedding model, and index on
both ends, or `Xref_tx ≠ Xref_rx` and recovery fails. Row order in the `.npz` and the
FAISS index must stay identical to `image_index.json`.

---

## 8. Open items / known noise

- `scripts/phase3_img2img.py`: `strength=0.0` is an invalid edge case in this diffusers
  build (empty timestep slice) and was dropped; keep strength > 0.
- `configs/base.yaml` `generation.prompt` is a placeholder SD prompt — replace with the
  paper's example `Kpub` if known (mechanics unchanged).
- `.gitignore` ignores `.venv temp pip_cache cache`, `data/_raw_coco/`,
  `data/retrieval_images/`, `models/clip-vit-base-patch32/`, `__pycache__/`, `*.pyc`, `*.log`.
- **CLIP is float32 by protocol.** fp16 CLIP changed embeddings by ~7e-4 between batch
  sizes, which would break Tx/Rx vector identity. Do not switch back to fp16.
- `models/clip-vit-base-patch32/` holds the 605 MB checkpoint. If it is missing, refetch
  with `scripts/phase5_fetch_model.py`; it resumes and verifies SHA-256.
- `imp_code/` and several `temp/` pip dirs are leftover junk; safe to prune.
- **Baseline reproduction is a real workload**, not a 2-day task. The paper uses PicX-Real,
  ControlNet, IPAdapter-plus, UniStega (3 classes), MDJCM and CodeFormer. Any substitute
  must be recorded in `docs/DEVIATIONS.md` with its measured impact.
