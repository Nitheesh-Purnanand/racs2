# RACS2 — Project Status & Change Log

> **Location**: `d:\racs2\project_status.md`  
> **Last Updated**: 2026-10-28  
> **Current Phase**: Phases 11-16 complete. Phase 15 froze the baseline; Phase 16 froze the
> baseline security table and XuNet steganalysis. Phase 17 (RACS retrieval stack) is next.

---

## 1. Executive Summary

**RACS2** (Retrieval-Augmented Coverless Steganography v2) is an Agentic AI system that implements secure, coverless, semantic steganography over wireless channels using vector retrieval, diffusion models (EDICT / PicX-Real), and error-resilient channel simulation.

---

## 2. Current Project State

| Component | Status | Details / Notes |
| :--- | :--- | :--- |
| **Workspace Cleanup** | ✅ Completed | Retained only core folders. Removed unused boilerplate. |
| **Project Analysis** | ✅ Completed | Comprehensive analysis in `racs2_complete_analysis.md`. |
| **Status Tracker** | ✅ Active | `project_status.md` maintained as live status + changelog. |
| **Environment & Dependencies** | ✅ Completed | venv + PyTorch 2.6.0+cu124 (CUDA working). Temp/pip cache redirected to D:. |
| **Phase 1 — Environment & Skeleton** | ✅ Completed | `configs/base.yaml`, `common/seed.py`, `common/device.py`, `scripts/phase1_verify.py`. All 7 checks pass. |
| **Phase 2 — VAE & Latent Space** | ✅ Completed | `common/paths.py`, `evaluation/metrics.py`, `models/vae.py`, `scripts/phase2_vae.py`. Round-trip verified on 5 images (PSNR ~32.8 dB avg). |
| **Phase 3 — Diffusion & Generation Cost** | ✅ Completed | `common/hf_lowmem.py`, `models/sd_pipeline.py`, `scripts/phase3_diffusion.py`. SD v1.5 loads fp16 on the 6 GB GPU; 6-point timing sweep + 5 verification checks pass. |
| **Phase 4 — Retrieval Image Database** | ✅ Completed | `retrieval/schema.py`, `retrieval/corpus.py`, `scripts/phase4_build_corpus.py`, `scripts/phase4_verify.py`. 1,000-image COCO val2017 corpus (512×512 RGB), versioned `image_index.json`, loader + integrity/determinism checks pass. |
| **Phase 5 — CLIP Embeddings** | ✅ Completed | `models/clip_encoder.py`, `scripts/phase5_clip.py`, `scripts/phase5_verify.py`, `scripts/phase5_fetch_model.py`. CLIP ViT-B/32, 1,000 × 512-d float32 L2-normalised vectors aligned with the corpus. Verify PASS: re-encode max|diff| 2.5e-07, caption→image top-1 0.476, within-topic 0.779 vs across 0.542. |
| **Phase 6 — Evaluation Harness & Splits** | ✅ Completed | `evaluation/splits.py` (role enforcement, tokens), `evaluation/report.py` (frozen report schema), extended `evaluation/metrics.py` (LPIPS + `distribution()`), `scripts/phase6_build_testset.py`, `scripts/phase6_eval_selftest.py`, `scripts/eval_report.py`, `scripts/dummy_runner.py`, `docs/DEVIATIONS.md`. Two frozen splits: **calibration** (UniStega, 22/class) and **evaluation** (COCO val2017 / Commons human portraits / Commons cats, 22/class). `split_version a57d9c8c...`, `tokens_version 20166234...`. Selftest **83/83 PASS**; dummy evaluation 264 cases across 12 cells. |
| **Phase 7 — Public Conditions (`Kpub` + `Xfeat`)** | ✅ Completed | `semantic/keygen.py` (BLIP + template), `semantic/features.py` (SegFormer-B0 segmentation, MediaPipe FaceLandmarker skeleton, style `none`), `semantic/cache.py`, `scripts/phase7_build_conditions.py`, `scripts/phase7_verify.py`. 132 records (22 per class per role), `config_hash 6e5eec49...`, `split_version e52d2ddf...` (live split; earlier notes used a57d9c8c before the split manifest was regenerated). Determinism exact for Kpub and features; `phase7_verify.json` PASS; grid inspected. |
| **Phase 8 — Reference Generator** | ✅ Completed | `reference/seed.py`, `reference/conditions.py`, `reference/generate.py`, `scripts/phase8_reference.py`, `scripts/phase8_verify.py`. SD v1.5 + seg/OpenPose ControlNet, 256 px, 20 DDIM steps. 132 references, exact rerun hashes, peak 2.84 GB, `phase8_verify.json` PASS. PicX-Real/IPAdapter and 512 px are deviations. |
| **Phase 9 — ControlNet Zero-Conv Fine-Tune** | ✅ Completed | `reference/train_controlnet.py`, `scripts/phase9_train_controlnet.py`, `scripts/phase9_verify.py`. 50 bf16 steps on calibration common+facial, only zero-conv blocks trainable (11.48M trainable / 349.8M frozen), probe MSE 0.1488 -> 0.1296, checkpoint `controlnet_zero_conv_50_88159c58.safetensors`, verify PASS. |
| **Phase 10 — EDICT** | ✅ Completed | `steganography/edict.py`, `scripts/phase10_edict.py`, `scripts/phase10_verify.py`. 256 px / 32x32 float32 latent, fp16 SD UNet, 20 steps, mixing 0.93, guidance 1.0. 20 random + 20 image latents; max relative error 3.58e-3, max abs 9.49e-3; verify PASS with the documented fp16 numerical-floor deviation. |
| **Phase 11 — Local Steganography Core** | ✅ Completed | `steganography/mask.py`, `hide.py`, `recover.py`, `pipeline.py`, `scripts/phase11_local.py`, `scripts/phase11_verify.py`. Token mask from the Phase 8 seed contract, bounded-strength EDICT hide/recover at 256 px, 20 steps, eta 0.05, strength 0.2, guidance 3.0. 20 matched + 20 mismatched evaluation images. Recovery PSNR 26.77 dB (matched) vs 26.62 dB (mismatched); stego PSNR 23.28 dB; deterministic reruns; verify 12/12 PASS. |
| **Phase 12 — AWGN Channel** | ✅ Completed | `communication/awgn.py`, `scripts/phase12_awgn.py`, `scripts/phase12_verify.py`. sigma^2 = P_signal / 10^(SNR/10) from the measured power, fixed seed per cell. Mean recovery PSNR 12.68 / 14.96 / 17.93 / 21.08 dB at 5/10/15/20 dB. Analytic variance max relative error 0.46 percent; seed replay exact; verify 7/7 PASS. |
| **Phase 13 — Semantic Codec** | ✅ Completed | `communication/semantic_codec.py`, `scripts/phase13_codec.py`, `scripts/phase13_verify.py`. Frozen SD-VAE analog JSCC substitute (paper checkpoints unavailable), explicit invertible power normalisation, pinned state-dict hash. Isolated PSNR 17.88 / 22.88 / 26.70 / 28.51 dB. Codec+recovery beats raw AWGN by +1.40 to +2.33 dB PSNR. Verify 9/9 PASS. |
| **Phase 14 — Optional CodeFormer Enhancement** | ✅ Completed | `enhancement/codeformer.py`, `scripts/phase14_enhance.py`, `scripts/phase14_verify.py`. CodeFormer/facelib stack unavailable in this environment, so the module is a documented deterministic pass-through kept **off by default** and restricted to the facial class; deltas are exactly 0 at all four SNR cells. Non-facial control unchanged. Verify 13/13 PASS. |
| **Phase 15 — Baseline Gate** | ✅ Completed | `pipelines/agentsemstecom.py`, `scripts/phase15_baseline.py`, `scripts/phase15_verify.py`. 15A calibration on UniStega (approximate, deviations recorded) and 15B authoritative evaluation baseline: 264 cases, 12 cells, `baseline_table.json` frozen, `phase15_verify.json` PASS. |
| **Phase 16 — Baseline Security & Steganalysis** | ✅ Completed | `security/token_conditions.py`, `security/steganalysis.py`, `scripts/phase16_security.py`, `phase16_steganalysis.py`, `phase16_manifest.py`, `phase16_verify.py`. Three receiver conditions (correct / wrong / no_token) on 264 frozen cases, N=22 per cell; aggregate PSNR 17.39 / 13.05 / 13.07 dB, correct-minus-wrong gap **4.34 dB** (threshold 4 dB), 6 low-SNR per-cell exceptions documented. XuNet (SRM front end, 5 blocks) trained on 2x2 128 px patches from 66 images, image-disjoint 46/10/10 split (368/80/80 samples); test accuracy **0.4875** (n=80) vs chance 0.5 and paper anchor 0.5091. Verify **17/17 PASS**. Report: `docs/phase16_report.md`. |

### Phase 1 gate results (2026-10-08)

```
[✓] torch 2.6.0+cu124
[✓] torch.cuda.is_available() == True
[✓] GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.00 GB VRAM)
[✓] Config loads, seed == 42
[✓] Image -> tensor shape (1, 3, 512, 512)
[✓] Tensor moved to cuda:0
[✓] Reproducibility test: PASS
```

### Phase 2 gate results (2026-10-08)

```
[✓] diffusers 0.41.0 + accelerate 1.15.0
[✓] VAE loaded fp16 on GPU (stabilityai/sd-vae-ft-mse)
[✓] input  shape (1, 3, 512, 512)
[✓] latent shape (1, 4, 64, 64)        <- 8x downscale, 4 channels
[✓] output shape (1, 3, 512, 512)
[✓] MSE/PSNR/SSIM for all 5 images (+ spread reported)
[✓] 5 side-by-side PNGs in experiments/results/
PSNR : mean=32.809 dB  (min=31.784, max=34.153)
SSIM : mean=0.935447   (min=0.933472, max=0.936783)
```

---

### Phase 3 gate results (2026-10-08)

```
[✓] SD v1.5 pipeline loads fp16 on the 6 GB GPU without OOM (6.3 s cold load)
[✓] DDIM scheduler active, attention slicing enabled
[✓] Cold start (download + warm-up) completes
[✓] Wall-clock time + peak VRAM measured for 3 step counts x 2 seeds
[✓] sec/step is roughly constant across step counts (0.267 -> 0.260 s/step)
[✓] Reference images saved for the example prompt (6 sweep + 2 verification PNGs)
[✓] experiments/results/phase3_timings.json written and well-formed
[✓] Peak VRAM 2.61 GB (budget 6 GB)
[✓] Verification PASS (determinism, seed effect, steps effect, guidance effect, non-trivial)
[✓] CFG cost measured: 0.430 s/step with guidance, 0.245 s/step without (1.76x)

steps  seed      sec   s/step   peakGB
   20    42     8.60    0.430     2.60
   20   123     8.64    0.432     2.60
   30    42    12.89    0.430     2.60
   30   123    12.62    0.420     2.60
   50    42    20.52    0.410     2.60
   50   123    21.00    0.420     2.60

mean seconds/step = 0.424   => one 50-step reference costs ~21 s
speedup without CFG = 1.76x  (classifier-free guidance = 2 U-Net passes/step)

Generated bytes are reproducible across runs (same seed -> same SHA-256), but
wall-clock varies with machine load: an earlier idle run measured 0.264 s/step
for the same 6 images (13 s at 50 steps). Treat s/step as machine-dependent and
report the output hashes alongside timings.
```

---

## 3. Directory Structure

```
d:\racs2\
├── common/               # seed.py, device.py, paths.py, hf_lowmem.py
├── configs/              # base.yaml
├── scripts/              # phase1_verify.py, phase2_vae.py, phase3_diffusion.py
├── data/
│   ├── secret_images/    # test_001.png .. test_005.png
│   ├── retrieval_images/ # 1,000 canonical 512x512 PNGs (Phase 4, gitignored)
│   └── metadata/         # image_index.json (versioned retrieval index)
├── retrieval/            # schema.py (index schema), corpus.py (loader)
├── evaluation/           # metrics.py (mse/psnr/ssim)
├── models/               # vae.py (SD VAE loader + roundtrip), sd_pipeline.py (SD v1.5)
├── experiments/
│   └── results/          # phase2_*_compare.png, phase3_*.png, phase3_timings.json, phase4_corpus_grid.png
├── cache/                # HuggingFace/diffusers/torch caches (on D:)
├── docs/                 # Briefs, concurrency guidelines, roadmap
├── phase1/               # Phase 1 PDFs
├── phase1_instructions.md
├── phase2_instructions.md
├── racs2_implementation_plan.md
├── racs2_complete_analysis.md
└── project_status.md      # This file
```

---

### Phase 4 gate results (2026-10-08)

```
[✓] Corpus built deterministically (subset_seed=42 -> same 1000 ids)
[✓] data/retrieval_images/ holds exactly 1000 canonical 512x512 RGB PNGs
[✓] data/metadata/image_index.json written (schema_version=1, count=1000)
[✓] Every indexed path exists; no duplicate image_ids
[✓] corpus_version recomputed == stored
[✓] Sampled (20) sha256 digests match the index
[✓] Loader iterates in stable order across loads
[✓] VERIFICATION RESULT: PASS

images       : 1000
bytes on disk: 429.3 MB (mean 429.3 kB/image)
corpus_version: cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff
contact sheet: experiments/results/phase4_corpus_grid.png
```

---

### Phase 5 implementation (2026-10-20)

**Complete. The corpus embedding artifacts exist and the verifier passes.**

Built:
- `models/clip_encoder.py` — frozen CLIP wrapper (`CLIPModel` + `CLIPProcessor`), fp16/fp32,
  image + text batching, float32 L2-normalised output, `hf_lowmem` installed before
  `transformers`. Handles both the transformers 4 tensor return and the transformers 5
  `BaseModelOutputWithPooling` return.
- `scripts/phase5_clip.py` — experiments 1–3 (text similarity, image neighbours,
  text→image), the semantic-vs-structural gate demo, the full 1,000-image pass,
  optional caption embeddings, and atomic `.npz` + manifest + experiments JSON output.
- `scripts/phase5_verify.py` — offline gate: shape/dtype, finite values, unit norms,
  row order vs `image_index.json`, provenance (`corpus_version`, `index_sha256`),
  re-encode determinism, caption→image top-1, retrieval sanity.
- `configs/base.yaml` — `embedding:` block (ViT-B/32, 512-d, 224 px, float32, prompts/queries).

Validation so far:
- `py_compile` passes for all three modules.
- Full pipeline runs end-to-end with a synthetic 512-d encoder: all 1,000 rows, all
  experiments, all grids, `.npz` (3.8 MB) and JSON artifacts are produced; only the
  semantic assertions fail, as expected for random vectors.
- `phase4_verify.py` re-run: PASS.

Gate results (2026-10-20, real CLIP ViT-B/32, float32):
```
[✓] CLIP ViT-B/32 loads from models/clip-vit-base-patch32 (weight sha256 a6308213...)
[✓] Experiment 1: within-topic 0.7791 > across-topic 0.5423 (gap +0.2368)
[✓] Experiment 2: image nearest-neighbour grid saved and sensible
[✓] Experiment 3: text -> top-5 corpus images saved and sensible
[✓] 1,000 x 512 float32 L2-normalised vectors (max |1-||v||| = 1.19e-07)
[✓] Row order identical to image_index.json (0 mismatches)
[✓] Provenance: corpus_version + index_sha256 match the live index
[✓] Re-encode determinism: max|diff| = 2.53e-07 (float32)
[✓] caption -> image top-1 = 0.476 (chance 0.001)
[✓] Retrieval sanity: nearest non-self mean 0.7707, max 0.8794
[✓] Gate demo: CLIP cosine 0.643 but edge IoU 0.199 (semantic != structure)
[✓] VERIFICATION RESULT: PASS

vectors       : (1000, 512)
throughput    : 18.2 s for 1,000 images (~55 img/s)
artifacts     : phase5_corpus_embeddings.npz (3.82 MB) + manifest + experiments.json
                + 3 PNG grids + phase5_verify.json
```

> **Why float32, not fp16**: fp16 CLIP differs by ~7e-4 between batch sizes, which
> breaks the requirement that Tx and Rx reproduce identical vectors. float32 is
> batch-size invariant (2.5e-07), so it is now the configured default. The checkpoint
> was fetched with `scripts/phase5_fetch_model.py` because `huggingface_hub`'s
> single-stream download reset mid-transfer on this network.

---

> Folders marked "(Phase N)" are created when that phase begins, per the plan's
> "only create what the current phase needs" principle.

---

## 4. Key Architectural Insights & Decisions

1. **Dual-Model Strategy**:
   - **Reference Generation**: PicX-Real / SD-based reference image generation for prompt-conditioned synthesis.
   - **Steganographic Inversion**: EDICT operating on Stable Diffusion v1.5 for bit-exact secret payload embedding/extraction.
2. **Retrieval-Augmented Semantic Alignment**:
   - Vector retrieval (ChromaDB / FAISS) over CLIP / OpenCLIP embeddings to select semantic keys.
3. **Entropy & Security Considerations**:
   - Secret payload capacity vs. steganography imperceptibility trade-off.
   - Vector selection entropy logging and evaluation.
4. **Determinism First**: Tx and Rx must reproduce identical references (`Xref_tx == Xref_rx`). Seeds, exact FAISS index (`IndexFlatIP`), and deterministic token selection are protocol-critical.
5. **Latent Space is the Error Floor**: All diffusion/EDICT stages run in SD's latent space, so VAE round-trip error (Phase 2) bounds downstream quality.

---

## 5. Change Log

### [2026-10-28] - Phase 16 Complete (Baseline Security & Steganalysis)

- **Security**: `security/token_conditions.py` implements the receiver half for three
  conditions on the frozen Phase 15 stego: `correct` (true token), `wrong` (a valid wrong
  token), and `no_token` (fixed public fallback token `00000000`, reference seed
  1102030775, all-zero mask). The transmitter is identical in all three.
- **Run**: `scripts/phase16_security.py` measured 264 cases (66 evaluation images x 4 SNRs),
  N=22 per class per cell. Aggregate recovery PSNR is **17.39 dB correct**, **13.05 dB
  wrong**, **13.07 dB no_token**; the correct-minus-wrong gap is **4.34 dB** (threshold
  4 dB). Six low-SNR cells fall below 4 dB because channel noise dominates and are
  itemised with a reason in `phase16_manifest.json`.
- **XuNet**: `security/steganalysis.py` implements the standard XuNet (5x5 SRM high-pass
  init, five conv blocks with BN/Tanh/pooling, global average pool, FC 128->2). Covers are
  the secret images; positives are the frozen stego. The 66 images are split 70/15/15 by
  `image_id` (46/10/10) and each image contributes 2x2 non-overlapping 128 px patches
  (368/80/80 samples). Seed 42, batch 8, early stopping on validation accuracy.
- **Detectability**: test accuracy **0.4875** on n=80 (chance 0.5), so the baseline is
  near-chance against this detector. Paper anchor 0.5091 is stored as approximate and is
  not treated as a target.
- **Artifacts**: `baseline_security.json`, `baseline_steganalysis.json`,
  `phase16_manifest.json`, `phase16_verify.json`, `phase16_security_grid.png`, and
  `experiments/steganalysis/xunet_baseline_0000002a.pt`
  (sha256 `243c21653e04e16402386ccbfe411ae23dbad2dcde363e2f0afabd187b9e6747`).
- **Verified**: `phase16_verify.py` 17/17 PASS. Every record is `role: evaluation` and
  `approximate: false`; no calibration image appears. The gate also re-checks the frozen
  `baseline_security.json` and `baseline_steganalysis.json` hashes against
  `phase16_manifest.json`, so artifact drift fails the freeze.
- **Deviations**: patch-based XuNet dataset at 128 px and the explicit no-token fallback are
  recorded in `docs/DEVIATIONS.md`.

### [2026-10-27] - Phases 11-14 Complete (stego core, channel, codec, enhancement)

- **Phase 11**: added `steganography/{mask,hide,recover,pipeline}.py` and
  `scripts/phase11_local.py` / `phase11_verify.py`. Full-inversion EDICT produced noise-like
  stego (11.3 dB), so hiding uses a bounded-strength EDICT chain (strength 0.2) that keeps the
  secret image intact while still conditioning on the reference `Kpub`. Matched recovery
  26.77 dB, stego 23.28 dB, mismatched 26.62 dB (gap 0.15 dB, documented). 12/12 PASS.
- **Phase 12**: added `communication/awgn.py` and `scripts/phase12_awgn.py` / `phase12_verify.py`.
  20 images x 4 SNR cells, per-cell seeds, real-valued AWGN with the complex-model variance.
  PSNR falls monotonically 21.08 -> 12.68 dB; 7/7 PASS.
- **Phase 13**: added `communication/semantic_codec.py` and `scripts/phase13_codec.py` /
  `phase13_verify.py`. SD-VAE substitute for the unavailable SwinJSCC/MDJCM checkpoints,
  isolated curve plus raw-vs-codec comparison (+1.40 to +2.33 dB); 9/9 PASS.
- **Phase 14**: added `enhancement/codeformer.py` and `scripts/phase14_enhance.py` /
  `phase14_verify.py`. CodeFormer weights unavailable; deterministic pass-through fallback,
  off by default and facial-only, with the miss recorded; 13/13 PASS.
- **Deviations**: bounded-strength EDICT and 256 px (Phase 11), SD-VAE codec substitute
  (Phase 13), CodeFormer fallback (Phase 14). See `docs/DEVIATIONS.md`.

### [2026-10-27] - Phases 8-10 Complete (References, ControlNet, EDICT)

- **Phase 8**: added `reference/` and `scripts/phase8_reference.py` / `phase8_verify.py`.
  Generated 132 token-seeded references with SD v1.5 + ControlNet at 256 px, 20 DDIM steps;
  exact rerun determinism, peak VRAM 2.84 GB, `phase8_verify.json` PASS.
- **Phase 9**: added `reference/train_controlnet.py` and `scripts/phase9_train_controlnet.py` /
  `phase9_verify.py`. 50 bf16 steps, 11.48M trainable / 349.8M frozen, probe MSE
  0.1488 -> 0.1296, checkpoint hash recorded, verify PASS.
- **Phase 10**: added `steganography/edict.py` and `scripts/phase10_edict.py` /
  `phase10_verify.py`. 12-cell sweep + 20 random / 20 image latents at the operating point;
  max relative error 3.58e-3, verify PASS with the fp16/256 px deviation recorded.
- **Deviations**: PicX-Real/IPAdapter and 512 px (Phase 8), 50-step bf16 fine-tune (Phase 9),
  fp16 EDICT at 256 px with a 3.58e-3 floor (Phase 10). See `docs/DEVIATIONS.md`.
- **Audit**: `scripts/phase8_10_audit.py` re-checks all 132 references, the full Phase 9
  checkpoint (26/26 zero-conv tensors changed vs the base, 11.48M params), Phase 10 algebra
  and determinism. Result: 30/30 PASS in `experiments/results/phase8_10_audit.json`.

### [2026-10-27] - Phase 7 Complete (Public Conditions)

- **Added** `semantic/keygen.py` (BLIP caption + fixed template, greedy/fp16),
  `semantic/features.py` (SegFormer-B0 150-class segmentation, MediaPipe FaceLandmarker
  skeleton with a deterministic equalized fallback for low-light faces, explicit `none` for
  style), and `semantic/cache.py` (atomic `.npz`, previews, manifest).
- **Added** `scripts/phase7_build_conditions.py` and `scripts/phase7_verify.py`.
- **Built** 132 records (calibration and evaluation, 22 per class), `config_hash
  6e5eec492b7af67cee8b35c18ff4d970f89f2397dd69a73e07fecb17004a4f64`, live `split_version
  e52d2ddfea4dba17ebde84a3b8c0129dca0ed928e69de3faad9746baeb7b330a`.
- **Verified** exact Kpub/feature determinism on the 24-image sample, 0 failures, all 13
  checks pass. Grid: `experiments/results/phase7_conditions_grid.png`.
- **Fixed** during the run: atomic `.npz` temp naming, missing `SegFormerExtractor.method`,
  MediaPipe 1.1 tasks API migration and the low-light detection fallback.
- **Added** instruction files `phase8_instructions.md` through `phase16_instructions.md`.

### [2026-10-23] - Phase 6 Complete (Evaluation Harness & Role-Tagged Splits)

- **Added** `evaluation/splits.py`: `Split.load(role=...)` with enforced roles. Loading the
  calibration split requires `allow_calibration=True`; a calibration record cannot be used
  under the evaluation role. Verifies every file hash on load and recomputes `split_version`.
- **Added** `evaluation/report.py`: the frozen report schema (`cells` by class and SNR,
  distributions per metric), with `approximate` forced true for calibration and false for
  evaluation, and rejection of mixed-role or duplicate cases.
- **Extended** `evaluation/metrics.py`: `lpips_metric` (LPIPS AlexNet, correct `[0,1] ->
  [-1,1]` conversion), `distribution()` (finite-only mean, `n_inf`, percentiles), and
  `all_metrics(..., with_lpips=...)` without changing existing behaviour.
- **Added** `scripts/phase6_build_testset.py`: builds both splits deterministically. Real
  UniStega release downloaded from the paper's linked DiffStega dataset. Calibration facial
  filtered to the 22 FFHQ/CelebA-HQ entries (8 COCO non-faces excluded). Evaluation uses
  COCO val2017 (disjoint from the retrieval corpus), Wikimedia Commons human portraits
  (title-filtered and CLIP-verified to single faces), and Commons cat photographs.
- **Added** `scripts/phase6_eval_selftest.py`: 83 checks over metrics, splits, manifest
  invariants, corruption paths, role enforcement, tokens, report shape, canonicalisation,
  cross-checks and config consistency. **83/83 PASS**.
- **Added** `scripts/eval_report.py` + `scripts/dummy_runner.py`: the single system x class x
  SNR measurement driver reused by Phases 15, 16, 21, 22 and 26.
- **Added** `docs/DEVIATIONS.md` and the `evaluation:` config block.
- **Artifacts**: `data/metadata/test_split.json`, `data/metadata/tokens.json`,
  `experiments/results/phase6_selftest.json`, `phase6_dummy_eval.json` (264 cases),
  `phase6_build_summary.json`, `phase6_splits_grid.png`.
- **Verified**: `split_version = a57d9c8c500e712fe6e8a15dbfb0eaa7cd2080ba1c5fd772610b90e9ecd5b9d3`,
  `tokens_version = 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1`.
  Rebuild reproduces the same hashes. Splits are pairwise disjoint and disjoint from the
  retrieval corpus by sha256.

### [2026-10-23] - Plan Reordered: Baseline-First

- **Rewrote** `racs2_implementation_plan.md` as v2. AgentSemSteCom is now reproduced and
  validated **first** (Phases 7-16), and only then is the RACS contribution built on top of
  the frozen baseline (Phases 17-22).
- **Why**: the previous order built RACS-only components first and deferred the baseline to
  one over-scoped phase. That left no control condition and no way to attribute differences
  to reference acquisition.
- **No completed work is discarded.** Phases 1-5 carry over unchanged. Phase 4-5 outputs are
  now explicitly **banked** and consumed at Phase 17 (FAISS), with an artifact-to-consumer
  reuse map in the plan's Section 4.
- **Phase 3 timing** is reclassified as the SD-only **lower bound**, not the baseline cost,
  since the paper's baseline adds ControlNet and IPAdapter-plus.
- **New baseline gate** (Phase 15) has two parts. 15A checks **approximate** reproduction on
  the UniStega calibration set: facial 5 dB PSNR 22.73, facial 20 dB LPIPS 0.2127, style
  5 dB LPIPS 0.3850, common 5 dB PSNR 19.71, with wide tolerance. 15B produces the
  **authoritative** baseline on our own evaluation split. Deviations go in
  `docs/DEVIATIONS.md`.
- **Phase 22** is now the primary comparison and the minimum viable research result.
- **Dataset policy added**: UniStega (the paper's dataset) is used **only as a read-only
  calibration set** to check approximate reproduction. All claims, the security matrix,
  ablations and the headline tables use our own authoritative evaluation split. Both splits
  are frozen by hash, pairwise disjoint from each other and from the retrieval corpus, and
  tagged with an enforced `role` field. Calibration tolerance is deliberately wide
  (+/- 2.5 dB PSNR, +/- 0.07 SSIM, +/- 0.05 LPIPS, +/- 40 percent MSE) and no exact
  reproduction is claimed.

### [2026-10-20] - Phase 5 Complete (CLIP Embeddings)

- **Added** `models/clip_encoder.py`: frozen CLIP ViT-B/32 wrapper over `transformers`
  (`CLIPModel` + `CLIPProcessor`), batched image/text encoding, float32 L2-normalised
  outputs, `hf_lowmem` installed before `transformers`, optional local `model_path`.
- **Added** `scripts/phase5_clip.py`: experiments 1–3, the semantic-vs-structural gate
  demo (edge IoU proxy), full 1,000-image pass, caption embeddings, atomic `.npz` +
  manifest + `phase5_experiments.json` + three PNG grids.
- **Added** `scripts/phase5_verify.py`: offline gate for shape/dtype/finite/norm,
  row order, provenance, re-encode determinism, caption→image top-1 and retrieval sanity.
- **Added** `scripts/phase5_fetch_model.py`: parallel HTTP-Range downloader with per-chunk
  retries and resume, needed because `huggingface_hub` reset mid-transfer on this network.
- **Added** `embedding:` block (model id, local path, dim, 224 px, batch sizes, dtype,
  prompts, queries) to `configs/base.yaml`.
- **Ran** CLIP ViT-B/32 on the 1,000-image corpus: 1,000 × 512 float32 unit-norm vectors
  in 18.2 s (~55 img/s), weight sha256 `a6308213...` verified.
- **Verified PASS**: row order exact, re-encode max|diff| 2.53e-07, caption→image top-1
  0.476, within-topic 0.7791 vs across 0.5423, nearest non-self mean 0.7707.
- **Decision**: float32 is the protocol default because fp16 differed ~7e-4 across
  batch sizes, which would break Tx/Rx vector identity.
- **Artifacts**: `phase5_corpus_embeddings.npz` (3.82 MB), `phase5_corpus_embeddings.json`,
  `phase5_experiments.json`, `phase5_verify.json`, three PNG grids.

### [2026-10-08] - Phase 4 Complete (Retrieval Image Database)

- **Built** the frozen retrieval corpus from COCO val2017: 1,000 images, deterministic subset (`subset_seed=42`), canonical 512×512 RGB PNGs, **429.3 MB** on disk.
- **Added** `retrieval/schema.py` (Step 6): the versioned index schema (`schema_version=1`), `corpus_version` (sha256 over the ordered `image_id` list), per-image `sha256`, atomic write, and `validate_index`.
- **Added** `retrieval/corpus.py` (Step 7): deterministic loader (`ids`, `records`, `get`, `load`, `iter_images`, `version`) using only stdlib + Pillow.
- **Added** `scripts/phase4_build_corpus.py` (Step 5) and `scripts/phase4_verify.py` (Step 8). Verified PASS: schema valid, 1000/1000 files present and RGB 512×512, no duplicate ids, stable order across loads, `corpus_version` recompute matches, sampled sha256 match; contact sheet → `experiments/results/phase4_corpus_grid.png`.
- **`corpus_version`** = `cc20af2154beb97e036b08401a350cecce252b72c0b5545dc8613c81abf89aff`.
- **Config**: `configs/base.yaml` gained a `retrieval:` block; `common/paths.py` gained `METADATA_DIR`, `IMAGE_INDEX_PATH`, `RAW_CORPUS_DIR`. Large corpus/raw dirs are now gitignored.

### [2026-10-08] - Phase 4 Instructions Written

- **Created** `phase4_instructions.md` from the implementation plan's Phase 4 (Build the Retrieval Image Database): pre-flight disk/connectivity checks, dependency confirmation (no new deps), corpus-source choice (COCO val2017, URLs verified HTTP 200), `configs/base.yaml` `retrieval` block, `retrieval/corpus.py` loader, `scripts/phase4_build_corpus.py` + `scripts/phase4_verify.py`, the versioned `data/metadata/image_index.json` schema (`corpus_version` + per-image `sha256`), the 10-item acceptance checklist, and the gate question.

### [2026-10-08] - Phase 3 Complete (+ Windows memory fix)

- **Root cause of the `MemoryError` / `OSError 1455` ("paging file is too small") when loading SD v1.5**: safetensors maps checkpoints copy-on-write on Windows, so the **whole file** counts against the system commit limit (RAM + page file). The UNet is 3.2 GB, but free commit on this 16 GB box (page file capped at ~16 GB because `C:` is full) was often < 1 GB.
- **Added** `common/hf_lowmem.py`: a `safetensors` shim that maps checkpoints **read-only** (`mmap.ACCESS_READ`, file-backed ⇒ no commit charge) and hands out zero-copy views; it also provides `safe_open`/`Slice` compatibility for transformers and falls back to stock safetensors on any error. Installed before `diffusers`/`transformers` import in `models/sd_pipeline.py` and `models/vae.py`.
- **Updated** `models/sd_pipeline.py`: loads with `device_map='cuda'` so fp16 weights stream straight to VRAM instead of being materialised in host RAM; frees memory between candidate repos; reports free commit on failure; retries without `device_map` if a build rejects it.
- **Wrote** `scripts/phase3_diffusion.py`: timing/VRAM sweep (3 step counts × 2 seeds) + 5 verification checks (determinism, seed/steps/guidance effects, non-triviality).
- **Verified**: full sweep passes; load 6.3 s, ~0.26-0.43 s/step (machine-load dependent), peak VRAM 2.61 GB, verification PASS, CFG costs 1.76x. Artifacts: `experiments/results/phase3_*.png`, `phase3_timings.json`.
- **Added** `scripts/phase3_img2img.py`: runs the *same* weights as img2img on our own images (reusing the text2img components, no second weight load) and shows what an image input produces at strength 0.25/0.5/0.75 -> the input is progressively overridden by the prompt. Records `mean|diff|` vs the input (`phase3_img2img.json`, `..._grid.png`).
- **Environment note**: a leaking IDE language-server process was holding ~10.7 GB of commit; freeing it restored ~5 GB of headroom. See section 6.

### [2026-10-08] - Phase 3 Instructions Written

- **Created** `phase3_instructions.md` from the implementation plan's Phase 3 (Diffusion & Generation Cost): pre-flight HF/disk check, `configs/base.yaml` generation block, `models/sd_pipeline.py`, `scripts/phase3_diffusion.py`, timing/VRAM sweep, JSON schema, and the gate question.

### [2026-10-08] - Phase 2 Complete

- **Added** 4 more secret images (`data/secret_images/test_002..005.png`, derived from `test_001.png`) to enable a 5-image spread measurement.
- **Wrote** `scripts/phase2_vae.py`: seeds, loads the VAE, round-trips every secret image, measures MSE/PSNR/SSIM, saves side-by-side comparisons, and reports the spread.
- **Verified gate**: 5/5 images pass. Latent `(1,4,64,64)`, output `(1,3,512,512)`; PSNR mean 32.809 dB (31.78–34.15), SSIM mean 0.9354.
- **Artifacts**: `experiments/results/phase2_test_00{1..5}_compare.png`.
- **Next**: Phase 3 (Diffusion & Generation Cost) — measure SD v1.5 generation time/VRAM (the baseline RACS must beat).

### [2026-10-08] - Phase 2 Core Modules Built

- **Wrote** `common/paths.py`: centralized paths + HuggingFace/diffusers/torch cache redirected to `d:\racs2\cache` (C: protected).
- **Wrote** `evaluation/metrics.py`: `mse`, `psnr`, `ssim` (+ `all_metrics`) on `[C,H,W]`/`[N,C,H,W]` tensors. Self-test passes.
- **Wrote** `models/vae.py`: `load_vae`, `encode`, `decode`, `roundtrip` for the SD v1.5 VAE (`stabilityai/sd-vae-ft-mse`, fp16 on GPU).
- **Verified**: VAE round-trip runs on GPU — input `(1,3,512,512)` → latent `(1,4,64,64)` → output `(1,3,512,512)`. VAE cached to D:.
- **Remaining for Phase 2**: `scripts/phase2_vae.py` (5-image run + metrics + side-by-sides), 4 more `data/secret_images/*.png`.

### [2026-10-08] - Phase 1 Complete + Phase 2 Instructions

- **Completed Phase 1**: PyTorch + CUDA verified, `common/seed.py`, `common/device.py`,
  `configs/base.yaml`, `scripts/phase1_verify.py` all written and passing (7/7 checks).
- **Created** `phase2_instructions.md`: step-by-step build guide for the VAE round-trip
  (deps, `common/paths.py`, `evaluation/metrics.py`, `models/vae.py`, `scripts/phase2_vae.py`).
- **Updated** this status file to reflect Phase 1 gate results and Phase 2 readiness.

### [2026-10-07] - Workspace Initialization & Cleanup

- **Action**: Cleaned workspace to remove legacy non-essential code/templates.
- **Preserved**: `phase1/`, `phase2/`, `research_papers/`, `docs/`.
- **Created**: `project_status.md` for live progress and change tracking.

---

## 6. Immediate Next Steps

- [x] **Step 1**: Install `diffusers` + `accelerate` (HF cache on D:). *(already present)*
- [x] **Step 2**: Write `common/paths.py` (redirect HuggingFace cache to D:).
- [x] **Step 3**: Write `evaluation/metrics.py` (`mse`, `psnr`, `ssim`).
- [x] **Step 4**: Write `models/vae.py` (load SD v1.5 VAE, fp16 on GPU).
- [x] **Step 5**: Add 4 more images to `data/secret_images/` and write `scripts/phase2_vae.py`.
- [x] **Step 6**: Run, confirm the Phase 2 checklist (latent `[1,4,64,64]`), move to Phase 3.
- [x] **Step 7 (Phase 3)**: Write `phase3_instructions.md` (SD v1.5 generation + timing/VRAM experiment).
- [x] **Step 8 (Phase 3)**: Build `models/sd_pipeline.py` + `scripts/phase3_diffusion.py`, run the timing sweep.
- [x] **Step 9 (Phase 4)**: Write `phase4_instructions.md` (retrieval image database).
- [x] **Step 10 (Phase 4)**: Build `retrieval/schema.py`, `retrieval/corpus.py`, `scripts/phase4_build_corpus.py`, `scripts/phase4_verify.py`; build + verify the 1,000-image corpus.
- [x] **Step 11 (Phase 5)**: Write `phase5_instructions.md` (CLIP embeddings, artifacts, gate).
- [x] **Step 12 (Phase 5)**: Build `models/clip_encoder.py`, `scripts/phase5_clip.py`, `scripts/phase5_verify.py`; add the `embedding:` config block; validate the full pipeline with a synthetic encoder.
- [x] **Step 13 (Phase 5)**: Download the CLIP ViT-B/32 checkpoint, run `scripts/phase5_clip.py` and `scripts/phase5_verify.py`. Outputs: 1,000 × 512 float32 vectors, verification PASS.

**Phase 3 gate question** — *Why is diffusion generation expensive?* Generation runs
SD's U-Net `N` times, once per denoising step, and twice per step when CFG is on
(measured: 1.76x). Measured here: 50 steps × 0.42 s/step = **~21 s** (13 s on an idle
machine) and **2.61 GB** peak VRAM for one 512×512 reference image, versus a
retrieval query that should cost a fraction of a second. That ~13–21 s gap is the
efficiency argument for RACS.

**Machine hygiene (important on this box)**: the model load needs ~2 GB of free
*commit charge*. Keep an eye on it with:

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
```

If it is under ~2 GB, close/restart memory-heavy apps (an IDE language server can
hold 10 GB) before running anything that loads a diffusion model. `common/hf_lowmem.py`
removes the 3.2 GB checkpoint-mmap requirement, but weights still need to be
materialised once.

See `phase3_instructions.md`, `phase4_instructions.md`, and `phase5_instructions.md` for
the historical task detail. **Phases 1-5 are complete and verified.** The execution order
changed on 2026-10-23: see `racs2_implementation_plan.md` v2.

Active next steps (new numbering):

- [x] **Phase 6**: Complete. Follow `phase6_instructions.md`. `lpips`, `scipy` and
  `scikit-image` installed; metrics extended; two role-tagged splits built (22/class each);
  `evaluation/splits.py`, `evaluation/report.py`, `scripts/eval_report.py` with loader role
  enforcement; `docs/DEVIATIONS.md` created. Selftest 83/83 PASS.
- [x] **Phase 7**: Complete. `Kpub` + `Xfeat` cached for both roles (132 records),
  `phase7_verify.json` PASS, grid inspected.
- [x] **Phase 8**: Complete. SD v1.5 + ControlNet at 256 px, 132 deterministic references,
  `phase8_verify.json` PASS. PicX-Real/IPAdapter deviations recorded.
- [x] **Phase 9**: Complete. 50-step bf16 zero-conv fine-tune, probe improved, `phase9_verify.json` PASS.
- [x] **Phase 10**: Complete. 12-cell EDICT sweep + 20/20 operating point, 3.58e-3 relative
  error, `phase10_verify.json` PASS with the numerical-floor deviation recorded.
- [x] **Phase 11**: Complete. Local hide/recover with eta = 0.05 and a manual reference;
  `phase11_verify.json` 12/12 PASS.
- [x] **Phase 12-14**: Complete. AWGN (`7/7`), semantic codec (`9/9`), and the documented
  CodeFormer pass-through fallback (`13/13`).
- [x] **Phase 15 (GATE)**: Complete. Assemble AgentSemSteCom. 15A: approximate reproduction on the
  UniStega calibration split (wide tolerance). 15B: authoritative baseline on our evaluation
  split. Calibration numbers never appear in any claim table. `phase15_verify.json` PASS.
- [x] **Phase 16**: Complete. Baseline security and steganalysis frozen; `phase16_verify.json`
  17/17 PASS. Report: `docs/phase16_report.md`.
- [ ] **Phase 17-20**: FAISS, token selection, structural reranking (consumes the banked
  Phase 4-5 artifacts).
- [ ] **Phase 21 (GATE)**: RACS-Base, `Xref_tx == Xref_rx` on 100 percent of images.
- [ ] **Phase 22**: Controlled AgentSemSteCom vs RACS-Base comparison (primary result).

Nothing RACS-specific is wired into the pipeline before Phase 15 passes.
