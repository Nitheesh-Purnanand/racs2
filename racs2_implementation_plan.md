# RACS2 — Learning-Oriented Implementation Plan (v2, baseline-first)

> **Revision v2, 2026-10-23.** The execution order has been inverted: **AgentSemSteCom is
> reproduced and validated first**, and only then is the RACS contribution built on top of
> the frozen baseline. Phases 1-5 are already complete and carry over unchanged. Every phase
> after 5 is renumbered. v1 remains in git history (commit `4af625a`).
>
> **Why the change.** v1 built RACS-only components (Phases 4-9) and deferred the baseline to
> a single over-scoped Phase 16. That breaks the scientific claim in
> `docs/PROJECT_UNDERSTANDING.md` ("everything downstream is unchanged, so any measured
> difference is attributable to reference acquisition"). Without a validated baseline there is
> no control, no debugging signal, and no anchor for the RACS numbers.

> **Principle**: Learn one thing, build a small version, test it, understand the output,
> integrate it, move on.
> **Anti-pattern**: Do NOT try to build the whole system at once.
> **Constraints**: RTX 3050 (6 GB VRAM), 16 GB RAM, 26 GB free disk, Windows, Python 3.10.11.

---

## 1. The ordering rule

```
1. Reproduce AgentSemSteCom  ->  land on the paper's numbers  ->  freeze it
2. Swap exactly one stage     ->  retrieval instead of diffusion  ->  RACS-Base
3. Compare both under identical conditions
```

The baseline is the control condition, not an afterthought. It must exist, be measured, and
be trusted before the contribution is built. Concretely:

- **Nothing RACS-specific is integrated before Phase 15 passes.** That means no FAISS
  retrieval feeding EDICT, no reranking, no token selection in the pipeline. Those components
  may exist on disk (they already do), but they are not wired in.
- **The shared downstream is built once and used by both systems.** VAE, EDICT, token
  perturbation, AWGN, semantic codec and metrics are identical in both paths. Only the
  reference acquisition differs.
- **Every deviation is logged.** If the paper cannot be reproduced exactly (missing weights,
  VRAM limits, substitute codec), it goes in `docs/DEVIATIONS.md` with the reason and the
  measured impact.

---

## 2. How This Plan Is Organized

The project is split into **10 Milestones (M1-M10)** containing **33 Phases**. Each phase is a
self-contained mini-project:

1. **Learn the theory** for that component.
2. **Build a tiny experiment** that demonstrates the concept in isolation.
3. **Test and measure**, saving artifacts to `experiments/results/`.
4. **Answer the gate question** in your own words.
5. **Integrate** into the growing pipeline.
6. **Report** in `project_status.md`.

Every phase ends with a concrete **gate**. Gates are numeric where the paper gives numbers.

**Status legend**: `DONE` complete and verified, `BANKED` built and verified but consumed in a
later phase, `NEXT` the immediate next phase, `TODO` not started.

---

## 3. Current state (2026-10-23)

| Phase | Scope | Status |
| :-- | :-- | :-- |
| 1 | Environment and project skeleton | **DONE** |
| 2 | VAE and latent space | **DONE** |
| 3 | SD v1.5 loading and generation cost | **DONE** |
| 4 | Retrieval corpus (1,000 COCO images) | **BANKED** (used in Phase 17) |
| 5 | CLIP embeddings (1,000 x 512 float32) | **BANKED** (used in Phase 17) |
| 6 | Evaluation harness and class-balanced splits | **DONE** (75/75 selftest) |
| 7-16 | AgentSemSteCom reproduction | **TODO** |
| 17-22 | RACS retrieval, RACS-Base, comparison | **TODO** |
| 23-33 | Variants, evaluation, write-up | **TODO** |

Verified numbers so far:

- VAE round-trip: PSNR 32.809 dB, SSIM 0.9354 on 5 images.
- SD v1.5, 512x512, fp16: 0.26 s/step, 50 steps = 12.98 s, 2.60 GB peak VRAM, CFG 1.75x.
- Corpus: 1,000 images, 512x512 RGB, 429.3 MB, `corpus_version cc20af21...89aff`.
- CLIP ViT-B/32: 1,000 x 512 float32, 18.2 s, caption to image top-1 0.476, verify PASS.

---

## 4. Nothing built so far is wasted: the reuse map

Every existing artifact has a named consumer in the new order. This is the contract.

| Artifact (already built) | Built in | Consumed by (new phase) | How it is reused |
| :-- | :-- | :-- | :-- |
| `configs/base.yaml`, `common/seed.py`, `common/device.py`, `common/paths.py` | 1 | Every phase | Unchanged. Config gains baseline blocks. |
| `evaluation/metrics.py` (MSE, PSNR, SSIM) | 2 | Phase 6 and all measurement | Extended with LPIPS and distribution reporting, not replaced. |
| `models/vae.py` (SD VAE fp16) | 2 | Phases 11, 12, 21, baseline and RACS | Unchanged. Both systems share the same VAE. |
| `experiments/results/phase2_*_compare.png` | 2 | Phase 32 figures | Reused as the VAE error-floor evidence. |
| `models/sd_pipeline.py` (fp16 loader, DDIM, attention slicing) | 3 | Phase 8 (reference generator) | Extended with ControlNet and IPAdapter, not rewritten. |
| `common/hf_lowmem.py` (read-only safetensors mmap) | 3 | Phase 8, 10, 11 | Required. The baseline stack is larger than SD v1.5 alone, so the commit-charge fix matters more, not less. |
| `experiments/results/phase3_timings.json` | 3 | Phase 30 efficiency analysis | Retained, but relabelled as the **SD-only lower bound**, not the baseline cost. |
| `scripts/phase3_img2img.py` | 3 | Phase 24 lightweight editing | Img2img is the editing primitive. The strength sweep is a direct prior for edit strength 0.5. |
| `retrieval/schema.py`, `retrieval/corpus.py` | 4 | Phase 17 FAISS, Phase 21 RACS | Index schema and deterministic loader reused as the protocol resource. |
| `data/retrieval_images/` (1,000 COCO PNGs) | 4 | Phase 17 corpus, Phase 6 evaluation `common` class | Dual use: retrieval database for RACS and a disjoint dev subset for the evaluation `common` class. Never used for calibration. |
| `data/metadata/image_index.json` | 4 | Phase 17, Phase 21 | Row order and `corpus_version` provenance must match the FAISS index. |
| `models/clip_encoder.py` | 5 | Phase 17, Phase 20 | Unchanged. float32 stays. |
| `experiments/results/phase5_corpus_embeddings.npz` | 5 | Phase 17 | Direct input to `IndexFlatIP`. |
| `scripts/phase5_verify.py` | 5 | Phase 17, Phase 21 | Its determinism-check pattern becomes the `Xref_tx == Xref_rx` checker. |
| `scripts/phase5_fetch_model.py` (resumable SHA-verified download) | 5 | Phase 8 model fetch | Reused for PicX-Real, ControlNet, IPAdapter, OneFormer, OpenPose, CodeFormer weights. |

**What changes in interpretation, not in code**

- The Phase 3 timing number is no longer "the baseline cost". It is the floor. The paper's
  baseline adds ControlNet, IPAdapter-plus and an independently trained generator, so the real
  comparator is more expensive. This makes the RACS efficiency claim more conservative, not
  less.
- The corpus and CLIP work is no longer "step 1 of RACS". It is banked. It costs nothing while
  the baseline is built, and it is consumed at Phase 17.

---

## 5. Old phase to new phase map

Use this to translate existing scripts, notes and commit messages.

| Old | New | Component |
| :-- | :-- | :-- |
| 1 | 1 | Environment and skeleton |
| 2 | 2 | VAE and latent space |
| 3 | 3 | SD v1.5 and generation cost |
| 4 | 4 | Retrieval corpus (banked) |
| 5 | 5 | CLIP embeddings (banked) |
| 6 | 17 | FAISS retrieval |
| 7 | 18 | Deterministic token selection |
| 8 | 19 | Structural representation |
| 9 | 20 | Structure-aware reranking |
| 10 | 10 | EDICT |
| 11 | 11 | Local steganography core |
| 12 | 21 | RACS-Base local integration |
| 13 | 12 | AWGN channel |
| 14 | 13 | Semantic codec |
| 15 | 21 | Full RACS-Base |
| 16 | 22 | Baseline comparison |
| 17 | 23 | VLM reranking |
| 18 | 24 | Lightweight editing |
| 19 | 25 | Retrieve-or-generate hybrid |
| 20 | 26 | Security testing |
| 21 | 27 | Steganalysis |
| 22 | 28 | Full evaluation matrix |
| 23 | 29 | Ablations |
| 24 | 30 | Efficiency analysis |
| 25 | 31 | Research analysis |
| 26 | 32 | Final tables and figures |
| 27 | 33 | Reproducibility package |
| (new) | 6-9, 14-16 | Baseline reproduction work that v1 never scheduled |

---

## 6. The baseline reproduction contract

Before Phase 15 is declared passed, all of the following must hold.

**Dataset policy: two roles, never mixed**

The paper's dataset (UniStega) is used **only as a calibration reference**, to check that the
reimplementation lands approximately on the published numbers. It is never the evaluation
dataset for any claim, never the RACS retrieval corpus, and never merged into the frozen
evaluation split.

- **Calibration set** (`data/calibration/unistega/`, read-only): a 100-image, three-class
  subset drawn from UniStega-Content, UniStega-Similar (face subset) and UniStega-Style,
  matching the paper's class definitions. Purpose: approximate reproduction only.
- **Evaluation set** (`data/test_set/`, authoritative): 100 images, three classes, from
  openly licensed sources. This is the only dataset used for baseline-versus-RACS claims,
  for the security matrix, for ablations, and for the headline tables.
- Both sets are frozen by hash before Phase 7, disjoint from `data/retrieval_images/`, and
  disjoint from each other. Tokens are fixed per image in both.

**Why approximate only.** UniStega gating, licensing and exact splits cannot be reconstructed
from the paper, and several modules (PicX-Real, ControlNet zero-conv, MDJCM) may be
substituted. A calibrated reimplementation can match the direction and rough magnitude of
Table II, but it cannot honestly claim exact reproduction. Calibration is a sanity anchor,
not a target to overfit.

**"Approximate" means**: the same qualitative ordering across classes and SNRs, monotone
improvement as SNR rises, and metrics inside the widened tolerance below. It does not mean
matching every cell.

**Inputs**

- Calibration: 100 images, 20+ per class, from UniStega. Evaluation: 100 images, 20+ per
  class, from our own sources. Both three-class.
- Four SNR points: 5, 10, 15, 20 dB.
- Fixed digital tokens per image in both sets, stored in `data/metadata/tokens.json` so Tx
  and Rx use the same values.

**Parameters** (from the paper, do not tune to hit numbers)

- SD v1.5 latent space in both directions, EDICT steps 50, EDICT mixing coefficient 0.93,
  perturbation strength eta = 0.05, channel coefficient h = 1.
- Reference generator: PicX-Real, or SD v1.5 + ControlNet + IPAdapter-plus if PicX-Real
  weights are unavailable.
- Implicit feature: segmentation for common, pose-to-skeleton for facial, none for style
  (style is conditioned on the public key only).
- Analog codec: SwinJSCC-style, C = 48, trained SNR 10 dB. Digital codec: MDJCM, DIV2K,
  modulation 64.
- Optional enhancement: CodeFormer at fidelity 0.5, facial class only.

**Calibration targets** (paper Table II and Section VI; approximate check only)

| Condition | Paper value |
| :-- | :-- |
| facial, 5 dB | PSNR 22.73 dB, SSIM 0.6989 |
| facial, 20 dB | PSNR 26.29 dB, LPIPS 0.2127 |
| style, 5 dB | MSE 0.721e-2, LPIPS 0.3850 |
| common, 5 dB | PSNR 19.71 dB, MSE 1.207e-2 |
| common, 10 dB | SSIM 0.5643 |

**Calibration tolerance (approximate, deliberately wide)**

- PSNR within +/- 2.5 dB of the paper value.
- SSIM within +/- 0.07.
- LPIPS within +/- 0.05.
- MSE within +/- 40 percent relative.
- Monotone improvement from 5 dB to 20 dB in every class.
- Class ordering preserved: facial at least as good as common on PSNR at 5 dB and 20 dB,
  as the paper reports.
- At least 90 percent of calibration images recover with PSNR above 18 dB at 10 dB SNR.

A calibration miss outside these bounds does not automatically fail the phase. It requires a
written diagnosis in `docs/DEVIATIONS.md` and, if the direction or ordering is wrong, a fix.
The authoritative numbers are always the evaluation-set numbers from Phase 15B.

---

## Milestone M1 — Foundation (Phases 1-2)

*Time estimate: 2-3 days. Status: complete.*

### Phase 1: Environment and Project Skeleton — **DONE**

> **What**: A working Python environment with PyTorch and GPU, a clean project structure, and
> a shared config.
> **Why**: Nothing else works without this. Reproducibility starts here.

**Theory**: virtual environments, tensors and dtypes, CUDA availability and memory accounting,
model checkpoints (`safetensors`, `.bin`), random seeds and determinism.

**Build tasks**: project skeleton, `common/seed.py`, `common/device.py`, `common/paths.py`,
`configs/base.yaml`, a first GPU tensor experiment.

**Gate**: Image tensors (B x C x H x W), CPU versus GPU, checkpoints, and seed determinism are
explained, and a script runs on the GPU and prints shapes.

**Verified**: 7/7 checks, PyTorch 2.6.0+cu124, RTX 3050 6 GB.

---

### Phase 2: Image Tensors, VAE and Latent Space — **DONE**

> **What**: Encode an image into Stable Diffusion's latent space and decode it back.
> **Why**: EDICT and every diffusion stage operate in latent space. This round-trip error is
> the floor that both the baseline and RACS inherit.

**Theory**: VAE encoder and decoder, 512x512x3 to 64x64x4 (8x spatial downscale, 4 channels),
normalisation to [-1, 1], MSE, PSNR, SSIM, and why reconstruction is lossy.

**Build tasks**: install `diffusers` and `accelerate`, load the SD VAE in fp16, run the
round-trip on 5 images, implement metrics, save side-by-side comparisons.

**Gate**: Round-trip measured on more than one image, with an explanation of latent space, the
compression factor, and the source of loss.

**Verified**: PSNR 32.809 dB (31.78-34.15), SSIM 0.9354, latent (1, 4, 64, 64).

**Reuse**: `models/vae.py` is used unchanged by the baseline (Phase 11 onward) and by RACS.

---

## Milestone M2 — Diffusion and Generation Cost (Phase 3)

*Time estimate: 1-2 days. Status: complete.*

### Phase 3: SD v1.5 Loading and Generation Cost — **DONE**

> **What**: Run Stable Diffusion, measure what it costs, and confirm the exact same weights
> can be loaded again inside the larger baseline stack.
> **Why**: v1 framed this as "the number RACS must beat". In v2 it is the **floor**: the
> paper's baseline adds ControlNet and IPAdapter-plus on top, so the true comparator costs
> more. Keeping this measurement is still useful, as a lower bound and as the loader that
> Phase 8 extends.

**Theory**: forward and reverse diffusion, timesteps and schedules, DDIM, classifier-free
guidance, and the cost model `time ~ steps x U-Net cost`.

**Build tasks**: load SD v1.5 fp16 with DDIM and attention slicing, sweep steps and seeds,
measure wall-clock and peak VRAM, save `phase3_timings.json`.

**Gate**: Generation cost explained from measured numbers, reference images saved.

**Verified**: 0.26 s/step, 12.98 s at 50 steps, 2.60 GB peak, CFG 1.75x.

**Reuse**: `models/sd_pipeline.py` and `common/hf_lowmem.py` are the starting point for the
Phase 8 reference generator. `phase3_timings.json` becomes the "generation only" row in
Phase 30. `scripts/phase3_img2img.py` seeds Phase 24.

---

## Milestone M3 — Banked Retrieval Inputs (Phases 4-5)

*Time estimate: already spent. Status: complete but not wired in.*

These two phases are finished. They are **not** on the critical path to the baseline. They are
parked here so the work is preserved and clearly accounted for.

### Phase 4: Retrieval Image Database — **BANKED**

> **What**: A versioned, deterministic corpus of 1,000 images.
> **Why**: This is the RACS reference source. It is shared protocol state, so versioning and
> hashes matter.

**Verified**: 1,000 COCO val2017 images, canonical 512x512 RGB PNGs, 429.3 MB,
`schema_version=1`, per-image `sha256`, `corpus_version cc20af21...89aff`, loader stable
across loads, verify PASS.

**Reuse**: Phase 17 builds the FAISS index directly on this corpus and its index file. Phase 6
may also draw its `common`-class secret images from the same pool for development, while the
frozen test split stays disjoint from the retrieval corpus.

---

### Phase 5: CLIP Embeddings — **BANKED**

> **What**: 1,000 x 512 float32 unit-norm CLIP ViT-B/32 vectors aligned row-for-row with the
> corpus index.
> **Why**: Semantic similarity for retrieval. CLIP captures semantics, not pose, layout or
> geometry, which is why structural reranking exists in Phase 20.

**Verified**: 18.2 s for 1,000 images, re-encode max difference 2.53e-07, caption to image
top-1 0.476, within-topic 0.779 versus across-topic 0.542, verify PASS. float32 is mandatory:
fp16 varied by about 7e-4 across batch sizes and would break Tx/Rx identity.

**Reuse**: Phase 17 loads `phase5_corpus_embeddings.npz` and builds `IndexFlatIP` with
`image_ids` from `image_index.json`. Phase 20 reuses `models/clip_encoder.py`.

---

## Milestone M4 — Measurement and Test Set (Phase 6)

*Time estimate: 2-3 days.*

### Phase 6: Evaluation Harness and Class-Balanced Test Set — **NEXT**

> **What**: The measurement rig and the frozen test split that every later phase reports on.
> **Why**: You cannot "land on the paper's numbers" without the paper's metrics, classes and
> SNRs. Building this before the baseline avoids retrofitting numbers later, and it is also
> exactly what RACS needs, so it is not baseline-only work.
>
> **Full build detail**: `phase6_instructions.md`.

**Theory**: LPIPS and why perceptual distance differs from PSNR and SSIM, distribution
reporting versus single-image anecdotes, class-conditional evaluation, and frozen test splits.

**Build tasks**

1. Install `lpips`, `scipy`, `scikit-image`. Confirm they import inside the venv.
2. Extend `evaluation/metrics.py` with `lpips` (AlexNet or VGG backbone, state which), and a
   `report(metric_dict)` helper that emits mean, median, std, min, max and N per
   (class, SNR) cell.
3. Build two dataset roles, never mixed:
   - `data/calibration/unistega/` (paper dataset, read-only, **approximate check only**):
     `common` = UniStega-Content, `facial` = UniStega-Similar face subset, `style` =
     UniStega-Style. 100 images, 20+ per class.
   - `data/test_set/` (our evaluation set, **authoritative**): 100 images, 20+ per class.
     `common` from the COCO pool but disjoint from `data/retrieval_images/`; `facial` and
     `style` from openly licensed sources. Do NOT draw this set from UniStega.
4. Freeze both in `data/metadata/test_split.json` with `role` = `calibration` or
   `evaluation`, sha256 per file, and a `split_version`. Calibration, evaluation and the
   retrieval corpus are pairwise disjoint.
5. Create `data/metadata/tokens.json`: one fixed digital token per image in both splits,
   versioned.
6. Write `scripts/phase6_eval_selftest.py`: metrics on identical images return PSNR infinity
   or a sentinel, LPIPS near 0, and the report shape is correct.
7. Write `scripts/eval_report.py`: a reusable driver that takes a system name and a split
   role, runs the SNR sweep, and writes `experiments/results/<system>_<class>_<snr>.json`.
   The loader must reject a `calibration` image when `role=evaluation`.

**Gate**: Metrics verified against known cases (identical, slightly noisy, structurally
different), both splits frozen and reproducible by hash, and the report driver produces a
well-formed table for a dummy system. Each split has at least 20 images per class. The
`role` field is enforced by the loader, and a test proves that a calibration image cannot be
loaded under `role=evaluation`. The evaluation set contains no UniStega images.

**Reuse**: `evaluation/metrics.py` from Phase 2 is extended, not replaced. `eval_report.py` is
used by Phase 15 (baseline), Phase 21 (RACS) and Phase 22 (comparison), so the comparison is
literally the same code path.

---

## Milestone M5 — AgentSemSteCom Reproduction (Phases 7-16)

*Time estimate: 12-16 days. This is the baseline. Nothing RACS-specific is wired in until it
passes.*

### Phase 7: Public Conditions (Kpub and Xfeat)

> **What**: The public semantic key and the implicit structural feature.
> **Why**: These are the paper's conditioning inputs. They are public in the paper's threat
> model, and both Tx and Rx must derive identical values. They also become RACS inputs later,
> so this work is shared.

**Theory**: LLM prompting as a semantic key generator, why the key carries no reversible
information, semantic segmentation versus pose skeletons as structural constraints, and the
difference between public and private state.

**Build tasks**

1. `semantic/keygen.py`: generate `Kpub` from a secret image. Note `Kpub` is a prompt sequence,
   not a caption of the secret verbatim. Options in order of preference: a local VLM or LLM,
   then a deterministic template plus CLIP-based content words. Whatever is chosen must be
   deterministic for a fixed image and recorded in the manifest.
2. `semantic/features.py`:
   - common: segmentation map (OneFormer or SegFormer). Store as a class-index map.
   - facial: detect and extract pose keypoints, then render a skeleton graph image.
   - style: return `None`, since style is conditioned on `Kpub` only.
3. Cache artefacts to `experiments/results/public_conditions/<image_id>.npz` plus a manifest.
4. Verify determinism: two runs on the same image give bit-identical `Kpub` and `Xfeat`.

**Gate**: For all three classes, `Kpub` and `Xfeat` are produced, cached, deterministic, and
visually inspected. A facial skeleton and a common-class segmentation map are shown next to
their source images.

**Reuse**: Phase 17 reuses `Kpub` and `Xfeat` as the RACS query. `semantic/features.py` also
provides the edge and segmentation tools used by Phase 19.

---

### Phase 8: Reference Generator (PicX-Real / SD + ControlNet + IPAdapter-plus)

> **What**: The paper's reference acquisition stage: deterministic noise from the digital
> token, DDIM sampling guided by `Kpub` and `Xfeat`, ControlNet structure injection, and
> IPAdapter-plus image guidance.
> **Why**: This is the single stage RACS replaces. It must exist and be deterministic before
> anything can be compared.

**Theory**: ControlNet side branches and zero convolutions, IPAdapter as a decoupled
cross-attention image prompt, deterministic noise from a hashed token, DDIM inversion and
sampling order, and why a training step is involved.

**Build tasks**

1. Fetch weights with `scripts/phase5_fetch_model.py` (resumable, SHA-verified): PicX-Real if
   available, otherwise SD v1.5 plus a segmentation-conditioned ControlNet and IPAdapter-plus.
2. Implement `reference/generate.py`:
   ```
   t -> seed = H(token) -> zD ~ N(0, I)
   zref = DDIM(zD, noise_pred, Kpub, Xfeat, T, 0)
   Xref = VAE.decode(zref)
   ```
3. Fit the 6 GB budget: fp16, `enable_attention_slicing`, `enable_vae_slicing`, sequential CPU
   offload, and keep `hf_lowmem` installed before `diffusers`/`transformers`.
4. Log peak VRAM, reference generation latency, and the seed derivation.
5. Determinism test: the same token and image must produce a bit-identical `Xref`.

**Gate**: References generated for all three classes, same token gives a bit-identical
`Xref` (hash equality across two runs), peak VRAM measured and inside budget, and the
generated reference is visually plausible for its class. If PicX-Real is unavailable, the
substitute is recorded in `docs/DEVIATIONS.md` with the measured visual and metric impact.

---

### Phase 9: ControlNet Zero-Conv Fine-Tuning (Paper Algorithm 2)

> **What**: Train the ControlNet side branch (and its zero convolutions) on the diffusion
> objective with the backbone frozen.
> **Why**: The paper trains this branch. A stock ControlNet is a different conditioning
> distribution and is a likely source of a systematic gap in Phase 15.

**Theory**: the denoising objective, zero-initialised convolutions, why freezing the backbone
preserves generative priors, and how to tell whether a fine-tune helped or just moved noise.

**Build tasks**

1. Implement `reference/train_controlnet.py` following Algorithm 2: sample `t`, add noise,
   compute the controlled block output, predict noise, backpropagate into the zero-conv
   parameters only.
2. Train a small run first (a few hundred steps) and verify the loss decreases and outputs
   change in the intended direction.
3. Scale up only if VRAM permits. Record batch size, steps, learning rate, and the exact
   checkpoint hash.
4. Compare zero-shot ControlNet versus fine-tuned on a 10-image probe using PSNR and SSIM of
   the recovered secret at 10 dB.
5. If a pretrained structure-conditioned ControlNet already matches the probe within
   tolerance, record that as a deviation and skip the full fine-tune.

**Gate**: Either a fine-tuned checkpoint exists with a documented improvement over zero-shot on
the probe, or a documented deviation shows the pretrained branch is sufficient. In both cases
the chosen checkpoint, its hash, and the decision are recorded.

---

### Phase 10: EDICT as an Isolated Mini-Project

> **What**: Exact Diffusion Inversion via Coupled Transformations, learned standalone.
> **Why**: This is the reversible hiding and recovery mechanism. It is the highest-risk
> unknown in the whole project, so it is proven before retrieval, reranking, or comparison.

> **Caution**: Do NOT integrate retrieval here. This phase is EDICT alone.

**Theory**: diffusion inversion, DDIM inversion drift, affine coupling of two latent
trajectories, the mixing coefficient, why exactness is possible, and where the numerical
floor actually is.

**Build tasks**

1. Implement `steganography/edict.py` following the paper: forward
   `latent -> EDICT forward -> transformed latent`, inverse the other way.
2. Round-trip test over 20 random latents plus 20 real image latents:
   `latent -> forward -> inverse -> latent`.
3. Report max absolute and relative error, and the recovery error after VAE decoding.
4. Sweep steps in {10, 20, 50} and mixing coefficient in {0.5, 0.7, 0.93, 0.99}.
5. Confirm 50 steps with 0.93 is a sensible operating point before adopting it.

**Gate**: Latent reconstruction error is at float32 machine precision scale (document the
exact value, expect about 1e-6 or better relative), and the choice of steps and mixing
coefficient is justified with the sweep.

---

### Phase 11: Local Steganography Core (VAE + EDICT + Token Perturbation)

> **What**: The complete hide and recover mechanism with a manually chosen reference, no
> channel, no retrieval.
> **Why**: If this does not work, nothing downstream can. It isolates the steganographic core
> from the reference-acquisition question.

**Build tasks**

1. `steganography/hide.py` and `steganography/recover.py`:
   ```
   Secret + Reference -> VAE encode -> EDICT forward -> token perturbation (eta = 0.05) -> Stego
   Stego -> VAE encode -> EDICT inverse -> inverse perturbation -> Recovered secret
   ```
2. Implement the binary mask from the token using the same PRNG derivation as Phase 8.
3. Use a manually chosen reference that is semantically close to the secret, so failures are
   not confused with bad reference selection.
4. Measure MSE, PSNR, SSIM and LPIPS between secret and recovered secret on 20 images.
5. Include a reference-sensitivity probe: swap to a mismatched reference and record the drop.
6. Do NOT add AWGN, codec, VLM or editing.

**Gate**: Local hide and recover works on 20 images, metrics are reported as distributions,
and the mismatched-reference probe shows a clear degradation. The eta = 0.05 operating point is
reproduced from the paper.

---

### Phase 12: AWGN Channel

> **What**: Add realistic channel noise between stego transmission and recovery.
> **Why**: The paper reports results over SNR 5 to 20 dB. This is the first point where the
> pipeline spans a channel.

**Theory**: SNR in dB, complex Gaussian noise, channel coefficient `h = 1`, and how noise power
follows from a target SNR.

**Build tasks**

1. `communication/awgn.py`: `received = h * signal + n`, `n ~ CN(0, sigma^2)`, with `sigma^2`
   derived from the target SNR and the measured signal power.
2. Run the Phase 11 pipeline at 5, 10, 15 and 20 dB.
3. Confirm the noise seed is controlled so runs are reproducible.
4. Report per-SNR distributions, not single images.

**Gate**: Recovery degrades monotonically as SNR falls, with the same noise seed reproducing
identical outputs. The SNR implementation is checked against the analytic noise power.

---

### Phase 13: Semantic Communication Codec

> **What**: Replace raw pixel transmission with the paper's JSCC codec.
> **Why**: The paper's numbers are measured through this codec. Comparing against it without
> the codec would be an apples-to-oranges baseline.

**Build tasks**

1. Obtain the paper's codec path: analog SwinJSCC-style with C = 48 trained at 10 dB, and the
   digital MDJCM trained on DIV2K with modulation 64. Use pretrained weights, do not train
   from scratch.
2. `communication/semantic_codec.py`: `stego -> encoder -> channel -> decoder -> stego_hat`.
3. Verify the codec in isolation on clean images: report reconstruction PSNR and LPIPS at each
   SNR before wiring it into the full pipeline.
4. Compare raw AWGN versus codec-plus-AWGN at the same SNR so the codec's contribution is
   separable.
5. If a specific pretrained checkpoint is unobtainable, substitute the closest available JSCC
   implementation and log it in `docs/DEVIATIONS.md`.

**Gate**: The codec runs end to end, its standalone degradation curve is measured, and the
codec-plus-channel path is measurably different from raw AWGN in the expected direction.

---

### Phase 14: Optional Task-Oriented Enhancement (CodeFormer)

> **What**: The paper's optional enhancement module for the facial class.
> **Why**: Fig. 8 in the paper reports a facial-class gain from this module. Without it, that
> result cannot be compared. It is optional for the main Table II path.

**Build tasks**

1. Load CodeFormer with fidelity weight 0.5.
2. Apply it after recovery, facial class only.
3. Measure the delta in PSNR, SSIM and LPIPS, with and without enhancement, at 5, 10, 15 and
   20 dB.
4. Confirm it does not degrade the non-facial classes (it should not be applied there).

**Gate**: The facial-class gain is reproduced in direction and roughly in magnitude, and the
module is cleanly switchable by config.

---

### Phase 15: Assemble AgentSemSteCom and Reproduce Table II — **BASELINE GATE**

> **What**: The full baseline, end to end, in the paper's order:
> ```
> Secret -> Kpub + Xfeat -> token-seeded diffusion + ControlNet + IPAdapter-plus -> Xref
>        -> VAE encode -> EDICT forward -> token perturbation -> Stego
>        -> semantic encode -> AWGN -> semantic decode -> Stego_hat
>        -> EDICT inverse -> inverse perturbation -> Recovered secret
> ```
> **Why**: This is the control condition. Everything after this phase is a comparison against
> this.
> **Rule**: Nothing RACS-specific is wired in here. If a retrieval module appears in this
> pipeline, the gate fails.

**Build tasks**

1. `pipelines/agentsemstecom.py`: one function that takes a secret image, class and SNR and
   returns the recovered image plus all intermediates.
2. `scripts/phase15_baseline.py` with a `--split calibration|evaluation` switch.
3. **15A, approximate reproduction**: run the calibration split, 3 classes x 4 SNRs, and
   write `experiments/results/phase15A_calibration_table.json`. Every entry is labelled
   `"approximate": true`.
4. **15B, authoritative baseline**: run the evaluation split through the identical code and
   write `experiments/results/baseline_table.json`. This is the frozen baseline that Phase
   22 compares against.
5. Manifests for both runs: model hashes, config hash, `split_version`, `role`, token file
   hash, and code commit.
6. Visual grids for both, prefixed `calibration_` and `baseline_`.
7. Fill `docs/DEVIATIONS.md` for anything that departs from the paper.

**Gate, part A: calibration (approximate only)**

- Determinism: the same image, token and seed reproduce a bit-identical `Xref` and a
  bit-identical stego image.
- Approximate agreement with the paper, deliberately wide:

| Cell | Paper value | Approximate tolerance |
| :-- | :-- | :-- |
| facial 5 dB | PSNR 22.73, SSIM 0.6989 | +/- 2.5 dB, +/- 0.07 |
| facial 20 dB | PSNR 26.29, LPIPS 0.2127 | +/- 2.5 dB, +/- 0.05 |
| style 5 dB | MSE 0.721e-2, LPIPS 0.3850 | +/- 40 percent, +/- 0.05 |
| common 5 dB | PSNR 19.71, MSE 1.207e-2 | +/- 2.5 dB, +/- 40 percent |
| common 10 dB | SSIM 0.5643 | +/- 0.07 |

- Monotone improvement from 5 dB to 20 dB in every class.
- Class ordering preserved (facial at least as good as common on PSNR at 5 and 20 dB).
- Any miss has a written diagnosis in `docs/DEVIATIONS.md`.

**Gate, part B: authoritative baseline (our evaluation split)**

- The evaluation split is used, and the loader rejects any `calibration` image.
- Determinism as above.
- Metrics reported as distributions (mean, median, std, N) per class and SNR cell.
- No paper tolerance applies here, because no paper numbers exist for this data. These are
  the numbers that Phase 22 uses.
- Artifacts are hashed and labelled `"approximate": false`.

> **Do not proceed to Phase 17 with an unexplained calibration gap or a non-deterministic
> evaluation baseline.** Calibration numbers never appear in Phase 22 or in any claim table.

---

### Phase 16: Baseline Security and Steganalysis

> **What**: The baseline's security profile: correct token versus wrong token versus no token,
> and the XuNet detection rate.
> **Why**: These are the paper's contribution numbers. They must be baseline numbers before
> they can be RACS numbers.

**Build tasks**

1. Three receiver conditions on the same stego images, from the **evaluation split only**:
   correct token, wrong token, no token.
2. Report PSNR, SSIM and LPIPS distributions for each, plus the correct-minus-wrong gap.
3. Treat the paper's 25.37 dB correct versus 18.91 dB wrong (gap 6.46 dB) as an
   **approximate anchor**, not an exact target, because the data differs.
4. Train or obtain XuNet on our evaluation stego set. The paper's 50.91 percent detection
   rate is an approximate anchor only.
5. Save `experiments/results/baseline_security.json` and `baseline_steganalysis.json`, both
   labelled `"approximate": false` for our data.

**Gate**: The correct-token advantage holds on our data (gap at least 4 dB), the wrong-token
and no-token conditions both underperform the correct one, and the XuNet detection rate is
reported with its training setup. Baseline results on the evaluation split are now frozen.
The paper's numbers are recorded beside ours as approximate anchors, clearly separated.

---

## Milestone M6 — RACS Retrieval Stack (Phases 17-20)

*Time estimate: 6-8 days. Now, and only now, the banked work is wired in.*

### Phase 17: FAISS Retrieval

> **What**: Exact inner-product vector search over the banked embeddings.
> **Why**: This is the retrieval engine that will stand in for diffusion generation.

**Theory**: vector search, cosine and inner product, `IndexFlatIP` versus approximate indexes,
and why exactness is required for Tx/Rx agreement.

**Build tasks**

1. Install `faiss-cpu`.
2. `retrieval/build_index.py`: load `phase5_corpus_embeddings.npz`, assert row order matches
   `image_index.json`, build `IndexFlatIP`, save the index plus `image_ids` and
   `corpus_version`.
3. `retrieval/retrieve.py`: `retrieve(query, top_k=10)` returning ordered `(image_id, score)`.
4. Sweep K in {1, 5, 10, 20, 50} and log candidates and scores.
5. Save `experiments/results/phase17_index_manifest.json` with the embedding hash, index hash,
   corpus version and K sweep.

**Gate**: Index row order matches the corpus exactly, retrieval is deterministic across
process restarts, top-1 self-query returns the query image at score 1.0, and the K sweep is
logged. Query to Top-K works for an image query and for a text query.

**Reuse**: This is the Phase 4 and Phase 5 payoff. No corpus or embedding work is redone.

---

### Phase 18: Deterministic Token Selection

> **What**: Use the digital token to select exactly one candidate from Top-K.
> **Why**: Tx and Rx must select the same reference. The token provides reproducibility and
> security.

**Theory**: deterministic hashing, `index = H(token) mod K`, why the token must not drive a
nondeterministic model in the selection path, and the 1/K guessing bound.

**Build tasks**

1. `retrieval/token_selection.py`: stable hash (SHA-256 based, not Python's randomised
   `hash()`), input is the ordered candidate list and the token.
2. Test determinism across processes and Python versions.
3. Test that distinct tokens spread across the pool, and measure the distribution for K = 1,
   5, 10, 20, 50.
4. Test that K = 1 always selects the same candidate.
5. Confirm the same selection function is used by Tx and Rx code paths.

**Gate**: Same token and candidates produce the same selection across processes, different
tokens spread measurably, K = 1 is stable, and the guessing probability is 1/K by
construction. Do not use the built-in `hash()`.

---

### Phase 19: Structural Representation

> **What**: Structural features that measure arrangement rather than semantics.
> **Why**: CLIP finds the right content but can return the wrong pose, layout or geometry.
> Structure is the reranking signal.

**Theory**: semantic segmentation, edge detection (Canny, HED), depth (MiDaS), pose landmarks,
and the difference between semantic similarity and structural compatibility.

**Build tasks**

1. Start with edges or segmentation, whichever is simpler to validate.
2. Extract features for a probe set of 50 images.
3. Find pairs with high CLIP similarity but low structural similarity and show them.
4. Define the structural metric: IoU for segmentation, MSE or correlation for edges.
5. Reuse the Phase 7 feature extractors where they apply.

**Gate**: A working structural representation exists, and at least three concrete examples
show CLIP saying "similar" while structure says "different", with numbers.

---

### Phase 20: Structure-Aware Reranking

> **What**: Composite semantic plus structural ranking of the Top-K candidates.
> **Why**: Pure CLIP retrieval is the baseline design, not the final one.

**Build tasks**

1. `retrieval/rerank.py`: `score = alpha * semantic + beta * structural`, with both terms
   normalised to a comparable range.
2. Sweep alpha and beta on a held-out probe. Do not tune on the test split.
3. A/B comparison: CLIP-only versus CLIP plus structure. Report rank changes and the resulting
   reference quality from Phase 11's local pipeline.
4. Freeze the chosen weights and record them in config and in the manifest.
5. Do NOT add a VLM here.

**Gate**: Reranking measurably changes the candidate pool, the chosen weights are frozen and
justified, and there is at least one case where it visibly helps. If it does not help, report
that honestly as a negative result; it is a valid finding.

---

## Milestone M7 — RACS-Base and the Comparison (Phases 21-22)

*Time estimate: 3-5 days.*

### Phase 21: RACS-Base End-to-End

> **What**: The full pipeline with exactly one stage swapped:
> ```
> Secret -> Kpub + Xfeat -> CLIP -> FAISS -> rerank -> token select -> Xref
>        -> [identical downstream as Phase 15]
> ```
> **Why**: This is the contribution. Because the downstream is byte-identical to the baseline,
> the difference is attributable to reference acquisition.

**Build tasks**

1. `pipelines/racs_base.py`, importing the same downstream modules as Phase 15. Any
   duplication of the downstream code is a bug.
2. Run the same frozen **evaluation** split, 3 classes, 4 SNRs, same tokens. Never the
   calibration split.
3. **Before looking at PSNR**, verify `Xref_tx == Xref_rx` bit-exactly for every image:
   - recompute the receiver-side reference from the same query, index, ranking and token;
   - compare SHA-256 of the tensors and of the saved PNG.
4. Measure reference acquisition latency and peak VRAM, and keep the baseline values from
   Phase 15 for comparison.
5. Save `experiments/results/racs_table.json` and `racs_manifest.json`.

**Gate**: `Xref_tx == Xref_rx` for 100 percent of images across all classes and SNRs, the
downstream code is provably the same module as the baseline, and quality metrics are measured
as distributions. If the invariant fails on any single image, stop and fix it. Nothing else
matters until it holds.

---

### Phase 22: Controlled Comparison — **PRIMARY RESULT**

> **What**: AgentSemSteCom versus RACS-Base under identical conditions.
> **Why**: This is the paper-style result: same secrets, same classes, same SNRs, same
> metrics, same codec, same tokens, same downstream.

**Build tasks**

1. `scripts/phase22_compare.py`: load both frozen result sets and emit the comparison table.
   Both must carry `role = evaluation`. The script must refuse to run on calibration data.
2. Table rows: system, class, SNR, PSNR, SSIM, MSE, LPIPS, reference latency, total latency,
   peak VRAM.
3. Security comparison: correct versus wrong token gap for both systems.
4. Visual grids: baseline versus RACS recovered images at 5 and 20 dB.
5. State the efficiency claim only if the measured latency difference supports it.
6. Save `experiments/results/phase22_comparison.json` and `.md`.

**Dataset rule**: this comparison runs on our evaluation split only. The paper's numbers may
appear in a separate, explicitly labelled "paper reference (approximate)" column, but they
are never mixed into the measured rows and never used to compute a delta.

**Gate**: The comparison table exists, both systems were measured with the same code path,
`Xref_tx == Xref_rx` holds, and the conclusion is stated in terms of measured numbers,
including any metric where RACS is worse. A negative or mixed result is valid and must be
reported as such.

> **This phase is the minimum viable research result.** Everything after it strengthens the
> paper but does not change whether the core claim can be evaluated.

---

## Milestone M8 — Intelligent Variants (Phases 23-25)

*Time estimate: 5-7 days.*

### Phase 23: VLM and Agentic Reranking (RACS-Agent)

> **What**: A VLM scores candidates before deterministic ranking.
> **Why**: A VLM can judge layout and compatibility in ways CLIP plus edges cannot.

> **Important**: The VLM must produce fixed scores, then a deterministic ranking, then token
> selection. No nondeterministic cloud model in the Tx/Rx selection path.

**Build tasks**: score each candidate with a VLM, cache scores by image hash, rerank
deterministically, compare RACS-Base versus RACS-Agent on quality and latency.

**Gate**: RACS-Agent is measured against RACS-Base. Either it improves reference quality
enough to justify its cost, or it does not and that is reported.

---

### Phase 24: Lightweight Editing (RACS-AgentEdit)

> **What**: A light edit of a semantically good but structurally imperfect reference.
> **Why**: It may close the gap to generation at a fraction of the cost.

> **Warning**: Do not turn this into full diffusion generation. If the edit costs as much as
> generation, the project's premise is defeated.

**Build tasks**: reuse `scripts/phase3_img2img.py`, apply low-strength img2img (start from the
measured strength curve, around 0.25-0.5), record edit latency, compare quality and cost.

**Gate**: Quality gain versus added latency is measured, and the edit stays substantially
cheaper than a full generation.

---

### Phase 25: Retrieve-or-Generate Hybrid (RACS-Hybrid)

> **What**: Fall back to generation when retrieval quality is too low.
> **Why**: Best of both, if the threshold is deterministic on both ends.

**Build tasks**: define a deterministic quality threshold on the ranking score, implement the
fallback, verify the threshold decision is identical at Tx and Rx, compare all four variants.

**Gate**: The hybrid is compared against always-generate, always-retrieve and retrieve-plus-edit
on quality and cost, with a fixed documented threshold.

---

## Milestone M9 — Security, Evaluation, Ablations (Phases 26-30)

*Time estimate: 5-7 days.*

### Phase 26: Security Matrix

Test correct token, wrong token and no token for both systems on the evaluation split. Report
distributions and the correct-minus-wrong gap. Gate: distributions, not single images, and the
gap is stated alongside the paper's 6.46 dB as an approximate anchor only.

### Phase 27: RACS Steganalysis

Run XuNet and SRNet on RACS stego images, using the same training setup as Phase 16.
Non-blocking. Gate: detection accuracy reported and compared with the baseline's 50.91
percent.

### Phase 28: Full Evaluation Matrix

Systems x classes x SNR x token conditions, with quality, efficiency and security metrics.
Gate: a single consolidated matrix whose cells trace back to raw result JSON files.

### Phase 29: Ablations

CLIP only versus CLIP plus structure; Top-K in {1, 5, 10, 20, 50}; with and without token
selection; with and without VLM; with and without editing; always-retrieve versus hybrid.
Gate: each ablation has a measured delta and a stated confidence, with distributions.

### Phase 30: Efficiency Analysis

Measure `T_generation`, `T_retrieval`, `T_rerank`, `T_editing`, `T_total` and peak GPU memory,
for every system variant. Gate: the efficiency claim is supported or refuted by these numbers,
including the Phase 3 SD-only floor as a reference row.

---

## Milestone M10 — Research Output (Phases 31-33)

*Time estimate: 3-5 days.*

### Phase 31: Research Analysis

Answer the 12 final questions from the overview: quality, efficiency, security, robustness,
retrieval effectiveness, adaptive architecture. Gate: every answer cites measured artifacts.

### Phase 32: Final Tables and Figures

Quality table, efficiency table, security table, ablation table. Figures: architecture
comparison, latency bars, PSNR versus SNR curves, Top-K effect, token security. Reuse the
Phase 2 and Phase 3 PNGs as supporting evidence rather than regenerating them.

### Phase 33: Reproducibility Package

Freeze dataset version, database, FAISS index, embedding model, preprocessing, seeds, token
map, Top-K, ranking weights, editing parameters, EDICT parameters, SNRs and test set. One
command reproduces the headline table. Gate: a clean checkout plus the package reproduces
Phase 22 within tolerance.

---

## 7. Sprint path (revised)

| Sprint | Phases | What you get | Duration |
| :-- | :-- | :-- | :-- |
| A | 6-9 | Test rig, public conditions, deterministic reference generator | 5-7 days |
| B | 10-14 | EDICT, local hide and recover, channel, codec, enhancement | 7-10 days |
| C | 15-16 | **AgentSemSteCom reproduced and frozen** | 3-5 days |
| D | 17-20 | Retrieval, token selection, structural reranking | 6-8 days |
| E | 21-22 | **RACS-Base and the primary comparison** | 3-5 days |
| F | 23-25 | Variants | 5-7 days |
| G | 26-30 | Security, steganalysis, matrix, ablations, efficiency | 5-7 days |
| H | 31-33 | Write-up and reproducibility | 3-5 days |

**Minimum viable result is the end of Sprint E (Phase 22).** Sprints F-H strengthen the paper.

Total remaining estimate: **37-54 working days** for the full plan, with the first honest
paper-comparable number at the end of Sprint C.

---

## 8. Critical path and stop rules

Critical path: `6 -> 8 -> 10 -> 11 -> 13 -> 15 -> 17 -> 18 -> 21 -> 22`.

- Phase 10 (EDICT) is the highest technical risk. If exact inversion does not reach machine
  precision, stop and understand why before building anything else.
- Phase 15 (baseline numbers) is the highest scientific risk. If calibration drifts outside
  the wide tolerance, diagnose it; if the direction or class ordering is wrong, fix it. If the
  authoritative evaluation baseline is non-deterministic, stop. Options in order: fix the
  pipeline, substitute the closest documented component, or narrow the claim. Do not silently
  proceed.
- Calibration data must never leak into an evaluation run. If a `role=evaluation` result
  contains any image from `data/calibration/`, the phase is invalid.
- Phase 21 (`Xref_tx == Xref_rx`) is a hard invariant. A single mismatch invalidates the
  security and recovery results.
- Phase 6 is cheap and unblocks everything, so it comes next.

---

## 9. Environment and machine hygiene

- Windows, RTX 3050 Laptop 6 GB, 16 GB RAM. venv at `.venv`, PyTorch 2.6.0+cu124.
- HF, diffusers and torch caches are redirected to `D:\racs2\cache`; pip temp to
  `D:\racs2\temp`. C: is full.
- The commit-charge trap is still live. `common/hf_lowmem.py` must be imported before
  `diffusers` and `transformers`. The baseline stack is heavier than SD v1.5 alone, so check
  free commit before loading:
  ```powershell
  Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
  ```
- Expect to use sequential CPU offload for the Phase 8 stack. Record actual peak VRAM rather
  than assuming it fits.

---

## 10. Deviation register

`docs/DEVIATIONS.md` is created in Phase 6 and appended to in every phase that cannot match
the paper exactly. Each entry records: phase, paper component, what was used instead, why, the
measured impact on PSNR, SSIM, MSE and LPIPS, and whether the impact is acceptable.

Deviation entries are required for, at minimum: the reference generator if PicX-Real weights
are unavailable, the ControlNet branch if it is not fine-tuned, the semantic codec if a
substitute is used, the acquisition and composition of the UniStega calibration set, the
sourcing of the evaluation classes, and the public-key generator if a local model replaces
the paper's LLM setup.

---

## 11. What this plan deliberately does not do

- It does not build RACS components into the pipeline before the baseline is validated. They
  exist on disk and are tested in isolation, which is why the banked work is safe.
- It does not tune any parameter to make a number match. Parameters come from the paper;
  deviations are recorded.
- It does not claim an efficiency win before Phase 30 measures it.
- It does not discard any completed work: see Section 4 for the artifact-to-consumer map.
