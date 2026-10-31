# RACS — Guided Build Roadmap & Working Agreement

This is the living plan for building RACS. It is a **learning plan first and an
implementation plan second**: the goal is that you understand and write the
project, not that it gets built for you.

Source documents:
- Spec (source of truth): `docs/papers/RACS_Final_Project_Overview_Compact.pdf`
- Baseline paper: `docs/papers/Secure_Intellicise_..._Communication.pdf`
- Original roadmap: `docs/papers/RACS_Implementation_Plan_Phase_by_Phase.pdf`

---

## 1. Working agreement

Roles:
- **You build.** All project code, commands, experiments, and decisions are yours.
- **I mentor.** I give a short theory brief, precise tasks, acceptance criteria,
  review your code and outputs, help debug, and keep the roadmap/contract current.

Rules:
1. No project code is written by me unless you explicitly ask for a specific
   snippet. Debugging means I point at the cause, you fix it.
2. One component at a time. Small version -> test -> understand -> integrate.
3. You should be able to *explain* a component before we integrate the next one.
4. Every phase ends with an artifact + a passing gate.
5. We keep a canonical contract (below) so Tx/Rx reproducibility is never
   ambiguous.

Per-phase loop (applies to every phase):
1. What we are building
2. Why RACS needs it
3. Theory to learn (with pointers)
4. You build a tiny experiment
5. You test it
6. You explain the output
7. We integrate it
8. We save the result
9. Next phase

Report-back format after each phase (keep it short):
- What I built / files added
- Command(s) I ran
- Observed output / numbers
- What surprised me / what is unclear
- My answer to the "explain" question for this phase

---

## 2. Canonical milestones and phases

The source implementation plan is internally inconsistent: its summary list has
21 phases, its body has 27, and numbering diverges after Phase 15. This roadmap
is the canonical numbering. Milestones group phases; gates are hard.

Environment facts (fixed for planning):
- Windows, Python 3.10.11, torch 2.6.0+cu124
- GPU RTX 3050 Laptop, **6 GB VRAM** (tight; fp16 + attention slicing expected)
- Disk D: **~29.7 GB free** (tight; plan corpus + checkpoint budget carefully)

### M1 Foundation — Phases 1-2
- Phase 1: environment, project skeleton, config, seeds, first GPU inference.
- Phase 2: image tensors, VAE encode/decode, latent vs pixel, round-trip MSE/PSNR/SSIM.
- Gate: you can explain tensors/shapes, CPU vs GPU, checkpoints, why seeds matter;
  VAE round-trip runs and is measured.

### M2 Diffusion — Phase 3
- Phase 3: forward/reverse diffusion, timestep, schedule, U-Net, conditioning,
  latent diffusion, DDIM, classifier-free guidance. Measure generation time + VRAM.
- Gate: you can explain why generation is expensive, with your own measured numbers.

### M3 Retrieval stack — Phases 4-7
- Phase 4: build retrieval image DB (grow 1k -> 5k -> 10k), metadata + embeddings.
- Phase 5: CLIP text and image embeddings; demonstrate semantic similarity.
- Phase 6: FAISS index build + retrieve Top-K; sweep K = 1/5/10/20/50.
- Phase 7: deterministic token selection `index = hash(token) % K`, fixed ordering.
- Gate: same token + same candidate list always yields the same selection;
  correct/wrong/different token produce expected selection changes.

### M4 Structural reranking — Phases 8-9
- Phase 8: structural representation (segmentation/edges/depth; pose/facial landmarks).
- Phase 9: score = alpha*semantic + beta*structural; compare CLIP-only vs CLIP+structure.
- Gate: structural reranking measurably changes the candidate pool vs CLIP-only.

### M5 EDICT — Phase 10
- Phase 10: EDICT as an isolated mini-project. latent -> forward -> inverse -> latent.
- Gate: latent reconstruction error is small; you can explain why EDICT enables
  reversible hiding/recovery.

### M6 Local steganography core — Phase 11
- Phase 11: VAE + EDICT + token perturbation, local only (no AWGN/JSCC/VLM/editing).
- Gate: local hide/recover works; measure MSE/PSNR/SSIM/LPIPS.

### M7 RACS + EDICT (RACS-Base local) — Phase 12
- Phase 12: retrieval -> rerank -> token select -> Xref -> EDICT -> stego -> inverse.
- Gate: end-to-end local recovery works with retrieved references.

### M8 Communication + Full RACS-Base — Phases 13-15
- Phase 13: AWGN, stepping 20 -> 15 -> 10 -> 5 dB.
- Phase 14: semantic communication (use pretrained implementations; do not train).
- Phase 15: Full RACS-Base.
- Gate (critical): **Xref_transmitter == Xref_receiver** before any PSNR is read.

### M9 Baseline comparison — Phase 16
- Phase 16: build/obtain faithful AgentSemSteCom baseline; same secrets, classes,
  SNR, metrics, downstream pipeline.
- Gate: quality + efficiency comparison table exists.

### M10 Intelligent variants — Phases 17-19
- Phase 17: VLM/agentic reranking (deterministic scoring; no cloud nondeterminism).
- Phase 18: lightweight editing (must not become full generation).
- Phase 19: retrieve-or-generate hybrid with a candidate-quality threshold.
- Gate: each variant compared against RACS-Base on quality and cost.

### M11 Security, steganalysis, experiments — Phases 20-24
- Phase 20: security (correct / wrong / no token) with proper test-set statistics.
- Phase 21: steganalysis (XuNet, optionally SRNet). Non-blocking.
- Phase 22: full evaluation matrix.
- Phase 23: ablations (retrieval, Top-K, token, VLM, editing, fallback).
- Phase 24: efficiency analysis (T_generation/T_retrieval/T_reranking/T_editing/T_total, VRAM).
- Gate: numbers support or refute each hypothesis, with distributions not anecdotes.

### M12 Analysis + packaging — Phases 25-27
- Phase 25: research analysis (the 12 questions).
- Phase 26: final tables and figures.
- Phase 27: reproducibility package (freeze every version/parameter).

---

## 3. Canonical names and contract

System names (frozen):
- `AgentSemSteCom` — diffusion-generation baseline.
- `RACS-Base` — retrieval (CLIP + structural rerank) + token selection + EDICT + comms.
- `RACS-Agent` — RACS-Base + VLM/agentic reranking.
- `RACS-AgentEdit` — RACS-Agent + lightweight editing.
- `RACS-Hybrid` — retrieve-or-generate fallback.

Determinism contract (must hold at transmitter and receiver):
database version, preprocessing, embedding model/checkpoint, FAISS index,
ranking function/weights, Top-K, token mapping, editing parameters,
EDICT parameters, SNR, test set. Invariant: `Xref_tx == Xref_rx`.

Parameter set (from the baseline paper, to be reused unless a phase says otherwise):
SD v1.5, EDICT steps 50, mixing p = 0.93, perturbation eta = 0.05,
edit strength 0.5 (reverse-step proportion; unrelated to RACS image editing),
IPAdapter-plus weight 1, awgn SNR 5/10/15/20 dB, h = 1, analog C = 48,
digital modulation 64.

---

## 4. Open decisions to resolve early (do not let these drift)

1. Embedding model + checkpoint (e.g. CLIP ViT-B/32 vs ViT-L/14 vs OpenCLIP).
2. FAISS index type: exact `IndexFlatIP` recommended for determinism.
3. Kpub determinism: greedy/seeded LLM decoding, or transmit Kpub explicitly.
4. DB size graduation: 1k -> 5k -> 10k -> target; final reported size.
5. Test-set leakage rule: evaluation secrets must be disjoint from retrieval DB,
   plus a near-duplicate check.
6. Retrieve-vs-generate gating criterion (deterministic, identical at Tx/Rx).
7. Editing cost/determinism cap (e.g. steps limit, seeded, same checkpoint).
8. Pre-registered success thresholds (acceptable quality change, minimum speedup).

---

## 5. Phase brief template

```
### Phase N - <name>
Goal:
Why RACS needs it:
Theory to learn:
Build task(s):
Acceptance gate:
Report back:
```
