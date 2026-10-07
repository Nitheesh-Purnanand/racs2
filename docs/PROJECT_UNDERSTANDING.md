# RACS — Project Understanding (working reference)

> Consolidated understanding of the source documents, for onboarding and as a
> shared reference. Source of truth remains
> `phase2/RACS_Final_Project_Overview_Compact.pdf`; plan/agreement in
> `docs/RACS_ROADMAP.md`; live status in `project_status.md`.

## 1. What RACS is

**RACS = Retrieval-Augmented Coverless Semantic Steganography.** It is a research
study built on the 2026 IEEE paper *"Secure Intellicise Wireless Network: Agentic
AI for Coverless Semantic Steganography Communication"* (Meng et al., TCCN 12(2026)).

It replaces exactly **one** stage of the baseline `AgentSemSteCom`:

| | Reference acquisition |
|---|---|
| AgentSemSteCom (baseline) | diffusion/DDIM generation of `Xref` (expensive) |
| RACS | deterministic multimodal **retrieval** (+ optional lightweight edit) |

Everything downstream (EDICT steganography, token perturbation, semantic
communication, channel, recovery) is kept **unchanged**, so any measured
difference is attributable to the reference-acquisition stage. RACS does not
assume retrieval is better; it is a quality/security/robustness/efficiency
comparison.

## 2. Baseline AgentSemSteCom (what RACS modifies)

Pipeline: `Secret Xs → Kpub (LLM) + Xfeat (structure) → diffusion/DDIM + ControlNet
+ IPAdapter-plus → Xref → EDICT + token perturbation → Xstego → JSCC semantic
codec → AWGN channel → semantic decoder → EDICT inversion + inverse perturbation
→ recovered secret`.

- `Kpub` = public semantic key (LLM-generated prompt describing the secret).
- `Xfeat` = public implicit structural feature (segmentation for common images;
  OpenPose skeleton for faces).
- Digital token does dual duty: seeds the reference generation noise **and**
  controls the latent perturbation mask.
- EDICT provides exact (invertible) diffusion so the hide/recover is reversible.

### Reported parameters (carry forward unless a phase says otherwise)
SD v1.5; EDICT steps 50; mixing p = 0.93; perturbation strength η = 0.05;
edit strength 0.5; IPAdapter-plus weight 1; AWGN SNR 5/10/15/20 dB; h = 1;
analog SemCom SwinJSCC, C = 48, trained SNR 10 dB; digital MDJCM, modulation 64;
CodeFormer fidelity 0.5.

### Reported results (baselines to compare against)
Facial 5 dB PSNR 22.73 / SSIM 0.6989; facial 20 dB PSNR 26.29 / LPIPS 0.2127;
style 5 dB MSE 0.721e-2 / LPIPS 0.3850; vs SemSteDiff (common, 5 dB) PSNR 19.71
vs 18.04, MSE 1.207e-2 vs 1.73e-2; at 10 dB SSIM 0.5643 vs 0.5183. Security:
correct-token PSNR 25.37 dB, wrong-token 18.91 dB (**gap 6.46 dB**). Steganalysis
(XuNet): AgentSemSteCom detection 50.91% (near chance) vs Baluja 95.12%,
ISN 56.23%, HiNet 55.71%.

## 3. RACS design

### Transmitter
```
Xs → Kpub + Xfeat → query → CLIP/VLM embedding → FAISS index → Top-K
   → semantic + structural reranking (score = α·semantic + β·structural [+ γ·steg])
   → valid candidate pool → digital token → deterministic selection
   → optional light editing → Xref → (unchanged EDICT + perturbation) → Xstego
   → (unchanged semantic comm / channel)
```
### Receiver
```
received → semantic decoder → reconstructed stego → SAME query/config
        → SAME db + embedding + FAISS + ranking + token → SAME Xref
        → EDICT inversion → inverse token perturbation → secret
```

- Token mapping must be fixed and deterministic, e.g. `index = H(token) mod K`.
  Never `token → arbitrary database image`; selection is from a quality-filtered pool.
- Database: initially ~50k–100k images (LAION subset) with `image_id`, image,
  embedding, caption/metadata, structural representation; smaller subsets for dev
  (1k → 5k → 10k). Indexing is offline; online is query + ANN + rerank + edit.
- Structural reranking motivation: CLIP captures semantics but not pose/layout/
  geometry/texture/steganographic suitability.
- VLM/agentic reranking (later) must score candidates then use a **fixed
  deterministic ranking**; no nondeterministic cloud LLM in the Tx/Rx path.

### Determinism contract (mandatory)
Tx and Rx must agree on: database version, preprocessing, embedding
model/checkpoint, FAISS index, ranking function/weights, Top-K, token mapping,
editing parameters, EDICT parameters, SNR, test set.
Invariant: **`Xref_tx == Xref_rx`** (if it fails, recovery can fail).

### System variants (frozen names)
`AgentSemSteCom` (generation) · `RACS-Base` (retrieval + rerank + token) ·
`RACS-Agent` (+VLM rerank) · `RACS-AgentEdit` (+light edit) · `RACS-Hybrid`
(retrieve-or-generate fallback).

### Hypotheses
H1 efficiency (lower latency/GPU/memory); H2 quality (acceptable PSNR/SSIM/MSE/
LPIPS); H3 security (correct- vs wrong-token gap preserved); H4 robustness
(usable across 5–20 dB).

### Success is not "beat the baseline everywhere"
A useful result may be: slightly lower PSNR, similar SSIM/LPIPS/security, but
substantially lower latency and memory. Negative results are valid and still
scientific (they identify what structural/agentic components are needed).

## 4. Phase plan and milestones

Canonical numbering (resolves the source plan's 21-vs-27 inconsistency) groups
27 phases into M1–M12:

| M | Phases | Theme | Gate highlight |
|---|---|---|---|
| M1 | 1–2 | Env, skeleton, tensors, VAE/latent | VAE round trip measured |
| M2 | 3 | Diffusion fundamentals + cost | explain cost with own numbers |
| M3 | 4–7 | DB, CLIP, FAISS, token selection | same token+list ⇒ same selection |
| M4 | 8–9 | Structural representation + reranking | reranking changes pool |
| M5 | 10 | EDICT isolated | small latent reconstruction error |
| M6 | 11 | Local steganography core | local hide/recover measured |
| M7 | 12 | RACS + EDICT local (RACS-Base) | end-to-end local recovery |
| M8 | 13–15 | AWGN, SemCom, Full RACS-Base | `Xref_tx == Xref_rx` |
| M9 | 16 | Baseline comparison | comparison table exists |
| M10 | 17–19 | VLM, editing, retrieve-or-generate | each variant vs RACS-Base |
| M11 | 20–24 | Security, steganalysis, eval, ablations, efficiency | distributions, not anecdotes |
| M12 | 25–27 | Analysis, tables/figures, reproducibility | every result tied to a config |

Per-phase loop: what/why → theory → tiny experiment → test → explain → integrate
→ save → next. Every phase ends with an artifact and a passing gate.

## 5. Standing decisions / open questions

Open (from roadmap §4): embedding model + checkpoint; FAISS index type (exact
`IndexFlatIP` recommended); Kpub determinism; DB size graduation; test-set
leakage rule; retrieve-vs-generate gating criterion; editing cost/determinism
cap; pre-registered success thresholds.

Environment constraints: RTX 3050 Laptop **6 GiB VRAM**; disk **D: ~44 GiB**,
**C: nearly full** (holds pagefile); host RAM tight (~3.4 GiB free).
