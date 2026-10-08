# RACS2 — Complete Project Analysis

> **Status**: Analysis only — no files have been modified or created in the project workspace.
> **Date**: 2026-10-07
> **Files inspected**: Every file in `d:\racs2\` (15 files across 4 directories)

---

# 1. Executive Summary

**RACS** (Retrieval-Augmented Coverless Semantic Steganography) is a research project that proposes replacing the computationally expensive diffusion-based reference-image generation stage of the **AgentSemSteCom** system (Meng et al., IEEE TCCN 2026) with a deterministic multimodal **retrieval** pipeline. The project is a scientific comparison study — not an unconditional claim that retrieval is better.

The project is currently in a **documentation-only state**. Phase 1 produced planning documents (PDFs). Phase 2 produced detailed specification and implementation plan documents. Phase briefs for Phases 2–3 exist, with Phase 3 marked as blocked on Hugging Face connectivity. **No source code, scripts, tests, configuration files, or data exist anywhere in the repository.**

The system is designed to run on severely constrained hardware: **RTX 3050 Laptop (6 GB VRAM)**, limited disk (~29–44 GB on D:), and tight host RAM (~3.4 GB free). This shapes every design decision.

---

# 2. What RACS Is

**RACS = Retrieval-Augmented Coverless Semantic Steganography.**

It is a modification to exactly **one stage** of the AgentSemSteCom pipeline:

| System | Reference Acquisition Stage |
|---|---|
| **AgentSemSteCom** (baseline) | Diffusion/DDIM generation of `Xref` (expensive: ~50 denoising steps through U-Net) |
| **RACS** | Deterministic multimodal retrieval from a pre-built image database + optional lightweight editing |

Everything **downstream** of reference acquisition — EDICT steganography, token perturbation, semantic communication, channel, recovery — remains **unchanged**. This is a deliberate experimental design: any measured difference can be attributed solely to the reference-acquisition stage.

### One-sentence definition (from the project spec):
> RACS is a retrieval-augmented coverless semantic steganography framework that replaces the computationally expensive diffusion-based reference-image generation stage of AgentSemSteCom with deterministic multimodal retrieval and optional lightweight structural editing, while preserving the downstream EDICT-based hiding/recovery and semantic communication pipeline for a direct quality, security, robustness, and efficiency comparison.

---

# 3. Project Goals

### Primary Research Question
> Can retrieval-based reference-image acquisition provide sufficiently compatible semantic and structural conditioning for reversible coverless steganography while reducing the computational cost of reference generation?

### Four Hypotheses
| ID | Hypothesis | Measurement |
|---|---|---|
| **H1** | **Efficiency**: Retrieval reduces reference latency, GPU computation, memory | T_retrieval vs T_generation, peak VRAM |
| **H2** | **Quality**: Retrieved reference maintains acceptable PSNR/SSIM/MSE/LPIPS | Same metrics as baseline paper |
| **H3** | **Security**: Token-controlled selection preserves correct-token/wrong-token gap | Recovery gap ≥ ~6.46 dB |
| **H4** | **Robustness**: RACS remains usable at 5–20 dB SNR | Metrics across SNR range |

### Success Criteria
- RACS does **not** need to beat AgentSemSteCom on every metric
- A useful result could be: slightly lower PSNR, similar SSIM/LPIPS/security, **substantially lower latency and memory**
- **Negative results are explicitly declared valid** — they show what structural/agentic components are needed

---

# 4. Phase 1 Understanding

### What Phase 1 Produced
Two PDF documents in [phase1/](file:///d:/racs2/phase1):
1. **Project_Understanding_Document.md.pdf** (285 KB) — Understanding of the original AgentSemSteCom paper
2. **RACS_Research_Plan_Full.md.pdf** (148 KB) — Full research plan for RACS

### What Phase 1 Did NOT Produce
- No source code
- No environment setup
- No experiments
- No test results

### Assessment
Phase 1 was purely a documentation/comprehension phase. The PDFs are not readable by tools (binary PDF format), but their content has been distilled into the Phase 2 text documents and the `docs/` markdown files.

---

# 5. Phase 2 Understanding

### What Phase 2 Produced
Five documents in [phase2/](file:///d:/racs2/phase2):
1. **RACS_Final_Complete_Project_Overview.pdf** (251 KB) — The full project specification
2. **RACS_Final_Project_Overview_Compact.pdf** (177 KB) — Compact version (declared "source of truth")
3. **RACS_Implementation_Plan_Phase_by_Phase.pdf** (178 KB) — 27-phase implementation plan
4. **_overview.txt** (24 KB) — Text extraction of the compact overview
5. **_implplan.txt** (25 KB) — Text extraction of the implementation plan

### Additionally created in `docs/`:
- [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) — Consolidated understanding
- [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) — Canonical milestone/phase roadmap (resolves inconsistencies)
- [CONCURRENCY.md](file:///d:/racs2/docs/CONCURRENCY.md) — Concurrent session safety rules
- [phase_briefs/phase2.md](file:///d:/racs2/docs/phase_briefs/phase2.md) — Phase 2 build brief (VAE)
- [phase_briefs/phase3.md](file:///d:/racs2/docs/phase_briefs/phase3.md) — Phase 3 build brief (Diffusion, BLOCKED)

### What Phase 2 Did NOT Produce
- No source code (the phase briefs mention scripts like `scripts/phase3_diffusion.py` and modules like `common/paths.py` and `common/seed.py`, but these do not exist in the current workspace)
- No virtual environment
- No installed dependencies
- No experiments

### Key Phase 2 Decisions
1. **Canonical 27-phase structure** (resolving the 21-vs-27 inconsistency in the source plan)
2. **12 milestones** (M1–M12) grouping the 27 phases
3. **System variant names frozen**: `AgentSemSteCom`, `RACS-Base`, `RACS-Agent`, `RACS-AgentEdit`, `RACS-Hybrid`
4. **Working agreement**: Learner builds, mentor guides; one component at a time; every phase ends with artifact + gate
5. **Determinism contract** established as mandatory

> [!IMPORTANT]
> The phase briefs reference files (`scripts/phase3_diffusion.py`, `common/paths.py`, `common/seed.py`, `steganography/vae/roundtrip.py`) that **do not exist** in this workspace. This suggests either a separate workspace existed previously or these were planned but never created.

---

# 6. Complete Architecture

## 6.1 AgentSemSteCom (Baseline) Architecture

```
Secret Xs
  ↓
Semantic Extraction: Kpub = fLLM(Xs)  [LLM-generated public semantic key]
Structure Extraction: Xfeat = G(Xs, R) [Segmentation/OpenPose/none]
  ↓
Token → H(token) → seed → zD ~ N(0,I)  [Deterministic Gaussian noise]
  ↓
Diffusion/DDIM + ControlNet + IPAdapter-plus  [Conditioned on Kpub, Xfeat, zD]
  ↓
VAE Decoder → Xref  [Reference image]
  ↓
VAE Encode(Xs) → zs → EDICT Forward → zT
  ↓
Token Perturbation: z'T = fn(zT)  [Binary mask controlled by token]
  ↓
EDICT Reverse (conditioned on Kpub, Xref) → zstego
  ↓
VAE Decode → Xstego  [Stego image]
  ↓
JSCC Semantic Encoder → S → AWGN Channel → S' → Semantic Decoder → X̂stego
  ↓
[Receiver: reverse the process with same Kpub, Xref, token]
  ↓
Recovered Secret X̂s
```

## 6.2 RACS Architecture

```
Secret Xs
  ↓
Semantic Extraction → Kpub
Structure Extraction → Xfeat
  ↓
Retrieval Query (from Kpub + Xfeat)
  ↓
CLIP/VLM Embedding
  ↓
FAISS Index → Top-K Candidates
  ↓
Semantic + Structural Reranking: Score = α·Ssemantic + β·Sstructural [+ γ·Ssteg]
  ↓
Valid Candidate Pool
  ↓
Digital Token → index = H(token) mod K → Deterministic Selection
  ↓
[Optional] Lightweight Editing
  ↓
Xref
  ↓
[UNCHANGED from here: EDICT + perturbation → Xstego → SemCom → Channel → Recovery]
```

## 6.3 Receiver Side (RACS)

```
Received semantic representation
  ↓
Semantic Decoder → Reconstructed stego
  ↓
SAME query/configuration
SAME database + embedding + FAISS
SAME ranking
Correct token
  ↓
SAME Xref  [Critical invariant: Xref_tx == Xref_rx]
  ↓
EDICT Inversion → Inverse token perturbation → Recovered Secret
```

---

# 7. Component-by-Component Explanation

## 7.1 Semantic Extraction Module
- **Input**: Secret image `Xs`
- **Output**: Public semantic key `Kpub` (text prompt describing content)
- **Method**: LLM invocation `Kpub = fLLM(Xs)`
- **RACS change**: None — carried forward unchanged
- **Concern**: LLM output must be **deterministic** at Tx and Rx (greedy/seeded decoding, or transmit `Kpub` explicitly)

## 7.2 Structural Feature Extraction Module
- **Input**: Secret image `Xs`, task requirements `R`
- **Output**: Implicit feature `Xfeat`
- **Method**: OneFormer (common images → segmentation), OpenPose (facial → skeleton), none (style)
- **RACS change**: None — carried forward unchanged
- **Role in RACS**: Used as a second signal for structural reranking

## 7.3 Reference Acquisition Module [**THE CHANGE**]

### Baseline (Generation)
- Diffusion/DDIM with ControlNet + IPAdapter-plus
- Token seeds the Gaussian noise initialization
- 50 EDICT steps, expensive U-Net evaluations
- ~2.5 GB VRAM, significant wall-clock time

### RACS (Retrieval)
- **Image Database**: ~50k-100k images (LAION subset); dev progression: 1k→5k→10k
  - Each image has: `image_id`, image, CLIP/VLM embedding, caption, structural representation
- **CLIP Embedding**: Text/image → vector; semantic similarity in embedding space
- **FAISS Index**: Approximate nearest-neighbor search; `IndexFlatIP` recommended for determinism
- **Structural Reranking**: `Score(I) = α·Ssemantic(I) + β·Sstructural(I)`
- **Token Selection**: `index = H(token) mod K` from quality-filtered candidate pool
- **Optional Editing**: SDEdit-style partial diffusion, lightweight image-to-image adjustment

## 7.4 EDICT Steganography Core
- **Purpose**: Exact invertible diffusion for hiding secret in stego
- **Method**: Coupled latent variables `(zt, ut)` with affine coupling layers
- **Key equations**: Forward noising (Eq. 11), Denoising diffusion (Eq. 12)
- **Mixing coefficient**: `p = 0.93`
- **Steps**: 50
- **RACS change**: None

## 7.5 Token Perturbation
- **Purpose**: Dual-stage security — seeds reference generation AND controls latent perturbation
- **Method**: Binary mask `M ∈ {0,1}^{C×H×W}`, perturbation strength `η = 0.05`
- **Inversion**: `z'T = zT ⊙ (1 - M(s,η)) - zT ⊙ M(s,η)` — sign-flip at selected positions
- **RACS change**: Token no longer seeds diffusion noise; instead controls candidate selection from pool. Perturbation role **unchanged**.

## 7.6 Semantic Communication
- **Analog**: SwinJSCC, C=48, trained SNR 10 dB
- **Digital**: MDJCM, DIV2K, modulation 64, SNR 10 dB
- **Channel**: AWGN, h=1, SNR 5/10/15/20 dB
- **RACS change**: None — use pretrained implementations

## 7.7 Task-Oriented Enhancement (Optional)
- **Method**: CodeFormer for facial enhancement, fidelity weight 0.5
- **RACS change**: None

---

# 8. End-to-End Data/Control Flow

## Complete RACS-Base Transmitter Flow
```
1. INPUT: Secret image Xs (512×512×3)
2. SEMANTIC EXTRACTION: Kpub = fLLM(Xs) → text prompt
3. STRUCTURE EXTRACTION: Xfeat = G(Xs, R) → segmentation/skeleton
4. QUERY CONSTRUCTION: Combine Kpub + Xfeat into retrieval query
5. EMBEDDING: CLIP text/image encoder → query embedding vector
6. RETRIEVAL: FAISS(query_embedding, top_k=K) → K candidate images
7. STRUCTURAL RERANKING: Score each candidate → ordered pool
8. TOKEN SELECTION: index = H(token) mod K → select one image
9. [OPTIONAL EDIT]: Light structural editing → Xref
10. VAE ENCODE SECRET: zs = EVAE(Xs) → latent (64×64×4)
11. EDICT FORWARD: (z0,u0) → ... → (zT,uT) over 50 steps
12. TOKEN PERTURBATION: z'T = fn(zT) using binary mask
13. EDICT REVERSE (conditioned on Kpub, Xref): z'T → zstego
14. VAE DECODE: Xstego = DVAE(zstego) → stego image (512×512×3)
15. SEMANTIC ENCODE: S = Esem(Xstego) → channel representation
16. TRANSMIT: S → AWGN channel → S'
```

## Complete RACS-Base Receiver Flow
```
1. RECEIVE: S' (corrupted semantic representation)
2. SEMANTIC DECODE: X̂stego = Dsem(S')
3. RETRIEVE SAME Xref: Same query → same CLIP → same FAISS → same ranking → same token → same Xref
   [CRITICAL: Xref_tx == Xref_rx — if this fails, everything fails]
4. VAE ENCODE STEGO: ẑstego = EVAE(X̂stego)
5. EDICT FORWARD (conditioned on Kpub, Xref): ẑstego → ẑ'T
6. INVERSE PERTURBATION: ẑT = fn⁻¹(ẑ'T) using same token/mask
7. EDICT REVERSE (unconditioned): ẑT → ẑ0
8. VAE DECODE: X̂s = DVAE(ẑ0) → recovered secret
9. [OPTIONAL ENHANCEMENT]: X̂*s = H(X̂s | R)
```

---

# 9. Research Paper Analysis

## Paper 1: "Secure Intellicise Wireless Network: Agentic AI for Coverless Semantic Steganography Communication"
**Source**: [Secure_Intellicise...pdf](file:///d:/racs2/research_papers/Secure_Intellicise_Wireless_Network_Agentic_AI_for_Coverless_Semantic_Steganography_Communication.pdf) + [_paper.txt](file:///d:/racs2/research_papers/_paper.txt)
**Authors**: Meng et al., IEEE TCCN vol. 12, 2026
**Role in RACS**: **The baseline system** — everything RACS modifies and compares against

### Problem Addressed
Semantic Communication (SemCom) faces semantic eavesdropping threats. Prior solutions like SemSteDiff use diffusion models but depend on private semantic keys that can be inferred. AgentSemSteCom eliminates private keys by introducing a digital token + agentic AI.

### Methodology
1. LLM-based public semantic key generation (replaces private keys)
2. Adaptive implicit feature extraction (segmentation/OpenPose)
3. Digital token–controlled deterministic diffusion noise initialization
4. EDICT-based exact invertible diffusion for steganography
5. Decoupled cross-attention for text + image guidance
6. Token-based latent perturbation for security
7. JSCC semantic communication over AWGN channels

### Key Technical Concepts
- **Coverless steganography**: No cover image needed; stego is generated
- **EDICT coupled transformations**: Mathematically exact inversion via dual latent chains (z,u)
- **Digital token dual role**: Seeds reference generation + controls perturbation mask
- **Decoupled cross-attention**: Separate attention for text (Kpub) and image (Xref) features

### Results
| Setting | Metric | Value |
|---|---|---|
| Facial, 5 dB | PSNR | 22.73 dB |
| Facial, 5 dB | SSIM | 0.6989 |
| Facial, 20 dB | PSNR | 26.29 dB |
| Common, 5 dB | PSNR | 19.71 (vs SemSteDiff 18.04) |
| Security gap | Correct vs Wrong token | 25.37 vs 18.91 = **6.46 dB gap** |
| XuNet detection | AgentSemSteCom | 50.91% (near chance) |

### Assumptions
- Key management center provides secure channel for token distribution
- Eavesdropper may have the semantic decoder but not the correct token
- AWGN channel model with h=1

### Limitations
- Tested on small dataset (UniStega: 100 images total)
- Reference generation is computationally expensive (the motivation for RACS)
- Common images show lower quality due to less precise structural features (segmentation vs skeleton)

### What RACS Uses
- Entire downstream pipeline (EDICT, perturbation, SemCom, recovery)
- All reported parameters
- All evaluation metrics and baselines
- Token perturbation mechanism

### What RACS Changes
- Only the reference acquisition stage (diffusion → retrieval)
- Token role: from noise seed → candidate selection index

---

## Paper 2: "Re-Imagen: Retrieval-Augmented Text-to-Image Generator"
**Source**: [2209.14491v3.pdf](file:///d:/racs2/research_papers/2209.14491v3.pdf) (arXiv 2209.14491)
**Role in RACS**: **Conceptual inspiration** for retrieval-augmented image generation

### Problem Addressed
Text-to-image generation struggles with rare/unseen entities. Re-Imagen retrieves relevant images from an external database and conditions generation on them.

### Key Concepts Relevant to RACS
- **Retrieval-augmented generation**: Use retrieved images as conditioning
- **CLIP-based retrieval**: Semantic similarity in embedding space
- **Database as shared resource**: Offline indexing, online retrieval
- **Multimodal conditioning**: Combining text and retrieved image features

### What RACS Borrows
- The fundamental idea of using retrieval instead of pure generation
- CLIP-based semantic embedding for retrieval
- Database design principles

### What RACS Does NOT Use
- Re-Imagen still generates images (retrieval augments generation)
- RACS uses retrieval to **replace** generation entirely (in base variant)
- Re-Imagen does not deal with steganography, security, or determinism

### How It Influences Architecture
- Validates the feasibility of retrieval-based reference acquisition
- Suggests CLIP + FAISS as the retrieval mechanism
- Motivates the RACS-Hybrid variant (retrieve-or-generate)

---

## Paper 3: EDICT (Exact Diffusion Inversion via Coupled Transformations)
**Source**: Referenced as [59] in the main paper; arXiv 2211.12446 (Wallace et al., CVPR 2023)
**Note**: The EDICT paper is **not directly included** in the repository but its full mathematical details are reproduced in Section V of the main paper.

### Key Concepts
- Dual coupled latent sequences `(zt, ut)` with affine coupling layers
- Mixing layer: `z't = p·zt + (1-p)·ut` synchronizes chains
- Forward noising (Eq. 11): deterministic, exact
- Denoising diffusion (Eq. 12): conditional, exact inverse
- No approximation errors (unlike DDIM inversion)

### Role in RACS
- EDICT is the core steganographic mechanism — it is NOT modified by RACS
- Its reversibility enables hiding and recovering secrets
- It operates in VAE latent space
- 50 steps, mixing coefficient p=0.93

---

# 10. Research → RACS Mapping

| Research Paper | Relevant Concept | RACS Component | How It Is Used |
|---|---|---|---|
| AgentSemSteCom (Meng 2026) | Public semantic key | Semantic Extraction | Carried forward unchanged; generates `Kpub` |
| AgentSemSteCom | Implicit structural features | Structure Extraction | Used for reranking; segmentation/OpenPose |
| AgentSemSteCom | Digital token (dual role) | Token Selection + Perturbation | Selection role changed: noise seed → `H(token) mod K`; perturbation unchanged |
| AgentSemSteCom | EDICT coupled diffusion | Steganography Core | Carried forward completely unchanged |
| AgentSemSteCom | SwinJSCC/MDJCM | Semantic Communication | Carried forward unchanged |
| AgentSemSteCom | CodeFormer | Task Enhancement | Optional, unchanged |
| AgentSemSteCom | Baseline metrics | Evaluation | Target values for comparison |
| Re-Imagen (2209.14491) | Retrieval-augmented generation | CLIP+FAISS Retrieval | Core RACS idea: retrieve instead of generate |
| Re-Imagen | CLIP embedding for retrieval | Embedding Module | Text/image → vector for ANN search |
| Re-Imagen | Database conditioning | Image Database | Offline indexing, online query |
| EDICT (Wallace 2023) | Exact diffusion inversion | EDICT Module | Unchanged — provides reversible steganography |
| EDICT | Coupled transformations | Forward/Reverse Process | Mathematical foundation for hiding/recovery |

---

# 11. Requirements

## MUST HAVE (Functional)
1. **FR-01**: Secret image → RACS pipeline → Xref → EDICT → Xstego → recovery → secret (end-to-end)
2. **FR-02**: `Xref_tx == Xref_rx` (determinism invariant)
3. **FR-03**: CLIP embedding of query → FAISS Top-K retrieval
4. **FR-04**: Structural reranking with configurable α, β weights
5. **FR-05**: Deterministic token selection: `index = H(token) mod K`
6. **FR-06**: EDICT forward/reverse with p=0.93, 50 steps
7. **FR-07**: Token perturbation with η=0.05
8. **FR-08**: AWGN channel simulation at 5/10/15/20 dB
9. **FR-09**: Quality metrics: PSNR, SSIM, MSE, LPIPS
10. **FR-10**: Faithful AgentSemSteCom baseline for comparison
11. **FR-11**: Correct token / wrong token / no token security test
12. **FR-12**: Efficiency metrics: T_generation, T_retrieval, T_reranking, peak VRAM

## MUST HAVE (Non-Functional)
1. **NFR-01**: Run on RTX 3050 6 GB VRAM (fp16, attention slicing)
2. **NFR-02**: Fit within D: ~29-44 GB disk
3. **NFR-03**: Reproducible: fixed seeds, deterministic pipeline
4. **NFR-04**: Every result tied to a configuration

## SHOULD HAVE
1. **FR-13**: VLM/agentic reranking (RACS-Agent variant)
2. **FR-14**: Lightweight editing (RACS-AgentEdit variant)
3. **FR-15**: Retrieve-or-generate hybrid (RACS-Hybrid variant)
4. **FR-16**: Steganalysis with XuNet
5. **FR-17**: Ablation studies (retrieval, Top-K, token, VLM, editing, fallback)
6. **FR-18**: Three image classes: common, facial, style

## NICE TO HAVE
1. **FR-19**: SRNet steganalysis
2. **FR-20**: Energy/CPU/GPU utilization measurement
3. **FR-21**: FLOPs estimation
4. **FR-22**: Steganographic suitability scoring (γ term)

## UNKNOWN / NEEDS CLARIFICATION
1. **UNK-01**: Which CLIP checkpoint? (ViT-B/32 vs ViT-L/14 vs OpenCLIP) — impacts VRAM and quality
2. **UNK-02**: How is Kpub made deterministic? (greedy decoding? transmitted explicitly?)
3. **UNK-03**: Final database size and source (LAION subset — which subset?)
4. **UNK-04**: Test-set leakage prevention: how to ensure evaluation secrets are disjoint from DB
5. **UNK-05**: Retrieve-vs-generate gating criterion (threshold for "poor quality")
6. **UNK-06**: Editing cost/determinism cap
7. **UNK-07**: Pre-registered success thresholds (what PSNR drop is "acceptable"?)
8. **UNK-08**: How to obtain/reproduce SwinJSCC and MDJCM pretrained models
9. **UNK-09**: Where does the UniStega dataset come from? (100 images; licensing)
10. **UNK-10**: IPAdapter-plus model and its determinism in the RACS context

---

# 12. Technical Design

## 12.1 Data Structures
- **Image Database**: Per-image record: `{image_id, image_path, clip_embedding, caption, structural_representation}`
- **FAISS Index**: `IndexFlatIP` (exact inner product search) for determinism
- **Configuration**: Single config with `image_size, device, seed, database_path, index_path, top_k, token, snr`
- **Latent tensors**: `64×64×4` (SD v1.5 VAE, 8× spatial compression, 4 latent channels)
- **Coupled EDICT states**: `(zt, ut)` pairs, both `64×64×4`
- **Binary perturbation mask**: `M ∈ {0,1}^{C×H×W}` where C=4, H=64, W=64

## 12.2 Algorithms
- **CLIP embedding**: ViT-based encoder → 512/768-d vector
- **FAISS ANN**: Inner product search for cosine similarity
- **Structural reranking**: `Score(I) = α·cos(clip_query, clip_I) + β·structural_sim(Xfeat_secret, Xfeat_I)`
- **Token hashing**: `index = H(token) mod K` where `H` is a deterministic hash function
- **EDICT forward** (Eq. 11 in paper): 4-step per timestep (unmix, invert coupling, noise predict × 2)
- **EDICT reverse** (Eq. 12 in paper): 4-step per timestep (coupling, mix, noise predict × 2)
- **Token perturbation**: Sign-flip at masked positions: `z'T[i] = -zT[i]` where `M[i]=1`

## 12.3 Protocols
- **Determinism contract**: Tx and Rx must agree on ALL of: database version, preprocessing, embedding model/checkpoint, FAISS index, ranking function/weights, Top-K, token mapping, editing parameters, EDICT parameters, SNR, test set
- **No nondeterministic cloud LLM** in the critical Tx/Rx path

## 12.4 Security Mechanisms
- **Token provides selection security**: `P(correct reference | no token) ≈ 1/K`, entropy ≈ `log2(K)` bits
- **Token provides perturbation security**: Wrong mask → EDICT inversion collapses
- **Coverless design**: No cover image → steganalysis cannot compare cover vs stego
- **Near-chance XuNet detection**: 50.91% (baseline) — RACS must preserve this

## 12.5 Networking
- AWGN channel: `Ŝ = h·S + n`, where `n ~ CN(0, σ²w)`, `h = 1`
- SNR range: 5, 10, 15, 20 dB
- SwinJSCC for analog, MDJCM for digital

## 12.6 Storage
- Model checkpoints: SD v1.5 (~2 GB fp16), CLIP (varies), ControlNet, IPAdapter
- Image database: 1k–100k images (storage TBD, ~5-50 GB)
- FAISS index: relatively small for <100k entries
- Experiment results: JSON + images

## 12.7 Concurrency
- Multiple agent sessions possible ([CONCURRENCY.md](file:///d:/racs2/docs/CONCURRENCY.md))
- Git-based conflict avoidance with narrow staging
- CPU tests as pre-commit verification

## 12.8 Performance/Scalability
- All constrained by 6 GB VRAM
- fp16 + attention slicing mandatory
- Database indexing is offline (one-time cost)
- Online retrieval: embedding + FAISS search + reranking (should be fast)
- Target: retrieval significantly faster than 50-step diffusion generation

---

# 13. Current Implementation Status

| Component | Status | Evidence |
|---|---|---|
| Project documentation | ✅ Complete | 15 files, well-structured |
| Research understanding | ✅ Complete | Detailed understanding docs |
| Phase roadmap | ✅ Complete | 27 phases, 12 milestones defined |
| Phase briefs | 🟡 Partial | Only Phases 2-3 written |
| Source code | ❌ Missing | Zero `.py`, `.json`, `.yaml` files |
| Virtual environment | ❌ Missing | No `.venv`, `requirements.txt`, `pyproject.toml` |
| Project skeleton | ❌ Missing | No directory structure |
| Configuration | ❌ Missing | No config files |
| Tests | ❌ Missing | No test files |
| Data/images | ❌ Missing | No data directory |
| Experiments | ❌ Missing | No experiment results |
| Git repository | ❌ Missing | No `.git` directory |

> [!WARNING]
> The phase briefs reference files that do not exist (`scripts/phase3_diffusion.py`, `common/paths.py`, `common/seed.py`). This suggests work was done in a different workspace that was not carried forward to `d:\racs2`.

### Conclusion
The project is **100% documentation, 0% implementation**. The implementation is at Phase 1, Step 0 (before environment setup).

---

# 14. Implementation Roadmap

## Dependency Graph (Phases)
```
Phase 1 (Env) ─────────────────────────────┐
  └──→ Phase 2 (VAE) ─────────────────────┐│
         └──→ Phase 3 (Diffusion) ────────┐││
Phase 4 (DB) ────────────────────────────┐│││
  └──→ Phase 5 (CLIP) ─────────────────┐││││
         └──→ Phase 6 (FAISS) ────────┐│││││
                └──→ Phase 7 (Token) ─┤│││││
Phase 8 (Structure) ─────────────────┐││││││
  └──→ Phase 9 (Reranking) ─────────┤│││││
Phase 10 (EDICT isolated) ──────────┤│││││
  └──→ Phase 11 (Local steg core) ──┤│││││
         └──→ Phase 12 (RACS+EDICT) ─┤ (M7)
                └──→ Phase 13 (AWGN) ─┤
                      └──→ Phase 14 (SemCom)
                            └──→ Phase 15 (Full RACS-Base) (M8)
Phase 16 (Baseline comparison) ──── (M9)
Phase 17 (VLM reranking) ──────────┐
Phase 18 (Editing) ────────────────┤ (M10)
Phase 19 (Hybrid) ─────────────────┘
Phases 20-24 (Security, eval, ablations) (M11)
Phases 25-27 (Analysis, tables, reproducibility) (M12)
```

## Implementation Order (dependency-aware)

### Sprint 1: Foundation (Phases 1-2)
1. **Phase 1**: Environment setup, project skeleton, config, seeds, GPU inference test
   - Depends on: Nothing
   - Depended on by: Everything
   - Risk: HF connectivity (already identified as blocker)
   
2. **Phase 2**: VAE encode/decode round-trip, metrics
   - Depends on: Phase 1
   - Depended on by: Phase 10 (EDICT), Phase 11 (steg core)
   - Risk: Low (simple component)

### Sprint 2: Understanding the Cost (Phase 3)
3. **Phase 3**: Diffusion generation, measure cost
   - Depends on: Phase 1-2
   - Depended on by: Phase 16 (baseline), Phase 19 (hybrid)
   - Risk: VRAM constraints, HF connectivity

### Sprint 3: Retrieval Stack (Phases 4-7)
4. **Phase 4**: Build image database
5. **Phase 5**: CLIP embeddings
6. **Phase 6**: FAISS index + retrieval
7. **Phase 7**: Deterministic token selection
   - Risk: Database source/size, CLIP checkpoint choice

### Sprint 4: Structural Intelligence (Phases 8-9)
8. **Phase 8**: Structural representation
9. **Phase 9**: Structural reranking
   - Risk: Structural similarity metric choice

### Sprint 5: EDICT (Phases 10-12)
10. **Phase 10**: EDICT as isolated mini-project
11. **Phase 11**: Local steganography core (VAE + EDICT + perturbation)
12. **Phase 12**: RACS-Base local (retrieval + steg)
    - Risk: EDICT implementation complexity; VRAM for full pipeline

### Sprint 6: Communication (Phases 13-15)
13. **Phase 13**: AWGN channel
14. **Phase 14**: Semantic communication (pretrained SwinJSCC/MDJCM)
15. **Phase 15**: Full RACS-Base end-to-end
    - Risk: Obtaining pretrained SemCom models; VRAM budget

### Sprint 7: Advanced Variants (Phases 16-19)
16. **Phase 16**: AgentSemSteCom baseline
17. **Phase 17**: VLM reranking
18. **Phase 18**: Lightweight editing
19. **Phase 19**: Retrieve-or-generate hybrid

### Sprint 8: Evaluation (Phases 20-27)
20-27: Security, steganalysis, evaluation matrix, ablations, efficiency, analysis, figures, reproducibility

---

# 15. Testing & Validation Strategy

## Per-Phase Gates (from roadmap)
| Phase | Gate Criterion |
|---|---|
| 1-2 | VAE round-trip runs and is measured |
| 3 | Explain why generation is expensive, with own numbers |
| 4-7 | Same token + same candidate list → same selection; correct/wrong/different produce expected changes |
| 8-9 | Structural reranking measurably changes pool vs CLIP-only |
| 10 | Small latent reconstruction error in EDICT round-trip |
| 11 | Local hide/recover works with MSE/PSNR/SSIM/LPIPS |
| 12 | End-to-end local recovery works with retrieved references |
| 13-15 | **Xref_tx == Xref_rx** before any PSNR is read |
| 16 | Comparison table exists |
| 17-19 | Each variant compared against RACS-Base |
| 20-24 | Numbers with distributions, not anecdotes |
| 25-27 | Every result tied to a config |

## Critical Test: Determinism Invariant
The single most important test: `Xref_transmitter == Xref_receiver`. If this fails, the entire system breaks. This must be verified automatically at every integration point.

## Metric Suite
- **Quality**: PSNR↑, SSIM↑, MSE↓, LPIPS↓
- **Efficiency**: T_generation, T_retrieval, T_reranking, T_editing, T_total, peak VRAM
- **Security**: Correct-token PSNR vs wrong-token PSNR (gap), no-token PSNR
- **Steganalysis**: XuNet detection accuracy (target ≈ 50%)

---

# 16. Security Analysis

## Original Security Model
1. **Token is private** — distributed via secure key management center
2. **Kpub and Xfeat are public** — they contain no reversible information
3. **Dual-stage protection**: Token seeds reference generation AND controls perturbation
4. **Three eavesdropper models** tested:
   - Eavesdropper 1: Has semantic decoder only → recovers stego (not secret)
   - Eavesdropper 2: Has decoder + Kpub + Xfeat + wrong token → severely degraded (18.60 dB at 5 dB SNR)
   - Eavesdropper 3: Has decoder + Kpub + Xfeat + no token → 20.86 dB at 5 dB SNR

## RACS Security Differences

### Changed
- Token no longer seeds diffusion noise; it selects from a candidate pool
- Security of reference selection: `P(correct ref | no token) ≈ 1/K`
- Token entropy for selection: `log2(K)` bits (much less than full diffusion noise seed entropy)

### Unchanged
- Token perturbation mechanism remains identical
- EDICT inversion with wrong perturbation still collapses

### New Attack Surfaces
1. **Database knowledge attack**: If an attacker knows the database, they can enumerate all K candidates and try each
2. **Ranking knowledge attack**: If ranking function/weights are known, attacker can narrow the candidate pool
3. **Side-channel**: Image IDs or database structure might leak information
4. **Smaller security margin**: Selection from K candidates has `log2(K)` bits entropy vs. full Gaussian noise seed (much higher entropy)

### Open Security Questions
- Is `log2(K)` bits sufficient for the selection security?
- Does the perturbation alone provide enough security even if the reference is known?
- What is the combined security when both selection and perturbation are token-controlled?
- How does database size affect the security analysis?

---

# 17. Critical Risks / Unknowns / Contradictions

## Technical Risks
| # | Risk | Severity | Mitigation |
|---|---|---|---|
| TR-01 | CLIP retrieval may find semantically similar but structurally incompatible images | High | Structural reranking, editing, larger database |
| TR-02 | 6 GB VRAM may be insufficient for full pipeline | High | fp16, attention slicing, sequential loading |
| TR-03 | EDICT implementation complexity — subtle bugs cause silent failures | High | Isolated Phase 10, extensive testing |
| TR-04 | SwinJSCC / MDJCM pretrained models may not be available | Medium | May need to simulate channel more simply |
| TR-05 | Determinism may break across platforms/versions | Medium | Pin every dependency, verify bit-identical outputs |

## Security Risks
| # | Risk | Severity |
|---|---|---|
| SR-01 | Token selection entropy (`log2(K)` bits) much lower than diffusion noise seed entropy | Medium |
| SR-02 | Database knowledge enables candidate enumeration attack | Medium |
| SR-03 | If retrieval always picks the "same best" image, effective K << nominal K | High |

## Research Risks
| # | Risk | Description |
|---|---|---|
| RR-01 | Retrieved references may be structurally incompatible → poor EDICT recovery | Core risk |
| RR-02 | PSNR/SSIM degradation may be unacceptable | Possible negative result |
| RR-03 | Efficiency gain may not be as large as hoped if reranking/editing costs grow | Undermines H1 |

## Architecture Risks
| # | Risk | Description |
|---|---|---|
| AR-01 | Kpub determinism is unresolved — LLM may give different outputs at Tx/Rx | Can break invariant |
| AR-02 | FAISS index type choice affects both determinism and performance | Must decide early |
| AR-03 | VLM in reranking path may introduce nondeterminism | Architecture constraint |

## Contradictions Between Documents
| # | Contradiction | Source A | Source B |
|---|---|---|---|
| C-01 | **Phase count**: 21 vs 27 phases | _implplan.txt summary (21) | _implplan.txt body (27) |
| C-02 | **Phase numbering diverges after Phase 15** | _implplan.txt | RACS_ROADMAP.md (resolves this) |
| C-03 | **Disk space**: "D: ~44 GiB" vs "D: ~29.7 GB free" | PROJECT_UNDERSTANDING.md line 133 | RACS_ROADMAP.md line 59 |
| C-04 | **DB size**: "50k-100k" (overview) vs "1k→5k→10k→larger" (roadmap) | _overview.txt (target) | RACS_ROADMAP.md (development progression) |
| C-05 | **Host RAM**: "~3.4 GiB free" (roadmap) but concurrency doc shows "15/15 tests pass, all modules compile" | RACS_ROADMAP.md | CONCURRENCY.md — implies a previous workspace with working code |
| C-06 | **Missing code**: Phase briefs reference existing scripts (`scripts/phase3_diffusion.py`, `common/paths.py`) that don't exist | phase_briefs/* | actual filesystem |

## Missing Requirements
- MR-01: No specification of which structural similarity metric to use for reranking
- MR-02: No specification of query construction (how Kpub + Xfeat become a retrieval query)
- MR-03: No specification of editing parameters or method selection criteria
- MR-04: No specification of retrieve-vs-generate threshold
- MR-05: No definition of what "acceptable quality change" means numerically

---

# 18. Requirement → Design → Implementation → Test Traceability

| Requirement | Design Decision | Architecture Component | Research Justification | Implementation Task | Test/Validation |
|---|---|---|---|---|---|
| FR-02 (Xref determinism) | Shared config, deterministic pipeline | Determinism Contract | Baseline paper: token → same ref at Tx/Rx | Phase 7, 12, 15 | Assert `Xref_tx == Xref_rx` bitwise |
| FR-03 (CLIP retrieval) | CLIP embedding → FAISS | Embedding + FAISS modules | Re-Imagen: CLIP for retrieval; Baseline: semantic extraction | Phase 5-6 | Same query → same Top-K |
| FR-04 (Structural reranking) | α·semantic + β·structural scoring | Reranking module | Baseline: CLIP ≠ structural compatibility | Phase 8-9 | Pool changes vs CLIP-only |
| FR-05 (Token selection) | `H(token) mod K` | Token Selection module | Baseline: token controls generation | Phase 7 | Same token → same selection |
| FR-06 (EDICT) | 50 steps, p=0.93, coupled latents | EDICT module | EDICT paper; Baseline Section V | Phase 10-11 | Latent reconstruction error < threshold |
| FR-09 (Quality metrics) | PSNR/SSIM/MSE/LPIPS | Evaluation module | Baseline Table II | Phase 2+ | Compare to baseline numbers |
| FR-11 (Security) | Correct/wrong/no token test | Security module | Baseline Section VI.C.2 | Phase 20 | Gap ≥ comparable to 6.46 dB |
| FR-12 (Efficiency) | Timing and VRAM measurement | Evaluation module | H1 hypothesis | Phase 3, 24 | T_retrieval << T_generation |
| NFR-01 (6 GB VRAM) | fp16, attention slicing | All GPU modules | Hardware constraint | Phase 1+ | Peak VRAM < 6 GB |

---

# 19. Final End-to-End Explanation

> For a senior engineer who has never seen this project:

**The Problem**: Imagine you need to transmit a secret image over a wireless channel without anyone knowing the image exists. The state-of-the-art system (AgentSemSteCom) does this by generating a "reference image" using an expensive diffusion model (~50 denoising steps), then using a mathematically exact invertible diffusion process (EDICT) to hide the secret image within a "stego image" that looks like the reference but secretly encodes the original. The stego image is sent over a noisy wireless channel using semantic communication. The receiver, who has the same token and public information, regenerates the same reference image and reverses the EDICT process to recover the secret.

**The RACS Idea**: Instead of *generating* the reference image (expensive), *retrieve* it from a pre-built database of images (fast). Use CLIP for semantic matching and FAISS for fast vector search. Add structural reranking so the retrieved image matches not just the meaning but also the pose/layout. Use a digital token to deterministically select one candidate from the pool, maintaining security.

**One Complete Flow**:
1. **Alice** has a secret face photo. Her system describes it as "a woman smiling" (`Kpub`) and extracts the skeletal pose (`Xfeat`).
2. The system encodes the description with CLIP and searches a pre-indexed database of 10,000 images using FAISS, finding the 20 most semantically similar faces.
3. Each candidate is scored by both semantic similarity (CLIP) and structural similarity (pose match). The top 10 pass the quality filter.
4. Alice's secret token `42` is hashed: `H(42) mod 10 = 7`. Candidate #7 is selected as `Xref`.
5. The secret image is encoded to latent space via VAE, forward-diffused via EDICT, perturbed using the token, then reverse-diffused conditioned on `Kpub` and `Xref`, and decoded back to a stego image.
6. The stego image is semantically encoded and transmitted over a noisy wireless channel (10 dB SNR).
7. **Bob** receives the noisy signal, decodes it to a stego image, and runs the *exact same retrieval process* with the same database, same CLIP, same FAISS, same ranking, and the same token `42` — arriving at the same `Xref`.
8. Bob reverses the EDICT process using `Xref` and the token, recovering the original secret face photo.

**The Key Constraint**: Alice and Bob MUST arrive at exactly the same `Xref`. If even one bit differs, EDICT inversion fails. This is why the entire retrieval pipeline must be perfectly deterministic.

**The Research Question**: This retrieval approach is much faster and uses less GPU memory. But is the retrieved reference "good enough" for EDICT to work? Does security hold? Does it work at low SNR? That's what RACS measures.

---

# 20. Questions Requiring Clarification

## Critical (Must resolve before implementation)
1. **Where is the previous codebase?** The phase briefs and CONCURRENCY.md reference existing scripts (`scripts/phase3_diffusion.py`, `common/paths.py`, `common/seed.py`) and passing tests (15/15). Where is this code?
2. **CLIP checkpoint**: Which CLIP model should be used? (ViT-B/32: 350 MB, 512-d; ViT-L/14: 890 MB, 768-d; OpenCLIP: various) — impacts VRAM, quality, and every downstream component.
3. **Kpub determinism**: How will we ensure the LLM generates the same `Kpub` at transmitter and receiver? Options: (a) greedy decoding with fixed seed, (b) transmit `Kpub` explicitly, (c) use a predefined `Kpub` for experiments.
4. **Image database source**: Which specific LAION subset? How are images selected? Licensing?
5. **HuggingFace connectivity**: Phase 3 is blocked on HF reachability. Is this still the case?

## Important (Should resolve early)
6. **Structural similarity metric**: What specific metric compares `Xfeat` of secret vs candidate? (IoU of segmentation maps? Pose keypoint distance? Learned metric?)
7. **FAISS index**: Confirm `IndexFlatIP` (exact, deterministic, but slow for large DB) vs approximate (faster, nondeterministic)?
8. **SwinJSCC / MDJCM availability**: Are pretrained models available? Do they need specific frameworks?
9. **Test-set leakage**: How to partition images so evaluation secrets never appear in the retrieval database?
10. **Pre-registered success thresholds**: What PSNR drop (in dB) is "acceptable"? What speedup (×) is "substantial"?

## Design Decisions
11. **Query construction**: How exactly are `Kpub` (text) and `Xfeat` (image/feature) combined into a retrieval query?
12. **Retrieve-or-generate threshold**: What metric/threshold determines "candidate quality is too low"?
13. **Editing method**: Which editing approach? SDEdit, ControlNet-guided editing, or something simpler?
14. **VLM model**: Which VLM for agentic reranking? How to make it deterministic?
15. **IPAdapter-plus in RACS**: The baseline uses IPAdapter-plus for reference image guidance during EDICT. Is this still used when Xref is retrieved instead of generated?

---

# MY UNDERSTANDING OF RACS IN ONE PAGE

**RACS (Retrieval-Augmented Coverless Semantic Steganography)** is a research study that modifies the AgentSemSteCom system (Meng et al., IEEE TCCN 2026) by replacing its computationally expensive diffusion-based reference image generation with deterministic multimodal retrieval from a pre-indexed database.

**The baseline system** hides a secret image inside a generated stego image using EDICT (exact invertible diffusion with coupled transformations), digital token-controlled noise seeding and perturbation, and semantic communication over AWGN channels. The receiver regenerates the same reference image using the same token and reverses the process. The bottleneck is generating the reference image via 50-step diffusion (~2.5 GB VRAM, significant latency).

**RACS replaces this single stage**: instead of generating Xref via diffusion, it embeds a query using CLIP, searches a FAISS-indexed database for Top-K semantically similar candidates, reranks by structural compatibility (pose/layout/segmentation similarity), and deterministically selects one using the digital token (`index = H(token) mod K`). Optional stages include VLM-based intelligent reranking, lightweight editing (SDEdit-style), and a retrieve-or-generate fallback.

**The critical invariant** is `Xref_tx == Xref_rx` — both transmitter and receiver must arrive at the same reference image. This demands perfect determinism: same database, same embeddings, same FAISS index, same ranking, same token mapping.

**The project compares five system variants**: AgentSemSteCom (baseline), RACS-Base (retrieval+rerank+token), RACS-Agent (+VLM), RACS-AgentEdit (+editing), RACS-Hybrid (retrieve-or-generate). Evaluation covers quality (PSNR/SSIM/MSE/LPIPS), efficiency (latency/VRAM), security (correct/wrong/no token gap), and robustness (5–20 dB SNR), across three image classes (common/facial/style).

**Current status**: The project is 100% documentation and 0% implementation. 27 phases grouped into 12 milestones are defined. Hardware is severely constrained (RTX 3050 6 GB VRAM, limited disk and RAM). The key open decisions are: CLIP checkpoint, Kpub determinism strategy, database source/size, structural similarity metric, and success thresholds. Negative results are explicitly valid — the project is a scientific comparison, not a sales pitch.

---

---

# PART 2: Critical Self-Review

> Acting as a senior systems architect and research reviewer, I now critically review the above analysis. For every major claim, I identify its source, verify it, and classify my understanding.

---

## 1. Confirmed Facts (directly supported by source material)

| # | Fact | Source |
|---|---|---|
| CF-01 | RACS replaces only the reference acquisition stage | [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L14-19; [_overview.txt](file:///d:/racs2/phase2/_overview.txt) L29-36 |
| CF-02 | Baseline is AgentSemSteCom from Meng et al. 2026 | [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L10-12; [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L1-4 |
| CF-03 | EDICT steps=50, p=0.93, η=0.05, edit strength=0.5 | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L973-984; [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L42-44 |
| CF-04 | Token has dual role: noise seed + perturbation control | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L122-126 (overview) and L758-790 (equations) |
| CF-05 | Security baseline: correct 25.37 dB vs wrong 18.91 dB = 6.46 dB gap | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L229-237; [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L50-52 |
| CF-06 | XuNet detection: 50.91% for AgentSemSteCom | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L1247; [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L52-53 |
| CF-07 | UniStega dataset: 100 images, three classes | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L929-946 |
| CF-08 | System variant names are frozen | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L132-137 |
| CF-09 | 27 phases, 12 milestones, canonical numbering | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L52-127 |
| CF-10 | Hardware: RTX 3050 6 GB VRAM | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L58 |
| CF-11 | `Xref_tx == Xref_rx` is mandatory invariant | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L142; [_overview.txt](file:///d:/racs2/phase2/_overview.txt) L631 |
| CF-12 | Phase count inconsistency (21 vs 27) is acknowledged and resolved | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L52-54 |
| CF-13 | No source code exists in this workspace | Filesystem scan: zero code files |
| CF-14 | Phase 3 is blocked on HF reachability | [phase3.md](file:///d:/racs2/docs/phase_briefs/phase3.md) L54-59 |
| CF-15 | Negative results are explicitly valid | [_overview.txt](file:///d:/racs2/phase2/_overview.txt) L721-735; [PROJECT_UNDERSTANDING.md](file:///d:/racs2/docs/PROJECT_UNDERSTANDING.md) L98-101 |
| CF-16 | The perturbation is a sign-flip: `z'T = zT ⊙ (1-M) - zT ⊙ M` | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L773-774 (Eq. 13) |
| CF-17 | ControlNet is used for structural conditioning in reference generation | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L594-624 |
| CF-18 | IPAdapter-plus is used for reference image guidance during steganography | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L983-984 |

## 2. Reasonable Inferences (logical deductions from the source material)

| # | Inference | Basis |
|---|---|---|
| RI-01 | The 2209.14491v3.pdf is Re-Imagen (retrieval-augmented generation) | arXiv ID confirmed via web search; relevant to RACS retrieval concept |
| RI-02 | A previous workspace/session existed with working code | CONCURRENCY.md references "15/15 tests pass, all modules compile"; phase briefs reference existing scripts |
| RI-03 | Token selection provides weaker security than token-based noise seeding | Selection from K candidates = log2(K) bits; Gaussian noise seed = much higher entropy. This is an inference but strongly supported by the math. |
| RI-04 | IPAdapter-plus may still be used in RACS for the EDICT steganography conditioning | The baseline uses it for reference image guidance during stego generation (not reference generation); RACS only changes reference acquisition |
| RI-05 | The Phase 2 brief's reference to `common.seed.set_seed` suggests a utility module was planned | Phase brief line 48 |
| RI-06 | Database query is likely text-based (CLIP text encoding of Kpub) | RACS architecture shows CLIP text/multimodal embedding; Kpub is text |

## 3. Unverified Assumptions

| # | Assumption | Why Unverified |
|---|---|---|
| UA-01 | RACS retrieval will be significantly faster than diffusion generation | Not yet measured; depends on DB size, reranking cost, editing cost |
| UA-02 | CLIP semantic similarity correlates with EDICT compatibility | Not established; EDICT requires specific structural properties that CLIP may not capture |
| UA-03 | A database of 50k-100k images provides sufficient coverage | No analysis of image diversity requirements |
| UA-04 | Structural reranking will improve EDICT recovery quality | Hypothesis, not fact; needs ablation |
| UA-05 | VLM reranking can be made deterministic | Depends on model and implementation |
| UA-06 | Lightweight editing preserves the efficiency advantage | Could approach full generation cost |
| UA-07 | The perturbation mechanism alone provides sufficient security even with known reference | The baseline relies on BOTH token roles; RACS weakens one |

## 4. Contradictions Between Documents

| # | Contradiction | Documents | Resolution |
|---|---|---|---|
| CD-01 | Phase count 21 vs 27 | _implplan.txt vs itself | RACS_ROADMAP.md resolves: 27 phases is canonical |
| CD-02 | Disk space "~44 GiB" vs "~29.7 GB free" | PROJECT_UNDERSTANDING.md vs RACS_ROADMAP.md | Likely measured at different times; RACS_ROADMAP.md is more recent |
| CD-03 | Referenced code files don't exist | phase_briefs vs filesystem | Previous workspace existed; code was not carried forward |
| CD-04 | CONCURRENCY.md mentions commits and tests | CONCURRENCY.md vs empty workspace | Confirms prior workspace existence |
| CD-05 | _implplan.txt has different phase numbering after Phase 15 | _implplan.txt (Phases 16-21) vs RACS_ROADMAP.md (Phases 16-27) | _implplan.txt collapses VLM/editing/hybrid and security/experiments into fewer phases; RACS_ROADMAP.md expands them. RACS_ROADMAP.md is canonical. |

## 5. Missing Requirements

| # | Missing Requirement |
|---|---|
| MR-01 | No specification of the structural similarity metric |
| MR-02 | No specification of how Kpub + Xfeat become a retrieval query vector |
| MR-03 | No editing method selected or parameterized |
| MR-04 | No retrieve-vs-generate threshold defined |
| MR-05 | No pre-registered acceptable PSNR/SSIM thresholds |
| MR-06 | No specification of which LLM generates Kpub (or whether it's bypassed for experiments) |
| MR-07 | No specification of how to handle the "style" image class (no implicit feature) during retrieval |
| MR-08 | No specification of the reference image generator model for baseline (PicX-Real is mentioned in the paper but not in RACS docs) |
| MR-09 | No specification of how to handle ControlNet in the RACS context (still used for EDICT conditioning? Or only for baseline reference generation?) |

## 6. Research Concepts That Need Careful Handling

| # | Concept | Risk of Misunderstanding |
|---|---|---|
| RC-01 | **EDICT is used in two different places in the baseline**: (1) Reference generation uses DDIM (not EDICT), (2) Steganography (hiding/recovery) uses EDICT | The overview sometimes conflates these. RACS replaces (1), not (2). |
| RC-02 | **IPAdapter-plus is used for the steganography stage**, not the reference generation stage | IPAdapter provides reference image guidance during EDICT denoising. It will still be needed in RACS. |
| RC-03 | **ControlNet is used for reference generation** (conditioning on Xfeat) | RACS doesn't generate references, so ControlNet is not needed for reference acquisition. But it IS needed for the baseline comparison and possibly for the editing variant. |
| RC-04 | **The paper uses PicX-Real (not SD v1.5) for reference generation** | [_paper.txt](file:///d:/racs2/research_papers/_paper.txt) L958-967: "We employ PicX-Real as an independent diffusion-based generator." SD v1.5 is used only for the steganography (EDICT) process. |
| RC-05 | **"Edit strength" is a steganography parameter, NOT an editing parameter** | [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) L146: "edit strength 0.5 (reverse-step proportion; unrelated to RACS image editing)" |

> [!IMPORTANT]
> RC-01 and RC-04 are critical distinctions. The reference generator uses **PicX-Real** with ControlNet, while the steganography module uses **SD v1.5** with EDICT. These are different models used in different stages. Several documents are imprecise about this distinction.

## 7. Architecture Decisions That Need Justification

| # | Decision | Needs Justification Because |
|---|---|---|
| AD-01 | `IndexFlatIP` for FAISS | Exact search is O(n) per query. At 100k entries, may be fast enough, but should verify timing. |
| AD-02 | CLIP text embedding (vs image embedding) for query | The secret image IS available at query time — why not use CLIP image embedding directly? |
| AD-03 | Token selection via `H(token) mod K` | Simple but may cause uneven distribution for small K; what hash function? |
| AD-04 | Database size graduation 1k→5k→10k | What determines when to grow? What's the minimum viable size? |
| AD-05 | α, β weights for reranking | How are these tuned? Grid search? Per-class? |

## 8. Security Concerns

| # | Concern | Severity |
|---|---|---|
| SC-01 | Token selection from K candidates is fundamentally weaker than Gaussian noise seeding | High — the original token provides ~32+ bits of seed entropy vs log2(K) ≈ 3-6 bits of selection entropy |
| SC-02 | If attacker knows the database and ranking, they can enumerate all K candidates | High — reduces to trying K references (feasible for K=10-50) |
| SC-03 | The perturbation mechanism (unchanged) may compensate, but this is not proven | Medium — needs formal or experimental analysis |
| SC-04 | Database images might leak information about the secret (if similar images are in the DB) | Low to Medium — mitigated by test-set leakage rules |

## 9. Implementation Risks

| # | Risk | Probability | Impact |
|---|---|---|---|
| IR-01 | HuggingFace model downloads blocked | Known | Blocks Phase 2+ |
| IR-02 | VRAM overflow during full pipeline | High | Crash; requires sequential model loading |
| IR-03 | Disk space exhaustion (29 GB for models + data + DB) | Medium | Must carefully budget |
| IR-04 | EDICT implementation bugs cause silent accuracy loss | High | Requires exhaustive numerical testing |
| IR-05 | Previous codebase not available — starting from scratch | Confirmed | Delays but not blockers |
| IR-06 | SwinJSCC/MDJCM may require incompatible frameworks | Medium | May need to use simplified channel model |
| IR-07 | OneFormer/OpenPose may have VRAM requirements exceeding budget | Medium | May need CPU fallback or smaller models |

## 10. Questions That Must Be Resolved Before Coding

### Tier 1: Blocking (cannot proceed without answers)
1. **Is there a previous codebase to recover?** The docs reference existing code.
2. **Is HuggingFace accessible now?** Phase 2-3 are blocked without model downloads.
3. **Which CLIP model/checkpoint?** Affects embedding dimensions, VRAM, and all downstream indexing.

### Tier 2: Important (affects architecture)
4. **How is Kpub generated/ensured at receiver?** Transmitted explicitly? Or regenerated?
5. **Query type**: Text embedding of Kpub, or image embedding of Xs, or both?
6. **Is IPAdapter-plus still used in the EDICT stage when Xref is retrieved?**
7. **Baseline reference generator**: PicX-Real (paper) or SD v1.5 (RACS docs sometimes imply)?

### Tier 3: Design choices (can defer but shouldn't drift)
8. Structural similarity metric for reranking
9. Database source, size, and licensing
10. Success thresholds for H1-H4

---

## Corrected Canonical Project Understanding

After critical review, I confirm the analysis is materially accurate, with these important corrections and clarifications:

1. **Two different diffusion models** are used in the baseline: **PicX-Real** for reference generation, **SD v1.5** for EDICT steganography. Several documents are imprecise about this.

2. **IPAdapter-plus** is used in the **steganography stage** (EDICT denoising), not the reference generation stage. It will still be needed in RACS.

3. **ControlNet** is used in the **reference generation stage** only. RACS retrieval does not need ControlNet, but the baseline comparison and editing variant might.

4. **"Edit strength" = 0.5** is a steganography parameter (proportion of reverse diffusion steps), NOT related to the lightweight editing stage of RACS.

5. **A previous workspace with working code existed** (evidenced by CONCURRENCY.md and phase brief references), but the code is not in this workspace.

6. **The token's security role is split** in RACS: selection security (weakened) + perturbation security (unchanged). Whether the combined security is sufficient is an open research question, not a confirmed fact.

7. **The 2209.14491v3.pdf** is the **Re-Imagen** paper (retrieval-augmented generation), not the EDICT paper. It provides conceptual validation for the retrieval approach but is not directly cited in the RACS design documents.
