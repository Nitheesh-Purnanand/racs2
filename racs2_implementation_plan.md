# RACS2 — Learning-Oriented Implementation Plan

> **Principle**: Learn one thing → build a small version → test it → understand the output → integrate it → move on.  
> **Anti-pattern**: Do NOT try to build the whole system at once.  
> **Your constraints**: RTX 3050 (6 GB VRAM), ~30 GB free disk, ~3.4 GB free RAM, Windows, Python 3.10.11

---

## How This Plan Is Organized

The project is split into **12 Milestones (M1–M12)**, containing **27 Phases**. Each phase is a self-contained "mini-project" where you:

1. **Learn the theory** — understand *what* you're building and *why* RACS needs it
2. **Build a tiny experiment** — write isolated code that demonstrates the concept
3. **Test and measure** — run it, collect numbers, save artifacts
4. **Explain** — answer the "gate question" in your own words
5. **Integrate** — connect it to the growing pipeline
6. **Report** — document what you built, what you measured, what surprised you

Every phase ends with a concrete **gate** — a test you must pass before moving on.

---

## The Big Picture (What You're Building)

```
EXISTING SYSTEM (AgentSemSteCom):
  Secret Image → LLM semantic key → Diffusion GENERATES reference → EDICT steganography → Channel → Recovery

RACS (Your Contribution):
  Secret Image → LLM semantic key → RETRIEVES reference from database → EDICT steganography → Channel → Recovery
                                     ↑
                              THIS IS WHAT YOU BUILD
```

You're replacing **one expensive stage** (diffusion generation) with **retrieval**. Everything else stays the same. This way, any performance difference is scientifically attributable to your change.

---

## Milestone M1 — Foundation (Phases 1–2)
*Time estimate: 2–3 days*

### Phase 1: Environment & Project Skeleton

> **What**: A working Python environment with PyTorch + GPU, a clean project structure, and a shared config.  
> **Why**: Nothing else works without this. Reproducibility starts here.

**Theory to learn:**
- Python virtual environments (`venv`)
- PyTorch: tensors, devices (CPU vs GPU), dtypes (fp32, fp16)
- CUDA: what it is, how `torch.cuda.is_available()` works, `torch.cuda.memory_allocated()`
- Model checkpoints: what `.safetensors` / `.bin` files are, HuggingFace Hub
- Random seeds: `torch.manual_seed()`, `random.seed()`, `np.random.seed()` — why determinism matters

**Build tasks:**
1. Create the project skeleton:
   ```
   d:\racs2\
   ├── configs/          # YAML/JSON config files
   ├── common/           # Shared utilities (seed, paths, device)
   ├── data/             # Images (secret, retrieval corpus)
   ├── models/           # Model loading helpers
   ├── semantic/         # Semantic/structural extraction
   ├── retrieval/        # CLIP, FAISS, embedding, ranking
   ├── reference/        # Reference acquisition (retrieval path, editing)
   ├── steganography/    # VAE, EDICT, token perturbation
   ├── communication/    # AWGN, SemCom codecs
   ├── security/         # Token tests, steganalysis
   ├── evaluation/       # Metrics (PSNR, SSIM, MSE, LPIPS, latency)
   ├── experiments/      # Experiment scripts + results
   │   └── results/
   ├── scripts/          # Standalone run scripts per phase
   └── tests/            # Unit tests
   ```
2. Create `common/seed.py` — a `set_seed(seed)` function that seeds Python, NumPy, PyTorch (CPU + CUDA).
3. Create `common/device.py` — auto-detect GPU, report VRAM.
4. Create `common/paths.py` — centralized paths, HuggingFace cache on D: (not C:).
5. Create `configs/base.yaml` — image_size, device, seed, database_path, index_path, top_k, token, snr.
6. **First experiment**: Load any image → convert to tensor → move to GPU → run through a tiny pretrained model (e.g., torchvision ResNet or just a matrix multiply) → get output → print shapes.

**Gate**: You can explain image tensors (B×C×H×W), CPU vs GPU, model checkpoints, and why seeds matter. A script runs on GPU and prints tensor shapes.

---

### Phase 2: Image Tensors, VAE & Latent Space

> **What**: Encode an image into Stable Diffusion's latent space and decode it back.  
> **Why**: EDICT and all diffusion stages operate in **latent space**, not pixel space. This is the error floor for everything downstream.

**Theory to learn:**
- VAE (Variational Autoencoder): encoder compresses, decoder reconstructs
- Pixel space (512×512×3) → Latent space (64×64×4): 8× spatial downscale, 4 latent channels
- Normalization: SD VAE expects pixel range [-1, 1]
- Metrics: MSE, PSNR = 10·log₁₀(1/MSE), SSIM
- **Key insight**: Reconstruction is NOT perfect. The round-trip error is a *floor* — EDICT/channel/retrieval errors add ON TOP of this.

**Build tasks:**
1. Install `diffusers` + `accelerate`.
2. Load the SD v1.5 VAE (`AutoencoderKL`, fp16 on GPU).
3. Take one image (512×512), normalize to [-1, 1].
4. Run: `image → VAE encode → latent → VAE decode → reconstructed image`.
5. Print every shape: input, latent, output.
6. Measure MSE / PSNR / SSIM (implement in `evaluation/`).
7. Repeat on 5 images. Report the spread, not one value.
8. Save a side-by-side comparison to `experiments/results/`.

**Gate**: VAE round-trip runs, is measured (MSE/PSNR/SSIM for >1 image), and you can explain: why latent space, what compression factor, why reconstruction is lossy.

> **Answer for yourself**: *Why does Stable Diffusion work in latent space instead of pixel space?*

---

## Milestone M2 — Diffusion Understanding (Phase 3)
*Time estimate: 1–2 days*

### Phase 3: Diffusion Models & Generation Cost

> **What**: Run Stable Diffusion to generate images and measure how expensive it is.  
> **Why**: You cannot argue RACS is cheaper until you have **measured** the generation cost. This phase gives you the baseline number that RACS must beat.

**Theory to learn:**
- Forward process: clean image → progressively add Gaussian noise → pure noise
- Reverse process: pure noise → U-Net predicts and removes noise → clean image (many steps!)
- Timesteps and noise schedules
- DDIM: deterministic, fewer-step variant of DDPM
- Classifier-free guidance: `guidance_scale` controls prompt adherence vs diversity
- **Cost model**: generation time ≈ num_steps × per-step U-Net cost. This is why it's expensive.

**Build tasks:**
1. Load SD v1.5 pipeline (fp16, DDIM scheduler, attention slicing for 6 GB VRAM).
2. Generate images with the paper's example prompt.
3. Experiment: vary seed, prompt, num_steps (20, 30, 50), guidance_scale.
4. Measure: wall-clock time and peak VRAM per setting.
5. Save timings to `experiments/results/phase3_timings.json`.

**Gate**: You can explain *why generation is expensive* using your own numbers (e.g., "50 steps × 0.8s/step = 40 seconds on my GPU"). You have reference images saved.

> **Key insight for RACS**: If generation takes 40 seconds and retrieval takes 0.1 seconds, that's the efficiency argument. But you need the numbers first.

---

## Milestone M3 — Retrieval Stack (Phases 4–7)
*Time estimate: 4–5 days — this is where RACS begins*

### Phase 4: Build the Retrieval Image Database

> **What**: Collect and organize a corpus of images that RACS will retrieve from.  
> **Why**: The database IS the RACS core. No database = no retrieval. Both transmitter and receiver must share the exact same database.

**Theory to learn:**
- Image datasets: LAION subset, or curated corpus (start small)
- Metadata per image: image_id, image_path, embedding (later), caption, structural metadata
- **Critical principle**: The database is a shared protocol resource. Tx and Rx must use the SAME version, preprocessing, embedding model, and index. If they disagree, `Xref_tx ≠ Xref_rx` and recovery fails.

**Build tasks:**
1. Curate a small dataset: start with **1,000 images** (e.g., from COCO, ImageNet subset, or Unsplash).
2. Organize as: `data/retrieval_images/` with an index file (`data/metadata/image_index.json`).
3. Store: image_id, image_path, caption/description (if available).
4. Build a loader that can iterate over the corpus.
5. Plan the growth path: 1k → 5k → 10k → larger if needed.

**Gate**: You have a working image corpus with metadata, and a loader that iterates over it.

---

### Phase 5: CLIP Embeddings

> **What**: Convert images and text into vector embeddings using CLIP.  
> **Why**: CLIP is how RACS measures semantic similarity. "Semantically similar" = "nearby in embedding space."

**Theory to learn:**
- CLIP: trained on image-text pairs, produces aligned embeddings
- Image → image embedding (vector), Text → text embedding (vector)
- Cosine similarity: `sim(a, b) = a·b / (|a|·|b|)`
- **Critical limitation**: CLIP captures *semantic* similarity but NOT: same pose, same layout, same geometry, same texture, same steganographic suitability. This limitation motivates structural reranking (Phase 8–9).

**Build tasks:**
1. Load CLIP (e.g., `ViT-B/32` or `ViT-L/14` via `transformers` or `open_clip`).
2. **Experiment 1**: Encode text prompts → embeddings. Compare similarity between related vs unrelated prompts.
3. **Experiment 2**: Encode images → embeddings. Find which images are "similar" to each other.
4. **Experiment 3**: Cross-modal — encode a text description and find the most similar images.
5. Embed your entire 1k image corpus. Save embeddings to disk.

**Gate**: You understand that CLIP = semantic similarity ≠ structural compatibility. You have embeddings for your corpus.

---

### Phase 6: FAISS Retrieval

> **What**: Build a vector search index so you can query "find the 10 most similar images to this query."  
> **Why**: This is the first real RACS component — the retrieval engine.

**Theory to learn:**
- Vector search: given a query vector, find the K nearest neighbors in a database
- Distance metrics: cosine similarity, inner product, L2 distance
- FAISS: Facebook's library for efficient similarity search
- `IndexFlatIP` (exact inner product) — recommended for determinism
- Approximate indices (IVF, HNSW) — faster but non-deterministic

**Build tasks:**
1. Build `retrieval/build_index.py`: takes embeddings → creates FAISS index → saves to disk.
2. Build `retrieval/retrieve.py`: `results = retrieve(query_embedding, top_k=10)`.
3. Test with varying K: 1, 5, 10, 20, 50. For each K, log candidate IDs and similarity scores.
4. Visualize: for a query image, display the Top-K retrieved candidates side by side.

**Gate**: You can retrieve semantically similar images from your corpus. Pipeline so far:
```
Secret image → semantic query → CLIP embedding → FAISS → Top-K candidate images
```

---

### Phase 7: Deterministic Token Selection

> **What**: Given the Top-K candidates, use a digital token to select exactly one candidate in a deterministic, reproducible way.  
> **Why**: The transmitter and receiver must select the SAME reference image. The token provides both reproducibility and security (wrong token → wrong selection → failed recovery).

**Theory to learn:**
- Deterministic hashing: `index = hash(token) % K`
- Why the token must NOT directly drive nondeterministic LLM behavior
- Separation of AI reasoning from protocol-critical determinism
- Security implication: without the token, an attacker's chance of guessing the right reference is ≈ 1/K

**Build tasks:**
1. Implement `retrieval/token_selection.py`:
   - Input: Top-K candidate list (ordered), digital token
   - Output: selected candidate index
   - Method: `selected = candidates[hash(token) % len(candidates)]`
2. **Test determinism**: same token + same candidate list → always the same selection.
3. **Test security**: different token → different selection.
4. **Test edge cases**: K=1 (always same), K=50 (more diversity).

**Gate**: Token selection is deterministic and reproducible. Correct/wrong/different tokens produce expected behavior.

> **Milestone M3 complete!** You now have the fundamental RACS retrieval pipeline:
> ```
> Secret → semantic query → CLIP → FAISS → Top-K → token selection → Xref
> ```

---

## Milestone M4 — Structural Reranking (Phases 8–9)
*Time estimate: 3–4 days*

### Phase 8: Structural Representation

> **What**: Extract structural features (edges, segmentation, pose) from images.  
> **Why**: CLIP finds "a person playing football" but might return the wrong pose, wrong layout, wrong spatial arrangement. Structural features measure *how things are arranged*, not just *what things are*.

**Theory to learn:**
- Semantic segmentation: pixel-level class labels
- Edge detection: Canny, HED
- Depth estimation: MiDaS / relative depth
- Pose landmarks: OpenPose (for human/face images)
- The difference between *semantic similarity* and *structural compatibility*

**Build tasks:**
1. Pick one structural representation to start (edges or segmentation — simplest).
2. Extract structural features for a few images.
3. Compare: two images that are semantically similar (same CLIP score) but structurally different.
4. Define a structural similarity metric (e.g., IoU for segmentation maps, MSE for edge maps).

**Gate**: You have a working structural representation and can demonstrate cases where CLIP says "similar" but structure says "different."

---

### Phase 9: Structure-Aware Reranking

> **What**: Combine semantic (CLIP) and structural similarity to produce better candidate rankings.  
> **Why**: Pure CLIP retrieval is the baseline, not the final design. Structural reranking should improve reference quality for EDICT.

**Theory to learn:**
- Composite scoring: `Score(I) = α × semantic_similarity + β × structural_similarity`
- Hyperparameter tuning of α, β
- Later extension: `+ γ × steganographic_suitability`

**Build tasks:**
1. For each FAISS Top-K candidate, compute structural similarity to the secret image.
2. Rerank by composite score.
3. **A/B comparison**: CLIP-only ranking vs CLIP + structure ranking. Do the top candidates change? Are they visually better?
4. Do NOT add VLM yet — first prove the simple structural signal helps.

**Gate**: Structural reranking measurably changes the candidate pool vs CLIP-only. You can show examples where it helps.

---

## Milestone M5 — EDICT (Phase 10)
*Time estimate: 4–5 days — this is the hardest new concept*

### Phase 10: EDICT as an Isolated Mini-Project

> **What**: Understand and implement EDICT (Exact Diffusion Inversion via Coupled Transformations).  
> **Why**: EDICT is the reversible hiding/recovery mechanism. It's what makes the steganography work. You MUST understand it separately before integrating with RACS.

> [!CAUTION]
> Do NOT try to integrate EDICT with RACS yet. Learn it as a standalone project first.

**Theory to learn:**
- Diffusion inversion: running the diffusion process "backwards" to recover the original latent
- DDIM inversion: approximate but has numerical drift
- EDICT: uses coupled latent variables to achieve *exact* (bit-level) inversion
- Forward: `latent → EDICT forward → transformed latent`
- Inverse: `transformed latent → EDICT inverse → original latent`
- Why this matters: the hiding/recovery process must be *reversible*

**Build tasks:**
1. Implement or reproduce EDICT from the paper.
2. **First experiment** (pure inversion test):
   ```
   latent → EDICT forward → EDICT inverse → latent
   ```
   Measure latent reconstruction error. It should be near zero.
3. Decode both latents to images. Compare visually and with metrics.
4. Experiment with EDICT parameters: steps (50), mixing coefficient (0.93).

**Gate**: Latent reconstruction error is small (near machine precision). You can explain why EDICT enables reversible hiding/recovery.

---

## Milestone M6 — Local Steganography Core (Phase 11)
*Time estimate: 3–4 days*

### Phase 11: VAE + EDICT + Token Perturbation (Local Only)

> **What**: Make the complete hiding/recovery mechanism work locally (no wireless channel, no retrieval, no VLM).  
> **Why**: If the core steganography doesn't work, nothing else matters.

**Build tasks:**
1. Implement the full local pipeline:
   ```
   Secret + Reference → VAE encode → EDICT forward + token perturbation → Stego image
   Stego → VAE encode → EDICT inverse + inverse perturbation → Recovered secret
   ```
2. Use a manually chosen reference image (not retrieved yet).
3. Include the token perturbation mechanism (η = 0.05).
4. Measure: MSE, PSNR, SSIM, LPIPS between original secret and recovered secret.
5. **Do NOT add**: AWGN, JSCC, VLM, editing. First prove the local core works.

**Gate**: Local hide/recover works. Metrics are measured and reasonable.

---

## Milestone M7 — RACS-Base Local (Phase 12)
*Time estimate: 2–3 days*

### Phase 12: Connect Retrieval to EDICT

> **What**: Replace the manually-chosen reference with a retrieval-selected reference.  
> **Why**: This is the RACS idea! You're now using your retrieval pipeline (Phases 4–9) to select references for EDICT (Phase 11).

**Build tasks:**
1. Wire the full local RACS pipeline:
   ```
   Secret → Kpub + Xfeat → CLIP → FAISS → rerank → token select → Xref
         → EDICT hide → Stego → EDICT recover → Secret
   ```
2. This is **RACS-Base** (no VLM, no editing, no generation fallback).
3. Measure quality: MSE, PSNR, SSIM, LPIPS.
4. If it fails, debug whether the problem is in retrieval, reference compatibility, or EDICT integration.

**Gate**: End-to-end local recovery works with retrieved references.

> **Why start simple?** If it fails, you know the problem is in retrieval OR reference compatibility OR EDICT integration — not in a complicated combination of 10 modules.

---

## Milestone M8 — Communication + Full RACS-Base (Phases 13–15)
*Time estimate: 4–5 days*

### Phase 13: AWGN Channel

> **What**: Add realistic channel noise to the stego image before recovery.  
> **Why**: Real wireless channels add noise. RACS must work under noise, not just in ideal conditions.

**Theory to learn:**
- SNR (Signal-to-Noise Ratio) in dB
- AWGN (Additive White Gaussian Noise): `received = signal + noise`
- How to compute noise power from target SNR

**Build tasks:**
1. Implement `communication/awgn.py`.
2. Add AWGN to the stego image at SNR = 20, 15, 10, 5 dB.
3. Measure recovery quality at each SNR level.

**Gate**: You see quality degrade gracefully as SNR decreases.

---

### Phase 14: Semantic Communication

> **What**: Replace raw AWGN with a learned semantic communication codec (SwinJSCC or similar).  
> **Why**: The baseline paper uses semantic communication, not raw pixel transmission. RACS must use the same codec for fair comparison.

**Build tasks:**
1. Use a pretrained semantic communication implementation (do NOT train from scratch).
2. Integrate: `stego → semantic encoder → channel → semantic decoder → reconstructed stego`.
3. Measure quality degradation from the codec at each SNR.

**Gate**: The semantic communication codec runs and its effect on quality is measured.

---

### Phase 15: Full RACS-Base

> **What**: The complete end-to-end pipeline with all components connected.  
> **Why**: This is the first major milestone — the system that answers the research question.

```
TRANSMITTER:
  Secret → Kpub + Xfeat → CLIP → FAISS → rerank → token select → Xref
        → EDICT + token perturbation → Stego → Semantic encoder → AWGN

RECEIVER:
  AWGN → Semantic decoder → Reconstructed stego
       → SAME database → SAME embedding → SAME ranking → SAME token → SAME Xref
       → EDICT inversion → Inverse perturbation → Recovered secret
```

**Critical test BEFORE looking at PSNR:**
```
Xref_transmitter == Xref_receiver
```
**If this fails, STOP and fix reproducibility.** Nothing else matters until this invariant holds.

**Gate**: Full pipeline runs. `Xref_tx == Xref_rx`. Quality metrics are measured across SNR levels.

---

## Milestone M9 — Baseline Comparison (Phase 16)
*Time estimate: 2–3 days*

### Phase 16: Compare Against AgentSemSteCom

> **What**: Build or obtain the best possible implementation of the original system and compare.  
> **Why**: Without a fair comparison, RACS results are meaningless.

**Build tasks:**
1. Use the same secret images, same image classes, same SNR values, same metrics, same downstream pipeline.
2. Compare:
   - **Quality**: PSNR, SSIM, MSE, LPIPS
   - **Efficiency**: reference acquisition time, total latency, peak GPU memory
   - **Security**: correct-token vs wrong-token gap
3. **Do not claim RACS is faster until measurements demonstrate it.**

**Gate**: A comparison table exists with honest numbers.

---

## Milestone M10 — Intelligent Variants (Phases 17–19)
*Time estimate: 5–7 days*

### Phase 17: VLM/Agentic Reranking (RACS-Agent)

> **What**: Use a Vision-Language Model to evaluate candidates more intelligently.  
> **Why**: A VLM can assess semantic compatibility, layout, structure, visual similarity, and editing requirements — things CLIP + simple structure metrics can't.

> [!IMPORTANT]
> **Do NOT put a nondeterministic cloud LLM in the transmitter/receiver selection path.**
> The VLM must produce fixed scores → deterministic ranking → token selection.

**Gate**: Compare RACS-Base vs RACS-Agent. Does the VLM actually improve retrieval enough to justify its cost?

---

### Phase 18: Lightweight Editing (RACS-AgentEdit)

> **What**: If the retrieved reference is semantically good but structurally imperfect, apply a light edit.  
> **Why**: This might bridge the quality gap between pure retrieval and full generation.

> [!WARNING]
> Do NOT turn this into full diffusion generation. The editing must be substantially cheaper than generation. Otherwise you've recreated the problem RACS is trying to solve.

**Gate**: Compare quality gain vs additional latency. Is RACS-AgentEdit better than RACS-Agent?

---

### Phase 19: Retrieve-or-Generate Hybrid (RACS-Hybrid)

> **What**: If retrieval quality is too low, fall back to diffusion generation.  
> **Why**: This gives the best of both worlds — fast retrieval when it works, full generation only when needed.

**Build tasks:**
1. Define a quality threshold: if candidate score < threshold, fall back to generation.
2. The threshold must be **deterministic** and **identical at Tx and Rx**.
3. Compare: always-generate vs always-retrieve vs retrieve+edit vs retrieve-or-generate.

**Gate**: The hybrid system is compared against all other variants on quality and cost.

---

## Milestone M11 — Security, Evaluation, Ablations (Phases 20–24)
*Time estimate: 5–7 days*

### Phase 20: Security Testing
Test correct token, wrong token, no token. Measure recovery quality degradation. Use a proper test set and report distributions, not single images.

### Phase 21: Steganalysis
Test whether stego images can be detected by XuNet / SRNet. Compare detection rates for AgentSemSteCom vs RACS. Non-blocking.

### Phase 22: Full Evaluation Matrix
Systems × Image classes × SNR × Token conditions → Quality + Efficiency + Security metrics.

### Phase 23: Ablation Studies
- CLIP only vs CLIP + structure
- Top-K sweep (1, 5, 10, 20, 50)
- With/without token selection
- With/without VLM
- With/without editing
- Always-retrieve vs retrieve-or-generate

### Phase 24: Efficiency Analysis
Measure: T_generation, T_retrieval, T_reranking, T_editing, T_total, peak GPU memory.

**Gate**: Numbers support or refute each hypothesis, with distributions not anecdotes.

---

## Milestone M12 — Research Output (Phases 25–27)
*Time estimate: 3–5 days*

### Phase 25: Research Analysis
Answer the 12 final questions from the overview (quality, efficiency, security, robustness, retrieval effectiveness, adaptive architecture).

### Phase 26: Final Tables & Figures
Quality table, efficiency table, security table, ablation table. Key figures: architecture comparison, latency bars, PSNR vs SNR curves, Top-K effect, token security.

### Phase 27: Reproducibility Package
Freeze: dataset version, database, FAISS index, embedding model, preprocessing, seeds, token mapping, Top-K, ranking weights, editing parameters, EDICT parameters, SNR, test set.

---

## Summary: The Sprint Path

If you want a fast first pass, here's the critical path to a working RACS-Base:

| Sprint | Phases | What You Get |
|--------|--------|-------------|
| **Sprint 1** | 1–2 | Environment + VAE understanding |
| **Sprint 2** | 3 | "Generation costs X seconds" — the number RACS must beat |
| **Sprint 3** | 4–7 | `query → CLIP → FAISS → token → selected image` — RACS retrieval works |
| **Sprint 4** | 8–9 | Structural reranking improves candidate quality |
| **Sprint 5** | 10–11 | EDICT steganography works locally (hide + recover) |
| **Sprint 6** | 12 | RACS + EDICT — the core contribution works locally |
| **Sprint 7** | 13–15 | Full RACS-Base with communication channel |
| **Sprint 8** | 16 | "Here's how RACS compares to the baseline" |

Everything after Sprint 8 is **extensions and evaluation** that make the research stronger, but Sprint 8 is the minimum viable research result.

---

## My Recommendation

> [!TIP]
> **Start with Phase 1 right now.** Create the project skeleton, set up seeds, run something on GPU. It's the smallest phase and builds confidence. Don't overthink it — just get Python + PyTorch + GPU working and printing tensor shapes.

The plan in [_implplan.txt](file:///d:/racs2/phase2/_implplan.txt) and [RACS_ROADMAP.md](file:///d:/racs2/docs/RACS_ROADMAP.md) are solid. I've restructured and simplified them here, but the substance is the same. The key adjustments I'd make:

1. **Don't rush to EDICT.** Phases 1–7 build your understanding progressively. EDICT (Phase 10) is the hardest concept — arrive there with a strong foundation.
2. **Start with a small corpus** (1k images). Scaling to 10k+ is a later problem.
3. **Use `IndexFlatIP`** for FAISS — it's exact and deterministic. Speed doesn't matter at 1k–10k scale.
4. **Keep the HuggingFace blocker in mind** — Phase 2 and 3 need model downloads. Test connectivity first.
5. **Phase 15's gate is the hardest**: `Xref_tx == Xref_rx`. Design for determinism from day one.
