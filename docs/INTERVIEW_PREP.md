# RACS2 — Interview Quick Revision (GenAI Project)

> **One line:** a Generative AI system that hides a secret image inside a diffusion-generated
> image and recovers it with a private key. We made the reference stage **retrieval-augmented**
> (82× faster, no GPU) and added an **exact-recovery** layer (pixel-identical secret, wrong
> keys rejected).

---

## 1. 30-second pitch

"I reimplemented AgentSemSteCom (IEEE TCCN 2026), an agentic GenAI pipeline for coverless
semantic steganography. An LLM writes a public prompt from the secret image, ControlNet and
Stable Diffusion generate a reference image, and EDICT, an exactly invertible diffusion
process, hides the secret so that only a receiver with the token can invert it. I made two
contributions. **RACS** replaces the per-message diffusion reference with CLIP + FAISS
retrieval, structural reranking and token-based selection: 3.92 s → 0.048 s, 2.6 GB → 0 GB
GPU, with equal or better quality and security. The **exact-recovery layer** carries the
encrypted, authenticated original inside the generative stego, so the secret comes back
pixel-identical (66/66) instead of at ~22 dB, and chi-square steganalysis drops to chance.
I also shipped it as a phone web app."

## 2. Is this a GenAI project? Yes. Here is why

| GenAI concept | Concrete use in RACS2 |
| :-- | :-- |
| Latent diffusion model | Stable Diffusion v1.5 / PicX-Real generates the reference `Xref` (baseline) and renders the stego |
| Diffusion inversion | EDICT maps image ↔ noise *exactly*, which is what makes hiding reversible |
| Conditional generation | ControlNet (segmentation / OpenPose) for structure, IP-Adapter-plus for image prompts |
| LLM / agentic step | BLIP caption → Qwen2.5-0.5B-Instruct rewrites it into the public prompt `Kpub` |
| Multimodal embeddings | CLIP ViT-B/32 puts text (`Kpub`) and images in one vector space |
| Retrieval-augmented generation | Retrieved `Xref` conditions the diffusion hiding step instead of a generated one |
| Generative restoration | CodeFormer (codebook + transformer prior) restores faces |
| Learned semantic communication | SwinJSCC neural codec + AWGN channel |

**Honest framing.** It is not an LLM chatbot or text RAG app. It is *generative-model
engineering*: diffusion, conditioning, inversion, multimodal retrieval and LLM prompting,
plus rigorous evaluation. Contribution 2 is classical cryptography and steganography layered
on the generative output. Say this plainly if asked; it shows you understand the system.

## 3. Architecture in 60 seconds

```
SENDER
Secret Xs ─► BLIP → Qwen LLM ─► Kpub (public text)
          ─► OneFormer / OpenPose ─► Xfeat (public structure)
Kpub ─► CLIP text vec ─► exact search over 1,000 images (top 50)
     ─► rerank 0.75·semantic + 0.25·structure (top 10)
     ─► SHA-256(token) mod 10 ─► Xref                       ← C1: RACS (replaces diffusion)
Xs + Kpub + Xref(IP-Adapter) + token noise/mask ─► EDICT ─► Xstego
Xstego ─► SwinJSCC ─► AWGN (5–20 dB)                        Path A (paper, approximate)
Xs ─► lossless WebP ─► PBKDF2 key ─► SHAKE-256 encrypt + HMAC
   ─► ±1 embed in the most textured 50% of Xstego ─► PNG    ← C2: Path B (exact)

RECEIVER
Path A: decode → same RACS retrieval → same Xref → remove mask → EDICT inverse → CodeFormer
Path B: recompute texture positions → read bits → verify HMAC → decrypt → exact Xs
```

**Core invariant:** `Xref_tx == Xref_rx` bit-for-bit. If the two sides get different
references, recovery fails. Verified 66/66 across separate processes.

## 4. Key numbers (memorise these)

| What | Value |
| :-- | :-- |
| Reference acquisition, diffusion → retrieval | **3.92 s → 0.048 s (82×)** |
| Reference GPU memory | **2.61 GB → 0 GB** |
| Recovery PSNR, baseline vs RACS (10 dB SNR subset) | 22.37 → **22.62 dB** (+0.25 paired) |
| SSIM / LPIPS | 0.616 → 0.627 / 0.238 → 0.227 |
| Security gap (correct − wrong token) | 8.67 → **8.94 dB** (paper: 6.46) |
| XuNet steganalysis accuracy | **0.50** (chance; paper 0.509) |
| C2 exact recovery | **100% pixel-identical (66/66)**, 100% wrong-key rejection |
| C2 payload after lossless WebP | 386 KB → **261 KB (−32%)** |
| C2 embedding rate / PSNR vs carrier | 0.25 bits/value / **57.2 dB** |
| C2 chi-square AUC / SPA AUC | **0.50** (chance) / 0.83 |
| Hardware | RTX 3050 Laptop, **6 GB VRAM** |

## 5. Quick-revise concepts

**Steganography vs encryption.** Encryption hides *what* the message says; steganography
hides *that a message exists*. **Coverless:** no existing cover is modified; the stego is
synthesised.

**Diffusion model.** It learns to reverse a gradual noising process. *Latent* diffusion runs
in a VAE latent (64×64×4 for 512 px), which makes it cheaper. DDIM gives deterministic sampling.

**Why EDICT, not DDIM inversion?** DDIM inversion is only approximate because of a
linearisation error. EDICT keeps two coupled latent sequences (like affine coupling layers),
so every step is algebraically invertible and the receiver can undo the sender's steps
exactly. What remains lossy is the VAE and the channel.

**ControlNet.** A trainable copy of the UNet encoder, attached through zero-initialised
convolutions, that injects spatial conditions (segmentation maps, poses) without damaging the
base model.

**IP-Adapter.** Decoupled cross-attention that adds *image* embeddings as a prompt next to the
text embeddings. Here it makes the stego depend on the reference image's pixels.

**CLIP.** Contrastive image–text model. Text and image vectors share a space, so cosine
similarity ranks images for a text query. Vectors are L2-normalised and searched with
inner product (`IndexFlatIP`).

**FAISS exact vs ANN.** We use *exact* flat search on purpose. Approximate indexes (IVF, HNSW)
can return different neighbours under tiny numerical differences, which would break the
`Xref_tx == Xref_rx` invariant.

**Reranking.** CLIP matches meaning but not layout. Final score = 0.75·semantic +
0.25·structural (edge-map similarity against the public `Xfeat`). The weights were fixed on
calibration images only, never on evaluation data.

**Token selection.** `SHA-256(token) mod K`. It is deterministic across processes and
machines, unlike Python's salted `hash()`. An attacker without the token has a 1/K chance of
picking the right reference.

**Metrics.** PSNR (pixel error, dB, higher is better), SSIM (structure, 0–1), LPIPS
(perceptual distance from deep features, lower is better), MSE. The security gap is the PSNR
with the right token minus the PSNR with a wrong one. Detector AUC: 0.5 means the detector
cannot tell stego from clean.

**Exact-recovery crypto.** PBKDF2-SHA256 with 200k iterations slows brute force of the
8-digit token. A SHAKE-256 keystream gives encryption. HMAC-SHA256 gives authentication: a
wrong key or a tampered image is *rejected*, never shown as garbage. This is
encrypt-then-MAC.

**Why ±1 (LSB matching), not LSB replacement?** Replacement creates "pairs of values"
artefacts that the chi-square attack detects. ±1 changes avoid that, which is why chi-square
AUC drops from 1.00 to 0.50.

**Why dither the carrier?** An upscaled image has an unnaturally smooth LSB plane, so any
embedding stands out. Adding σ = 1 camera-like noise makes clean and stego carriers look alike
(SPA AUC 1.00 → 0.83).

**Texture-adaptive positions without side information.** Texture is computed on the top 5
bits only, and ±1 never changes those bits, so the receiver recomputes the same positions from
the stego alone.

## 6. Engineering highlights

- **6 GB VRAM budget:** fp16, model offloading, CodeFormer and the IP-Adapter image encoder on
  the CPU, a read-only safetensors mmap shim (`common/hf_lowmem.py`) so checkpoint loads don't
  exhaust the Windows paging file.
- **Process isolation:** FAISS and torch in one process trigger `OMP Error #15`, and CLIP text
  encoding segfaulted once diffusion was loaded. The fix was a separate worker process
  (`app/public_worker.py`).
- **Reproducibility:** SHA-256-pinned weights (`models/PAPER_ASSETS.json`), frozen role-tagged
  splits (calibration vs evaluation), config hashes and code commit in every report,
  determinism checks (reference/stego/recovered equal across runs).
- **Controlled experiment:** `RACSBase` subclasses `Baseline` and overrides only
  `reference_path`, so any difference is attributable to the reference stage.
- **Honest deviations:** 256 px, 20 EDICT steps and SwinJSCC C = 64 instead of the paper's
  setting, all recorded in `docs/DEVIATIONS.md` ("no entry, no claim").

## 7. Interview questions with answers

### Fundamentals

**Q1. What problem does the project solve?**
Covert image transmission. The baseline regenerates a reference image with diffusion for
every message (slow, GPU-heavy) and recovers the secret only approximately. RACS makes the
reference cheap; C2 makes recovery exact.

**Q2. What does "coverless" mean?**
You don't take an innocent photo and modify it. The stego image is produced by a generative
model, so there is no original cover to compare against.

**Q3. Why does the receiver need the same reference image?**
The stego is produced by EDICT conditioned on `Xref` through IP-Adapter. Inverting with a
different conditioning gives a different latent trajectory, so the secret does not come back.

**Q4. What is public and what is private?**
Public: `Kpub` (text prompt) and `Xfeat` (structure). Private: the 8-digit token, which seeds
the reference choice, the noise and the latent mask, and (in C2) derives the encryption keys.

**Q5. Why is the baseline's recovery approximate even though EDICT is exact?**
The VAE encoder and decoder are lossy, and the AWGN channel plus the neural codec add noise.
EDICT is exact only in latent space.

### GenAI depth

**Q6. Explain the forward and reverse processes of diffusion.**
Forward: add Gaussian noise over T steps until the image is pure noise. Reverse: a UNet
predicts the noise at each step and a scheduler (DDIM) removes it. Classifier-free guidance
mixes conditional and unconditional predictions.

**Q7. DDIM inversion vs EDICT?**
DDIM inversion assumes the noise prediction at step t ≈ step t+1, so errors accumulate. EDICT
uses two interleaved latents updated alternately with a mixing coefficient (0.93), which is
invertible in closed form.

**Q8. How does ControlNet avoid ruining the pretrained model?**
It copies the encoder, trains only the copy, and connects it through zero-initialised
convolutions, so at the start of training it adds exactly nothing.

**Q9. What does IP-Adapter add over text prompting?**
A separate cross-attention path for image embeddings, so a reference image can steer
generation without fine-tuning the UNet.

**Q10. Where is the LLM, and why is it "agentic"?**
BLIP perceives the image (caption) and Qwen2.5 reasons and rewrites the caption into a
generation prompt. That is a perceive → reason → act loop that produces a tool input (the
prompt) for the diffusion model. Greedy decoding keeps it deterministic.

**Q11. Is RACS "RAG"?**
It is retrieval-augmented *generation* in the multimodal sense. The diffusion hiding step is
conditioned on a retrieved image instead of a freshly generated one. It is not text RAG, since
nothing is retrieved into an LLM context.

### Retrieval

**Q12. Why exact search instead of HNSW/IVF?**
Determinism. Both ends must rank identically. With 1,000 images exact search is microseconds,
so approximation buys nothing and risks the core invariant.

**Q13. Why rerank? Why 0.75/0.25?**
CLIP ignores layout. Adding structural similarity against public `Xfeat` picks references that
fit the secret's composition. The weights came from a calibration-only sweep over
α ∈ {0, 0.25, 0.5, 0.75, 1}; reranking changed the top-10 pool for 67% of images.

**Q14. Couldn't an attacker just run the same retrieval?**
Yes, they get the same top 10, because it uses only public data. The token picks one of the 10
and also seeds the noise and mask, so retrieval alone does not reveal the secret. The measured
wrong-token PSNR is ~14 dB, which is garbled.

**Q15. How would you scale the corpus to 1M images?**
Still use deterministic search: either exact search on GPU, or IVF with fixed, versioned
training and `nprobe` shipped to both sides, plus a hash of the index file. Add a corpus
version to the protocol.

### Security and steganography

**Q16. Why encrypt-then-MAC?**
The receiver verifies integrity before decrypting, so wrong keys and tampered files are
rejected cleanly and no padding or oracle tricks are possible.

**Q17. An 8-digit token is only 10^8 keys. Isn't that weak?**
Yes, and it is listed as a limitation. PBKDF2 with 200k iterations makes each guess expensive.
For production, use a longer passphrase or a key exchange (e.g. X25519).

**Q18. What does XuNet accuracy 0.50 mean?**
A CNN steganalyser trained to separate stego from clean images does no better than a coin
flip, i.e. it cannot detect the stego. Caveat: n = 8 test images.

**Q19. What breaks C2?**
Any lossy recompression (WhatsApp photo mode, JPEG). The payload is destroyed, but the HMAC
detects it, so the user gets an error rather than a wrong image. That is why the stego is sent
as a file.

**Q20. Why is SPA still at 0.83?**
Sample pair analysis exploits the statistics of pairs of neighbouring values. Dither weakens it
but doesn't remove it. Next steps would be syndrome-trellis coding (lower change rate) or
HILL/WOW cost functions.

### Evaluation and engineering

**Q21. How did you make the comparison fair?**
The same images, tokens, SNRs, codec and downstream code, and one code path. `RACSBase`
overrides only the reference method. Calibration and evaluation splits are role-tagged and
frozen by SHA-256.

**Q22. Biggest engineering challenge?**
Running SD, ControlNet, IP-Adapter, CodeFormer and SwinJSCC on 6 GB. The fixes were fp16,
moving models between CPU and GPU, an mmap safetensors shim, and isolating FAISS/CLIP in a
subprocess after OpenMP and segfault conflicts.

**Q23. What are the limitations?**
Small evaluation subsets (2 images per class for the C1 quality numbers; the invariant is
checked on 66), classical detectors only for C2 (no SRNet), a short token, a 256 px research
setting, and one paper branch (MDJCM) not reproducible without a public checkpoint.

**Q24. How would you productionise it?**
Put the model server behind a queue (one GPU worker), add an HTTPS reverse proxy, upload
limits (done: `RACS_MAX_UPLOAD_MB`), auth and rate limiting, delete uploads after a TTL,
health checks, containerise with pinned CUDA, and offer lossless mode on CPU-only nodes.

**Q25. What would you do next?**
Run the full 66-image C1 evaluation, train SRNet against C2, use STC embedding, add key
exchange instead of a shared token, and use a larger, versioned retrieval corpus.

## 8. Rapid-fire (one-liners)

- **Kpub?** The LLM-written public text prompt.
- **Xfeat?** Public structure: a segmentation map (scenes) or a pose (faces).
- **Xref?** The reference image that conditions hiding; must match on both ends.
- **Why SHA-256, not `hash()`?** Python's `hash()` is salted per process.
- **CLIP dim?** 512 (ViT-B/32).
- **Corpus?** 1,000 COCO val2017 images at 512 px, center-cropped.
- **Top-k?** Pool 50 → rerank → top 10 → token picks 1.
- **SNRs?** 5, 10, 15, 20 dB AWGN.
- **EDICT mixing?** 0.93, 20 steps (paper: 50).
- **CodeFormer fidelity?** 0.5, faces only.
- **Exact rate?** 66/66.
- **Speed-up?** 82×.

## 9. Resume bullets

- Built a retrieval-augmented GenAI pipeline for coverless semantic steganography. It
  replaces per-message Stable Diffusion + ControlNet reference generation with CLIP + FAISS
  retrieval, structural reranking and token selection, cutting reference latency **82×
  (3.92 s → 0.05 s)** and GPU memory **2.6 GB → 0** with no quality or security loss.
- Designed an exact-recovery layer (PBKDF2 + SHAKE-256 + HMAC, texture-adaptive ±1
  embedding) over EDICT diffusion stego: **100% pixel-identical recovery**, wrong keys
  rejected, chi-square steganalysis at chance.
- Reproduced an IEEE TCCN 2026 paper end to end on a 6 GB GPU (EDICT, IP-Adapter, Qwen2.5,
  SwinJSCC, CodeFormer), with SHA-256-pinned assets, frozen splits and PSNR/SSIM/LPIPS/XuNet
  evaluation, and shipped it as a phone web app.
