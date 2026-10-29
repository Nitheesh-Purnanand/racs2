# Contribution 2 — Exact-Recovery Layer for Generative Semantic Steganography

## Problem

AgentSemSteCom (and RACS, which keeps its downstream) hides the secret by EDICT
diffusion inversion. Recovery is **approximate by construction**: on our evaluation
split the correctly-keyed receiver reaches 22.37 dB mean PSNR (Phase 15), and faces
in particular come back visibly altered. For many uses (documents, evidence, medical
or ID images) an approximate secret is not acceptable.

## Idea

Keep the generative stego as the *visible carrier* — it still looks like the paper's
output (secret layout, retrieved-reference appearance) — and carry the **exact secret**
inside it with a keyed, authenticated, content-adaptive embedding layer:

1. **Lossless compaction.** The secret is re-encoded as lossless WebP when that is
   smaller (every pixel identical): 386 KB → 261 KB mean on the evaluation split (-32 %).
2. **Encryption and authentication.** PBKDF2-SHA256 (200k iterations) over the shared
   token derives a SHAKE-256 keystream, an HMAC-SHA256 key and the embedding-order seed.
   A wrong key is *rejected* (MAC), never decoded into a plausible-looking wrong image.
3. **Texture-adaptive ±1 embedding.** One bit per value, only in the 50 % most textured
   channel values, at 0.25 bits per value (the carrier is enlarged to fit). Texture is
   computed on the top 5 bits, and the ±1 step never changes those bits, so the receiver
   recomputes the *same* positions from the stego alone — no side information.
   Changes are ±1 (LSB matching) rather than LSB replacement, except at the two edges of
   an 8-value block where only one direction preserves the top bits.
4. **Dithered carrier.** The 256 px generative stego is upscaled and given camera-like
   noise (σ = 1 grey level) before embedding, because an upscaled image has an unnaturally
   clean LSB plane. The dither belongs to the cover source, so payload-free carriers are
   produced the same way.

The receiver needs only the stego PNG and the token: no diffusion model, no reference.

## Evaluation

`scripts/c2_make_covers.py` (66 EDICT carriers, strength 0.5, RACS-retrieved Xref) then
`scripts/c2_evaluate.py`. Results: `experiments/results/c2_exact_recovery.{md,json}`.
Every stego is written to and read back from a real PNG before extraction.

| layer | pixel-exact | wrong key rejected | payload KB | bits/value | PSNR vs carrier dB | chi-square AUC | SPA AUC | embed s | extract s |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A naive 2-LSB, original file | 100% | 100% | 386 | 1.997 | 44.11 | 1.000 | 1.000 | 0.27 | 0.27 |
| B naive 2-LSB + lossless compaction | 100% | 100% | 261 | 1.996 | 44.11 | 1.000 | 1.000 | 0.22 | 0.22 |
| C adaptive +-1 + compaction | 100% | 100% | 261 | 0.250 | 57.16 | 0.500 | 1.000 | 2.16 | 2.13 |
| D **C + dithered carrier (proposed)** | 100% | 100% | 261 | 0.250 | 57.16 | 0.500 | 0.829 | 2.51 | 2.17 |

Detectors are the classical structural attacks (Westfeld chi-square, Dumitrescu sample
pair analysis, `security/lsb_detect.py`); AUC is computed between each stego and the
same carrier without payload, so 0.5 means the detector cannot tell them apart.

## Findings

- **Exact recovery**: 66/66 secrets pixel-identical through a real PNG round trip, for every
  layer, against 22.37 dB approximate recovery from the generative pipeline alone.
- **Wrong key**: 66/66 rejected by the MAC (no wrong-but-plausible output).
- **Compaction** cuts the payload by 32 % (386 -> 261 KB), which is what makes a low
  embedding rate affordable.
- **Chi-square is defeated** by the adaptive +-1 layer: AUC 1.000 -> 0.500 (chance).
- **SPA is weakened, not defeated**: the estimated embedding rate drops from 0.40 (naive) to
  0.09, and AUC from 1.000 to 0.829 once the carrier carries realistic sensor noise
  (0.0 -> 0.045 SPA on clean carriers). Without the dither, the unnaturally clean upscaled
  carrier keeps SPA at AUC 1.000 even at 0.25 bits/value — the cover source matters as much
  as the embedding.
- **Distortion**: 57.2 dB PSNR against the payload-free carrier (naive: 44.1 dB).
- **Cost**: about 2.5 s to embed and 2.2 s to extract on the CPU, versus ~20 s of diffusion
  for the generative stego itself and no model at all on the receive side.

## Limitations

- The layer is fragile by design: any recompression (sending a "photo" on a messenger)
  destroys the payload. The MAC detects this and the receiver reports it; it never
  returns a corrupted image. Send the stego as a file.
- Detectability is measured against classical attacks only. A trained CNN steganalyser
  (e.g. SRNet) on this cover source is future work.
- The 8-digit token is short; PBKDF2 slows brute force but a longer key is advisable.
- The carrier is larger than the generative stego (mean 2.85 Mpx vs 0.07 Mpx at 256 px).

## Code

`app/lossless.py` (`compact`, `embed_adaptive`, `extract_adaptive`, `prepare_carrier`),
`security/lsb_detect.py`, `scripts/c2_make_covers.py`, `scripts/c2_evaluate.py`.
Used by the phone app (`app/server.py`, hybrid and lossless modes).
