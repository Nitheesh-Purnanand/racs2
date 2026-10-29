# Contribution 2 — Exact-recovery layer for generative steganography

Evaluation split, n = 66 secrets. Carrier: paper-style EDICT stego (strength 0.5) with a RACS-retrieved reference.

Generative recovery alone (Phase 15, approximate): **22.37 dB** mean PSNR. With the layer below the secret is recovered **exactly**.

| layer | pixel-exact | wrong key rejected | payload KB | bits/value | PSNR vs carrier dB | chi-square AUC | SPA AUC | embed s | extract s |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| A naive 2-LSB, original file | 100% | 100% | 386 | 1.997 | 44.11 | 1.000 | 1.000 | 0.27 | 0.27 |
| B naive 2-LSB + lossless compaction | 100% | 100% | 261 | 1.996 | 44.11 | 1.000 | 1.000 | 0.22 | 0.22 |
| C adaptive +-1 + compaction | 100% | 100% | 261 | 0.250 | 57.16 | 0.500 | 1.000 | 2.16 | 2.13 |
| D **C + dithered carrier (proposed)** | 100% | 100% | 261 | 0.250 | 57.16 | 0.500 | 0.829 | 2.51 | 2.17 |

AUC = probability the detector scores the stego above its own clean carrier; 0.5 is chance, 1.0 is always detected. Classical structural detectors only; a trained CNN steganalyser is future work.
