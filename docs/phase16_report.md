# Phase 16 Report — Baseline Security and Steganalysis

> **Status**: complete, frozen, verify **17/17 PASS**.
> **Data rule**: evaluation split only. No calibration record appears in either table.
> `approximate: false` for every number below; paper values are approximate anchors only.

## 1. Receiver conditions

The transmitter is identical in all three conditions: the same frozen Phase 15 stego image
is delivered to the receiver. Only the receiver's token, derived reference seed, mask, and
inverse-sampler noise change.

| Condition | Token | Reference seed | Mask | Noise |
| :-- | :-- | :-- | :-- | :-- |
| `correct` | true token from `tokens.json` | `derive_seed(token)` | Phase 15 token delta | Phase 15 token noise |
| `wrong` | valid `wrong_token` from `tokens.json` | `derive_seed(wrong_token)` | `token_delta(wrong_token)` | `token_noise(wrong_token)` |
| `no_token` | fixed public fallback `00000000` | `1102030775` | all-zero (no perturbation) | `token_noise("00000000")` |

`no_token` is not a wrong-token run: it never reads `tokens.json`, and its mask is exactly
zero. Fallback mask seed is `1693947931`. Every per-case token, seed, mask hash, and noise
hash is recorded in `phase16_manifest.json` (264 cases x 3 conditions).

## 2. Security result

264 cases (66 evaluation images x 4 SNRs), N=22 per class per SNR.

| Class | SNR | correct | wrong | no_token | gap vs wrong | gap vs no_token |
| :-- | --: | --: | --: | --: | --: | --: |
| common | 5 | 13.14 | 11.49 | 11.50 | 1.65 | 1.64 |
| common | 10 | 15.65 | 12.52 | 12.61 | 3.13 | 3.04 |
| common | 15 | 17.78 | 13.03 | 13.05 | 4.74 | 4.72 |
| common | 20 | 19.19 | 13.31 | 13.39 | 5.88 | 5.80 |
| facial | 5 | 14.36 | 12.41 | 12.50 | 1.95 | 1.86 |
| facial | 10 | 17.47 | 13.77 | 13.71 | 3.69 | 3.75 |
| facial | 15 | 20.21 | 14.41 | 14.30 | 5.80 | 5.91 |
| facial | 20 | 22.47 | 14.76 | 14.77 | 7.71 | 7.70 |
| style | 5 | 13.60 | 11.56 | 11.60 | 2.04 | 2.00 |
| style | 10 | 16.53 | 12.72 | 12.75 | 3.81 | 3.78 |
| style | 15 | 18.46 | 13.16 | 13.19 | 5.31 | 5.27 |
| style | 20 | 19.78 | 13.43 | 13.42 | 6.35 | 6.36 |

PSNR in dB, means over N=22. Aggregate (weighted by cell N):
**17.39 dB correct / 13.05 dB wrong / 13.07 dB no_token**.

- Correct-minus-wrong gap: **4.34 dB**, above the 4 dB threshold.
- Correct-minus-no_token gap: **4.32 dB**.
- Wrong and no_token are nearly identical in every cell, as expected: both remove the
  secret-dependent mask and neither reconstructs the true reference.
- Six low-SNR cells fall below 4 dB (common 5/10, facial 5/10, style 5/10). The documented
  cause is the channel-noise floor: AWGN dominates the latent perturbation at 5 dB and
  compresses the difference at 10 dB. They are listed with a reason in
  `phase16_manifest.json`.

Grid: `experiments/results/phase16_security_grid.png`, 3 rows (common, facial, style) x
4 columns (secret, correct, wrong, no token), first image per class at 10 dB. Visually, the
correct column retains scene and face structure, while wrong and no-token are structured
failures with heavy colour noise. Neither failure column looks like a second valid recovery.

## 3. XuNet steganalysis

- **Architecture**: XuNet with a 5x5 SRM high-pass initialisation, five blocks
  (conv3-8 5x5, conv8-16 5x5, conv16-32 1x1, conv32-64 1x1, conv64-128 1x1, each with
  BN + Tanh + average pooling), global average pool, FC 128->2. The full architecture
  string is stored in the JSON and the checkpoint; it is not a generic CNN.
- **Data**: covers are the 66 secret evaluation images; positives are their frozen 10 dB
  stego. Four non-overlapping 128 px patches per image (2x2 grid).
- **Split**: grouped by `image_id`, 46/10/10 images, 368/80/80 samples. Image-disjoint by
  construction; class balance 50/50.
- **Training**: seed 42, batch 8, Adam lr 1e-3, 50 epochs configured, early stopping after
  10 epochs without validation improvement. Best epoch 16 (validation 0.5375); stopped
  at epoch 26.
- **Test**: accuracy **0.4875** (n=80, chance 0.5), precision 0.4921, recall 0.775,
  confusion `tp=31, tn=8, fp=32, fn=9`. Train 0.5082, validation 0.5375. The detector has
  no useful advantage on this baseline, which is the expected near-chance result.
- **Checkpoint**: `experiments/steganalysis/xunet_baseline_0000002a.pt`
  sha256 `243c21653e04e16402386ccbfe411ae23dbad2dcde363e2f0afabd187b9e6747`
  (72 KB, best-epoch state dict).

## 4. Anchors and frozen artifacts

| Item | Value | Label |
| :-- | :-- | :-- |
| Paper correct PSNR | 25.37 dB | approximate anchor |
| Paper wrong PSNR | 18.91 dB | approximate anchor |
| Paper gap | 6.46 dB | approximate anchor |
| Paper XuNet accuracy | 0.5091 | approximate anchor |
| Our correct/wrong/no_token PSNR | 17.39 / 13.05 / 13.07 dB | measured, `approximate: false` |
| Our XuNet test accuracy | 0.4875 | measured, `approximate: false` |

Frozen hashes (re-checked by `scripts/phase16_verify.py`):

- `baseline_security.json`: `8ab321a55d92a7f0982271435fa756e72995363e02cbb8d39dd5820029b5761f`
- `baseline_steganalysis.json`: `78db750b26c71ec25cdd3edddc3ed8e37efa9584f10d9041139ceac64fefd42b`
- Phase 15 `baseline_table.json`: `a6db25365c61b8cb00d5cbf13aa6e11980c4cb55cf4c5ebbca13c071177a7244`
- Split version: `e52d2ddfea4dba17ebde84a3b8c0129dca0ed928e69de3faad9746baeb7b330a`

XuNet config hash: `6716f17ac9787d428d86a267df180d34416f682580210889c69910e20aa87047`.

## 5. Gate question

> What does the correct-versus-wrong-token gap demonstrate, and why is the paper's
> 25.37 dB versus 18.91 dB only an anchor?

The gap shows that recovery depends on the secret token rather than on a generic decoder.
A receiver holding the true token can subtract the exact mask and invert the perturbation
with the matching noise draw. A receiver with a valid but wrong token derives a different
mask, reference seed, and noise, so the subtraction is misaligned and the inverse sampler
does not return the secret. A receiver with no token cannot subtract any mask at all. Both
failures collapse to a generic reconstruction, which is why their PSNR values are close to
each other and clearly below the correct condition.

This is signal-level key dependence, not a cryptographic security proof. Detectability is a
separate question, which is why XuNet is reported independently.

The paper's 25.37 versus 18.91 dB cannot be our target. Its data (UniStega), reference
generator (PicX-Real/IPAdapter), channel codec (SwinJSCC/MDJCM), resolution, and training
state all differ from our reproduction. Only the direction and approximate magnitude are
comparable. Storing the paper values beside ours with explicit approximate labels prevents
them from being mistaken for measurements on our evaluation split, and the 4 dB threshold
applies to our own aggregate, not to the paper's 6.46 dB.

## 6. Reproduce and verify

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase16_security.py        # ~38 min, GPU; writes security table + grid
.venv\Scripts\python.exe scripts\phase16_steganalysis.py --train
.venv\Scripts\python.exe scripts\phase16_manifest.py        # freeze per-case metadata + artifact hashes
.venv\Scripts\python.exe scripts\phase16_verify.py          # offline gate, 17 checks
```

Do not re-run Phase 16 to tune it. The Phase 15 stego is frozen, and the frozen artifacts
are hash-pinned by the manifest and the verifier.
