# RACS2 Runbook: Phases 7-16

Scope: from public conditions to the frozen AgentSemSteCom baseline and its security profile.
Phase 4 (retrieval corpus) and Phase 5 (CLIP embeddings) are banked and consumed at Phase 17;
they are not rerun here. Phase 6 must already be complete.

All commands use the project root and the project virtual environment:

```powershell
cd D:\racs2
.venv\Scripts\python.exe --version
```

Do not use the system `python`. If a command fails with `ModuleNotFoundError`, the wrong
interpreter is being used.

**Important**: this runbook describes the **frozen verified state** (the artifacts currently in
`experiments/results/`). The working tree also contains an in-progress upgrade to paper components
(SwinJSCC, CodeFormer, PicX-Real, OneFormer/OpenPose, Qwen Kpub, IP-Adapter). Those assets are not
all present, so rerunning Phases 7-15 with today's `configs/base.yaml` will not reproduce the
frozen numbers. See section 15 before rerunning anything.

---

## 1. Order and entry condition

```text
Phase 6  splits + tokens + evaluation harness     DONE
   |
Phase 7  Kpub + Xfeat public conditions          -> 132 records
   |
Phase 8  reference generation                    -> 132 references
   |
Phase 9  ControlNet zero-conv fine-tune          -> checkpoint (calibration only)
   |
Phase 10 EDICT isolated invertibility            -> operating point
   |
Phase 11 local hide/recover                      -> matched/mismatched probe
   |
Phase 12 AWGN channel                            -> SNR curve
   |
Phase 13 semantic codec                          -> codec vs raw AWGN
   |
Phase 14 optional CodeFormer enhancement         -> off by default
   |
Phase 15 baseline gate (15A calibration + 15B evaluation) -> frozen table
   |
Phase 16 baseline security + XuNet               -> frozen security table
```

Entry checks:

```powershell
.venv\Scripts\python.exe scripts\phase6_eval_selftest.py
.venv\Scripts\python.exe scripts\phase5_verify.py
```

Expected: `83 passed, 0 failed`; Phase 5 verify PASS. Live protocol identifiers:

```text
split_version  : e52d2ddfea4dba17ebde84a3b8c0129dca0ed928e69de3faad9746baeb7b330a
tokens_version : 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
counts         : calibration 22/22/22, evaluation 22/22/22 (common/facial/style)
```

> The older runbook value `a57d9c8c...` predates the regenerated split manifest. Use the live
> value above; every frozen artifact in Phases 7-16 records `e52d2ddf...`.

---

## 2. Quick reference

| Phase | Command | Expected |
| :-- | :-- | :-- |
| 7 | `.venv\Scripts\python.exe scripts\phase7_build_conditions.py` | 132 cached records; `phase7_verify.json` PASS |
| 7 | `.venv\Scripts\python.exe scripts\phase7_verify.py` | 13/13 checks |
| 8 | `.venv\Scripts\python.exe scripts\phase8_reference.py` | 132 references at 256 px; `phase8_verify.json` PASS |
| 8 | `.venv\Scripts\python.exe scripts\phase8_verify.py` | 12/12 checks |
| 9 | `.venv\Scripts\python.exe scripts\phase9_train_controlnet.py` | 50 steps; checkpoint written; `phase9_verify.json` PASS |
| 10 | `.venv\Scripts\python.exe scripts\phase10_edict.py` | sweep + operating point; 445 s |
| 10 | `.venv\Scripts\python.exe scripts\phase10_verify.py` | 14/14 checks |
| 11 | `.venv\Scripts\python.exe scripts\phase11_local.py` | 20 matched + 20 mismatched; 469 s |
| 11 | `.venv\Scripts\python.exe scripts\phase11_verify.py` | 12/12 checks (artifact in `results/phase11/`) |
| 12 | `.venv\Scripts\python.exe scripts\phase12_awgn.py` | 4 SNR cells; 544 s |
| 12 | `.venv\Scripts\python.exe scripts\phase12_verify.py` | 7/7 checks |
| 13 | `.venv\Scripts\python.exe scripts\phase13_codec.py` | isolated + comparison; 1063 s |
| 13 | `.venv\Scripts\python.exe scripts\phase13_verify.py` | 9/9 checks |
| 14 | `.venv\Scripts\python.exe scripts\phase14_enhance.py` | pass-through deltas; 13 s |
| 14 | `.venv\Scripts\python.exe scripts\phase14_verify.py` | 13/13 checks |
| 15A | `.venv\Scripts\python.exe scripts\phase15_baseline.py --split calibration` | 264 cases; ~1022 s; approximate |
| 15B | `.venv\Scripts\python.exe scripts\phase15_baseline.py --split evaluation --resume` | 264 cases; ~2337 s |
| 15 | `.venv\Scripts\python.exe scripts\phase15_verify.py` | 12/12 checks, Gate A/B |
| 16 | `.venv\Scripts\python.exe scripts\phase16_security.py` | 264 cases x 3 conditions; ~2264 s |
| 16 | `.venv\Scripts\python.exe scripts\phase16_steganalysis.py --train` | XuNet checkpoint + test accuracy |
| 16 | `.venv\Scripts\python.exe scripts\phase16_verify.py` | 17/17 checks |

A full rerun from Phase 7 to 16 takes roughly 2.5-3.5 hours on the 6 GB GPU, dominated by Phases
8, 11-13, 15B and 16.

---

## 3. Phase 7: Public conditions (`Kpub` + `Xfeat`)

Build both roles (omit `--split` to do calibration and evaluation):

```powershell
.venv\Scripts\python.exe scripts\phase7_build_conditions.py
.venv\Scripts\python.exe scripts\phase7_verify.py
```

Useful variants:

```powershell
# rebuild one role, overwrite existing cache
.venv\Scripts\python.exe scripts\phase7_build_conditions.py --split evaluation --force
# smoke test only
.venv\Scripts\python.exe scripts\phase7_build_conditions.py --split calibration --limit 2
```

Data flow:

```text
secret image (512 px)
  -> BLIP caption (greedy, num_beams=1)
  -> fixed natural-language template
  -> Kpub string
  -> class-specific structural feature:
       common : SegFormer-B0 ADE20K segmentation map (512)
       facial : MediaPipe face-mesh skeleton
       style  : none (Kpub only)
  -> per-image npz cache + manifest
```

Verified output:

```text
records        : 132 (22 per class per role)
config_hash    : 6e5eec492b7af67cee8b35c18ff4d970f89f2397dd69a73e07fecb17004a4f64
split_version  : e52d2ddf...
verify checks  : 13/13 PASS
artifacts      : experiments/results/public_conditions/{image_id}.npz
                 experiments/results/public_conditions/manifest.json
                 experiments/results/phase7_conditions_grid.png
                 experiments/results/phase7_verify.json
```

Gate: all 132 records exist, counts are 22/22/22 per role, caches are deterministic (exact rerun
hashes), and `Kpub` is non-degenerate. **Do not regenerate conditions after Phase 15** unless you
intend to rerun Phases 8-16; every downstream artifact depends on the cache.

> The paper-path config (`local_llm`, OneFormer, OpenPose) is the in-progress upgrade. The frozen
> cache was built with BLIP + SegFormer + MediaPipe and is registered in `docs/DEVIATIONS.md`.

---

## 4. Phase 8: Reference generator

```powershell
.venv\Scripts\python.exe scripts\phase8_reference.py
.venv\Scripts\python.exe scripts\phase8_verify.py
```

Variants: `--split evaluation`, `--class facial`, `--limit 2`, `--force`.

Data flow:

```text
Kpub + Xfeat
  -> SD v1.5 fp16
     common : + seg ControlNet
     facial : + openpose ControlNet
     style  : SD v1.5 alone
  -> 256 x 256 reference image, 20 DDIM steps, guidance 7.5, eta 0
  -> save PNG + per-image hash
```

Verified output:

```text
references     : 132 (22 per class per role)
backend        : sd_controlnet
output_size    : 256, steps 20, guidance 7.5, eta 0.0
config_hash    : 75359372bebded8da9e411e5e9b516264a4845b14caa6f3aa9858625794bbfdc
peak VRAM      : 2.84 GB
verify checks  : 12/12 PASS
artifacts      : experiments/results/references/*.png
                 experiments/results/phase8_references.json
                 experiments/results/phase8_reference_grid.png
                 experiments/results/phase8_verify.json
```

Determinism: rerunning the same record must reproduce the same reference hash. If it does not,
stop; Phases 11-16 inherit the instability.

> The paper-path config (`backend: picx_real`, 50 steps) is the in-progress upgrade. The frozen
> references are SD v1.5 + ControlNet at 20 steps; the deviation is registered.

---

## 5. Phase 9: ControlNet zero-conv fine-tune

```powershell
.venv\Scripts\python.exe scripts\phase9_train_controlnet.py
.venv\Scripts\python.exe scripts\phase9_verify.py
```

Optional: `--max-steps 50`, `--size 256`, `--resume <checkpoint>`.

Training setup and verified result:

```text
role            : calibration only
trainable       : 11,479,680 parameters (zero-conv blocks)
frozen          : 349,799,440 parameters
steps           : 50, batch 1, bf16, lr 1e-5
loss            : 0.30998 -> 0.00304 (min 0.00290)
probe (n=10)    : zero-shot MSE 0.14883 -> fine-tuned 0.12959
peak VRAM       : 2.89 GB
checkpoint      : experiments/checkpoints/controlnet_zero_conv_50_88159c58.safetensors
checkpoint sha  : 65ce82a73638137525958dc2db952944f203e8427c62594930e19b4f3695dc16
verify checks   : 10/10 PASS
artifacts       : experiments/results/phase9_training.json, phase9_verify.json
```

Gate: only zero-conv parameters changed against the base ControlNet, the probe improves, and the
checkpoint hash is recorded. This is a short proof of the Algorithm 2 path, not a converged model.

Phase 8 references do **not** automatically use this checkpoint; the reference manifest records
which ControlNet fingerprint was used. Do not mix the two silently.

---

## 6. Phase 10: EDICT isolated invertibility

```powershell
.venv\Scripts\python.exe scripts\phase10_edict.py
.venv\Scripts\python.exe scripts\phase10_verify.py
```

Variants: `--quick`, `--latents N`, `--operating-only`, `--split evaluation`, `--limit N`.

Data flow:

```text
random latent or image latent (4 x 32 x 32)
  -> EDICT forward, 20 steps, mixing 0.93, guidance 1.0, fp16 UNet
  -> paired latent
  -> EDICT inverse
  -> compare to the original latent
```

Verified output:

```text
operating point : 20 steps, mixing 0.93
max abs error   : 9.49e-3
max rel error   : 3.58e-3
finite          : true
elapsed         : 446 s
verify checks   : 14/14 PASS
artifacts       : experiments/results/phase10_edict.json
                 experiments/results/phase10_edict_sweep.png
                 experiments/results/phase10_verify.json
```

Gate: inversion is deterministic and the error is small and explained. The paper's ~1e-6 floor
is not reachable at 256 px fp16; the measured floor is registered in `docs/DEVIATIONS.md`.
Do not proceed to Phase 11 while the inverse is unstable.

---

## 7. Phase 11: Local steganography core

```powershell
.venv\Scripts\python.exe scripts\phase11_local.py
.venv\Scripts\python.exe scripts\phase11_verify.py
```

Variants: `--class common|facial|style`, `--limit N`, `--reference matched|mismatched|both`,
`--force`.

Data flow:

```text
secret image + reference image + Kpub + token
  -> VAE encode secret
  -> token-derived noise
  -> bounded EDICT forward_img2img (strength 0.2, 20 steps, mixing 0.93, guidance 3.0)
  -> sparse token delta (5 percent of latent elements, eta 0.05)
  -> decode stego
  -> add / subtract the same delta and noise
  -> inverse EDICT
  -> recovered image + MSE/PSNR/SSIM/LPIPS
```

Verified output:

```text
n               : 20 matched + 20 mismatched (facial)
matched PSNR    : 26.77 dB mean
mismatched PSNR : 26.62 dB mean
stego PSNR      : 23.28 dB mean (vs secret)
elapsed         : 469 s
verify checks   : 12/12 PASS
artifacts       : experiments/results/phase11_local.json
                 experiments/results/phase11/*.png and *.pt
                 experiments/results/phase11/phase11_verify.json
                 experiments/results/phase11/phase11_grid.png
```

Gate: hide/recover works, reruns are deterministic, and the token mask is consistent between
hide and recover. Note the matched-minus-mismatched gap is only 0.15 dB: the semantic-reference
probe is weak, which is recorded. Do not present that gap as a security result (Phase 16 does
that properly).

---

## 8. Phase 12: AWGN channel

```powershell
.venv\Scripts\python.exe scripts\phase12_awgn.py
.venv\Scripts\python.exe scripts\phase12_verify.py
```

Variants: `--limit N`, `--force`.

Data flow:

```text
stego pixels [0,1]
  -> signal power P = mean(x^2)
  -> sigma^2 = P / 10^(SNR/10)
  -> fixed per-cell seed (sha256("phase12|<role>|<class>|<snr>"))
  -> received = x + N(0, sigma^2), not clipped before recovery
  -> recover
  -> metrics vs secret
```

Verified output:

| SNR | PSNR | SSIM | LPIPS |
| :-- | --: | --: | --: |
| 5 dB | 12.68 | 0.1321 | 0.8143 |
| 10 dB | 14.96 | 0.1929 | 0.7444 |
| 15 dB | 17.93 | 0.2955 | 0.6012 |
| 20 dB | 21.08 | 0.4316 | 0.4380 |

```text
monotone PSNR   : true
variance check  : analytic vs measured max relative error 0.46 percent
elapsed         : 544 s
verify checks   : 7/7 PASS
artifacts       : experiments/results/phase12_awgn.json
                 experiments/results/phase12/phase12_awgn_curve.png
                 experiments/results/phase12/phase12_verify.json
```

Gate: PSNR rises monotonically with SNR, seeds replay exactly, and the measured noise variance
matches the analytic value. The received signal is not clipped before recovery; clipping is for
PNG display only.

---

## 9. Phase 13: Semantic codec

```powershell
.venv\Scripts\python.exe scripts\phase13_codec.py
.venv\Scripts\python.exe scripts\phase13_verify.py
```

Optional `--limit N`.

Data flow (frozen verified path):

```text
stego pixels [0,1]
  -> SD VAE encode -> 4 x 32 x 32 latent symbols
  -> explicit power normalisation (invertible)
  -> AWGN at the cell SNR
  -> power denormalisation
  -> VAE decode
  -> compare: raw-pixel AWGN vs codec + AWGN
```

Verified output:

```text
isolated PSNR       : 17.88 / 22.88 / 26.70 / 28.51 dB at 5/10/15/20 dB
codec minus raw     : +1.45 / +2.27 / +2.33 / +1.40 dB
elapsed             : 1063 s
verify checks       : 9/9 PASS
artifacts           : experiments/results/phase13_codec.json
                      experiments/results/phase13/phase13_codec_grid.png
                      experiments/results/phase13/phase13_verify.json
```

Gate: the codec beats raw AWGN at every cell and the normalisation is exactly invertible.
The frozen artifact uses the SD-VAE substitute because the paper checkpoints were unavailable;
`docs/DEVIATIONS.md` records the compression-ratio and training-SNR mismatch.

> The working tree now points at SwinJSCC (`models/swinjscc/...`). That checkpoint is absent;
> rerunning Phase 13 today will fail. Fetch it and re-freeze, or restore the frozen config before
> rerunning. Do not mix SwinJSCC numbers with frozen SD-VAE artifacts.

---

## 10. Phase 14: Optional CodeFormer enhancement

```powershell
.venv\Scripts\python.exe scripts\phase14_enhance.py
.venv\Scripts\python.exe scripts\phase14_verify.py
```

Variants: `--limit N`, `--snr 5 10 15 20`.

Verified output (pass-through fallback, module off):

```text
enabled_by_default : false
class              : facial only
backend            : passthrough
input/output       : 256 -> 256
facial PSNR delta  : 0.0 dB at every SNR
facial LPIPS delta : 0.0
non-facial control : identical (unchanged)
deterministic      : true
elapsed            : 12.7 s
verify checks      : 13/13 PASS
artifacts          : experiments/results/phase14_codeformer.json
                     experiments/results/phase14/phase14_verify.json
```

Gate: the fallback is honest, deterministic, off by default, and restricted to facial. A zero
delta is the expected result, not a failure. If real CodeFormer weights are added later, this
phase must be rerun and the baseline frozen again before any comparison.

> The working tree now enables CodeFormer (`models/codeformer/codeformer.pth`). That file is
> absent; rerunning today will raise or fall back depending on the code path. Resolve before
> rerunning.

---

## 11. Phase 15: Baseline gate (15A + 15B)

15A is calibration-only and approximate. 15B is the authoritative evaluation baseline.

```powershell
.venv\Scripts\python.exe scripts\phase15_baseline.py --split calibration
.venv\Scripts\python.exe scripts\phase15_baseline.py --split evaluation --resume
.venv\Scripts\python.exe scripts\phase15_verify.py
```

Use `--class` and `--limit` only for smoke tests. Never mix a limited run into a frozen table.
`--resume` skips cases whose per-case JSON already exists; it is safe and recommended.

Data flow:

```text
secret -> VAE encode -> bounded EDICT hide (token noise + delta)
       -> stego -> semantic codec -> AWGN (5/10/15/20 dB)
       -> receiver inverse -> recovered -> metrics vs secret
```

Verified output:

```text
15A calibration : 264 cases, 0 failures, approximate=true, 1022 s
15B evaluation  : 264 cases, 0 failures, approximate=false, 2337 s
cells           : 12 (3 classes x 4 SNR), N=22 each
determinism     : reference/stego/recovered equal on 6 samples + fresh process
verify checks   : 12/12 PASS; gate_a_pass=false, gate_b_pass=true
artifacts       : experiments/results/baseline_table.json
                  experiments/results/baseline_manifest.json
                  experiments/results/phase15A_calibration_table.json
                  experiments/results/phase15A_manifest.json
                  experiments/results/baseline_cases/{role}/*.json and *.pt
                  experiments/results/baseline_grid.png
                  experiments/results/calibration_baseline_grid.png
                  experiments/results/phase15_verify.json
```

Frozen evaluation means (PSNR / SSIM / LPIPS):

| class | 5 dB | 10 dB | 15 dB | 20 dB |
| :-- | :-- | :-- | :-- | :-- |
| common | 13.14 / .2463 / .6829 | 15.65 / .3517 / .5742 | 17.78 / .4565 / .4476 | 19.19 / .5302 / .3527 |
| facial | 14.36 / .2844 / .6474 | 17.47 / .4253 / .5290 | 20.21 / .5570 / .3915 | 22.47 / .6544 / .2708 |
| style | 13.60 / .2539 / .6377 | 16.53 / .3736 / .4960 | 18.46 / .4593 / .3752 | 19.78 / .5178 / .2887 |

Gate A compares against paper anchors with tolerance. Eight of nine absolute tolerances miss;
every miss is itemised and accepted in `docs/DEVIATIONS.md`. Gate A is a regime check, not an
equality claim. Gate B (the real gate) requires monotone SNR, facial >= common ordering,
determinism, correct component choices, no retrieval imports, and both grids.

**Freeze rule**: once 15B passes, `baseline_table.json` and `baseline_manifest.json` are read-only.
Any regeneration invalidates Phases 16, 21 and 22 and requires rerunning them.

---

## 12. Phase 16: Baseline security and steganalysis

```powershell
.venv\Scripts\python.exe scripts\phase16_security.py
.venv\Scripts\python.exe scripts\phase16_steganalysis.py --train
.venv\Scripts\python.exe scripts\phase16_verify.py
```

Receiver conditions:

```text
correct   : true token, its seed and its mask
wrong     : a valid wrong_token from tokens.json
no_token  : fixed public fallback token 00000000, reference seed 1102030775,
            all-zero mask (a distinct condition, not a random-token attack)
```

Data flow:

```text
frozen Phase 15 stego
  -> receiver runs with correct / wrong / no_token
  -> metrics vs secret per cell (N=22)
  -> correct-minus-wrong and correct-minus-no-token gaps
  -> XuNet: stego vs cover, 2x2 128 px patches, image-disjoint 46/10/10 split
```

Verified output:

```text
security cases  : 264 (66 evaluation images x 4 SNR)
aggregate PSNR  : correct 17.39 / wrong 13.05 / no_token 13.07 dB
correct-wrong   : 4.34 dB (threshold 4 dB)
per-cell        : 6 low-SNR exceptions documented
XuNet samples   : 368/80/80 train/val/test
XuNet accuracy  : 0.4875 (n=80) vs chance 0.5
elapsed         : 2264 s (security run)
verify checks   : 17/17 PASS
artifacts       : experiments/results/baseline_security.json
                  experiments/results/baseline_steganalysis.json
                  experiments/results/phase16_manifest.json
                  experiments/results/phase16_security_grid.png
                  experiments/results/phase16_verify.json
                  experiments/steganalysis/xunet_baseline_*.pt
                  docs/phase16_report.md
```

Gate: correct-token PSNR exceeds wrong and no-token in aggregate, the gap is at least 4 dB in
aggregate, per-cell exceptions are itemised, the XuNet split is image-disjoint, and the checkpoint
hash is recorded. A near-chance detector is the expected baseline result; do not tune it.

---

## 13. Cross-phase invariants

1. **Role isolation.** Calibration never enters an evaluation table. `role=evaluation` results
   must not contain any `data/calibration/` image.
2. **Split and tokens are frozen.** `split_version e52d2ddf...`, `tokens_version 20166234...`.
   Regenerating either invalidates Phases 7-16 and 21-22.
3. **Same seed contract.** Tokens drive the mask and noise in Phases 11-16 through
   `reference/seed.py`; no second implementation.
4. **256 px protocol.** All references, EDICT, hide/recover and Phase 15/16 cases are 256 px.
   512 px is only the canonical dataset edge.
5. **No clipping before recovery.** AWGN and codec operate on real values; PNG clipping is display
   only.
6. **Float32 latents, fp16 UNet.** EDICT latents are float32; the UNet is fp16. Changing either
   changes the numerical floor.
7. **Read-only frozen artifacts.** After Phase 15/16, `baseline_table.json`,
   `baseline_manifest.json`, `baseline_security.json` and their verify files are never edited.
8. **Every artifact records provenance.** `config_hash`, `split_version`, `tokens_version`,
   code commit, model/checkpoint hashes, per-case hashes.
9. **Deviations are registered.** No entry, no claim. Append to `docs/DEVIATIONS.md`; never edit
   or delete a prior row.
10. **Do not tune on evaluation.** Component choices and tolerances are fixed before 15B.

---

## 14. Verified state (frozen artifacts)

```text
Phase 7 : PASS 13/13  config 6e5eec49...  132 condition records
Phase 8 : PASS 12/12  config 75359372...  132 references, 256 px, 2.84 GB peak
Phase 9 : PASS 10/10  50 steps, 11.48M trainable, probe 0.1488 -> 0.1296
Phase 10: PASS 14/14  20 steps / 0.93, max rel error 3.58e-3
Phase 11: PASS 12/12  matched 26.77 dB, mismatched 26.62 dB, stego 23.28 dB
Phase 12: PASS  7/7   12.68 / 14.96 / 17.93 / 21.08 dB at 5/10/15/20 dB
Phase 13: PASS  9/9   isolated 17.88 / 22.88 / 26.70 / 28.51 dB; +1.40 to +2.33 dB over raw
Phase 14: PASS 13/13  pass-through, 0 dB delta, disabled by default
Phase 15: PASS 12/12  Gate B true; 264 evaluation cases; 12 cells; N=22
Phase 16: PASS 17/17  correct/wrong/no_token 17.39 / 13.05 / 13.07 dB; XuNet 0.4875
split_version  : e52d2ddfea4dba17ebde84a3b8c0129dca0ed928e69de3faad9746baeb7b330a
tokens_version : 201662347a7f6dc1b90dd94b96feca867c93e1b273fed445752da1792b4294a1
baseline commit: 807d52e (Phase 16 freeze)
```

---

## 15. Rerunning after the paper-component upgrade

The working tree changes `configs/base.yaml` to paper components but several assets are missing:

```text
models/swinjscc/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C64.model   MISSING
models/codeformer/codeformer.pth                                        MISSING
models/ip-adapter/models/ip-adapter-plus_sd15.safetensors               MISSING
```

Consequences today:

- `phase15_verify.py` fails on component/config-hash drift (and the fresh-process probe).
- `phase13_codec.py` and `phase14_enhance.py` will fail or change behaviour.
- `phase7_verify.py` / `phase8_verify.py` will fail because the live config hash no longer matches
  the frozen caches/references.

Choose one state and record it:

**Option A - keep the frozen baseline**

```powershell
git stash push -- configs/base.yaml
.venv\Scripts\python.exe scripts\phase15_verify.py
.venv\Scripts\python.exe scripts\phase16_verify.py
git stash pop
```

This restores the verified configuration without touching code or artifacts. Do not run training
or generation while stashed.

**Option B - upgrade to paper components**

1. Fetch every missing asset and verify its SHA-256.
2. Rerun Phases 7-8 to rebuild conditions and references (new config hash).
3. Rerun Phases 10-14 to re-measure the operating point, channel, codec and enhancer.
4. Rerun 15A and 15B in full, then 16, and register every changed component in
   `docs/DEVIATIONS.md` as a new version (for example `baseline v2`).
5. Never mix `baseline v1` and `v2` artifacts in one table.

Phases 21-22 must compare systems measured under the **same** baseline version.

---

## 16. Troubleshooting

- **Wrong interpreter** (`ModuleNotFoundError`): use `.venv\Scripts\python.exe`.
  Check with `.venv\Scripts\python.exe -c "import sys; print(sys.executable)"`.
- **CUDA OOM**: close other GPU apps and browser tabs. The 6 GB card fits one SD/ControlNet path
  at a time. Do not run two phase scripts concurrently.
- **Windows SD load crash** (`0xC0000005`): free commit charge and retry. `phase15_verify.py`
  spawns its fresh-process probe with `-X faulthandler`; this is a known environment issue.
- **HF download resets**: use `scripts/phase5_fetch_model.py` for CLIP, or the local model paths in
  config. Do not rely on an interrupted `from_pretrained` download.
- **Missing LPIPS weights on first run**: needs network access once; afterwards it is cached.
- **Phase 11/12/13/14 verify path**: some verify files live under `experiments/results/phase11/`,
  `phase12/`, `phase13/`, `phase14/`, not at the results root. Check the phase directory.
- **Phase 15 resume confusion**: `--resume` reads `baseline_cases/{role}/*.json`; delete a case
  file only if you intentionally want to recompute it.
- **Determinism failure**: check the seed setter, the token map version, and whether the reference
  was regenerated. Do not "fix" determinism by averaging runs.
- **Gate A misses**: expected and documented. Do not tune the evaluation split to close them.
- **`phase15_verify` fresh-process failure**: rerun with no other GPU work; if it persists, record
  the command output and stop rather than marking the gate passed.

---

## 17. Handoff to Phases 17-20

When Phase 16 passes:

- Phases 15/16 artifacts are read-only. No RACS artifact may change them.
- Phase 17 starts from the banked Phase 4 corpus and Phase 5 embeddings.
- Phase 20's reranking A/B and Phase 21 depend on whether the downstream consumes reference
  pixels. In the frozen baseline it does not (IP-Adapter off, EDICT text-conditioned only), so
  read `PHASE_RUNBOOK_17_TO_20.md` section 6 before claiming a reference-quality improvement.

Next runbook: `PHASE_RUNBOOK_17_TO_20.md`.
