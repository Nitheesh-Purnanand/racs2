# Phase 11: Local Steganography Core (VAE + EDICT + Token Perturbation)

> **Goal**: Implement hide and recover with a manually chosen, semantically close reference,
> no channel and no retrieval.
> **Why**: If this does not work, nothing downstream can. It isolates the steganographic core
> from the reference-acquisition question.

> **Ordering rule**: Phase 11 consumes Phase 10 EDICT and the Phase 8 seed contract. It must not
> add AWGN, the semantic codec, VLM reranking or editing.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase10_verify.py
.venv\Scripts\python.exe scripts\phase2_vae.py --help
```

Expected: EDICT passes at float32 machine precision and the VAE round-trip is available.

---

## Step 1: Config

```yaml
steganography:
  eta: 0.05
  mask_fraction: 0.05          # fraction of latent elements perturbed
  mask_domain: "racs2-mask-v1"
  edict_steps: 50
  edict_mixing: 0.93
  latent_dtype: "float32"
  num_images: 20
  results_dir: "experiments/results/phase11"
  manifest_path: "experiments/results/phase11_local.json"
```

`eta = 0.05` is the paper's operating point. It is the perturbation strength, not a learning
rate.

---

## Step 2: Structure

```text
steganography/
  mask.py               # token -> deterministic binary mask
  hide.py               # secret + reference -> stego
  recover.py            # stego -> recovered secret
  pipeline.py           # shared loaders and helpers
scripts/
  phase11_local.py      # 20-image run + mismatched-reference probe
  phase11_verify.py
experiments/results/phase11/
  <image_id>_secret.png, _reference.png, _stego.png, _recovered.png
  phase11_local.json
  phase11_metrics.json
  phase11_grid.png
  phase11_verify.json
```

---

## Step 3: Implement `steganography/mask.py`

Use the Phase 8 mask seed contract exactly:

```python
import numpy as np

def token_delta(token: str, shape: tuple[int, ...] = (1, 4, 64, 64),
                fraction: float = 0.05) -> np.ndarray:
    seed = derive_mask_seed(token)          # from reference.seed
    rng = np.random.default_rng(seed)
    flat = rng.permutation(int(np.prod(shape)))[: int(round(fraction * np.prod(shape)))]
    delta = np.zeros(int(np.prod(shape)), dtype=np.float32)
    delta[flat] = rng.standard_normal(len(flat)).astype(np.float32)
    return delta.reshape(shape)
```

Rules:

- One mask per token, independent of the image.
- `fraction` and `mask_domain` are frozen in config and recorded in the manifest.
- The mask is applied identically by transmitter and receiver.

---

## Step 4: Implement hide and recover

### Hide

```text
secret X, reference Xref (manually chosen, semantically close)
  -> z_secret = VAE.encode(X) * 0.18215
  -> z_ref    = VAE.encode(Xref) * 0.18215
  -> z_hidden = EDICT_forward(z_secret)        # conditioned on the public Kpub/Xfeat of Xref
  -> delta = token_delta(token)                # deterministic from the token alone
  -> z_stego = z_hidden + eta * delta          # receiver can rebuild delta without Xref
  -> stego = VAE.decode(z_stego / 0.18215)
```

### Recover

```text
stego
  -> z_stego = VAE.encode(stego) / 0.18215
  -> delta = token_delta(token)
  -> z_hidden = z_stego - eta * delta
  -> z_secret_hat = EDICT_inverse(z_hidden)
  -> X_hat = VAE.decode(z_secret_hat / 0.18215)
```

Implement the perturbation sign and placement exactly once in `hide.py`, and import that
function in `recover.py` so the two cannot diverge. If the paper's exact perturbation is
ambiguous, use the symmetric token-derived additive form above, record it as a deviation, and
keep it identical in both directions. The delta must be derivable from the token alone; never
use the reference in the perturbation, or the receiver would need the reference to invert it.

The reference must be selected by hand for this phase: pick the nearest image from the same
class in the evaluation split, or a fixed curated pair. Do not use FAISS or retrieval.

---

## Step 5: Implement `scripts/phase11_local.py`

```powershell
python scripts\phase11_local.py --class facial --limit 20
python scripts\phase11_local.py --reference matched|mismatched
```

Algorithm:

```text
1. set_seed(42)
2. load 20 evaluation images of one class, fixed order
3. for each image:
     - choose the manually specified matched reference
     - hide, then recover
     - measure MSE, PSNR, SSIM, LPIPS(secret, recovered)
     - also measure stego imperceptibility PSNR/SSIM/LPIPS(secret, stego)
     - save the four PNGs and the hashes of z_hidden, z_stego, mask
4. repeat with a mismatched reference for the same images
5. write distributions (mean, median, std, min, max, N) per condition
6. write the grid: secret | reference | stego | recovered
```

Report the correct-reference and mismatched-reference results side by side. The mismatched
probe is the evidence that reference semantics matter.

---

## Step 6: Implement `scripts/phase11_verify.py`

Checks:

```text
[ ] 20 images recovered, no failures
[ ] correct-reference mean PSNR is finite and clearly above the mismatched-reference mean
[ ] all metrics reported as distributions with N=20
[ ] stego PSNR is high enough to be imperceptible (record the measured value)
[ ] mask is deterministic: same token -> same mask, two runs
[ ] hide/recover is deterministic: same inputs -> same stego and recovered hashes
[ ] no channel, codec, retrieval or VLM import appears in the stego modules
[ ] phase11_local.json and phase11_verify.json exist and the gate is written
```

Exit non-zero on failure.

---

## Step 7: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase11_local.py
.venv\Scripts\python.exe scripts\phase11_verify.py
```

Expected: local hide/recover works on 20 images, metrics are distributions, and the mismatched
reference degrades clearly. Inspect `phase11_grid.png` and confirm the recovered image preserves
the secret's identity and layout.

---

## Checklist

```text
[ ] delta derivation matches the Phase 8 contract and is deterministic
[ ] hide and recover share the perturbation implementation
[ ] reference is manually chosen, not retrieved
[ ] 20 images processed with per-image artifacts
[ ] MSE/PSNR/SSIM/LPIPS reported as distributions
[ ] mismatched-reference probe shows a clear drop
[ ] eta = 0.05 and steps/mixing match Phase 10
[ ] phase11_verify.json has pass=true
[ ] no AWGN, codec, VLM, editing or FAISS
```

**When this passes, Phase 11 is done.** Phase 12 adds the channel; nothing else changes.

---

## The Gate Question (answer in your own words)

> *Why isolate the local hide/recover step before adding the channel, and why does the
> mismatched-reference probe matter?*

Answer with your artifacts:

- With no channel and no retrieval, any failure is attributable to the VAE, EDICT or the
  perturbation, not to noise or a bad reference. That makes debugging tractable.
- The matched reference establishes the achievable quality floor. The mismatched probe shows
  how much of that quality depends on semantic proximity, which is exactly the quantity RACS
  later tries to improve.
- If matched and mismatched were identical, reference selection would be irrelevant and the
  whole retrieval premise would be in question.
- The distributions and the grid are the evidence.

---

## Report back

- Mask scheme, fraction, and one worked `token -> mask hash` example
- Hide/recover equations actually implemented
- Metric distributions for matched and mismatched references
- Stego imperceptibility numbers
- Determinism hashes for mask, stego and recovered images
- Grid path and what was visually confirmed
- Any deviation and its register row
- Your answer to the gate question

---

## Notes and risks

- **Do not tune eta on the evaluation split.** Choose it from the paper or from calibration.
- **A strong recovery metric with a visible stego artifact is a fail.** Report both sides.
- **The perturbation must be exactly invertible.** If it is not, the "channel" will look
  better than it is.
- **Reference choice is a confound.** Freeze the pair list and write it to the manifest before
  looking at results.
