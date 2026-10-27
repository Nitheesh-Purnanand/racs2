# Phase 12: AWGN Channel

> **Goal**: Insert a realistic additive white Gaussian noise channel between stego transmission
> and recovery, over SNR 5, 10, 15 and 20 dB.
> **Why**: The paper reports results across this SNR range. This is the first point where the
> pipeline spans a channel and the first place to verify the noise model analytically.

> **Ordering rule**: Phase 12 extends Phase 11 only. It must not add the semantic codec, the
> VLM, editing or retrieval. The codec arrives in Phase 13.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase11_verify.py
```

Expected: local hide/recover passes on 20 images with distributions recorded.

---

## Step 1: Config

```yaml
channel:
  kind: "awgn"
  snr_db: [5, 10, 15, 20]
  h: 1.0
  complex_noise: true
  seed: 1234
  results_dir: "experiments/results/phase12"
  manifest_path: "experiments/results/phase12_awgn.json"
```

The channel seed is separate from the reference seed. It must be fixed per SNR cell so the
same cell is reproducible.

---

## Step 2: Structure

```text
communication/
  __init__.py
  awgn.py               # SNR -> noise power, complex noise, signal scaling
scripts/
  phase12_awgn.py       # full pipeline across SNR, distributions, grids
  phase12_verify.py
experiments/results/phase12/
  phase12_awgn.json
  phase12_awgn_curve.png
  phase12_verify.json
```

---

## Step 3: Implement `communication/awgn.py`

### API

```python
def signal_power(x: torch.Tensor) -> torch.Tensor: ...
def noise_variance_for_snr_db(x: torch.Tensor, snr_db: float) -> torch.Tensor: ...
def awgn_channel(x: torch.Tensor, *, snr_db: float, seed: int,
                 h: complex = 1 + 0j) -> tuple[torch.Tensor, dict]: ...
```

### Model

```text
received = h * signal + n,   n ~ CN(0, sigma^2)
sigma^2 = P_signal / 10^(SNR_dB / 10)
```

For real-valued tensors (the stego image is real), the paper's complex baseband model reduces
to real Gaussian noise with the same variance and `h = 1`. Implement both paths, but default to
the representation that the downstream codec expects and document it.

Rules:

- Generate noise with `torch.Generator(device=...).manual_seed(seed + snr_index)`; never the
  global RNG.
- Do not clip the received signal before passing it to the receiver unless the pipeline
  normalizes to `[0, 1]`; record the convention.
- Return the measured signal power and the applied variance in a metadata dict for verification.

---

## Step 4: Implement `scripts/phase12_awgn.py`

```powershell
python scripts\phase12_awgn.py --class facial --limit 20
python scripts\phase12_awgn.py --split evaluation
```

Algorithm:

```text
1. set_seed(42); load the frozen 20-image probe set from Phase 11
2. for each SNR in {5, 10, 15, 20}:
     a. hide the secret with the matched reference -> stego
     b. pass stego through the AWGN channel with the fixed seed
     c. recover from received -> recovered
     d. measure MSE, PSNR, SSIM, LPIPS versus the secret
3. also run the no-noise control from Phase 11 for the same images
4. write distributions per (class, SNR) and the monotonicity check
5. plot mean PSNR and mean LPIPS versus SNR
```

Do not use a single example image as the result. Report mean, median, std, min, max, and N.

---

## Step 5: Implement `scripts/phase12_verify.py`

Checks:

```text
[ ] analytic variance: measured noise variance within 1 percent of P/10^(SNR/10)
[ ] same seed + same input -> identical received tensor and identical recovery
[ ] different seeds at the same SNR -> different noise, same distribution
[ ] mean PSNR is non-increasing as SNR falls, for every class tested
[ ] no cell has fewer than 20 images
[ ] no codec, retrieval or VLM import appears in communication/awgn.py
[ ] phase12_awgn.json and the curve PNG exist
```

Exit non-zero on failure.

---

## Step 6: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase12_awgn.py
.venv\Scripts\python.exe scripts\phase12_verify.py
```

Expected: recovery degrades monotonically from 20 dB to 5 dB, the fixed seed reproduces exactly,
and the measured noise power matches the analytic value.

---

## Checklist

```text
[ ] complex/real convention is documented and used consistently
[ ] sigma^2 is derived from the measured signal power
[ ] noise seed is per-cell and controlled
[ ] 20 images per cell, metrics as distributions
[ ] monotone degradation is demonstrated
[ ] identical reruns produce identical received tensors
[ ] phase12_verify.json has pass=true
[ ] no codec, retrieval, VLM or editing
```

**When this passes, Phase 12 is done.** Phase 13 inserts the codec between stego and channel.

---

## The Gate Question (answer in your own words)

> *Why must the noise power be derived from the measured signal power, and why is a fixed seed
> per SNR cell necessary?*

Answer with your artifacts:

- SNR is a ratio, not an absolute noise level. If you hard-code a variance, the actual SNR
  changes with image content and the reported dB value is wrong.
- Deriving `sigma^2 = P_signal / 10^(SNR/10)` makes the cell comparable across images and
  classes.
- A fixed seed per cell isolates the effect of SNR from the effect of a lucky or unlucky noise
  draw. Without it, a 0.3 dB difference could be noise sampling, not the channel.
- The analytic check and the seed-replay check are the evidence.

---

## Report back

- SNR formula and the real/complex convention
- Analytic versus measured noise variance per cell
- Metric distributions and the degradation curve
- Seed replay result
- Your answer to the gate question

---

## Notes and risks

- **Do not compare SNR cells with different seeds and call the difference a channel effect.**
- **Normalization matters.** State whether the stego is in `[0, 1]`, `[-1, 1]` or raw VAE
  output, and keep it constant.
- **Clipping can flatter the result.** If you clip, report it.
