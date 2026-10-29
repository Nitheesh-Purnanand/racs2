# Phase 13: Semantic Communication Codec

> **Goal**: Replace raw pixel transmission with the paper's JSCC-style semantic codec, and
> measure the codec's own degradation curve before wiring it into the full pipeline.
> **Why**: The paper's numbers are measured through this codec. Comparing against them without
> it would be an apples-to-oranges baseline.

> **Ordering rule**: Phase 13 consumes the Phase 12 channel. It must not modify EDICT, the
> reference generator or the local stego core. If pretrained weights are unavailable, substitute
> the closest implementation and record the deviation.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase12_verify.py
.venv\Scripts\python.exe -c "import shutil; print(round(shutil.disk_usage('D:/').free/1024**3,1),'GB free')"
```

Expected: Phase 12 passes and there is room for one additional encoder/decoder pair.

---

## Step 1: Choose the codec and record it

The paper uses two paths:

- **Analog SwinJSCC-style**, compression ratio `C = 48`, trained at 10 dB.
- **Digital MDJCM**, trained on DIV2K, modulation 64.

Preferred order:

1. the paper's released weights, if obtainable;
2. the closest public SwinJSCC checkpoint with the same C and training SNR;
3. a documented substitute (for example a smaller JSCC autoencoder) with the deviation row.

Do not train a codec from scratch in this phase. If no pretrained checkpoint is usable, stop
and record the blocker rather than silently training one.

Add a `codec:` block:

```yaml
codec:
  kind: "swinjscc"                # swinjscc | mdjcm | substitute
  checkpoint: "models/codec/<file>"
  checkpoint_sha256: "..."
  compression_ratio: 48
  trained_snr_db: 10
  modulation: 64
  input_size: 512
  dtype: "float16"
  results_dir: "experiments/results/phase13"
  manifest_path: "experiments/results/phase13_codec.json"
```

---

## Step 2: Structure

```text
communication/
  semantic_codec.py     # load, encode, channel, decode
scripts/
  phase13_codec.py      # isolation test + codec-vs-raw comparison
  phase13_verify.py
experiments/results/phase13/
  phase13_codec.json
  phase13_codec_curve.png
  phase13_codec_grid.png
  phase13_verify.json
```

---

## Step 3: Implement `communication/semantic_codec.py`

```python
def build_codec(config: dict): ...

def codec_roundtrip(
    x: torch.Tensor,
    *,
    snr_db: float,
    seed: int,
    codec,
    config: dict,
) -> dict:
    """Returns {"output": tensor, "symbols_shape": ..., "latency_s": ..., "meta": ...}."""
```

Data flow:

```text
stego [0,1] -> normalize -> encoder -> latent symbols
            -> power normalization -> channel (AWGN at target SNR)
            -> decoder -> denormalize -> stego_hat [0,1]
```

Rules:

- The channel inside the codec uses the same SNR definition and seed contract as Phase 12.
- Power normalization must be explicit and invertible; record the average symbol power.
- The codec is frozen, `eval()`, no gradients.
- The encoder/decoder must not see the secret or the reference, only the stego tensor.

---

## Step 4: Validate the codec in isolation

Before touching the full pipeline:

```text
1. Take 20 clean images (same fixed probe set as Phase 12).
2. Pass each through codec + channel at 5, 10, 15 and 20 dB.
3. Report reconstruction PSNR and LPIPS per cell.
4. Plot the curve and save the grid.
```

If the codec cannot reconstruct a clean image above a sane floor (for example mean PSNR below
20 dB at 20 dB SNR), stop and diagnose the checkpoint, normalization or channel convention.

---

## Step 5: Compare raw AWGN versus codec-plus-AWGN

Run the same 20 images and the same SNR cells through:

```text
A. Phase 11 stego -> Phase 12 AWGN -> Phase 11 recover      (raw)
B. Phase 11 stego -> codec -> AWGN -> codec decode -> recover (codec)
```

Report per-cell PSNR, SSIM and LPIPS for both, plus the delta. The purpose is to make the
codec's contribution separable, not to claim a win before Phase 15.

---

## Step 6: Implement `scripts/phase13_verify.py`

Checks:

```text
[ ] checkpoint exists, sha256 matches, and the codec is in eval mode
[ ] isolated round-trip is finite on all 20 images and all 4 SNR cells
[ ] codec reconstruction improves as SNR rises
[ ] same seed -> identical codec output, two runs
[ ] codec-plus-channel and raw AWGN differ in the expected direction, or the difference is
    explained in the report
[ ] no reference generator, retrieval or VLM import in the codec module
[ ] phase13_codec.json, curve and grid exist
```

Exit non-zero on failure.

---

## Step 7: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase13_codec.py
.venv\Scripts\python.exe scripts\phase13_verify.py
```

Inspect `phase13_codec_grid.png`: clean input, codec output at 20 dB and at 5 dB. Confirm the
degradation is visual, not just numeric.

---

## Checklist

```text
[ ] codec source and checkpoint are pinned and hashed
[ ] no training; pretrained weights only
[ ] normalization and power convention are documented
[ ] isolated clean-image curve is measured before pipeline integration
[ ] raw-versus-codec comparison is measured at the same SNR and seeds
[ ] reruns are deterministic
[ ] any substitute is recorded in docs/DEVIATIONS.md
[ ] phase13_verify.json has pass=true
```

**When this passes, Phase 13 is done.** Phase 14 is optional and only touches the facial path.

---

## The Gate Question (answer in your own words)

> *Why measure the codec in isolation before inserting it into the steganographic pipeline?*

Answer with your artifacts:

- A codec introduces its own reconstruction error. If it is wired in first, a bad recovery
  cannot be attributed to the channel, the codec or EDICT.
- The isolated curve establishes the codec's floor per SNR. Any extra loss in the full pipeline
  is then attributable to the stego path.
- The raw-versus-codec comparison makes the codec's contribution falsifiable. If the codec does
  not help in this setting, that is a result, not something to hide.
- The pinned checkpoint hash makes the comparison reproducible.

---

## Report back

- Codec kind, checkpoint path, sha256, training SNR and compression ratio
- Normalization and power conventions
- Isolated reconstruction curve (PSNR, LPIPS per SNR)
- Raw versus codec-plus-channel table
- Determinism check
- Any deviation and its register row
- Your answer to the gate question

---

## Notes and risks

- **Do not train the codec on the evaluation split.** Pretrained weights only.
- **Checkpoint compatibility is the main failure mode.** Match the input size, channel count
  and normalization before doubting the model.
- **A codec that improves clean reconstruction but hurts recovery is still informative.** Record
  it; do not tune it on the evaluation set.
