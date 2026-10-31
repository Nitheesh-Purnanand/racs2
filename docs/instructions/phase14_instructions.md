# Phase 14: Optional Task-Oriented Enhancement (CodeFormer)

> **Goal**: Add the paper's optional facial enhancement after recovery, with fidelity weight
> 0.5, and measure its delta on the facial class only.
> **Why**: Fig. 8 reports a facial-class gain from this module. Without it that result cannot be
> compared. It is optional for the main Table II path and must be cleanly switchable.

> **Ordering rule**: Phase 14 runs after Phase 12 or 13. It must not affect the common or style
> paths and must not change any upstream component.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase12_verify.py
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Expected: the channel path passes and the facial-class artifacts from Phase 11/12 exist.

---

## Step 1: Config

```yaml
enhancement:
  enabled: false                 # default off for the main Table II path
  kind: "codeformer"
  model_id: "sczhou/CodeFormer"
  fidelity_weight: 0.5
  classes: ["facial"]
  input_size: 512
  dtype: "float16"
  results_dir: "experiments/results/phase14"
  manifest_path: "experiments/results/phase14_codeformer.json"
```

The switch must be a single config boolean read by the pipeline, not an import-time branch.

---

## Step 2: Structure

```text
enhancement/
  __init__.py
  codeformer.py          # loader + enhance()
scripts/
  phase14_enhance.py     # with/without delta at 4 SNR, facial only
  phase14_verify.py
experiments/results/phase14/
  phase14_codeformer.json
  phase14_enhance_grid.png
  phase14_verify.json
```

---

## Step 3: Implement `enhancement/codeformer.py`

```python
def build_enhancer(config: dict): ...
def enhance(image: PIL.Image, *, enhancer, config: dict) -> PIL.Image: ...
```

Rules:

- Load the pretrained CodeFormer weights; do not fine-tune.
- `fidelity_weight = 0.5` from the paper. It is part of the config hash.
- `torch.inference_mode()`, `eval()`, fixed dtype and device.
- Apply only when `class_name == "facial"` and `enhancement.enabled` is true.
- Preserve the output size (512x512) and RGB mode.
- Record the model hash and the fidelity weight in the manifest.

---

## Step 4: Implement `scripts/phase14_enhance.py`

```powershell
python scripts\phase14_enhance.py --limit 20
python scripts\phase14_enhance.py --snr 5 10 15 20
```

Algorithm:

```text
1. reuse the frozen 20-image facial probe set and the Phase 12/13 recovered images
2. for each SNR in {5, 10, 15, 20}:
     a. recovered = recovery(stego through channel)
     b. enhanced  = CodeFormer(recovered)
     c. measure PSNR, SSIM, LPIPS(secret, recovered) and (secret, enhanced)
3. write per-cell deltas (enhanced - recovered) as distributions
4. write the grid: secret | recovered | enhanced, for 5 dB and 20 dB
```

Do not apply the module to common or style. Run a control on one common image only to prove it
is not applied when disabled; do not report it as a result.

---

## Step 5: Implement `scripts/phase14_verify.py`

Checks:

```text
[ ] enhancer loads only when enabled, and the default config keeps it off
[ ] fidelity weight is 0.5 and recorded in the manifest
[ ] facial class shows a gain in the expected direction at 5 and 20 dB, or the miss is
    documented
[ ] common and style paths are byte-identical with enhancement on and off
[ ] same input -> identical enhanced output, two runs
[ ] output shape and mode are 512x512 RGB
[ ] phase14_codeformer.json, grid and verify JSON exist
```

Exit non-zero on failure.

---

## Step 6: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase14_enhance.py
.venv\Scripts\python.exe scripts\phase14_verify.py
```

Expected: a facial gain in PSNR/SSIM/LPIPS, especially at low SNR, with no change to the other
classes. If the gain is absent, record the measured delta and keep the module off by default.

---

## Checklist

```text
[ ] pretrained CodeFormer, fidelity 0.5, frozen
[ ] config switch cleanly enables/disables the module
[ ] facial only
[ ] deltas measured at 5, 10, 15 and 20 dB
[ ] no regression for common/style paths
[ ] deterministic reruns
[ ] deviation row if the gain does not reproduce
[ ] phase14_verify.json has pass=true
```

**When this passes, Phase 14 is done.** Phase 15 is the baseline gate and may include or exclude
this module, but the main Table II path keeps it off unless the paper's table includes it.

---

## The Gate Question (answer in your own words)

> *Why is a task-oriented enhancer legitimate here, and why must it be off by default and
> facial-only?*

Answer with your artifacts:

- CodeFormer is an optional post-processor from the paper, applied to the recovered image. It is
  not part of the hiding mechanism and does not change the reference or the latent.
- It is off by default so the main baseline is not silently improved by an extra model. Any
  claim that uses it must show the with/without delta.
- It is facial-only because the paper reports it for faces and because a face model applied to
  a cat or a landscape can hallucinate structure and corrupt the comparison.
- The switchable config and the class control are the evidence.

---

## Report back

- CodeFormer checkpoint and sha256, fidelity weight
- Per-SNR with/without deltas (PSNR, SSIM, LPIPS)
- Common/style unaffected check
- Determinism result
- Whether the default config enables it
- Your answer to the gate question

---

## Notes and risks

- **Never enhance before recovery.** It must act on the recovered secret, after the inverse
  stego path.
- **A gain on the evaluation split is a result, not a tuning signal.** Do not adjust fidelity
  weight per image.
- **Keep it optional.** The primary comparison must be reproducible without it.
