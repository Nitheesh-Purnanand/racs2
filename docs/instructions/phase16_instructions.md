# Phase 16: Baseline Security and Steganalysis

> **Goal**: Measure the baseline's security profile from the receiver side (correct token,
> wrong token, no token) and train or obtain XuNet to measure detectability. Freeze the
> baseline security numbers.
> **Why**: These are the paper's contribution numbers. They must exist for the baseline before
> they can be compared with RACS.

> **Data rule**: Evaluation split only. Calibration data may not enter a security table. Every
> number written here is `"approximate": false` for our data; paper values are approximate
> anchors recorded beside ours, clearly separated.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase15_verify.py
.venv\Scripts\python.exe -c "import torch, sklearn; print(torch.cuda.is_available(), sklearn.__version__)"
```

Expected: Phase 15 passes and `baseline_table.json` is frozen.

---

## Step 1: Config

```yaml
security:
  split_role: "evaluation"
  conditions: ["correct", "wrong", "no_token"]
  snr_db: [5, 10, 15, 20]
  classes: ["common", "facial", "style"]
  output_security: "experiments/results/baseline_security.json"
  output_steganalysis: "experiments/results/baseline_steganalysis.json"
  manifest_path: "experiments/results/phase16_manifest.json"
  steganalysis:
    model: "xunet"
    train_fraction: 0.7
    val_fraction: 0.15
    test_fraction: 0.15
    epochs: 50
    batch_size: 8
    learning_rate: 0.001
    seed: 42
    cover_source: "evaluation_secret_images"
```

The three receiver conditions must be defined before the run:

- **correct**: the true token, its seed and its mask.
- **wrong**: a valid `wrong_token` from `tokens.json` for that image. It produces a different
  reference seed and mask, so recovery should fail.
- **no_token**: the receiver has no token at all. Use a fixed public fallback seed (for example
  `derive_seed("00000000")`) and an all-zero mask. This is the "no secret" baseline, not a
  random-token attack.

Do not reuse the wrong-token run as the no-token run. They answer different questions.

---

## Step 2: Structure

```text
security/
  __init__.py
  token_conditions.py   # correct / wrong / no-token receiver paths
  steganalysis.py       # XuNet model, dataset, train/eval
scripts/
  phase16_security.py
  phase16_steganalysis.py
  phase16_verify.py
experiments/steganalysis/
  xunet_baseline_<config_hash8>.pt
experiments/results/
  baseline_security.json
  baseline_steganalysis.json
  phase16_security_grid.png
  phase16_verify.json
```

---

## Step 3: Implement `security/token_conditions.py`

```python
def receive(
    *,
    stego: PIL.Image,
    token: str | None,
    token_mode: str,
    config: dict,
    context: dict,
) -> dict:
    """Runs the receiver half of Phase 15 with the requested token condition."""
```

Rules:

- The transmitter side is identical for all three conditions. Only the receiver's token,
  seed and mask change.
- `no_token` must not read `tokens.json` for the image; it uses the public fallback.
- Record the token mode, the seed actually used, and the mask hash in every case.
- Run the same 22 images per class and the same SNR cells as the baseline.

---

## Step 4: Implement `scripts/phase16_security.py`

Algorithm:

```text
1. set_seed(42)
2. load the evaluation split and the frozen Phase 15 baseline cases
3. for each image, class, snr:
     a. take the baseline stego image
     b. run receive(correct), receive(wrong), receive(no_token)
     c. measure MSE, PSNR, SSIM, LPIPS against the secret
4. aggregate per (class, snr, token_mode): distributions with N=22
5. compute the correct-minus-wrong and correct-minus-no-token gaps
6. write baseline_security.json and the grid (secret | correct | wrong | no token)
```

Report the gaps per cell, not only the overall mean. A correct-token advantage that disappears
in one class is a finding, not something to average away.

### Output schema

```json
{
  "schema_version": 1,
  "approximate": false,
  "split_role": "evaluation",
  "split_version": "a57d9c8c...",
  "baseline_table_sha256": "...",
  "cells": [
    {
      "class": "facial", "snr_db": 10,
      "correct": {"psnr": {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 22},
                  "ssim": {}, "lpips": {}},
      "wrong":   {"psnr": {}, "ssim": {}, "lpips": {}},
      "no_token":{"psnr": {}, "ssim": {}, "lpips": {}},
      "gap_correct_minus_wrong_db": 0.0,
      "gap_correct_minus_no_token_db": 0.0
    }
  ],
  "paper_anchor": {"correct_psnr_db": 25.37, "wrong_psnr_db": 18.91,
                   "gap_db": 6.46, "source": "paper Table II/III, approximate"},
  "failures": []
}
```

The paper anchor is stored as metadata only. It is never compared as an exact target, because the
data and components differ.

---

## Step 5: Implement `security/steganalysis.py` and XuNet

### Model

Implement XuNet as described by the paper or use a faithful public implementation. The exact
architecture must be recorded. Do not substitute a generic CNN and call it XuNet.

### Dataset

```text
positive: baseline stego images from the evaluation split
negative: the corresponding covers (the secret images), same transforms
split: 70/15/15 by image_id, grouped so that all SNRs and conditions for an image stay in
       one split; no image appears in both train and test
input: 512x512 or the model's expected size, RGB, normalized consistently
```

If the evaluation set is too small for a stable split, use patches extracted from the same
image and keep the split at the image level. Report the exact counts.

### Training

- Fixed seed 42; deterministic dataloader ordering.
- Batch size 8 or the largest that fits 6 GB; gradient accumulation allowed.
- Early stopping on validation accuracy, with the epoch and checkpoint recorded.
- Save the best checkpoint and its sha256.
- Report train/val/test accuracy and the confusion matrix. Detection rate is test accuracy on
  the balanced stego-vs-cover task; state the chance rate (50 percent).

---

## Step 6: Implement `scripts/phase16_steganalysis.py`

```powershell
python scripts\phase16_steganalysis.py --train
python scripts\phase16_steganalysis.py --eval-only --checkpoint <file>
```

Outputs:

```text
baseline_steganalysis.json:
  dataset counts, split hashes, model config, epochs, best epoch,
  accuracy, precision, recall, confusion matrix, chance rate,
  checkpoint path + sha256, paper anchor 50.91 percent
```

The paper's 50.91 percent is an approximate anchor only. Do not force the result toward it.

---

## Step 7: Implement `scripts/phase16_verify.py`

Checks:

```text
[ ] Phase 15 baseline table hash matches the one recorded here
[ ] only evaluation records appear in security and steganalysis manifests
[ ] all three token conditions exist for every cell, N == 22
[ ] correct-token PSNR is higher than wrong-token and no-token in aggregate
[ ] the correct-minus-wrong gap is at least 4 dB in aggregate; per-cell exceptions documented
[ ] wrong-token and no-token both underperform correct
[ ] XuNet split is image-disjoint and the class balance is recorded
[ ] checkpoint exists, sha256 matches, and test accuracy is reported with N
[ ] paper anchors are present and clearly labelled approximate
[ ] both JSON files are labelled approximate: false for our data
[ ] no calibration image appears anywhere
[ ] phase16_security_grid.png exists
```

Exit non-zero on failure. A gap below 4 dB is a stop condition, not a number to round.

---

## Step 8: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase16_security.py
.venv\Scripts\python.exe scripts\phase16_steganalysis.py --train
.venv\Scripts\python.exe scripts\phase16_verify.py
```

Expected summary:

```text
conditions: correct / wrong / no_token, 22 per class per SNR
correct-token gap: >= 4 dB over wrong and no-token
XuNet test accuracy: reported with counts and checkpoint hash
paper anchors: present, approximate
verify: PASS
```

Inspect `phase16_security_grid.png`. The wrong-token and no-token columns should look like
structured failures, not like a second valid recovery.

---

## Checklist

```text
[ ] receiver conditions defined and implemented unambiguously
[ ] transmitter identical across conditions
[ ] evaluation split only; calibration absent
[ ] per-cell distributions with N=22
[ ] correct-minus-wrong gap >= 4 dB aggregate
[ ] no-token is a distinct condition with a documented fallback seed
[ ] XuNet is the paper's architecture, not a generic CNN
[ ] steganalysis split is image-disjoint and reported
[ ] checkpoint and sha256 recorded
[ ] paper anchors labelled approximate and kept separate
[ ] baseline_security.json and baseline_steganalysis.json frozen
[ ] phase16_verify.json has pass=true
```

**When this passes, the baseline is frozen end to end.** Phase 17 may now build the RACS
retrieval stack. No RACS artifact may retroactively change a Phase 15 or Phase 16 number.

---

## The Gate Question (answer in your own words)

> *What does the correct-versus-wrong-token gap demonstrate, and why is the paper's 25.37 dB
> versus 18.91 dB only an anchor?*

Answer with your artifacts:

- The gap shows that recovery depends on the secret token, not merely on a generic decoder. A
  receiver without the token cannot reconstruct the reference or invert the perturbation, so
  its output degrades toward a generic reconstruction.
- The gap is evidence of key-dependent security at the signal level. It is not a proof of
  cryptographic security, and the steganalysis accuracy is a separate detectability measure.
- The paper's numbers come from the paper's data, model revisions and codec. Our data and
  components differ, so the only honest comparison is direction and approximate magnitude.
- Recording the anchor beside ours, with `approximate` flags, prevents the paper's number from
  being mistaken for a measurement on our evaluation split.

---

## Report back

- Receiver condition definitions and the fallback seed for no-token
- Per-class per-SNR security distributions and the correct-minus-wrong gap
- Whether the 4 dB aggregate threshold passed
- XuNet architecture, split sizes, training setup, test accuracy and confusion matrix
- Checkpoint path and sha256
- Paper anchors and how they are labelled
- Grid path and what was visually confirmed
- Your answer to the gate question

---

## Notes and risks

- **Do not tune the security experiment.** The stego set is frozen by Phase 15.
- **Wrong token must be a valid token.** A malformed token would test input validation, not
  key dependence.
- **No-token is not "zero token".** Define it explicitly and keep it fixed across all cells.
- **XuNet is sensitive to split leakage.** Group by image before splitting; otherwise the
  detection rate is inflated by near-duplicate patches.
- **A near-chance detector is the expected baseline result.** Do not secretly train on the test
  split to make the number look better or worse.
