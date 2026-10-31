# Phase 15: Assemble AgentSemSteCom and Reproduce Table II — **BASELINE GATE**

> **Goal**: Assemble the complete paper baseline in the paper's order and freeze it:
>
> ```text
> Secret -> Kpub + Xfeat -> token-seeded diffusion + ControlNet + IPAdapter-plus -> Xref
>        -> VAE encode -> EDICT forward -> token perturbation -> Stego
>        -> semantic encode -> AWGN -> semantic decode -> Stego_hat
>        -> EDICT inverse -> inverse perturbation -> Recovered secret
> ```
>
> **Why**: This is the control condition. Every RACS result in Phases 17-22 is a comparison
> against this frozen baseline.
>
> **Hard rule**: Nothing RACS-specific is wired in here. No FAISS, no retrieval, no structural
> reranking, no VLM, no editing. If a retrieval module appears in this pipeline, the gate fails.

> **Data rule**: 15A uses the calibration split and is labelled approximate. 15B uses the
> evaluation split and is authoritative. Calibration numbers never appear in a claim table.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2

.venv\Scripts\python.exe scripts\phase8_verify.py
.venv\Scripts\python.exe scripts\phase9_verify.py
.venv\Scripts\python.exe scripts\phase10_verify.py
.venv\Scripts\python.exe scripts\phase11_verify.py
.venv\Scripts\python.exe scripts\phase12_verify.py
.venv\Scripts\python.exe scripts\phase13_verify.py
.venv\Scripts\python.exe scripts\phase14_verify.py

.venv\Scripts\python.exe scripts\phase6_eval_selftest.py
```

Expected: Phase 6 selftest is 83/83. Every earlier verify JSON has `"pass": true`, or the
skipped phase has a written deviation and a decision in `docs/DEVIATIONS.md`.

If any component is missing, stop. A baseline assembled around a broken part is not a baseline.

---

## Step 1: Config

Add a `baseline:` block. It must reference the frozen choices from earlier phases, not copy
new values.

```yaml
baseline:
  reference_config: "reference"        # Phase 8 block
  edict_config: "edict"                # Phase 10 block
  stego_config: "steganography"        # Phase 11 block
  channel_config: "channel"            # Phase 12 block
  codec_config: "codec"                # Phase 13 block
  enhancement_config: "enhancement"    # Phase 14 block, disabled by default
  snr_db: [5, 10, 15, 20]
  classes: ["common", "facial", "style"]
  calibration_output: "experiments/results/phase15A_calibration_table.json"
  evaluation_output: "experiments/results/baseline_table.json"
  calibration_manifest: "experiments/results/phase15A_manifest.json"
  evaluation_manifest: "experiments/results/baseline_manifest.json"
  calibration_grid: "experiments/results/calibration_baseline_grid.png"
  evaluation_grid: "experiments/results/baseline_grid.png"
```

`enhancement.enabled` stays `false` unless the paper's Table II cell explicitly includes it.
If it is turned on, the manifest and the report must say so.

---

## Step 2: Structure

```text
pipelines/
  __init__.py
  agentsemstecom.py     # one function, full baseline
scripts/
  phase15_baseline.py   # --split calibration|evaluation
  phase15_verify.py     # gate A + gate B
experiments/results/
  phase15A_calibration_table.json
  phase15A_manifest.json
  baseline_table.json
  baseline_manifest.json
  calibration_baseline_grid.png
  baseline_grid.png
  phase15_verify.json
```

---

## Step 3: Implement `pipelines/agentsemstecom.py`

### API

```python
def run_case(
    *,
    secret: PIL.Image,
    class_name: str,
    image_id: str,
    token: str,
    wrong_token: str,
    snr_db: float,
    split_role: str,
    config: dict,
    token_mode: str = "correct",
) -> dict:
    """Returns intermediates and metrics for one baseline case."""
```

Required intermediates in the returned dict:

```text
kpub, kpub_sha256
feature, feature_kind, feature_sha256
seed, latents_sha256
reference, reference_sha256
secret_latent, hidden_latent, stego_latent, recovered_latent
stego, stego_sha256
received, received_sha256
recovered, recovered_sha256
metrics: mse, psnr, ssim, lpips
```

### Construction rules

- Load the reference pipeline, EDICT, VAE, codec and optional enhancer **once** per run, not
  per image.
- Read `Kpub` and `Xfeat` from the Phase 7 cache. Do not regenerate them differently here.
- Use the Phase 8 token seed and the Phase 11 token mask. Do not reimplement either.
- Use the Phase 12 SNR definition and the Phase 13 codec normalization. Do not fork the channel.
- The secret is only used at VAE encode and for metrics. It is never an input to the reference
  generator in the paper path.
- Enhancement, when enabled, is applied after recovery and only for `class_name == "facial"`.

### Determinism

- `set_seed(42)`, fixed per-case token seed, fixed channel seed per `(split_role, class, snr)`.
- `torch.inference_mode()` everywhere.
- Rerun the same case twice in-process and once in a fresh process; the `reference_sha256`,
  `stego_sha256` and `recovered_sha256` must be identical.

---

## Step 4: Implement `scripts/phase15_baseline.py`

### CLI

```powershell
python scripts\phase15_baseline.py --split calibration
python scripts\phase15_baseline.py --split evaluation
python scripts\phase15_baseline.py --split evaluation --class facial --limit 2
python scripts\phase15_baseline.py --split evaluation --resume
```

### Algorithm

```text
1. set_seed(42)
2. load config and the role-specific split through evaluation.splits.Split
   - calibration requires allow_calibration=True
   - evaluation must never load calibration records
3. load the Phase 7 manifest, tokens, and the frozen component manifests
4. load every component once
5. for role, class, image_id in fixed order, for snr in [5, 10, 15, 20]:
     a. run_case(...)
     b. write per-case intermediate hashes and metrics
     c. append to the table and the case list
6. aggregate per (class, snr): mean, median, std, min, max, N, n_inf
7. write the table JSON and the manifest atomically
8. write the visual grid
9. print the cell summary and exit non-zero on any failure
```

Use the Phase 6 report schema (`evaluation/report.py`) so the output is directly comparable with
Phases 21 and 22. Do not invent a new metric schema.

### Output schema

```json
{
  "schema_version": 1,
  "approximate": false,
  "split_role": "evaluation",
  "split_version": "a57d9c8c...",
  "tokens_version": "20166234...",
  "config_hash": "...",
  "code_commit": "...",
  "models": {"reference": "...", "controlnet": "...", "edict": "...",
             "codec": "...", "enhancement": "disabled"},
  "cells": [
    {
      "class": "facial",
      "snr_db": 5,
      "n": 22,
      "mse":  {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
      "psnr": {"mean": 0.0, "median": 0.0, "std": 0.0},
      "ssim": {"mean": 0.0, "median": 0.0, "std": 0.0},
      "lpips":{"mean": 0.0, "median": 0.0, "std": 0.0}
    }
  ],
  "determinism": {"sample": 6, "reference_equal": true,
                   "stego_equal": true, "recovered_equal": true},
  "failures": []
}
```

`approximate` is forced `true` for 15A and `false` for 15B. A calibration document with
`approximate: false` is a gate failure.

---

## Step 5: Gate A — calibration, approximate reproduction

Run the calibration split, 3 classes x 4 SNRs. Every cell carries `"approximate": true`.

| Cell | Paper value | Approximate tolerance |
| :-- | :-- | :-- |
| facial 5 dB | PSNR 22.73, SSIM 0.6989 | +/- 2.5 dB, +/- 0.07 |
| facial 20 dB | PSNR 26.29, LPIPS 0.2127 | +/- 2.5 dB, +/- 0.05 |
| style 5 dB | MSE 0.721e-2, LPIPS 0.3850 | +/- 40 percent, +/- 0.05 |
| common 5 dB | PSNR 19.71, MSE 1.207e-2 | +/- 2.5 dB, +/- 40 percent |
| common 10 dB | SSIM 0.5643 | +/- 0.07 |

Additional Gate A requirements:

```text
[ ] determinism: same image, token and seed -> bit-identical Xref and stego
[ ] monotone improvement from 5 dB to 20 dB in every class
[ ] class ordering preserved: facial PSNR at least as good as common at 5 and 20 dB
[ ] every miss has a written diagnosis in docs/DEVIATIONS.md
[ ] no RACS/retrieval import in the pipeline
[ ] the pipeline used the same frozen component configs as Phases 8-14
```

Do not tune anything on the evaluation split to close a calibration gap. Diagnose the miss,
write it down, and only then decide whether to accept it.

---

## Step 6: Gate B — authoritative baseline on the evaluation split

Run the identical code with `--split evaluation`. This writes `baseline_table.json`.

Requirements:

```text
[ ] the loader rejects any calibration record; record the role in every case
[ ] 22 images per class per SNR cell (264 cases per class across 4 SNRs)
[ ] metrics are distributions (mean, median, std, N), not single-image values
[ ] determinism on at least 2 images per class, including a fresh process
[ ] code commit, config hash, split version, token version and model hashes recorded
[ ] every artifact is labelled approximate: false
[ ] no paper tolerance is applied; no paper number is compared to an evaluation cell
[ ] baseline_table.json is frozen and its sha256 is recorded
```

Once this file is written, it is the baseline. Any later change requires regenerating the file
and re-running Phase 22.

---

## Step 7: Implement `scripts/phase15_verify.py`

Checks, writing `phase15_verify.json`:

```text
[ ] both table files exist with the correct `approximate` flag
[ ] 15A used calibration only; 15B used evaluation only
[ ] split_version and tokens_version match the live manifests
[ ] every cell has N == 22 and finite metrics
[ ] Gate A tolerances pass one by one, with the measured value in the detail
[ ] Gate A monotonicity and class ordering pass
[ ] determinism holds in the recorded sample
[ ] codec, channel, EDICT, reference and enhancement choices match the component manifests
[ ] no retrieval/FAISS/RACS symbol is imported by pipelines/agentsemstecom.py
[ ] both grids exist
[ ] docs/DEVIATIONS.md contains every miss and every substitution
```

Exit non-zero if Gate A or Gate B fails. A Gate A miss that is explicitly diagnosed and accepted
in `docs/DEVIATIONS.md` may be reported as a pass with a warning, but the warning must appear in
the summary.

---

## Step 8: Run and check

```powershell
.venv\Scripts\python.exe scripts\phase15_baseline.py --split calibration
.venv\Scripts\python.exe scripts\phase15_baseline.py --split evaluation
.venv\Scripts\python.exe scripts\phase15_verify.py
```

Expected summary:

```text
15A calibration: 36 cells, approximate=true, tolerances PASS (or diagnosed misses)
15B evaluation : 36 cells, N=22 each, approximate=false
determinism    : PASS
peak VRAM      : inside budget
verify         : PASS
```

Open both grids and confirm each class row is coherent, no cell is blank, and the facial row
degrades more gracefully at high SNR than the style row, as the paper reports.

---

## Checklist

```text
[ ] all upstream verifiers pass or have accepted deviations
[ ] pipelines/agentsemstecom.py has no retrieval, FAISS, VLM or editing import
[ ] Kpub/Xfeat come from the Phase 7 cache
[ ] seed and mask come from the Phase 8/11 contracts
[ ] channel and codec come from the Phase 12/13 contracts
[ ] enhancement is off by default and facial-only when on
[ ] 15A is labelled approximate and uses calibration only
[ ] 15B is labelled approximate: false and uses evaluation only
[ ] 36 cells per split, 22 images per cell
[ ] determinism passes in-process and in a fresh process
[ ] manifests record config hash, split version, token version, code commit, model hashes
[ ] both grids inspected
[ ] docs/DEVIATIONS.md is complete
[ ] baseline_table.json is frozen and hashed
[ ] phase15_verify.json has pass=true
```

**When both gates pass, Phase 15 is done and the baseline is frozen.** Do not start Phase 17
with an unexplained calibration gap or a non-deterministic evaluation baseline.

---

## The Gate Question (answer in your own words)

> *Why is the calibration split only an approximate anchor, and why can the evaluation numbers
> not be compared to the paper's Table II?*

Answer with your artifacts:

- The paper's dataset, model revisions and training state are not bit-reproducible here. We use
  the official release as calibration data, but our model substitutions (BLIP, SegFormer,
  MediaPipe, the codec checkpoint) and our hardware change the distribution.
- A wide tolerance checks that the pipeline is in the right regime and that the failure modes
  are not structural. It does not claim equality.
- The evaluation split is our own frozen data, disjoint from calibration and from the retrieval
  corpus. There are no paper numbers for it, by construction.
- Mixing the two would let a calibration-tuned value masquerade as an evaluation result. The
  `approximate` flag, the role field and the separate files prevent that.

---

## Report back

- Pipeline order actually implemented, with file references
- Component versions and hashes (reference, ControlNet, EDICT, codec, enhancement)
- 15A table: every cell, paper value, measured value, pass/miss
- 15B table: per class and SNR, distributions
- Determinism evidence (hashes, fresh-process check)
- Peak VRAM and per-image latency
- Grid paths and visual confirmation
- Every deviation and its register row
- Your answer to the gate question

---

## Notes and risks

- **This is the scientific control.** A bug here invalidates every later comparison, including
  a favourable one.
- **Do not tune on evaluation.** No per-image fidelity weight, no per-class temperature, no
  cherry-picked seeds.
- **The calibration gap is a diagnosis, not a knob.** Write down the cause before changing a
  component.
- **Freeze after gate B.** If you regenerate `baseline_table.json`, Phase 22 must be rerun from
  scratch.
