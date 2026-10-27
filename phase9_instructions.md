# Phase 9: ControlNet Zero-Conv Fine-Tuning (Paper Algorithm 2)

> **Goal**: Train the ControlNet side branch, and only its zero-initialised convolutions, on the
> diffusion denoising objective with the SD backbone frozen.
> **Why**: The paper trains this branch. A stock ControlNet is a different conditioning
> distribution and is a likely source of a systematic gap in Phase 15.

> **Ordering rule**: Phase 9 may use calibration-split images and Phase 8 references. It must
> never train on the evaluation split, must not touch EDICT, and must not use retrieval.

---

## Step 0: Pre-flight checks

```powershell
cd D:\racs2
.venv\Scripts\python.exe scripts\phase8_verify.py
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3)"
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeCommitGB';e={[math]::Round($_.FreeVirtualMemory/1MB,2)}}
```

Expected: Phase 8 passes, free commit is above 2 GB, and there is room for one 512x512 training
step in fp16.

If the stock ControlNet already matches the probe in Step 5, you are allowed to skip the
fine-tune and record a deviation. The decision must be written down either way.

---

## Step 1: Config

Add a `controlnet_train:` block to `configs/base.yaml`.

```yaml
controlnet_train:
  base_model_id: "runwayml/stable-diffusion-v1-5"
  controlnet_id: "lllyasviel/control_v11p_sd15_seg"
  train_role: "calibration"          # evaluation data is never used for training
  classes: ["common", "facial"]      # style has no Xfeat
  image_size: 512
  batch_size: 1
  grad_accum_steps: 4
  learning_rate: 0.00001
  max_steps: 500
  warmup_steps: 50
  sampler: "ddpm"
  dtype: "float16"
  checkpoint_dir: "experiments/checkpoints"
  probe_size: 10
  probe_snr_db: 10
```

Freeze every value before training. A resumed run must reproduce the same loss curve within
run-to-run GPU nondeterminism; the checkpoint hash is the provenance.

---

## Step 2: Structure

```text
reference/
  train_controlnet.py      # model loading, frozen backbone, zero-conv training loop
  probe.py                 # zero-shot vs fine-tuned probe
scripts/
  phase9_train_controlnet.py
  phase9_verify.py
experiments/checkpoints/
  controlnet_zero_conv_<steps>_<config_hash8>.safetensors
experiments/results/
  phase9_training.json
  phase9_probe.json
  phase9_probe_grid.png
  phase9_verify.json
```

---

## Step 3: Implement `reference/train_controlnet.py`

### Model setup

```python
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
controlnet.requires_grad_(False)
for name, param in controlnet.named_parameters():
    if "zero_conv" in name or "controlnet_down_blocks" in name:
        param.requires_grad_(True)
```

If the checkpoint does not expose `zero_conv` in parameter names, select the parameters created
by `ControlNetModel`'s zero-initialised output blocks explicitly and print the trainable count.
Assert that the frozen backbone has zero trainable parameters.

### Algorithm 2 loop

```text
for step in range(max_steps):
    image, condition, kpub = next(calibration_batch)
    x0 = vae.encode(image).latent_dist.sample() * 0.18215
    t ~ Uniform(0, T-1) (seeded)
    noise ~ N(0, I) (seeded)
    xt = scheduler.add_noise(x0, noise, t)
    down, mid = controlnet(xt, t, encoder_hidden_states=text(kpub),
                           controlnet_cond=condition, return_dict=False)
    pred = unet(xt, t, encoder_hidden_states=text(kpub),
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid).sample
    loss = MSE(pred, noise)
    loss.backward()                     # grads only reach trainable zero-conv params
    clip_grad_norm_(trainable_params, 1.0)
    optimizer.step(); optimizer.zero_grad()
```

Use `torch.autocast("cuda", dtype=torch.float16)` and a `GradScaler`. Set the VAE, text encoder
and UNet to `eval()` and `requires_grad_(False)`. Keep `common.hf_lowmem.install()` before the
diffusers import.

### 6 GB budget

- batch size 1, gradient accumulation for the effective batch,
- attention slicing and VAE slicing on,
- sequential CPU offload is allowed for the frozen modules if peak VRAM exceeds 5.5 GB,
- checkpoint only trainable tensors with `safetensors.torch.save_file`.

---

## Step 4: Implement `scripts/phase9_train_controlnet.py`

CLI:

```powershell
python scripts\phase9_train_controlnet.py --max-steps 200
python scripts\phase9_train_controlnet.py --resume experiments/checkpoints/<file>.safetensors
```

Write `phase9_training.json` with:

```json
{
  "schema_version": 1,
  "config_hash": "...",
  "trainable_parameters": 1234567,
  "frozen_parameters": 859520964,
  "steps": 500,
  "batch_size": 1,
  "grad_accum_steps": 4,
  "learning_rate": 0.00001,
  "loss_first": 0.192,
  "loss_last": 0.071,
  "peak_vram_gb": 5.1,
  "checkpoint_path": "...",
  "checkpoint_sha256": "..."
}
```

First run a few hundred steps. The loss must decrease and the checkpoint must change the
controlled output. If the loss is flat or NaN, stop; do not scale up.

---

## Step 5: Probe: zero-shot versus fine-tuned

Use `reference/probe.py` and `scripts/phase9_verify.py`.

- 10 calibration images per trainable class (`probe_size`), fixed by sorted `image_id`.
- SNR 10 dB.
- Generate the reference with the stock branch and with the fine-tuned branch, using the same
  token seeds and conditions.
- Run the Phase 11 local hide/recover path at the time of the probe, or, if Phase 11 is not
  ready, compare reference quality and noise-prediction error only. Label the probe
  `"stage": "reference_only"`.
- Report PSNR, SSIM and LPIPS of the recovered or generated image, as distributions.

Decision rule:

- If fine-tuned improves mean PSNR by at least 0.2 dB or mean LPIPS by at least 0.01 without
  hurting another class, adopt it.
- Otherwise record a deviation: "pretrained structure-conditioned ControlNet is sufficient",
  with the measured probe table.

---

## Step 6: Gate and artifacts

`scripts/phase9_verify.py` writes `experiments/results/phase9_verify.json` and checks:

```text
[ ] trainable parameter count > 0 and frozen count > 0
[ ] training loss decreased over the run and no NaN/Inf appeared
[ ] checkpoint exists and its sha256 matches the manifest
[ ] probe ran on calibration only and used fixed seeds
[ ] either an improvement is documented or a deviation is recorded
[ ] no evaluation image appears in the training/probe manifests
[ ] docs/DEVIATIONS.md has the decision row
```

Append the deviation row before declaring the phase done.

---

## Checklist

```text
[ ] config block added and hashed
[ ] only zero-conv / control branch parameters are trainable
[ ] backbone, VAE and text encoder are frozen and in eval mode
[ ] training ran on calibration data only
[ ] loss decreased and the checkpoint is hash-recorded
[ ] probe compares zero-shot and fine-tuned with identical seeds
[ ] decision (adopt or deviation) is written in docs/DEVIATIONS.md
[ ] phase9_verify.json has pass=true
[ ] no EDICT, no retrieval, no evaluation training
```

---

## The Gate Question (answer in your own words)

> *Why freeze the backbone and train only the zero-initialised side branch?*

Answer with your artifacts:

- The pretrained UNet already encodes the generative prior. Training all weights on a small
  calibration set would destroy that prior and overfit.
- Zero convolutions start at zero, so the branch initially reproduces the base model exactly,
  then learns a residual structural correction. This is a stable fine-tune, not a new model.
- Freezing makes the update small enough for 6 GB and makes the checkpoint small enough to
  hash and audit.
- The probe and the loss curve are the evidence that the branch learned something useful rather
  than just moving noise.

---

## Report back

- Trainable versus frozen parameter counts
- Steps, batch size, learning rate, loss first/last, peak VRAM
- Checkpoint path and sha256
- Probe table (zero-shot vs fine-tuned, per class, 10 dB)
- Decision and the `docs/DEVIATIONS.md` row
- Your answer to the gate question

---

## Notes and risks

- **Training on evaluation data invalidates the baseline.** Never do it, not even for a probe.
- **A flat loss is a signal, not a nuisance.** Check that the condition tensor is non-zero and
  that the control residuals actually reach the UNet.
- **Do not train the VAE or text encoder.** Their frozen state is part of the protocol.
- **Report what you actually ran.** If you stop at 200 steps instead of 500, record 200.
