# Phase 2 — Image Tensors, VAE & Latent Space

> **Goal**: Encode an image into Stable Diffusion's latent space and decode it back,
> then **measure** how lossy that round-trip is.
> **Why**: EDICT and every diffusion stage operate in **latent space**, not pixel
> space. This round-trip error is the *error floor* for everything downstream.

---

## Step 1: Install the Phase 2 Dependencies

Still in the project + venv, with temp/cache pointed at D: (run these every new terminal):

```powershell
cd d:\racs2
.venv\Scripts\activate

$env:TMP = "D:\racs2\temp"
$env:TEMP = "D:\racs2\temp"
$env:PIP_CACHE_DIR = "D:\racs2\pip_cache"

# diffusers = the library that provides AutoencoderKL (the SD VAE)
# accelerate = lets diffusers place tensors on GPU efficiently
pip install diffusers accelerate
```

### Verify it worked:
```powershell
python -c "import diffusers, accelerate; print(diffusers.__version__); print(accelerate.__version__)"
```
✅ You should see two version numbers, no errors.

> If pip fails for disk-space reasons, confirm `$env:TMP` and `$env:PIP_CACHE_DIR`
> point to D: **before** running the install.

---

## Step 2: Create the Phase 2 Structure

Phase 1 gave you `configs/`, `common/`, `scripts/`, `data/`. Phase 2 adds:

```
d:\racs2\
├── common/
│   └── paths.py                # HF cache on D: (keeps C: from filling up)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # mse, psnr, ssim
├── models/
│   ├── __init__.py
│   └── vae.py                  # load the SD v1.5 VAE
├── scripts/
│   └── phase2_vae.py           # the test script
├── experiments/
│   └── results/                # where we save comparison images
└── data/
    └── secret_images/
        ├── test_001.png        # from Phase 1
        ├── test_002.png        # add 4 more images
        ├── test_003.png
        ├── test_004.png
        └── test_005.png
```

Create the folders and empty `__init__.py` files:

```powershell
New-Item -ItemType Directory -Force -Path "d:\racs2\evaluation"
New-Item -ItemType Directory -Force -Path "d:\racs2\models"
New-Item -ItemType Directory -Force -Path "d:\racs2\experiments\results"
New-Item -ItemType File -Force -Path "d:\racs2\evaluation\__init__.py"
New-Item -ItemType File -Force -Path "d:\racs2\models\__init__.py"
```

Then put **4 more photos** into `data\secret_images\` named `test_002.png` … `test_005.png`.
(They can be any photos. We measure the spread across 5 images, not one.)

---

## Step 3: Write `common/paths.py`

Your C: drive is nearly full, so HuggingFace must **not** download the VAE into the
default C: cache. This module centralizes paths and pins the HF cache to D:.

**What it should do:**
- Define `PROJECT_ROOT` (parent of the `common/` folder).
- Define `DATA_DIR`, `SECRET_IMAGES_DIR`, `RESULTS_DIR`, `HF_CACHE_DIR`.
- Create the cache/results folders if missing.
- Set `os.environ["HF_HOME"]` (and `HF_HUB_CACHE`) **before** any model import.

**It should work like this:**
```python
import common.paths  # side effect: HF cache redirected to D:
from common.paths import RESULTS_DIR, SECRET_IMAGES_DIR
```

---

## Step 4: Write `evaluation/metrics.py`

Three functions. All take **two tensors** (or images) in `[0, 1]` and return a float.

**`mse(a, b)`** — mean squared error: `mean((a - b) ** 2)`.

**`psnr(a, b)`** — peak signal-to-noise ratio in dB:
`10 * log10(1.0 / mse(a, b))` for unit-range data. Handle `mse == 0` by returning `inf`.

**`ssim(a, b, window_size=11)`** — structural similarity. Either:
- implement it (Gaussian window, `C1=(0.01*L)^2`, `C2=(0.03*L)^2` with `L=1`), or
- use `torchmetrics`/`skimage` if you install one. Implementing it yourself is the better learning exercise.

Accept both `[C,H,W]` and `[1,C,H,W]` by squeezing the batch dim if present.

**It should work like this:**
```python
from evaluation.metrics import mse, psnr, ssim
print(psnr(x, y), ssim(x, y))
```

---

## Step 5: Write `models/vae.py`

Load the exact VAE that Stable Diffusion v1.5 uses, as fp16 on the GPU.

**Functions:**
- `load_vae(device=None, dtype=torch.float16)` — returns an `AutoencoderKL` from
  `diffusers`, loaded from the SD v1.5 VAE repo (e.g. `stabilityai/sd-vae-ft-mse`),
  moved to the GPU, set to `eval()` with gradients disabled.
- `encode(vae, image)` and `decode(vae, latent)` helpers that call
  `vae.encode(x).latent_dist.sample()` / `vae.decode(z).sample`.
  - Remember the SD VAE needs a **scaling factor** (`vae.config.scaling_factor`,
    typically `0.18215`): latent is multiplied by it on encode and divided on decode.
- `roundtrip(vae, image)` — `image -> encode -> latent -> decode -> image`.

> **Determinism note**: `latent_dist.sample()` draws randomness. For a reproducible
> round-trip, seed first (`set_seed`) **or** use `latent_dist.mode()` (the mean).
> Document which one you chose and why.

**It should work like this:**
```python
from models.vae import load_vae, roundtrip
vae = load_vae()
recon, latent = roundtrip(vae, image)  # image: [1,3,512,512] in [-1,1]
```

---

## Step 6: Write `scripts/phase2_vae.py`

The test script. It should do these things **in order**:

```
1. import common.paths (redirect HF cache to D:), then set_seed(42, verbose=True)
2. device = get_device(); print it
3. vae = load_vae(device)
4. Load configs/base.yaml to read image_size (512) and dtype
5. Get the list of images in data/secret_images/ (expect 5)
6. For EACH image:
   a. Load with PIL, convert RGB, resize to 512x512
   b. To tensor [1,3,512,512] in [0,1], then normalize to [-1,1]
        (normalize: x = x * 2 - 1)
   c. Move to GPU, cast to fp16
   d. recon, latent = roundtrip(vae, image)
   e. Print shapes:  input [1,3,512,512] -> latent -> output
   f. Clamp recon back to [0,1], cast to fp32 on CPU
   g. Compute mse, psnr, ssim vs the original image
   h. Print: image name, latent shape, MSE, PSNR, SSIM
   i. Save a side-by-side (original | reconstruction) to
        experiments/results/phase2_<name>_compare.png
7. Print a SUMMARY: mean and min/max (spread) of PSNR and SSIM across all images
8. Assert latent shape == [1, 4, 64, 64] and output shape == [1, 3, 512, 512]
```

**Side-by-side tip**: stack the two images horizontally (`torch.cat([a, b], dim=3)`
on `[1,3,512,512]` gives `[1,3,512,1024]`), clamp to `[0,1]`, and save via
`torchvision.utils.save_image`.

---

## Step 7: Run and Check

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase2_vae.py
```

The first run downloads the VAE (~330 MB) into the D: HF cache, so it will take
a minute. Later runs are fast.

### ✅ Checklist (all must pass):

```
[ ] diffusers + accelerate import cleanly
[ ] VAE loads as fp16 on the GPU without OOM
[ ] Input shape  [1, 3, 512, 512]
[ ] Latent shape [1, 4, 64, 64]        (8x downscale, 4 channels)
[ ] Output shape [1, 3, 512, 512]
[ ] MSE / PSNR / SSIM printed for EACH of the 5 images
[ ] Summary shows the spread (not just one number)
[ ] 5 side-by-side PNGs saved to experiments/results/
[ ] PSNR is high (roughly 30+ dB) and reconstructions look very close
```

**When all checks pass, Phase 2 is done.** Paste the output and we move to Phase 3.

---

## The Gate Question (answer in your own words)

> *Why does Stable Diffusion work in latent space instead of pixel space?*

Be ready to explain, in your own words:
- What `B×C×H×W` means and how the VAE turns `512×512×3` into `64×64×4` (8× spatial downscale, 4 channels).
- Why the reconstruction is **not** identical, and why that error is a **floor** for EDICT/channel/retrieval.
- Why pixel range must be `[-1, 1]` for the SD VAE, and why we use fp16 on a 6 GB GPU.
