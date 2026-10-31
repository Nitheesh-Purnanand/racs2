# Phase 1 — Environment & First GPU Test (Simplified)

> **Goal**: PyTorch + CUDA working, one config file, two tiny utility files.  
> **Only create what Phase 1 needs. Everything else comes later.**

---

## Step 1: Fix PyTorch Installation

Your C: drive is almost full. Pip downloads to C: by default, which causes the crash.  
Run these lines **one by one** in your terminal:

```powershell
# 1. Make sure you're in the project and venv is active
cd d:\racs2
.venv\Scripts\activate

# 2. Point pip's temp/cache to D: (run these every time you open a new terminal)
$env:TMP = "D:\racs2\temp"
$env:TEMP = "D:\racs2\temp"
$env:PIP_CACHE_DIR = "D:\racs2\pip_cache"

# 3. Create those folders
New-Item -ItemType Directory -Force -Path "D:\racs2\temp"
New-Item -ItemType Directory -Force -Path "D:\racs2\pip_cache"

# 4. Install PyTorch with CUDA (this is ~2.5 GB, will take a few minutes)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --only-binary :all: --index-url https://download.pytorch.org/whl/cu124

# 5. Install the small packages
pip install numpy pillow pyyaml
```

### Verify it worked:
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
✅ Expected output:
```
2.6.0+cu124
True
```

If you get `False` for CUDA or the wrong version, let me know before continuing.

---

## Step 2: Create the Phase 1 Structure

Only these files. Nothing more.

```
d:\racs2\
├── configs/
│   └── base.yaml          # Settings
├── common/
│   ├── __init__.py         # Empty file
│   ├── seed.py             # Seeding function
│   └── device.py           # GPU detection
├── scripts/
│   └── phase1_verify.py    # Test script
└── data/
    └── secret_images/
        └── test_001.png    # Any photo you have
```

That's it. **6 files total.** Create the folders and empty `__init__.py`:

```powershell
New-Item -ItemType Directory -Force -Path "d:\racs2\configs"
New-Item -ItemType Directory -Force -Path "d:\racs2\common"
New-Item -ItemType Directory -Force -Path "d:\racs2\scripts"
New-Item -ItemType Directory -Force -Path "d:\racs2\data\secret_images"
New-Item -ItemType File -Force -Path "d:\racs2\common\__init__.py"
```

Then put **any photo** (jpg, png, whatever) into `data\secret_images\` and name it `test_001.png`.

---

## Step 3: Write `configs/base.yaml`

Create this file with these contents:

```yaml
seed: 42
image_size: 512
device: "cuda"
dtype: "float16"
```

That's the entire config for now. We'll add more settings as phases need them.

---

## Step 4: Write `common/seed.py`

Write a function `set_seed(seed)` that makes everything reproducible.

**What it should do:**
- Seed Python's `random` module
- Seed `numpy`
- Seed `torch` (CPU and CUDA)
- Set `torch.backends.cudnn.deterministic = True`
- Set `torch.backends.cudnn.benchmark = False`

**Why**: In RACS, the transmitter and receiver must produce identical results. If random numbers differ, recovery fails.

**It should work like this:**
```python
from common.seed import set_seed
set_seed(42)
```

---

## Step 5: Write `common/device.py`

Write two functions:

**`get_device()`** — returns `torch.device("cuda")` if GPU available, else `torch.device("cpu")`.

**`print_gpu_info()`** — prints:
- GPU name
- Total VRAM (in GB)
- Currently used VRAM (in GB)

**Useful torch functions:**
- `torch.cuda.is_available()`
- `torch.cuda.get_device_name(0)`
- `torch.cuda.get_device_properties(0).total_mem`
- `torch.cuda.memory_allocated(0)`

**It should work like this:**
```python
from common.device import get_device, print_gpu_info
device = get_device()   # torch.device("cuda")
print_gpu_info()        # prints GPU name and VRAM info
```

---

## Step 6: Write `scripts/phase1_verify.py`

This is your test script. It should do these things **in order**:

```
1. Import and call set_seed(42)
2. Import and call get_device() — print the device
3. Import and call print_gpu_info()
4. Load configs/base.yaml with yaml.safe_load() — print the seed value
5. Load data/secret_images/test_001.png with PIL
6. Resize it to 512x512
7. Convert to a PyTorch tensor — shape should be [1, 3, 512, 512]
8. Print: shape, dtype, min value, max value
9. Move tensor to GPU
10. Print GPU memory after the move
11. Reproducibility test:
    - set_seed(42), make a random tensor with torch.randn(3, 3)
    - set_seed(42), make another random tensor with torch.randn(3, 3)
    - Check they are equal with torch.equal()
    - Print PASS or FAIL
```

---

## Step 7: Run and Check

```powershell
cd d:\racs2
.venv\Scripts\python.exe scripts\phase1_verify.py
```

### ✅ Checklist (all must pass):

```
[ ] torch version is 2.6.0+cu124
[ ] torch.cuda.is_available() is True
[ ] GPU name prints (e.g. "NVIDIA GeForce RTX 3050 Laptop GPU")
[ ] Config loads and seed == 42
[ ] Image loads and tensor shape is [1, 3, 512, 512]
[ ] Tensor moves to GPU without error
[ ] Reproducibility test prints PASS
```

**When all 7 checks pass, Phase 1 is done.** Paste me the output and we move to Phase 2.
