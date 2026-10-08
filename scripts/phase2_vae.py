"""
Phase 2 — VAE Round-Trip Verification for RACS2.

Runs the Stable Diffusion VAE round-trip (pixel -> latent -> pixel) over every
image in `data/secret_images/`, measures MSE / PSNR / SSIM against the original,
and saves a side-by-side comparison for each image to `experiments/results/`.

Every diffusion / EDICT stage downstream operates in this latent space, so the
reconstruction error measured here is the error floor for the whole pipeline.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase2_vae.py
"""

import sys
from pathlib import Path

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect the HuggingFace cache to D: before any model import.
import common.paths  # noqa: E402,F401

import yaml
import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

from common.seed import set_seed
from common.device import get_device
from common.paths import SECRET_IMAGES_DIR, RESULTS_DIR, PROJECT_ROOT as _ROOT
from evaluation.metrics import mse, psnr, ssim
from models.vae import load_vae, roundtrip

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def load_image_tensor(path: Path, image_size: int, device, dtype) -> torch.Tensor:
    """Loads an image as a [1, 3, S, S] tensor in [-1, 1] on the target device."""
    image = Image.open(path).convert("RGB")
    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # -> [3, S, S] in [0, 1]
    ])
    tensor = to_tensor(image).unsqueeze(0)         # [1, 3, S, S] in [0, 1]
    tensor = tensor.to(device=device, dtype=dtype)
    tensor = tensor * 2.0 - 1.0                     # normalize to [-1, 1]
    return tensor


def main() -> None:
    # 1. Seeding + device
    section("1-2. Seeding & device")
    set_seed(42, verbose=True)
    device = get_device()
    print(f"Selected device: {device}")

    # 3. Load the SD VAE
    section("3. Load VAE")
    vae = load_vae(device)
    dtype = vae.dtype
    print(f"VAE dtype: {dtype}")

    # 4. Config
    section("4. Config")
    with open(_ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    image_size = int(config.get("image_size", 512))
    print(f"image_size = {image_size}")

    # 5. Gather images
    section("5. Images")
    image_paths = sorted(
        p for p in SECRET_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Found {len(image_paths)} image(s) in {SECRET_IMAGES_DIR}")
    for p in image_paths:
        print(f"  - {p.name}")
    assert image_paths, "No images found in data/secret_images/."

    # 6. Round-trip each image
    section("6. Per-image round-trip")
    rows = []
    for path in image_paths:
        original = load_image_tensor(path, image_size, device, dtype)
        recon, latent = roundtrip(vae, original)

        # Back to [0, 1] fp32 on CPU for measurement / saving.
        orig_01 = ((original.float() + 1.0) / 2.0).clamp(0, 1).cpu()
        recon_01 = ((recon.float() + 1.0) / 2.0).clamp(0, 1).cpu()

        # Assertions on shapes.
        assert tuple(original.shape) == (1, 3, image_size, image_size)
        assert tuple(latent.shape) == (1, 4, image_size // 8, image_size // 8)
        assert tuple(recon.shape) == (1, 3, image_size, image_size)

        m = mse(orig_01, recon_01)
        p = psnr(orig_01, recon_01)
        s = ssim(orig_01, recon_01)
        rows.append({"name": path.stem, "mse": m, "psnr": p, "ssim": s})

        print(f"\n[{path.name}]")
        print(f"  input  shape: {tuple(original.shape)}")
        print(f"  latent shape: {tuple(latent.shape)}")
        print(f"  output shape: {tuple(recon.shape)}")
        print(f"  MSE={m:.6f}  PSNR={p:.3f} dB  SSIM={s:.6f}")

        # Side-by-side (original | reconstruction).
        compare = torch.cat([orig_01, recon_01], dim=3)  # [1, 3, S, 2S]
        out_path = RESULTS_DIR / f"phase2_{path.stem}_compare.png"
        vutils.save_image(compare, out_path)
        print(f"  saved: {out_path}")

    # 7. Summary / spread
    section("7. Summary (spread across images)")
    psnrs = [r["psnr"] for r in rows]
    ssims = [r["ssim"] for r in rows]
    mses = [r["mse"] for r in rows]
    print(f"Images tested : {len(rows)}")
    print(f"MSE  : mean={sum(mses)/len(mses):.6f}  min={min(mses):.6f}  max={max(mses):.6f}")
    print(f"PSNR : mean={sum(psnrs)/len(psnrs):.3f}  min={min(psnrs):.3f}  max={max(psnrs):.3f} dB")
    print(f"SSIM : mean={sum(ssims)/len(ssims):.6f}  min={min(ssims):.6f}  max={max(ssims):.6f}")

    section("Phase 2 verification complete")


if __name__ == "__main__":
    main()
