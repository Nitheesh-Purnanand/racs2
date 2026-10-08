"""
Phase 1 Verification Script for RACS2.

Exercises the full Phase 1 surface end to end:
  1. Reproducible seeding (common.seed.set_seed)
  2. Device selection (common.device.get_device)
  3. GPU reporting (common.device.print_gpu_info)
  4. Config loading (configs/base.yaml)
  5. Image loading + preprocessing to a [1, 3, 512, 512] tensor
  6. Tensor transfer to GPU
  7. Reproducibility check with torch.equal

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\phase1_verify.py
"""

import sys
from pathlib import Path

import yaml
import torch
from PIL import Image
from torchvision import transforms

# Make the project root importable regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.seed import set_seed
from common.device import get_device, print_gpu_info


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    # 1. Deterministic seeding
    section("1. Reproducible seeding")
    set_seed(42, verbose=True)

    # 2. Device selection
    section("2. Device selection")
    device = get_device()
    print(f"Selected device: {device}")

    # 3. GPU info
    section("3. GPU information")
    print_gpu_info()

    # 4. Config loading
    section("4. Config loading")
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")
    print(f"Seed from config: {config['seed']}")
    assert config["seed"] == 42, "Config seed mismatch!"

    # 5. Image loading
    section("5. Image loading")
    image_size = config.get("image_size", 512)
    image_path = PROJECT_ROOT / "data" / "secret_images" / "test_001.png"
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Original size: {image.size}, mode: {image.mode}")

    # 6. Resize to 512x512
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # -> [3, H, W], floats in [0, 1]
    ])
    tensor = transform(image)

    # 7. Add batch dimension -> [1, 3, 512, 512]
    tensor = tensor.unsqueeze(0)

    # 8. Report tensor stats
    section("6. Tensor statistics")
    print(f"Shape : {tuple(tensor.shape)}")
    print(f"Dtype : {tensor.dtype}")
    print(f"Min   : {tensor.min().item():.6f}")
    print(f"Max   : {tensor.max().item():.6f}")
    assert tuple(tensor.shape) == (1, 3, image_size, image_size), "Unexpected tensor shape!"

    # 9. Move to GPU
    section("7. Tensor -> GPU")
    tensor = tensor.to(device)
    print(f"Tensor device after move: {tensor.device}")

    # 10. GPU memory after the move
    section("8. GPU memory after move")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"Allocated VRAM: {allocated:.4f} GB")
        print(f"Reserved VRAM : {reserved:.4f} GB")
    else:
        print("CUDA not available; skipped GPU memory report.")

    # 11. Reproducibility test
    section("9. Reproducibility test")
    set_seed(42)
    a = torch.randn(3, 3)
    set_seed(42)
    b = torch.randn(3, 3)
    if torch.equal(a, b):
        print("Reproducibility test: PASS")
    else:
        print("Reproducibility test: FAIL")
        print(f"a = {a}")
        print(f"b = {b}")
        raise SystemExit(1)

    section("Phase 1 verification complete")


if __name__ == "__main__":
    main()
