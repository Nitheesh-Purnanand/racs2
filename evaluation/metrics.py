"""
Image Quality Metrics for RACS2.

Implements the core fidelity metrics used throughout the project to compare a
reference image against a reconstruction: MSE, PSNR, and SSIM. All functions
operate on PyTorch tensors and accept either a single image ``[C, H, W]`` or a
batched image ``[N, C, H, W]``. Values are expected in ``[0, 1]`` (unit range),
which matches the convention used by the VAE round-trip and downstream stages.

Usage:
    from evaluation.metrics import mse, psnr, ssim
    print(mse(a, b), psnr(a, b), ssim(a, b))
"""

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _as_batch(x: Tensor) -> Tensor:
    """Ensures the tensor has a batch dimension -> [N, C, H, W]."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got shape {tuple(x.shape)}")


def _prepare(a: Tensor, b: Tensor) -> tuple:
    """Validates shapes and returns both as float batches on a common device."""
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        raise TypeError("mse/psnr/ssim expect torch tensors.")
    a = _as_batch(a)
    b = _as_batch(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return a.float(), b.float()


def mse(a: Tensor, b: Tensor) -> float:
    """Mean squared error between two unit-range images."""
    a, b = _prepare(a, b)
    return F.mse_loss(a, b, reduction="mean").item()


def psnr(a: Tensor, b: Tensor, data_range: float = 1.0) -> float:
    """
    Peak signal-to-noise ratio in dB for unit-range images.

    PSNR = 10 * log10(data_range^2 / MSE). Returns ``inf`` for identical inputs.
    """
    a, b = _prepare(a, b)
    err = F.mse_loss(a, b, reduction="mean").item()
    if err == 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor((data_range ** 2) / err)).item()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> Tensor:
    """Builds a [C, 1, window, window] 2-D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()

    window_1d = gauss.unsqueeze(0)
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return window_2d.expand(channels, 1, window_size, window_size).to(dtype)


def ssim(
    a: Tensor,
    b: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM) averaged over the image.

    Uses a Gaussian sliding window and the standard SSIM constants
    C1 = (0.01 * L)^2, C2 = (0.03 * L)^2.
    """
    a, b = _prepare(a, b)
    device, dtype = a.device, a.dtype
    channels = a.shape[1]

    window = _gaussian_window(window_size, sigma, channels, device, dtype)
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)

    mu_a_sq = mu_a.pow(2)
    mu_b_sq = mu_b.pow(2)
    mu_ab = mu_a * mu_b

    sigma_a_sq = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def all_metrics(a: Tensor, b: Tensor, data_range: float = 1.0) -> dict:
    """Convenience: returns {'mse', 'psnr', 'ssim'} in one call."""
    return {
        "mse": mse(a, b),
        "psnr": psnr(a, b, data_range=data_range),
        "ssim": ssim(a, b, data_range=data_range),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    print("metrics on slightly-noisy image:")
    for k, v in all_metrics(x, y).items():
        print(f"  {k}: {v:.6f}")
    print("identity check (expect mse=0, psnr=inf, ssim=1):")
    for k, v in all_metrics(x, x).items():
        print(f"  {k}: {v:.6f}")
