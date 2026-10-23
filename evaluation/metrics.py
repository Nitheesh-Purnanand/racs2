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

import math

import numpy as np
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


_LPIPS_CACHE: dict = {}


def _lpips_model(net: str = "alex", device=None):
    """Loads and caches the LPIPS network. Imports lpips lazily."""
    import lpips as lpips_lib

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (net, str(device))
    if key not in _LPIPS_CACHE:
        model = lpips_lib.LPIPS(net=net, verbose=False).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def lpips_metric(a: Tensor, b: Tensor, net: str = "alex", device=None,
                 batch_size: int = 1) -> float:
    """
    Learned Perceptual Image Patch Similarity. Lower is better, 0 for identical.

    Input tensors are unit-range ``[0, 1]`` (project convention). LPIPS expects
    ``[-1, 1]``, so the conversion happens here and the input range is asserted.
    """
    a, b = _prepare(a, b)
    if a.numel() == 0:
        raise ValueError("lpips_metric received an empty tensor")
    if float(a.min()) < -0.001 or float(a.max()) > 1.001:
        raise ValueError("lpips_metric expects unit-range [0, 1] inputs")
    if float(b.min()) < -0.001 or float(b.max()) > 1.001:
        raise ValueError("lpips_metric expects unit-range [0, 1] inputs")
    if a.shape[1] == 1:
        a = a.repeat(1, 3, 1, 1)
        b = b.repeat(1, 3, 1, 1)

    model = _lpips_model(net=net, device=device)
    run_device = next(model.parameters()).device
    step = max(1, int(batch_size))
    values = []
    with torch.inference_mode():
        for start in range(0, a.shape[0], step):
            x = a[start:start + step].to(run_device) * 2.0 - 1.0
            y = b[start:start + step].to(run_device) * 2.0 - 1.0
            values.append(model(x, y).reshape(-1))
    return torch.cat(values).mean().item()


def distribution(values, percentiles=(5, 25, 50, 75, 95)) -> dict:
    """
    Summarises a list of metric values, handling ``inf`` and missing entries.

    Means and standard deviations are computed over finite values only. ``n_inf``
    counts infinities (identical images give infinite PSNR) and ``n_missing``
    counts ``None`` entries.
    """
    seq = list(values)
    finite, n_inf, n_missing = [], 0, 0
    for value in seq:
        if value is None:
            n_missing += 1
            continue
        number = float(value)
        if math.isnan(number):
            n_missing += 1
        elif math.isinf(number):
            n_inf += 1
        else:
            finite.append(number)

    out = {"n": len(seq), "n_inf": n_inf, "n_missing": n_missing,
           "n_finite": len(finite)}
    if finite:
        array = np.asarray(finite, dtype=np.float64)
        out.update({
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean_finite": float(array.mean()),
        })
        for percentile in percentiles:
            key = f"p{int(percentile):02d}"
            out[key] = float(np.percentile(array, percentile))
    else:
        out.update({"mean": None, "median": None, "std": None,
                    "min": None, "max": None, "mean_finite": None})
        for percentile in percentiles:
            out[f"p{int(percentile):02d}"] = None
    return out


def all_metrics(a: Tensor, b: Tensor, data_range: float = 1.0,
                with_lpips: bool = False, lpips_net: str = "alex") -> dict:
    """
    Convenience: returns {'mse', 'psnr', 'ssim'} in one call.

    Pass ``with_lpips=True`` to add the perceptual metric. The default keeps the
    Phase 2-3 behaviour unchanged.
    """
    result = {
        "mse": mse(a, b),
        "psnr": psnr(a, b, data_range=data_range),
        "ssim": ssim(a, b, data_range=data_range),
    }
    if with_lpips:
        result["lpips"] = lpips_metric(a, b, net=lpips_net)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    print("metrics on slightly-noisy image:")
    for k, v in all_metrics(x, y, with_lpips=True).items():
        print(f"  {k}: {v:.6f}")
    print("identity check (expect mse=0, psnr=inf, ssim=1):")
    for k, v in all_metrics(x, x, with_lpips=True).items():
        print(f"  {k}: {v:.6f}")
    print("distribution check:", distribution([1.0, 2.0, 3.0, float("inf")]))
