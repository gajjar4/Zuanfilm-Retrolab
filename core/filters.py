"""Small dependency-free Torch filtering primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel (approximately 3 sigma radius)."""
    sigma = float(max(sigma, 0.0))
    if sigma < 0.05:
        return torch.ones((1,), device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_blur_nchw(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian to a BCHW tensor using grouped convolutions."""
    if sigma <= 0.05:
        return image
    _, channels, height, width = image.shape
    # Prevent excessive padding on very small inputs.
    sigma = min(float(sigma), max(0.35, min(height, width) * 0.24))
    kernel = gaussian_kernel1d(sigma, device=image.device, dtype=image.dtype)
    radius = (kernel.numel() - 1) // 2
    if radius == 0:
        return image

    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1).contiguous()

    out = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, horizontal, groups=channels)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, vertical, groups=channels)
    return out


def gaussian_blur_scaled(
    image: torch.Tensor,
    sigma_full: float,
    *,
    max_working_sigma: float = 10.0,
    max_working_edge: int = 2048,
) -> torch.Tensor:
    """Blur BCHW data with resolution-aware downsampling for broad optical glows.

    Area reduction keeps broad-field energy stable, while the working blur stays
    compact enough to avoid enormous spatial convolution kernels.
    """
    if sigma_full <= 0.05:
        return image
    _, _, height, width = image.shape
    long_edge = max(height, width)
    factor_sigma = max(1, int(math.ceil(float(sigma_full) / max_working_sigma)))
    factor_edge = max(1, int(math.ceil(long_edge / float(max_working_edge))))
    factor = max(factor_sigma, factor_edge)

    if factor == 1:
        return gaussian_blur_nchw(image, sigma_full)

    work_h = max(1, int(math.ceil(height / factor)))
    work_w = max(1, int(math.ceil(width / factor)))
    small = F.interpolate(image, size=(work_h, work_w), mode="area")
    small = gaussian_blur_nchw(small, sigma_full / factor)
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
