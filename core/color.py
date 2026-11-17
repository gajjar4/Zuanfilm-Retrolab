"""sRGB transfer functions and luminance helpers implemented in PyTorch."""

from __future__ import annotations

import torch


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB-encoded RGB values to linear-light RGB.

    Input and output are (..., 3) tensors. Input is expected to be in [0, 1].
    """
    x = rgb.clamp(0.0, 1.0)
    low = x / 12.92
    high = ((x + 0.055) / 1.055).pow(2.4)
    return torch.where(x <= 0.04045, low, high)


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear-light RGB values to sRGB encoding."""
    x = rgb.clamp_min(0.0)
    low = x * 12.92
    high = 1.055 * x.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, low, high).clamp(0.0, 1.0)


def luminance_rec709(rgb_linear: torch.Tensor) -> torch.Tensor:
    """Return Rec.709 linear luminance from (..., 3) linear RGB."""
    return (
        0.2126 * rgb_linear[..., 0]
        + 0.7152 * rgb_linear[..., 1]
        + 0.0722 * rgb_linear[..., 2]
    )
