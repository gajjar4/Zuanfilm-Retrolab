"""Center-preserving optical soft focus with corner-weighted blur."""

from __future__ import annotations

import torch

from .color import linear_to_srgb, srgb_to_linear
from .filters import gaussian_blur_scaled


def apply_soft_focus(
    rgb_srgb: torch.Tensor,
    *,
    amount: float,
    radius: float,
) -> torch.Tensor:
    """Blend a Gaussian-softened image progressively toward the corners.

    The exact image center receives no blur. Blur contribution increases with
    aspect-correct radial distance and reaches ``amount`` at the four corners.
    ``radius`` is the maximum Gaussian sigma in reference pixels at a 1024 px
    long edge. The blur is computed in linear RGB and uses the existing scaled
    Gaussian primitive to bound working resolution and convolution size.
    """
    if amount <= 0.0 or radius <= 0.05:
        return rgb_srgb
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC RGB image")

    strength = min(max(float(amount), 0.0), 1.0)
    height, width = rgb_srgb.shape[:2]
    sigma = min(max(float(radius), 0.0), 80.0) * max(height, width) / 1024.0
    if sigma <= 0.05:
        return rgb_srgb

    linear = srgb_to_linear(rgb_srgb[..., :3])
    nchw = linear.permute(2, 0, 1).unsqueeze(0).contiguous()
    blurred = gaussian_blur_scaled(nchw, sigma)[0].permute(1, 2, 0)

    device = rgb_srgb.device
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radial = torch.sqrt(xx.square() + yy.square()).div(2.0 ** 0.5).clamp(0.0, 1.0)

    # Keep the inner image crisp; the smoothly rising mix emphasizes the
    # outer frame and corners instead of applying a uniform global blur.
    t = ((radial - 0.12) / 0.88).clamp(0.0, 1.0)
    radial_weight = t.square() * (3.0 - 2.0 * t)
    blend = (strength * radial_weight.pow(1.35)).unsqueeze(-1)
    result_linear = linear * (1.0 - blend) + blurred * blend
    result = linear_to_srgb(result_linear)
    if rgb_srgb.shape[-1] > 3:
        return torch.cat((result, rgb_srgb[..., 3:]), dim=-1)
    return result
