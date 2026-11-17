"""Soft-knee, multi-scale highlight bloom implemented with Torch."""

from __future__ import annotations

import torch

from .color import linear_to_srgb, luminance_rec709, srgb_to_linear
from .filters import gaussian_blur_scaled


def _soft_knee_highlights(luminance: torch.Tensor, threshold: float, knee: float = 0.12) -> torch.Tensor:
    """Continuous highlight extraction with a soft knee that never extends below black."""
    threshold = max(float(threshold), 0.0)
    knee = max(float(knee), 0.0)
    # At low thresholds, shorten the transition band so its lower edge remains
    # at or above zero. A zero threshold intentionally includes every non-black
    # tone while keeping a completely black pixel at exactly zero extraction.
    effective_knee = min(knee, 2.0 * threshold)
    if effective_knee <= 1e-8:
        return luminance.clamp_min(0.0)
    half_knee = effective_knee * 0.5
    x = (luminance - (threshold - half_knee)).clamp(0.0, effective_knee)
    shoulder = (x.square() / (2.0 * effective_knee)) + (luminance - (threshold + half_knee)).clamp_min(0.0)
    return shoulder


def apply_bloom(
    rgb_srgb: torch.Tensor,
    *,
    amount: float,
    threshold: float,
    radius: float,
) -> torch.Tensor:
    """Add a broad, color-preserving optical glow to one HWC sRGB image."""
    amount = min(max(float(amount), 0.0), 3.0)
    if amount <= 0.0:
        return rgb_srgb
    linear = srgb_to_linear(rgb_srgb)
    luminance = luminance_rec709(linear)
    extracted_luma = _soft_knee_highlights(luminance, threshold)
    ratio = extracted_luma / luminance.clamp_min(1e-5)
    highlights = (linear * ratio[..., None]).clamp_min(0.0)

    height, width, _ = linear.shape
    scale = max(height, width) / 1024.0
    radius_px = max(0.0, float(radius)) * scale
    nchw = highlights.permute(2, 0, 1).unsqueeze(0)
    core = gaussian_blur_scaled(nchw, radius_px)
    tail = gaussian_blur_scaled(nchw, radius_px * 2.5)
    glow = 0.65 * core + 0.35 * tail
    # A restrained exposure gain keeps the node useful on already bright AI outputs.
    out = linear + 0.38 * float(amount) * glow[0].permute(1, 2, 0)
    return linear_to_srgb(out.clamp(0.0, 1.0))
