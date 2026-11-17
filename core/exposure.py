"""Soft photographic lightness adjustment for Zuanfilm Retrolab."""

from __future__ import annotations

import math

import torch

from .color import linear_to_srgb, srgb_to_linear


def apply_exposure(rgb_srgb: torch.Tensor, *, amount: float = 0.0) -> torch.Tensor:
    """Move image lightness smoothly toward black or white.

    ``amount`` is clamped to -100..100. Zero is an exact identity bypass.
    Negative settings linearly blend linear-light RGB toward black, reaching
    5% of the source lightness at -100. Positive settings use a 2.5-power
    ease-in for a more gradual response through low and middle values, reaching
    95% of the remaining distance toward white at +100.

    The adjustment is deliberately a normalized creative lightness control,
    not an EV-based camera exposure simulation.
    """
    if not isinstance(rgb_srgb, torch.Tensor) or rgb_srgb.ndim < 1 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected an RGB tensor with at least three channels")
    try:
        strength = float(amount)
    except (TypeError, ValueError, OverflowError):
        strength = 0.0
    if not math.isfinite(strength):
        strength = 0.0
    strength = min(max(strength, -100.0), 100.0)
    if strength == 0.0:
        return rgb_srgb

    rgb = rgb_srgb[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    linear = srgb_to_linear(rgb)
    if strength < 0.0:
        # Keep the already approved negative darkening response unchanged.
        progress = abs(strength) / 100.0 * 0.95
        adjusted = linear * (1.0 - progress)
    else:
        # Use a 2.5-power ease-in for more gradual positive control. This preserves
        # the +100 endpoint while spreading useful tonal travel toward the top.
        normalized = strength / 100.0
        progress = (normalized ** 2.5) * 0.95
        adjusted = linear + (1.0 - linear) * progress
    return linear_to_srgb(adjusted)
