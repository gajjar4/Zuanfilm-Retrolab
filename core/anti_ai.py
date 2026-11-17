"""Deterministic photographic cleanup for over-processed AI renders.

This is a public, self-contained Retrolab correction. It is not a chemical
emulsion simulator or a port of any separate/private film-emulation system.
"""

from __future__ import annotations

import math

import torch

from .color import linear_to_srgb, luminance_rec709, srgb_to_linear


def _smoothstep(edge0: float, edge1: float, value: torch.Tensor) -> torch.Tensor:
    t = ((value - edge0) / max(edge1 - edge0, 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _red_orange_weight(rgb_srgb: torch.Tensor) -> torch.Tensor:
    """Soft hue mask spanning reds through orange, weighted by saturation."""
    rgb = rgb_srgb[..., :3].clamp(0.0, 1.0)
    maximum = rgb.amax(dim=-1)
    minimum = rgb.amin(dim=-1)
    delta = maximum - minimum
    safe_delta = delta.clamp_min(1e-8)
    r, g, b = rgb.unbind(dim=-1)

    hue_r = torch.remainder((g - b) / safe_delta, 6.0)
    hue_g = ((b - r) / safe_delta) + 2.0
    hue_b = ((r - g) / safe_delta) + 4.0
    hue_degrees = torch.where(maximum == r, hue_r, torch.where(maximum == g, hue_g, hue_b)) / 6.0
    hue_degrees = torch.where(delta > 1e-8, hue_degrees, torch.zeros_like(hue_degrees))
    saturation = delta / maximum.clamp_min(1e-6)

    # The center (0.035) covers red through warm orange. Circular distance
    # keeps the red region continuous across the HSV 0/1 boundary.
    distance = torch.abs(hue_degrees - 0.035)
    distance = torch.minimum(distance, 1.0 - distance)
    hue_weight = torch.exp(-0.5 * (distance / 0.075).square())
    sat_weight = _smoothstep(0.025, 0.22, saturation)
    return hue_weight * sat_weight


def apply_anti_ai(rgb_srgb: torch.Tensor, *, amount: float = 0.0) -> torch.Tensor:
    """Gently normalize common AI-render color/tone excesses.

    Amount is 0–100. The correction compresses global contrast, lifts the
    lower toe and midtones, softens highlight roll-off, and selectively reduces
    red/orange chroma. It is deterministic, global (no local halos), and has an
    exact identity bypass at zero.
    """
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC image with at least three channels")
    try:
        strength = float(amount) / 100.0
    except (TypeError, ValueError, OverflowError):
        strength = 0.0
    if not math.isfinite(strength):
        strength = 0.0
    strength = min(max(strength, 0.0), 1.0)
    if strength <= 0.0:
        return rgb_srgb

    source_srgb = rgb_srgb[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    linear = srgb_to_linear(source_srgb)
    luminance = luminance_rec709(linear).clamp(0.0, 1.0)

    # A pivoted, smoothly weighted contrast reduction avoids turning deep
    # blacks into grey. A small toe lift softens the bottom end, a broad
    # midtone bell gently opens faces, and the shoulder rolls highlights off.
    contrast_weight = _smoothstep(0.015, 0.96, luminance)
    compressed = luminance + (0.46 - luminance) * (0.14 * strength * contrast_weight)
    toe_weight = torch.exp(-0.5 * (luminance / 0.105).square())
    midtone_weight = torch.exp(-0.5 * ((luminance - 0.46) / 0.25).square())
    highlight_weight = _smoothstep(0.74, 1.0, luminance).square()
    mapped = (
        compressed
        + 0.012 * strength * toe_weight
        + 0.026 * strength * midtone_weight
        - 0.038 * strength * highlight_weight
    ).clamp(0.0, 1.0)

    ratio = mapped / luminance.clamp_min(1e-6)
    ratio = torch.minimum(ratio, 1.0 / linear.amax(dim=-1).clamp_min(1e-6))
    toned = linear * ratio[..., None]
    toned_y = luminance_rec709(toned).clamp(0.0, 1.0)
    chroma = toned - toned_y[..., None]

    red_orange = _red_orange_weight(source_srgb)
    shadow_desaturation = 1.0 - _smoothstep(0.025, 0.34, luminance)
    global_chroma = 1.0 - 0.08 * strength
    selective_chroma = 1.0 - 0.34 * strength * red_orange
    shadow_chroma = 1.0 - 0.08 * strength * shadow_desaturation
    chroma_scale = (global_chroma * selective_chroma * shadow_chroma).clamp(0.0, 1.0)
    corrected = toned_y[..., None] + chroma * chroma_scale[..., None]
    return linear_to_srgb(corrected)
