"""Tone-reversal Sabattier character and asymmetric Mackie contours."""

from __future__ import annotations

import torch

from .color import linear_to_srgb, luminance_rec709, srgb_to_linear
from .filters import gaussian_blur_scaled


def _solarization_lut(luminance: torch.Tensor, beta: float, shield: float) -> torch.Tensor:
    """Apply x*10^(-beta*x^k), normalized to the curve's actual maximum."""
    x = luminance.clamp(0.0, 1.0)
    beta = max(float(beta), 0.0)
    shield = max(float(shield), 0.25)
    if beta <= 0.0:
        return x

    peak = (1.0 / (beta * shield * 2.302585092994046)) ** (1.0 / shield)
    peak = min(max(peak, 1e-6), 1.0)
    peak_value = peak * (10.0 ** (-beta * (peak ** shield)))
    end_value = 10.0 ** (-beta)
    normalizer = max(peak_value, end_value, 1e-8)
    return x * torch.pow(torch.tensor(10.0, device=x.device), -beta * x.pow(shield)) / normalizer


def apply_sabattier(
    rgb_srgb: torch.Tensor,
    *,
    solarization: float,
    highlight_confinement: float,
    mackie_lines: float,
) -> torch.Tensor:
    """Apply hue-preserving highlight reversal and optional Mackie edge lines.

    UI mappings: solarization [0,1] -> beta [0,2]; confinement [0,1] -> k
    [0.5,4.0]; Mackie amount [0,1] -> positive-lobe gain [0,0.65]. Edge width
    and asymmetry are fixed implementation constants.
    """
    solarization = min(max(float(solarization), 0.0), 1.0)
    mackie_lines = min(max(float(mackie_lines), 0.0), 1.0)
    if solarization <= 0.0 and mackie_lines <= 0.0:
        return rgb_srgb

    linear = srgb_to_linear(rgb_srgb)
    luminance = luminance_rec709(linear)
    if solarization > 0.0:
        beta = 2.0 * solarization
        shield = 0.5 + 3.5 * min(max(float(highlight_confinement), 0.0), 1.0)
        folded = _solarization_lut(luminance, beta, shield)
    else:
        folded = luminance

    if mackie_lines > 0.0:
        height, width = luminance.shape
        sigma = 2.0 * max(height, width) / 1024.0
        smooth = gaussian_blur_scaled(luminance[None, None], sigma)[0, 0]
        high_pass = luminance - smooth
        gain_pos = 0.65 * mackie_lines
        gain_neg = gain_pos / 6.0
        edge_delta = torch.where(high_pass > 0.0, gain_pos * high_pass, gain_neg * high_pass)
        processed_luma = (folded + edge_delta).clamp_min(0.0)
    else:
        processed_luma = folded

    # Ratio recombination changes luminance while retaining the original hue.
    ratio = processed_luma / luminance.clamp_min(1e-6)
    # Apply one common RGB ratio and gamut-limit that ratio before channel
    # clipping. This preserves hue instead of clipping saturated channels first.
    max_safe_ratio = 1.0 / linear.amax(dim=-1).clamp_min(1e-6)
    ratio = torch.minimum(ratio, max_safe_ratio)
    out = linear * ratio[..., None]
    return linear_to_srgb(out)
