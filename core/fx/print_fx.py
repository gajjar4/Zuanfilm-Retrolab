"""Print effects: halftone (mono and CMYK), gradient map and risograph."""

from __future__ import annotations

import math

import torch

from ..filters import gaussian_blur_scaled
from .common import FxContext, hash_noise, hex_to_rgb, luminance, pixel_grid, ref_px, sample, smoothstep, to_hwc, to_nchw

# Process inks and traditional screen angles (degrees).
_CMYK_INKS = (("#00AEEF", 15.0), ("#EC008C", 75.0), ("#FFF200", 0.0), ("#231F20", 45.0))


def _dot_coverage(density: torch.Tensor, cell: float, angle_deg: float, gain: float, offset: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """Coverage (0..1) of round halftone dots for a single-channel density map (H, W)."""
    height, width = density.shape
    x, y = pixel_grid(height, width, density.device, density.dtype)
    x, y = x - offset[0], y - offset[1]
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    u, v = x * ca + y * sa, -x * sa + y * ca
    uc, vc = (torch.floor(u / cell) + 0.5) * cell, (torch.floor(v / cell) + 0.5) * cell
    xc, yc = uc * ca - vc * sa + offset[0], uc * sa + vc * ca + offset[1]
    smooth = gaussian_blur_scaled(density.view(1, 1, height, width), cell * 0.35)
    value = sample(smooth, xc, yc)[0, 0].clamp(0.0, 1.0)
    radius = cell * 0.7071 * value.sqrt() * (1.0 + gain)
    distance = ((u - uc).square() + (v - vc).square()).sqrt()
    return 1.0 - smoothstep(-0.75, 0.75, distance - radius)


def halftone(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    cell = max(2.0, ref_px(p["cell"], rgb))
    device, dtype = rgb.device, rgb.dtype
    paper = hex_to_rgb(p["paper"], device, dtype)
    if p["mode"] == "CMYK":
        k = 1.0 - rgb.amax(-1)
        cmy = (1.0 - rgb - k.unsqueeze(-1)) / (1.0 - k).clamp_min(1e-4).unsqueeze(-1)
        channels = (cmy[..., 0], cmy[..., 1], cmy[..., 2], k)
        out = paper.expand_as(rgb).clone()
        for density, (ink_hex, angle) in zip(channels, _CMYK_INKS):
            cover = _dot_coverage(density, cell, angle + p["angle"], p["gain"]).unsqueeze(-1)
            out = out * torch.lerp(torch.ones_like(out), hex_to_rgb(ink_hex, device, dtype).expand_as(out), cover)
        return out.clamp(0.0, 1.0)
    cover = _dot_coverage(1.0 - luminance(rgb), cell, p["angle"], p["gain"]).unsqueeze(-1)
    ink = hex_to_rgb(p["ink"], device, dtype)
    return torch.lerp(paper.expand_as(rgb), ink.expand_as(rgb), cover)


def gradient_map(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    device, dtype = rgb.device, rgb.dtype
    stops = [hex_to_rgb(p["shadows"], device, dtype), hex_to_rgb(p["midtones"], device, dtype), hex_to_rgb(p["highlights"], device, dtype)]
    t = luminance(rgb).clamp(0.0, 1.0).unsqueeze(-1)
    if not p["use_midtones"]:
        mapped = torch.lerp(stops[0].expand_as(rgb), stops[2].expand_as(rgb), t)
    else:
        low = torch.lerp(stops[0].expand_as(rgb), stops[1].expand_as(rgb), (t * 2.0).clamp(0.0, 1.0))
        mapped = torch.lerp(low, stops[2].expand_as(rgb), (t * 2.0 - 1.0).clamp(0.0, 1.0))
    return torch.lerp(rgb, mapped, p["mix"])


def risograph(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    cell = max(2.0, ref_px(p["cell"], rgb))
    device, dtype = rgb.device, rgb.dtype
    height, width = rgb.shape[:2]
    paper = hex_to_rgb(p["paper"], device, dtype).expand_as(rgb)
    lum = luminance(rgb)
    # Ink A carries the shadows; ink B carries warm, red-leaning detail.
    density_a = (1.0 - lum).clamp(0.0, 1.0) ** 2.2
    density_b = ((rgb[..., 0] - lum) * 2.4 + (1.0 - lum).clamp(0.0, 1.0) ** 1.5 * 0.55).clamp(0.0, 1.0)
    shift = ref_px(p["misregister"], rgb)
    cover_a = _dot_coverage(density_a, cell, 22.0, 0.05).unsqueeze(-1)
    cover_b = _dot_coverage(density_b, cell, 67.0, 0.05, offset=(shift, shift * 0.6)).unsqueeze(-1)
    ink_a, ink_b = hex_to_rgb(p["ink_a"], device, dtype), hex_to_rgb(p["ink_b"], device, dtype)
    out = paper * torch.lerp(torch.ones_like(paper), ink_a.expand_as(paper), cover_a * 0.92)
    out = out * torch.lerp(torch.ones_like(paper), ink_b.expand_as(paper), cover_b * 0.85)
    if p["grain"] > 0.0:
        noise = hash_noise(height, width, 11, device, dtype)
        out = out * (1.0 - (noise - 0.5) * 0.25 * p["grain"])
    return out.clamp(0.0, 1.0)
