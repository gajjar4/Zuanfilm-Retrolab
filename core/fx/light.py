"""Light effects: anamorphic streaks, god rays and prism double exposure."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..color import linear_to_srgb, srgb_to_linear
from .common import FxContext, hex_to_rgb, luminance, pixel_grid, ref_px, rgb_to_hsv, hsv_to_rgb, sample, sample_count, smoothstep, to_hwc, to_nchw


def _highlights(linear: torch.Tensor, threshold: float) -> torch.Tensor:
    lum = luminance(linear.clamp(0.0, 1.0))
    return linear * smoothstep(threshold, min(1.0, threshold + 0.25), lum).unsqueeze(-1)


def anamorphic_streaks(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    if p["amount"] <= 0.0:
        return rgb
    height, width = rgb.shape[:2]
    linear = srgb_to_linear(rgb)
    bright = _highlights(linear, p["threshold"])
    length = max(1.0, ref_px(p["length"], rgb))
    # Squeeze horizontally so a long streak becomes a short exponential kernel.
    factor = max(1, int(length // 24))
    small = F.interpolate(to_nchw(bright), size=(height, max(1, width // factor)), mode="area")
    taps = max(2, int(math.ceil(length / factor)))
    offsets = torch.arange(-taps, taps + 1, device=rgb.device, dtype=rgb.dtype)
    kernel = torch.exp(-offsets.abs() / (taps / 3.0))
    kernel = (kernel / kernel.sum()).view(1, 1, 1, -1).expand(3, 1, 1, -1)
    streak = F.conv2d(F.pad(small, (taps, taps, 0, 0), mode="replicate"), kernel, groups=3)
    streak = to_hwc(F.interpolate(streak, size=(height, width), mode="bilinear", align_corners=False))
    tint = hex_to_rgb(p["tint"], rgb.device, rgb.dtype)
    energy = luminance(streak).unsqueeze(-1) * tint * (p["amount"] * 6.0)
    return linear_to_srgb((linear + energy).clamp_min(0.0))


def god_rays(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    if p["amount"] <= 0.0 or p["length"] <= 0.0:
        return rgb
    height, width = rgb.shape[:2]
    linear = srgb_to_linear(rgb)
    bright = to_nchw(_highlights(linear, p["threshold"]))
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    cx, cy = p["cx"] * width, p["cy"] * height
    count = sample_count(p["length"] * max(height, width), 8, 48)
    total = torch.zeros_like(bright)
    weight_sum = 0.0
    for i in range(count):
        scale = 1.0 - p["length"] * i / (count - 1)
        weight = (1.0 - i / count) ** 1.5
        total += sample(bright, cx + (x - cx) * scale, cy + (y - cy) * scale) * weight
        weight_sum += weight
    rays = to_hwc(total / weight_sum) * (p["amount"] * 4.0)
    return linear_to_srgb((linear + rays).clamp_min(0.0))


def prism(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    offset = ref_px(p["offset"], rgb)
    if offset <= 0.0 or p["mix"] <= 0.0:
        return rgb
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    angle = math.radians(p["angle"])
    ghost = to_hwc(sample(to_nchw(rgb), x - math.cos(angle) * offset, y - math.sin(angle) * offset))
    hsv = rgb_to_hsv(ghost)
    hsv = torch.stack(((hsv[..., 0] + p["hue"] / 360.0) % 1.0, hsv[..., 1], hsv[..., 2]), dim=-1)
    ghost = hsv_to_rgb(hsv)
    screened = 1.0 - (1.0 - rgb) * (1.0 - ghost)
    return torch.lerp(rgb, screened, p["mix"]).clamp(0.0, 1.0)
