"""Blur effects: directional, zoom, spin, tilt-shift and lens blur with bokeh."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..color import linear_to_srgb, srgb_to_linear
from ..filters import gaussian_blur_scaled
from .common import FxContext, luminance, pixel_grid, ref_px, sample, sample_count, smoothstep, to_hwc, to_nchw


def _average_samples(linear: torch.Tensor, coords: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    nchw = to_nchw(linear)
    total = torch.zeros_like(nchw)
    for x, y in coords:
        total += sample(nchw, x, y)
    return linear_to_srgb(to_hwc(total / len(coords)).clamp_min(0.0))


def directional_blur(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    distance = ref_px(p["distance"], rgb)
    if distance < 0.5:
        return rgb
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    angle = math.radians(p["angle"])
    dx, dy = math.cos(angle), math.sin(angle)
    count = sample_count(distance)
    coords = []
    for i in range(count):
        t = (i / (count - 1) - 0.5) * distance
        coords.append((x + dx * t, y + dy * t))
    return _average_samples(srgb_to_linear(rgb), coords)


def zoom_blur(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    amount = p["amount"]
    if amount <= 0.0:
        return rgb
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    cx, cy = p["cx"] * width, p["cy"] * height
    count = sample_count(amount * max(height, width) * 0.5, 4, 40)
    coords = []
    for i in range(count):
        scale = 1.0 - amount * i / (count - 1)
        coords.append((cx + (x - cx) * scale, cy + (y - cy) * scale))
    return _average_samples(srgb_to_linear(rgb), coords)


def spin_blur(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    angle = math.radians(p["angle"])
    if abs(angle) < 1e-4:
        return rgb
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    cx, cy = p["cx"] * width, p["cy"] * height
    count = sample_count(angle * max(height, width) * 0.5, 4, 40)
    coords = []
    for i in range(count):
        theta = (i / (count - 1) - 0.5) * angle
        c, s = math.cos(theta), math.sin(theta)
        coords.append((cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c))
    return _average_samples(srgb_to_linear(rgb), coords)


def tilt_shift(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    sigma = ref_px(p["blur"], rgb)
    if sigma <= 0.05:
        return rgb
    height, width = rgb.shape[:2]
    linear = srgb_to_linear(rgb)
    blurred = to_hwc(gaussian_blur_scaled(to_nchw(linear), sigma))
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    angle = math.radians(p["angle"])
    # Distance from the focus line through the handle, normalised by height.
    distance = ((y - p["cy"] * height) * math.cos(angle) - (x - p["cx"] * width) * math.sin(angle)).abs() / height
    half = p["band"] * 0.5
    mask = smoothstep(half, half + max(p["feather"], 0.01), distance).unsqueeze(-1)
    return linear_to_srgb(torch.lerp(linear, blurred, mask).clamp_min(0.0))


def _bokeh_kernel(radius: int, shape: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    if shape == "Hexagon":
        ax, ay = x.abs(), y.abs()
        inside = (ay <= radius * 0.866) & (ax * 0.866 + ay * 0.5 <= radius * 0.866)
    elif shape == "Cat eye":
        inside = (x.square() + y.square() <= radius ** 2) & ((x - radius * 0.45).square() + y.square() <= radius ** 2)
    else:
        inside = x.square() + y.square() <= radius ** 2
    kernel = inside.to(dtype)
    return kernel / kernel.sum().clamp_min(1.0)


def lens_blur(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    radius = ref_px(p["radius"], rgb)
    if radius < 0.75:
        return rgb
    height, width = rgb.shape[:2]
    # Work at a resolution where the bokeh disc stays within a 24 px radius.
    factor = max(1, math.ceil(radius / 24.0))
    linear = srgb_to_linear(rgb)
    small = F.interpolate(to_nchw(linear), size=(max(1, height // factor), max(1, width // factor)), mode="area")
    r = max(1, int(round(radius / factor)))
    kernel = _bokeh_kernel(r, p["shape"], rgb.device, rgb.dtype)
    weight = 1.0 + p["highlights"] * smoothstep(0.55, 1.0, luminance(to_hwc(small).clamp(0.0, 1.0))).square()
    weight = weight.view(1, 1, *weight.shape)
    padded = lambda t: F.pad(t, (r, r, r, r), mode="replicate")
    channels = small.shape[1]
    color = F.conv2d(padded(small * weight), kernel.view(1, 1, *kernel.shape).expand(channels, 1, -1, -1), groups=channels)
    norm = F.conv2d(padded(weight), kernel.view(1, 1, *kernel.shape))
    blurred = F.interpolate(color / norm.clamp_min(1e-6), size=(height, width), mode="bilinear", align_corners=False)
    blurred = to_hwc(blurred)
    focus = p["focus"]
    if focus > 0.0:
        x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
        long_edge = max(height, width)
        distance = ((x - p["cx"] * width).square() + (y - p["cy"] * height).square()).sqrt() / long_edge
        mask = smoothstep(focus * 0.5, focus * 0.5 + 0.18, distance).unsqueeze(-1)
        blurred = torch.lerp(linear, blurred, mask)
    return linear_to_srgb(blurred.clamp_min(0.0))
