"""Shared helpers for Retrolab FX effects.

Every effect receives one HWC RGB frame in display-referred sRGB [0, 1] and an
``FxContext``. Spatial sizes are authored in reference pixels at a 1024 px long
edge, exactly like the main Retrolab node, so the reduced live preview and the
full-resolution render produce the same look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FxContext:
    frame_index: int = 0
    # Long edge of the full-resolution source. The preview proxy passes the
    # source size so per-pixel sizes and seeded patterns stay consistent.
    reference_long_edge: int = 1024


def ref_px(value: float, image: torch.Tensor) -> float:
    """Convert reference pixels (1024 px long edge) to pixels of ``image``."""
    return float(value) * max(image.shape[0], image.shape[1]) / 1024.0


def hex_to_rgb(value: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        rgb = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        rgb = [0.0, 0.0, 0.0]
    return torch.tensor(rgb, device=device, dtype=dtype)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def to_nchw(rgb: torch.Tensor) -> torch.Tensor:
    return rgb.permute(2, 0, 1).unsqueeze(0)


def to_hwc(nchw: torch.Tensor) -> torch.Tensor:
    return nchw[0].permute(1, 2, 0)


def pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates (x, y), each shaped (H, W)."""
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    return xs.view(1, -1).expand(height, width), ys.view(-1, 1).expand(height, width)


def sample(nchw: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample an NCHW image at pixel coordinates (H, W) with edge clamping."""
    height, width = nchw.shape[-2:]
    grid = torch.stack((x / width * 2.0 - 1.0, y / height * 2.0 - 1.0), dim=-1).unsqueeze(0)
    return F.grid_sample(nchw, grid.to(nchw.dtype), mode=mode, padding_mode="border", align_corners=False)


def smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash_noise(height: int, width: int, seed: int, device: torch.device, dtype: torch.dtype, channels: int = 1) -> torch.Tensor:
    """Deterministic uniform noise in [0, 1) from integer pixel coordinates."""
    y = torch.arange(height, device=device, dtype=torch.int64).view(-1, 1, 1)
    x = torch.arange(width, device=device, dtype=torch.int64).view(1, -1, 1)
    c = torch.arange(channels, device=device, dtype=torch.int64).view(1, 1, -1)
    value = (x * 73856093) ^ (y * 19349663) ^ (c * 83492791) ^ (int(seed) * 2654435761)
    value = value & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    value = (value >> 16) ^ value
    return ((value & 0xFFFFFF).to(dtype) / float(0x1000000))


def hash_scalar(seed: int, index: int) -> float:
    """Deterministic uniform value in [0, 1) for per-frame or per-band choices."""
    value = (int(seed) * 2654435761 + int(index) * 40503 + 0x9E3779B9) & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    value = (value >> 16) ^ value
    return (value & 0xFFFFFF) / float(0x1000000)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(-1)
    maxc = rgb.amax(-1)
    minc = rgb.amin(-1)
    delta = maxc - minc
    safe = delta.clamp_min(1e-8)
    hue = torch.where(maxc == r, ((g - b) / safe) % 6.0,
          torch.where(maxc == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    hue = torch.where(delta > 1e-8, hue / 6.0, torch.zeros_like(hue))
    sat = torch.where(maxc > 1e-8, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    return torch.stack((hue % 1.0, sat, maxc), dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv.unbind(-1)
    k = torch.stack((5.0 + h * 6.0, 3.0 + h * 6.0, 1.0 + h * 6.0), dim=-1) % 6.0
    t = torch.minimum(k, 4.0 - k).clamp(0.0, 1.0)
    return v.unsqueeze(-1) - v.unsqueeze(-1) * s.unsqueeze(-1) * t


BLEND_MODES = (
    "Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light",
    "Lighten", "Darken", "Difference", "Add", "Color", "Luminosity",
)


def composite(base: torch.Tensor, layer: torch.Tensor, mode: str, opacity: float) -> torch.Tensor:
    """Photoshop-style layer compositing of ``layer`` over ``base`` (both sRGB HWC)."""
    if mode == "Multiply":
        mixed = base * layer
    elif mode == "Screen":
        mixed = 1.0 - (1.0 - base) * (1.0 - layer)
    elif mode == "Overlay":
        mixed = torch.where(base <= 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))
    elif mode == "Hard Light":
        mixed = torch.where(layer <= 0.5, 2.0 * base * layer, 1.0 - 2.0 * (1.0 - base) * (1.0 - layer))
    elif mode == "Soft Light":
        d = torch.where(base <= 0.25, ((16.0 * base - 12.0) * base + 4.0) * base, base.clamp_min(0.0).sqrt())
        mixed = torch.where(layer <= 0.5, base - (1.0 - 2.0 * layer) * base * (1.0 - base), base + (2.0 * layer - 1.0) * (d - base))
    elif mode == "Lighten":
        mixed = torch.maximum(base, layer)
    elif mode == "Darken":
        mixed = torch.minimum(base, layer)
    elif mode == "Difference":
        mixed = (base - layer).abs()
    elif mode == "Add":
        mixed = base + layer
    elif mode in ("Color", "Luminosity"):
        lb, ll = luminance(base).unsqueeze(-1), luminance(layer).unsqueeze(-1)
        mixed = layer + (lb - ll) if mode == "Color" else base + (ll - lb)
    else:
        mixed = layer
    mixed = mixed.clamp(0.0, 1.0)
    if opacity >= 1.0:
        return mixed
    return torch.lerp(base, mixed, float(opacity))


def area_resize(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return to_hwc(F.interpolate(to_nchw(rgb), size=(max(1, height), max(1, width)), mode="area"))


def nearest_resize(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return to_hwc(F.interpolate(to_nchw(rgb), size=(height, width), mode="nearest"))


def sample_count(span_px: float, minimum: int = 2, maximum: int = 48) -> int:
    return int(min(maximum, max(minimum, math.ceil(span_px / 1.5))))
