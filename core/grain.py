"""Deterministic, resolution-aware photographic grain using native Torch ops."""

from __future__ import annotations

import math

import torch

from .color import luminance_rec709, srgb_to_linear
from .blending import blend_signed_layer, normalize_blend_mode
from .filters import gaussian_blur_nchw

_MASK32 = 0xFFFFFFFF


def _hash_u32(value: torch.Tensor) -> torch.Tensor:
    """32-bit integer avalanche hash implemented with signed int64 tensors."""
    x = value & _MASK32
    x = (x ^ 61) ^ (x >> 16)
    x = (x + (x << 3)) & _MASK32
    x = x ^ (x >> 4)
    x = (x * 0x27D4EB2D) & _MASK32
    x = x ^ (x >> 15)
    return x & _MASK32


def _normal_field(
    height: int, width: int, seed: int, stream: int, device: torch.device,
    *, origin_x: int = 0, origin_y: int = 0,
) -> torch.Tensor:
    """Counter-based spatial Gaussian field, repeatable and globally coordinate-anchored."""
    yy = (torch.arange(height, device=device, dtype=torch.int64) + int(origin_y)).view(height, 1)
    xx = (torch.arange(width, device=device, dtype=torch.int64) + int(origin_x)).view(1, width)
    seed_i = int(seed) & 0x7FFFFFFF
    base = (xx * 73856093) ^ (yy * 19349663) ^ (int(stream) * 83492791) ^ (seed_i * 2654435761)
    h1 = _hash_u32(base)
    h2 = _hash_u32(base + 0x9E3779B9 + int(stream) * 40503)
    u1 = (h1.to(torch.float32) + 0.5) / 4294967296.0
    u2 = (h2.to(torch.float32) + 0.5) / 4294967296.0
    return torch.sqrt(-2.0 * torch.log(u1.clamp_min(1e-7))) * torch.cos(2.0 * math.pi * u2)


def apply_film_grain(
    rgb_srgb: torch.Tensor,
    *,
    amount: float,
    grain_size: float,
    color_grain: float,
    seed: int,
    blend_mode: str = "Additive",
    reference_dimensions: tuple[int, int] | None = None,
    coordinate_origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Add seeded grain to one HWC sRGB image using the selected blend mode.

    The grain field is spatially correlated, zero-centred and normalized per
    channel. Its amplitude is modulated by linear-light luminance to retain a
    photographic tonal response. A small common component produces luminance
    grain; the color_grain control blends in independent channel variation.
    """
    if amount <= 0.0:
        return rgb_srgb
    height, width, _ = rgb_srgb.shape
    reference_height, reference_width = reference_dimensions or (height, width)
    scale = max(reference_height, reference_width) / 1024.0
    origin_x, origin_y = int(coordinate_origin[0]), int(coordinate_origin[1])
    sigma = max(0.0, float(grain_size) * scale * 0.50)

    independent = []
    for channel in range(3):
        noise = _normal_field(height, width, seed, 100 + channel, rgb_srgb.device, origin_x=origin_x, origin_y=origin_y)
        if sigma > 0.35:
            noise = gaussian_blur_nchw(noise[None, None], sigma)[0, 0]
        noise = noise - noise.mean()
        noise = noise / noise.square().mean().sqrt().clamp_min(1e-5)
        independent.append(noise)
    independent_noise = torch.stack(independent, dim=-1)

    common = _normal_field(height, width, seed, 99, rgb_srgb.device, origin_x=origin_x, origin_y=origin_y)
    if sigma > 0.35:
        common = gaussian_blur_nchw(common[None, None], sigma)[0, 0]
    common = common - common.mean()
    common = common / common.square().mean().sqrt().clamp_min(1e-5)

    color_grain = float(min(max(color_grain, 0.0), 1.0))
    noise_rgb = (
        common[..., None] * (1.0 - color_grain)
        + independent_noise * color_grain
    )

    luminance = luminance_rec709(srgb_to_linear(rgb_srgb))
    tonal_weight = 0.28 + 0.72 * (4.0 * luminance * (1.0 - luminance)).clamp(0.0, 1.0)
    # Maximum standard deviation remains 6.6% of display-referred code value at Amount=3.
    amplitude = 0.022 * float(min(max(amount, 0.0), 3.0))
    signed_grain = noise_rgb * (amplitude * tonal_weight[..., None])
    return blend_signed_layer(
        rgb_srgb[..., :3].to(torch.float32).clamp(0.0, 1.0),
        signed_grain,
        normalize_blend_mode(blend_mode),
    )
