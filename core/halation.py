"""Fresnel/TIR annular halation with a diffuse scattering tail (Torch only)."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, srgb_to_linear

DEFAULT_HALATION_COLOR = "#FF9959"
_DEFAULT_CHANNEL_WEIGHTS = (1.0, 0.32, 0.10)


def normalize_halation_color(value: str | None) -> str:
    """Normalize an sRGB hex color, falling back to the original halation tint."""
    if not isinstance(value, str):
        return DEFAULT_HALATION_COLOR
    candidate = value.strip()
    raw = candidate[1:] if candidate.startswith("#") else candidate
    if len(raw) == 3 and all(ch in "0123456789abcdefABCDEF" for ch in raw):
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        return DEFAULT_HALATION_COLOR
    return "#" + raw.upper()

_KERNEL_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_CACHE_LIMIT = 6


def _next_fast_length(value: int) -> int:
    """Smallest integer >= value with prime factors limited to 2, 3 and 5."""
    candidate = int(value)
    while True:
        remainder = candidate
        for prime in (2, 3, 5):
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return candidate
        candidate += 1


def build_halation_kernel(
    ring_radius_px: float,
    tail_length_px: float,
    balance: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build a unit-energy radial kernel with a Fresnel/TIR annular component.

    The radius is a reference-pixel control. The refractive index (1.5), ring / tail
    blend (0.65) are implementation constants. The selected halation color controls
    normalized linear RGB channel weights; the default preserves the legacy response.
    """
    ring_radius_px = max(float(ring_radius_px), 1e-3)
    tail_length_px = max(float(tail_length_px), 1e-3)
    balance = min(max(float(balance), 0.0), 1.0)
    half_size = max(int(math.ceil(4.0 * ring_radius_px)), int(math.ceil(6.0 * tail_length_px)), 8)
    coords = torch.arange(-half_size, half_size + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())

    refractive_index = 1.5
    two_d = ring_radius_px * math.sqrt(refractive_index**2 - 1.0)
    t = radius / two_d
    theta = torch.atan(t)
    sin_t = refractive_index * torch.sin(theta)
    subcritical = sin_t < 1.0
    theta_transmitted = torch.asin(sin_t.clamp(0.0, 1.0))
    cos_i = torch.cos(theta)
    cos_t = torch.cos(theta_transmitted)
    rs = (refractive_index * cos_i - cos_t) / (refractive_index * cos_i + cos_t).clamp_min(1e-8)
    rp = (refractive_index * cos_t - cos_i) / (refractive_index * cos_t + cos_i).clamp_min(1e-8)
    fresnel = torch.where(subcritical, 0.5 * (rs.square() + rp.square()), torch.ones_like(radius))

    ring = fresnel / (1.0 + t.square()).square()
    ring = ring / ring.sum().clamp_min(1e-12)
    # The scattering tail fills outward from the annulus instead of placing a
    # second, centre-heavy Gaussian-like source on top of the TIR ring.
    radial_ramp = (radius / ring_radius_px).clamp(0.0, 1.0).square()
    tail = torch.exp(-(radius - ring_radius_px).clamp_min(0.0) / tail_length_px)
    tail = tail * radial_ramp / (radius + 0.5)
    tail = tail / tail.sum().clamp_min(1e-12)
    kernel = balance * ring + (1.0 - balance) * tail
    return kernel / kernel.sum().clamp_min(1e-12)


def _cached_kernel(ring_radius: float, tail_length: float, balance: float, device: torch.device) -> torch.Tensor:
    key = (round(ring_radius, 3), round(tail_length, 3), round(balance, 3), device.type, device.index)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        _KERNEL_CACHE.move_to_end(key)
        return kernel
    kernel = build_halation_kernel(ring_radius, tail_length, balance, device=device)
    _KERNEL_CACHE[key] = kernel
    while len(_KERNEL_CACHE) > _CACHE_LIMIT:
        _KERNEL_CACHE.popitem(last=False)
    return kernel


def _fft_convolve_same(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Linear 2D convolution of a single HW field with an odd-sized kernel."""
    height, width = field.shape
    kh, kw = kernel.shape
    fft_h = _next_fast_length(height + kh - 1)
    fft_w = _next_fast_length(width + kw - 1)
    field_fft = torch.fft.rfft2(field, s=(fft_h, fft_w))
    kernel_fft = torch.fft.rfft2(kernel, s=(fft_h, fft_w))
    full = torch.fft.irfft2(field_fft * kernel_fft, s=(fft_h, fft_w))
    y0, x0 = (kh - 1) // 2, (kw - 1) // 2
    return full[y0:y0 + height, x0:x0 + width]


def _halation_channel_weights(color: str, device: torch.device) -> torch.Tensor:
    """Convert a selected sRGB color to normalized linear RGB channel weights.

    The factory default is special-cased to retain the original implementation's
    exact (1.00, 0.32, 0.10) response rather than a rounded 8-bit approximation.
    """
    value = normalize_halation_color(color)
    if value == DEFAULT_HALATION_COLOR:
        return torch.tensor(_DEFAULT_CHANNEL_WEIGHTS, dtype=torch.float32, device=device)
    hex_value = value[1:]
    encoded = torch.tensor(
        [int(hex_value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=torch.float32,
        device=device,
    )
    linear = srgb_to_linear(encoded)
    peak = linear.max()
    return torch.where(peak > 1e-8, linear / peak.clamp_min(1e-8), torch.zeros_like(linear))


def apply_halation(
    rgb_srgb: torch.Tensor,
    *,
    amount: float,
    ring_radius: float,
    color: str = DEFAULT_HALATION_COLOR,
) -> torch.Tensor:
    """Apply red-dominant annular halation to one HWC sRGB image.

    The public Amount range is 0–2. The fixed 0.25 exposure coefficient
    therefore permits up to 0.50 additive linear-light exposure. The default
    color retains the original (1.00, 0.32, 0.10) red-dominant channel response.
    """
    if amount <= 0.0:
        return rgb_srgb

    linear = srgb_to_linear(rgb_srgb)
    height, width, _ = linear.shape
    long_edge = max(height, width)
    # Broad halation can be evaluated at a reduced working resolution. Max pooling
    # preserves small bright sources rather than averaging them out before exposure.
    factor = max(1, int(math.ceil(long_edge / 1536.0)))
    if factor > 1:
        work_h = max(1, int(math.ceil(height / factor)))
        work_w = max(1, int(math.ceil(width / factor)))
        source = F.max_pool2d(
            linear.permute(2, 0, 1).unsqueeze(0),
            kernel_size=factor,
            stride=factor,
            ceil_mode=True,
        )
        source = F.interpolate(source, size=(work_h, work_w), mode="nearest")
        work = source[0].permute(1, 2, 0)
    else:
        work_h, work_w = height, width
        work = linear

    work_scale = max(work_h, work_w) / 1024.0
    ring_px = float(ring_radius) * work_scale
    tail_px = 40.0 * work_scale
    kernel = _cached_kernel(ring_px, tail_px, 0.65, linear.device)

    # All light contributes in the physical model; the kernel itself is unit-energy.
    energy = work.clamp(0.0, 1.0)
    channel_weights = _halation_channel_weights(color, linear.device)
    halo_channels = []
    for channel in range(3):
        halo = _fft_convolve_same(energy[..., channel], kernel)
        halo_channels.append(halo * channel_weights[channel])
    halo_rgb = torch.stack(halo_channels, dim=-1).clamp_min(0.0)

    if factor > 1:
        halo_rgb = F.interpolate(
            halo_rgb.permute(2, 0, 1).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

    exposure_fraction = 0.25 * float(amount)
    return linear_to_srgb((linear + exposure_fraction * halo_rgb).clamp(0.0, 1.0))
