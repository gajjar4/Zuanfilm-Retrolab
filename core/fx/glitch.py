"""Glitch effects: pixel sort, block shift and JPEG crunch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import FxContext, hash_scalar, luminance, ref_px, to_hwc, to_nchw

# Standard JPEG (Annex K) luminance and chrominance quantisation tables.
_LUMA_Q = (
    (16, 11, 10, 16, 24, 40, 51, 61), (12, 12, 14, 19, 26, 58, 60, 55), (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62), (18, 22, 37, 56, 68, 109, 103, 77), (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101), (72, 92, 95, 98, 112, 100, 103, 99),
)
_CHROMA_Q = (
    (17, 18, 24, 47, 99, 99, 99, 99), (18, 21, 26, 66, 99, 99, 99, 99), (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99), (99, 99, 99, 99, 99, 99, 99, 99), (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99), (99, 99, 99, 99, 99, 99, 99, 99),
)


def pixel_sort(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    """Sort every run of pixels whose brightness lies in [low, high] along one axis.

    Fully vectorised: each pixel gets a key inside its own run's index span, so a
    single argsort per line reorders runs while out-of-range pixels stay put.
    """
    vertical = p["direction"] == "Vertical"
    image = rgb if vertical else rgb.transpose(0, 1)
    lum = luminance(image)
    length = image.shape[0]
    inside = (lum >= p["low"]) & (lum <= p["high"])
    pos = torch.arange(length, device=rgb.device).view(-1, 1).expand_as(lum)
    prev = torch.cat((torch.zeros_like(inside[:1]), inside[:-1]), dim=0)
    nxt = torch.cat((inside[1:], torch.zeros_like(inside[:1])), dim=0)
    start = torch.where(inside & ~prev, pos, torch.zeros_like(pos)).cummax(dim=0).values
    end_marker = torch.where(inside & ~nxt, pos, torch.full_like(pos, length))
    end = end_marker.flip(0).cummin(dim=0).values.flip(0)
    run = (end - start).to(rgb.dtype)
    order_value = (1.0 - lum) if p["reverse"] else lum
    key = torch.where(inside, start.to(rgb.dtype) + order_value * run, pos.to(rgb.dtype))
    index = torch.argsort(key, dim=0, stable=True)
    sorted_image = torch.gather(image, 0, index.unsqueeze(-1).expand_as(image))
    return sorted_image if vertical else sorted_image.transpose(0, 1)


def block_shift(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    height, width = rgb.shape[:2]
    seed = int(p["seed"]) + (ctx.frame_index * 104729 if p["animate"] else 0)
    band = max(2, int(round(ref_px(p["band"], rgb))))
    max_shift = ref_px(p["shift"], rgb)
    out = rgb.clone()
    for index, top in enumerate(range(0, height, band)):
        if hash_scalar(seed, index) > p["amount"]:
            continue
        offset = int(round((hash_scalar(seed + 1, index) - 0.5) * 2.0 * max_shift))
        rows = slice(top, min(height, top + band))
        out[rows] = torch.roll(rgb[rows], shifts=offset, dims=1)
        if p["split"] > 0.0:
            channel_offset = int(round(offset * p["split"]))
            out[rows, :, 0] = torch.roll(rgb[rows, :, 0], shifts=offset + channel_offset, dims=1)
    return out


def _dct_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(8, device=device, dtype=dtype).view(-1, 1)
    n = torch.arange(8, device=device, dtype=dtype).view(1, -1)
    matrix = torch.cos(math.pi * (2.0 * n + 1.0) * k / 16.0) * math.sqrt(2.0 / 8.0)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix


def _quant_table(table, quality: int, device, dtype) -> torch.Tensor:
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    q = torch.floor((torch.tensor(table, device=device, dtype=dtype) * scale + 50.0) / 100.0)
    return q.clamp(1.0, 255.0)


def _jpeg_plane(plane: torch.Tensor, q: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
    """Quantise one (H, W) plane in 8x8 DCT blocks, exactly as a baseline JPEG encoder does."""
    height, width = plane.shape
    ph, pw = math.ceil(height / 8) * 8, math.ceil(width / 8) * 8
    padded = F.pad(plane.view(1, 1, height, width), (0, pw - width, 0, ph - height), mode="replicate")[0, 0]
    blocks = padded.view(ph // 8, 8, pw // 8, 8).permute(0, 2, 1, 3) - 128.0
    coeffs = dct @ blocks @ dct.T
    restored = dct.T @ (torch.round(coeffs / q) * q) @ dct + 128.0
    return restored.permute(0, 2, 1, 3).reshape(ph, pw)[:height, :width]


def jpeg_crunch(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    """Blocky, smeared low-quality JPEG artefacts: YCbCr, 4:2:0 chroma and 8x8 DCT quantisation."""
    device, dtype = rgb.device, rgb.dtype
    dct = _dct_matrix(device, dtype)
    quality = int(p["quality"])
    luma_q, chroma_q = _quant_table(_LUMA_Q, quality, device, dtype), _quant_table(_CHROMA_Q, quality, device, dtype)
    out = rgb.clamp(0.0, 1.0) * 255.0
    height, width = out.shape[:2]
    for _ in range(int(p["passes"])):
        r, g, b = out.unbind(-1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
        chroma = F.avg_pool2d(torch.stack((cb, cr)).unsqueeze(0), 2, ceil_mode=True)[0]
        y = _jpeg_plane(y, luma_q, dct)
        chroma = torch.stack([_jpeg_plane(plane, chroma_q, dct) for plane in chroma])
        chroma = F.interpolate(chroma.unsqueeze(0), scale_factor=2, mode="nearest")[0, :, :height, :width]
        cb, cr = chroma[0] - 128.0, chroma[1] - 128.0
        out = torch.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), dim=-1).clamp(0.0, 255.0)
    return out / 255.0
