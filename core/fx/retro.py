"""Retro effects: chromatic aberration, VHS, CRT, film frames and gate weave."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..color import linear_to_srgb, srgb_to_linear
from ..filters import gaussian_blur_scaled
from .common import FxContext, hash_noise, hash_scalar, hex_to_rgb, pixel_grid, ref_px, sample, smoothstep, to_hwc, to_nchw
from .pixelfont import text_mask


def chromatic_aberration(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    amount = ref_px(p["amount"], rgb)
    if amount <= 0.0:
        return rgb
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    cx, cy = p["cx"] * width, p["cy"] * height
    half = max(height, width) * 0.5
    nchw = to_nchw(rgb)
    # Red spreads outward and blue inward, proportional to distance from the handle.
    red = sample(nchw[:, 0:1], cx + (x - cx) * (1.0 + amount / half), cy + (y - cy) * (1.0 + amount / half))
    blue = sample(nchw[:, 2:3], cx + (x - cx) * (1.0 - amount / half), cy + (y - cy) * (1.0 - amount / half))
    return torch.cat((to_hwc(red), rgb[..., 1:2], to_hwc(blue)), dim=-1)


def _ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(-1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.stack((y, (b - y) * 0.564, (r - y) * 0.713), dim=-1)


def _rgb(ycc: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycc.unbind(-1)
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.stack((r, g, b), dim=-1)


def vhs(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    height, width = rgb.shape[:2]
    seed = int(p["seed"]) + ctx.frame_index * 7919
    ycc = _ycbcr(rgb)
    bleed = ref_px(p["bleed"], rgb)
    if bleed > 0.5:
        # Tape chroma is low bandwidth: smear Cb/Cr horizontally and lag it to the right.
        taps = max(1, int(round(bleed)))
        chroma = to_nchw(ycc[..., 1:])
        kernel = torch.full((2, 1, 1, 2 * taps + 1), 1.0 / (2 * taps + 1), device=rgb.device, dtype=rgb.dtype)
        chroma = F.conv2d(F.pad(chroma, (taps, taps, 0, 0), mode="replicate"), kernel, groups=2)
        chroma = torch.roll(chroma, shifts=taps // 2 + 1, dims=-1)
        ycc = torch.cat((ycc[..., :1], to_hwc(chroma)), dim=-1)
    out = _rgb(ycc)
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    if p["tracking"] > 0.0:
        band_center = hash_scalar(seed, 1) * height
        band = max(4.0, height * 0.05 * (0.5 + p["tracking"]))
        inside = 1.0 - smoothstep(0.0, band, (y - band_center).abs())
        jitter = (hash_noise(height, 1, seed, rgb.device, rgb.dtype)[..., 0] - 0.5) * ref_px(18.0, rgb) * p["tracking"]
        shifted = to_hwc(sample(to_nchw(out), x - jitter * inside, y))
        out = shifted + inside.unsqueeze(-1) * 0.12 * p["tracking"]
    if p["scanlines"] > 0.0:
        period = max(2.0, ref_px(3.0, rgb))
        lines = 0.5 + 0.5 * torch.cos(y / period * 2.0 * math.pi)
        out = out * (1.0 - p["scanlines"] * 0.45 * lines).unsqueeze(-1)
    if p["noise"] > 0.0:
        grain = hash_noise(height, width, seed, rgb.device, rgb.dtype)
        out = out + (grain - 0.5) * 0.35 * p["noise"]
    return out.clamp(0.0, 1.0)


def crt(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    u, v = x / width * 2.0 - 1.0, y / height * 2.0 - 1.0
    curve = p["curvature"] * 0.25
    r2 = u.square() + v.square()
    su, sv = u * (1.0 + curve * r2), v * (1.0 + curve * r2)
    warped = to_hwc(sample(to_nchw(rgb), (su + 1.0) * 0.5 * width, (sv + 1.0) * 0.5 * height))
    outside = ((su.abs() > 1.0) | (sv.abs() > 1.0)).unsqueeze(-1)
    out = srgb_to_linear(warped)
    if p["glow"] > 0.0:
        glow = to_hwc(gaussian_blur_scaled(to_nchw(out), ref_px(6.0, rgb)))
        out = out + glow * p["glow"] * 0.6
    period = max(2.0, ref_px(3.2, rgb))
    lines = 0.5 + 0.5 * torch.cos(y / period * 2.0 * math.pi)
    out = out * (1.0 - p["scanlines"] * 0.55 * lines).unsqueeze(-1)
    if p["mask"] > 0.0:
        stripe = torch.floor(x / max(1.0, ref_px(1.2, rgb))) % 3
        phosphor = torch.stack([(stripe == i).to(rgb.dtype) for i in range(3)], dim=-1)
        out = out * (1.0 - p["mask"] * 0.5 + phosphor * p["mask"] * 0.5 * 1.5)
    vignette = 1.0 - smoothstep(0.55, 1.45, (su.square() + sv.square()).sqrt()).unsqueeze(-1) * 0.6
    out = linear_to_srgb((out * vignette).clamp_min(0.0))
    return torch.where(outside, torch.zeros_like(out), out)


def _cover(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize-and-crop ``rgb`` to fill a (height, width) window."""
    src_h, src_w = rgb.shape[:2]
    scale = max(height / src_h, width / src_w)
    rh, rw = max(height, round(src_h * scale)), max(width, round(src_w * scale))
    resized = to_hwc(F.interpolate(to_nchw(rgb), size=(rh, rw), mode="bilinear", align_corners=False, antialias=True))
    top, left = (rh - height) // 2, (rw - width) // 2
    return resized[top:top + height, left:left + width]


def film_frame(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    height, width = rgb.shape[:2]
    style = p["style"]
    device, dtype = rgb.device, rgb.dtype
    if style == "35mm":
        side, rail = round(width * 0.075), round(height * 0.025)
        canvas = hex_to_rgb("#0B0907", device, dtype).expand(height, width, 3).clone()
        canvas[rail:height - rail, side:width - side] = _cover(rgb, height - 2 * rail, width - 2 * side)
        hole_w, hole_h = max(2, round(side * 0.42)), max(2, round(side * 0.3))
        step = max(hole_h * 2, round(hole_h * 2.4))
        hole = hex_to_rgb("#E9E2D2", device, dtype)
        for top in range(step // 2, height - hole_h, step):
            for left in ((side - hole_w) // 2, width - side + (side - hole_w) // 2):
                canvas[top:top + hole_h, left:left + hole_w] = hole
        mask = torch.rot90(text_mask(p["text"], round(side * 0.3), device, dtype), k=-1)
        th, tw = min(mask.shape[0], height), min(mask.shape[1], side)
        top, left = (height - th) // 2, width - tw - max(1, round(side * 0.05))
        ink = hex_to_rgb("#FF9C3A", device, dtype)
        region = canvas[top:top + th, left:left + tw]
        canvas[top:top + th, left:left + tw] = torch.lerp(region, ink.expand_as(region), mask[:th, :tw].unsqueeze(-1))
        return canvas
    # Instant prints: cream border, deeper bottom margin, optional caption.
    side = round(width * (0.055 if style == "Polaroid" else 0.045))
    top_margin = side if style == "Polaroid" else round(height * 0.06)
    bottom = round(height * (0.2 if style == "Polaroid" else 0.15))
    canvas = hex_to_rgb("#F3EFE6", device, dtype).expand(height, width, 3).clone()
    canvas[top_margin:height - bottom, side:width - side] = _cover(rgb, height - top_margin - bottom, width - 2 * side)
    if p["text"].strip():
        mask = text_mask(p["text"], round(bottom * 0.22), device, dtype)
        th, tw = min(mask.shape[0], bottom), min(mask.shape[1], width - 2 * side)
        top, left = height - bottom + (bottom - th) // 2, side + round(width * 0.02)
        region = canvas[top:top + th, left:left + tw]
        ink = hex_to_rgb("#3B342E", device, dtype)
        canvas[top:top + th, left:left + tw] = torch.lerp(region, ink.expand_as(region), mask[:th, :tw].unsqueeze(-1))
    return canvas


def gate_weave(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    amount = ref_px(p["amount"], rgb)
    seed = int(p["seed"])
    frame = ctx.frame_index
    # Smoothly interpolated per-frame offsets so projected footage drifts, not jitters.
    def wander(axis: int) -> float:
        a, b = hash_scalar(seed + axis, frame // 3), hash_scalar(seed + axis, frame // 3 + 1)
        t = (frame % 3) / 3.0
        return (a + (b - a) * t - 0.5) * 2.0
    dx, dy = wander(0) * amount * 0.6, wander(1) * amount
    height, width = rgb.shape[:2]
    x, y = pixel_grid(height, width, rgb.device, rgb.dtype)
    out = to_hwc(sample(to_nchw(rgb), x - dx, y - dy))
    flicker = 1.0 + (hash_scalar(seed + 7, frame) - 0.5) * 0.12 * p["flicker"]
    return (out * flicker).clamp(0.0, 1.0)
