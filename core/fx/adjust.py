"""Adjustments: curves, HSL per colour, colour wheels, clarity and sharpen."""

from __future__ import annotations

import math

import torch

from ..filters import gaussian_blur_scaled
from .common import FxContext, hsv_to_rgb, luminance, ref_px, rgb_to_hsv, smoothstep, to_hwc, to_nchw

HSL_BANDS = (("red", 0.0), ("orange", 30.0), ("yellow", 60.0), ("green", 120.0),
             ("aqua", 180.0), ("blue", 225.0), ("purple", 270.0), ("magenta", 315.0))


def curves(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    """Lightroom-style parametric tone curve plus per-channel midtone balance."""
    t = rgb.clamp(0.0, 1.0)
    bump = lambda centre, width: torch.exp(-((t - centre) / width).square())
    shaped = (t
              + p["shadows"] * 0.18 * bump(0.15, 0.16)
              + p["darks"] * 0.15 * bump(0.38, 0.18)
              + p["lights"] * 0.15 * bump(0.62, 0.18)
              + p["highlights"] * 0.18 * bump(0.86, 0.16))
    contrast = p["contrast"]
    if contrast != 0.0:
        s = shaped.clamp(0.0, 1.0)
        shaped = s + contrast * (s - 0.5) * 4.0 * s * (1.0 - s) * 0.5
    balance = torch.tensor([p["red"], p["green"], p["blue"]], device=rgb.device, dtype=rgb.dtype)
    shaped = shaped + balance * 0.12 * 4.0 * t * (1.0 - t)
    black, white = p["black"], 1.0 - p["white"]
    shaped = black + shaped.clamp(0.0, 1.0) * (white - black)
    return shaped.clamp(0.0, 1.0)


def hsl(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    hsv = rgb_to_hsv(rgb.clamp(0.0, 1.0))
    hue, sat, val = hsv.unbind(-1)
    degrees = hue * 360.0
    hue_shift = torch.zeros_like(hue)
    sat_scale = torch.zeros_like(hue)
    lum_shift = torch.zeros_like(hue)
    for name, centre in HSL_BANDS:
        dist = (degrees - centre + 180.0) % 360.0 - 180.0
        weight = torch.exp(-(dist / 26.0).square()) * smoothstep(0.02, 0.2, sat)
        hue_shift += weight * p[f"{name}_hue"] * 30.0
        sat_scale += weight * p[f"{name}_sat"]
        lum_shift += weight * p[f"{name}_lum"] * 0.25
    out = hsv_to_rgb(torch.stack((((degrees + hue_shift) / 360.0) % 1.0, (sat * (1.0 + sat_scale)).clamp(0.0, 1.0), val), dim=-1))
    return (out + lum_shift.unsqueeze(-1) * 4.0 * val.unsqueeze(-1) * (1.0 - val.unsqueeze(-1))).clamp(0.0, 1.0)


def _wheel(x: float, y: float, device, dtype) -> torch.Tensor:
    """Colour offset for a wheel puck at (x, y) in the unit disc."""
    strength = min(1.0, math.hypot(x, y))
    if strength < 1e-4:
        return torch.zeros(3, device=device, dtype=dtype)
    angle = math.atan2(-y, x)
    hue = torch.tensor([(angle / (2.0 * math.pi)) % 1.0, 1.0, 1.0], device=device, dtype=dtype)
    colour = hsv_to_rgb(hue)
    return (colour - colour.mean()) * strength


def color_wheels(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    device, dtype = rgb.device, rgb.dtype
    lift = _wheel(p["lift_x"], p["lift_y"], device, dtype) * 0.25 + p["lift"] * 0.2
    gamma = _wheel(p["gamma_x"], p["gamma_y"], device, dtype) * 0.6 + p["gamma"] * 0.6
    gain = _wheel(p["gain_x"], p["gain_y"], device, dtype) * 0.5 + p["gain"] * 0.5
    t = rgb.clamp(0.0, 1.0)
    out = (t * (1.0 + gain) + lift * (1.0 - t)).clamp(0.0, 1.0)
    return out.pow(torch.exp(-gamma)).clamp(0.0, 1.0)


def clarity(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    amount = p["amount"]
    if amount == 0.0:
        return rgb
    lum = luminance(rgb).unsqueeze(-1)
    base = to_hwc(gaussian_blur_scaled(to_nchw(lum), ref_px(p["radius"], rgb)))
    midtones = 4.0 * lum * (1.0 - lum)
    return (rgb + (lum - base) * amount * 1.5 * midtones).clamp(0.0, 1.0)


def sharpen(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    if p["amount"] <= 0.0:
        return rgb
    blurred = to_hwc(gaussian_blur_scaled(to_nchw(rgb), ref_px(p["radius"], rgb)))
    detail = rgb - blurred
    if p["threshold"] > 0.0:
        detail = detail * smoothstep(p["threshold"] * 0.5, p["threshold"], detail.abs().amax(-1, keepdim=True))
    return (rgb + detail * p["amount"]).clamp(0.0, 1.0)
