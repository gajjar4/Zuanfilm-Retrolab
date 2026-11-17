"""Deterministic, dependency-free procedural light-leak geometries.

All full-resolution calculations use PyTorch tensors. Small, seeded control
vectors are generated with Python's standard library / torch CPU generator and
then evaluated on the input image's device.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, srgb_to_linear

LEAK_MODES = (
    "Soft Edge",
    "First-Frame Burn",
    "120 Fat Roll",
    "Light Piping",
    "Holga 135",
)

POSITIONS = (
    "Top", "Bottom", "Left", "Right",
    "Top Left", "Top Right", "Bottom Left", "Bottom Right",
)

_DEFAULT_COLOR = "#C2AF8E"
# Relative propagation lengths: shorter wavelengths are attenuated more
# quickly as light travels through the film base.
_CHANNEL_PATH_RATIOS = (1.0, 0.68, 0.43)


def normalize_hex_color(value: str | None) -> str:
    """Normalize a CSS-style RGB hex color; fall back safely on invalid input."""
    if not isinstance(value, str):
        return _DEFAULT_COLOR
    candidate = value.strip()
    if candidate.startswith("#"):
        candidate = candidate[1:]
    if len(candidate) == 3 and all(ch in "0123456789abcdefABCDEF" for ch in candidate):
        candidate = "".join(ch * 2 for ch in candidate)
    if len(candidate) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in candidate):
        return _DEFAULT_COLOR
    return "#" + candidate.upper()


def _hex_to_linear_rgb(value: str, device: torch.device) -> torch.Tensor:
    hex_value = normalize_hex_color(value)[1:]
    rgb = [int(hex_value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    encoded = torch.tensor(rgb, dtype=torch.float32, device=device)
    return srgb_to_linear(encoded)


def _low_frequency_warp(
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    *,
    amplitude_ref: float = 85.0,
    feature_ref: float = 380.0,
) -> torch.Tensor:
    """Create a seeded, broad two-dimensional boundary displacement field."""
    long_edge = max(height, width)
    scale = long_edge / 1024.0
    feature_px = max(24.0, feature_ref * scale)
    gh = max(3, int(math.ceil(height / feature_px)) + 2)
    gw = max(3, int(math.ceil(width / feature_px)) + 2)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    coarse_a = torch.rand((1, 1, gh, gw), generator=gen, dtype=torch.float32) - 0.5
    coarse_b = torch.rand(
        (1, 1, max(3, gh * 2 - 1), max(3, gw * 2 - 1)), generator=gen
    ) - 0.5
    coarse_a = coarse_a.to(device=device)
    coarse_b = coarse_b.to(device=device)
    a = F.interpolate(coarse_a, size=(height, width), mode="bicubic", align_corners=False)[0, 0]
    b = F.interpolate(coarse_b, size=(height, width), mode="bicubic", align_corners=False)[0, 0]
    field = 0.76 * a + 0.24 * b
    field = field - field.mean()
    field = field / field.abs().amax().clamp_min(1e-6)
    return field.clamp(-1.0, 1.0) * (float(amplitude_ref) * scale)


def _low_frequency_line(
    length: int,
    seed: int,
    device: torch.device,
    *,
    feature_ref: float = 190.0,
) -> torch.Tensor:
    """Seeded smooth one-dimensional variation along a film edge."""
    count = max(4, min(96, int(math.ceil(length / max(feature_ref, 24.0))) + 2))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    coarse = (torch.rand((1, 1, count), generator=gen) - 0.5).to(device=device)
    line = F.interpolate(coarse, size=length, mode="linear", align_corners=False)[0, 0]
    line = line - line.mean()
    line = line / line.abs().amax().clamp_min(1e-6)
    return line.clamp(-1.0, 1.0)


def _edge_distance_and_axis(
    height: int, width: int, position: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, str]:
    y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1).expand(height, width)
    x = torch.arange(width, device=device, dtype=torch.float32).view(1, width).expand(height, width)
    if position == "Top":
        return y, x, "top"
    if position == "Bottom":
        return (height - 1) - y, x, "bottom"
    if position == "Left":
        return x, y, "left"
    if position == "Right":
        return (width - 1) - x, y, "right"

    is_top = position.startswith("Top")
    is_left = position.endswith("Left")
    cy = 0.0 if is_top else float(height - 1)
    cx = 0.0 if is_left else float(width - 1)
    distance = torch.sqrt((y - cy).square() + (x - cx).square())
    edge = "top" if is_top else "bottom"
    return distance, x, edge


def _sprocket_modulation(
    distance: torch.Tensor,
    along_edge: torch.Tensor,
    long_edge: int,
    seed: int,
) -> torch.Tensor:
    """Diffusing KS-1870 135 perforation pattern with a seeded phase."""
    pitch = (4.75 / 36.0) * long_edge
    sigma0 = (0.50 / 36.0) * long_edge
    duty = 1.98 / 4.75
    phase = ((int(seed) & 0xFFFF) / 65536.0) * pitch
    path = distance + (2.1 / 36.0) * long_edge
    sigma = (sigma0 + 0.35 * torch.sqrt(path.clamp_min(0.0) * pitch)).clamp_min(1e-4)
    signed = torch.remainder(along_edge - phase, pitch) - 0.5 * pitch
    denom = sigma * math.sqrt(2.0)
    half = 0.5 * duty * pitch
    modulation = torch.zeros_like(distance)
    for offset in (-2, -1, 0, 1, 2):
        centre = offset * pitch
        left = torch.erf((signed - (centre - half)) / denom)
        right = torch.erf((signed - (centre + half)) / denom)
        modulation = modulation + 0.5 * (left - right)
    return (modulation / duty).clamp_min(0.0)


def _leak_profile(
    height: int,
    width: int,
    *,
    mode: str,
    position: str,
    seed: int,
    device: torch.device,
    influence_radius: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (spatial profile, optical path distance, per-mode exposure gain).

    ``influence_radius`` is a dimensionless multiplier for each mode's
    inward propagation distance. 1.0 preserves the original 0.1.3 look; 10.0
    expands the same seeded geometry toward the opposite image border.
    """
    long_edge = max(height, width)
    scale = long_edge / 1024.0
    radius = min(max(float(influence_radius), 0.25), 10.0)
    actual_position = position if position in POSITIONS else "Bottom"

    # Sprocket perforations are physically tied to the long edge of 135 film;
    # corners resolve to the corresponding top / bottom edge.
    if mode == "Holga 135" and actual_position in ("Top Left", "Top Right", "Bottom Left", "Bottom Right"):
        actual_position = "Top" if actual_position.startswith("Top") else "Bottom"

    distance, along, edge = _edge_distance_and_axis(height, width, actual_position, device)
    along_length = width if edge in ("top", "bottom") else height
    if mode == "Soft Edge":
        warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=85.0, feature_ref=380.0)
        path = (distance + warp).clamp_min(0.0)
        profile = torch.exp(-path / (200.0 * scale * radius))
        return profile, path, 0.30

    if mode == "First-Frame Burn":
        warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=155.0, feature_ref=260.0)
        path = (distance + warp).clamp_min(0.0)
        width_px = max(95.0 * scale * radius, 1.0)
        broad = torch.exp(-torch.pow(path / (1.75 * width_px), 1.28))
        hot_core = torch.exp(-path / max(0.34 * width_px, 1.0))
        lateral = 0.84 + 0.16 * torch.cos((along / max(along_length, 1)) * math.pi * 2.0 + (seed % 997) * 0.013)
        profile = ((0.70 * broad + 0.30 * hot_core) * lateral).clamp(0.0, 1.0)
        return profile, path, 0.55

    if mode == "120 Fat Roll":
        warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=125.0, feature_ref=190.0)
        line = _low_frequency_line(along_length, seed ^ 0x31F2, device, feature_ref=210.0)
        if edge in ("top", "bottom"):
            line_field = line.view(1, -1).expand(height, width)
        else:
            line_field = line.view(-1, 1).expand(height, width)
        path = (distance + warp + line_field * (30.0 * scale)).clamp_min(0.0)
        width_px = (72.0 + 28.0 * (line_field + 1.0)) * scale * radius
        main_band = torch.exp(-path / width_px.clamp_min(1.0))
        secondary_band = torch.exp(-((path - 2.15 * width_px) / (0.62 * width_px).clamp_min(1.0)).square())
        profile = (0.72 * main_band + 0.28 * secondary_band) * (0.78 + 0.22 * (line_field + 1.0) * 0.5)
        return profile.clamp(0.0, 1.0), path, 0.43

    if mode == "Light Piping":
        warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=52.0, feature_ref=310.0)
        line = _low_frequency_line(along_length, seed ^ 0x4C17, device, feature_ref=155.0)
        if edge in ("top", "bottom"):
            line_field = line.view(1, -1).expand(height, width)
        else:
            line_field = line.view(-1, 1).expand(height, width)
        path = (distance + warp).clamp_min(0.0)
        travel = torch.exp(-path / (0.88 * long_edge * radius))
        veil = 0.20 * travel * (0.62 + 0.38 * (line_field + 1.0) * 0.5)
        profile = veil
        rng = random.Random((int(seed) ^ 0x25A9) & 0x7FFFFFFF)
        # Long, irregular light channels; their locations are stable for a seed.
        for _ in range(5):
            centre = rng.uniform(0.04, 0.96) * along_length
            width_px = rng.uniform(0.006, 0.022) * along_length
            amplitude = rng.uniform(0.32, 0.78)
            streak = torch.exp(-0.5 * ((along - centre) / max(width_px, 1.0)).square())
            profile = profile + amplitude * streak * travel
        return profile.clamp(0.0, 1.6), path, 0.40

    if mode == "Holga 135":
        warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=85.0, feature_ref=380.0)
        path = (distance + warp).clamp_min(0.0)
        modulation = _sprocket_modulation(path, along, long_edge, seed)
        envelope = torch.exp(-path / (200.0 * scale * radius))
        return (modulation * envelope).clamp(0.0, 1.6), path, 0.30

    # Compatibility for old saved workflows: legacy Pinhole, Mechanical Marks,
    # and other removed/unknown values safely resolve to Soft Edge.
    warp = _low_frequency_warp(height, width, seed, device, amplitude_ref=85.0, feature_ref=380.0)
    path = (distance + warp).clamp_min(0.0)
    return torch.exp(-path / (200.0 * scale * radius)), path, 0.30


def apply_light_leak(
    rgb_srgb: torch.Tensor,
    *,
    mode: str,
    position: str,
    exposure: float,
    seed: int,
    color: str = _DEFAULT_COLOR,
    influence_radius: float = 1.0,
) -> torch.Tensor:
    """Apply a seeded leak to one HWC sRGB image using additive linear exposure.

    The selected color defines the incident light spectrum. Wavelength-dependent
    attenuation through the film base changes that spectrum along the path.
    All geometries are deterministic for a fixed image shape, seed and settings.
    """
    if exposure <= 0.0:
        return rgb_srgb
    mode = "Holga 135" if mode == "135 Sprocket" else mode
    mode = mode if mode in LEAK_MODES else "Soft Edge"
    height, width, channels = rgb_srgb.shape
    if channels < 3:
        raise ValueError("Light Leak expects at least three RGB channels")

    profile, path, gain = _leak_profile(
        height, width, mode=mode, position=position, seed=seed,
        device=rgb_srgb.device, influence_radius=influence_radius
    )
    incident = _hex_to_linear_rgb(color, rgb_srgb.device)
    long_edge = max(height, width)
    scale = long_edge / 1024.0
    # Spectral attenuation follows the same propagation-distance control as
    # the geometric envelope, so high radius values remain visible at the
    # opposite border instead of fading back to zero-colored exposure.
    radius = min(max(float(influence_radius), 0.25), 10.0)
    lambda_red = max(200.0 * scale * radius, 1.0)
    attenuation = torch.stack(
        [torch.exp(-path / max(lambda_red * ratio, 1e-4)) for ratio in _CHANNEL_PATH_RATIOS],
        dim=-1,
    )
    incident_rgb = incident.view(1, 1, 3)
    if mode == "First-Frame Burn":
        # The first frame can receive a hot, near-white core at the film edge;
        # the selected color remains visible through the broader burn and tail.
        hot_core = torch.exp(-path / max(80.0 * scale, 1.0)).unsqueeze(-1)
        hot_mix = 0.78 * hot_core
        incident_rgb = incident_rgb * (1.0 - hot_mix) + hot_mix
    field_rgb = profile.unsqueeze(-1) * attenuation * incident_rgb

    linear = srgb_to_linear(rgb_srgb[..., :3])
    # Mode-specific factors keep the photographic mode distinct while
    # the shared Exposure control remains easy to understand.
    out = linear + (gain * min(max(float(exposure), 0.0), 10.0)) * field_rgb
    return linear_to_srgb(out.clamp(0.0, 1.0))
