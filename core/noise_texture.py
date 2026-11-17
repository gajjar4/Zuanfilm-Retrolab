"""Seeded photographic surface textures for Zuanfilm Retrolab.

These bounded 2D fields are image-space finishing textures: layered emulsion
mottle, silver-halide clustering, film-base scratches, dust/fibres and foxing
are distinct procedural constructions rather than relabelled generic fBm.
"""

from __future__ import annotations

import math

import torch

from .blending import BLEND_MODES as NOISE_BLEND_MODES, blend_signed_layer, normalize_blend_mode


def normalize_noise_blend_mode(value: object) -> str:
    return normalize_blend_mode(value)


NOISE_TEXTURES = (
    "Gaussian",
    "Value fBm",
    "Gradient fBm",
    "Simplex fBm",
    "Ridged fBm",
    "Worley Microcells",
    "Domain Warp",
    "Silver Halide Clumps",
    "Emulsion Mottle",
    "Film Scratches",
    "Dust & Fibers",
    "Paper Foxing",
)

_MASK32 = 0xFFFFFFFF


def normalize_noise_texture(value: object) -> str:
    candidate = str(value or "Gaussian")
    # A workflow saved with the retired directional carrier maps to the
    # closest irregular, non-directional emulsion surface rather than failing.
    if candidate == "Gabor Texture":
        return "Emulsion Mottle"
    return candidate if candidate in NOISE_TEXTURES else "Gaussian"


def _hash_u32(ix: torch.Tensor, iy: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    x = (ix * 73856093) ^ (iy * 19349663) ^ ((int(seed) & _MASK32) * 83492791) ^ (int(stream) * 2654435761)
    x = x & _MASK32
    x = (x ^ 61) ^ (x >> 16)
    x = (x + (x << 3)) & _MASK32
    x = x ^ (x >> 4)
    x = (x * 0x27D4EB2D) & _MASK32
    return (x ^ (x >> 15)) & _MASK32


def _hash01(ix: torch.Tensor, iy: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    return (_hash_u32(ix, iy, seed, stream).to(torch.float32) + 0.5) / 4294967296.0


def _fade(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _value_noise(x: torch.Tensor, y: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    ix = torch.floor(x).to(torch.int64)
    iy = torch.floor(y).to(torch.int64)
    tx = _fade(x - ix.to(x.dtype))
    ty = _fade(y - iy.to(y.dtype))
    v00 = _hash01(ix, iy, seed, stream) * 2.0 - 1.0
    v10 = _hash01(ix + 1, iy, seed, stream) * 2.0 - 1.0
    v01 = _hash01(ix, iy + 1, seed, stream) * 2.0 - 1.0
    v11 = _hash01(ix + 1, iy + 1, seed, stream) * 2.0 - 1.0
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return top + (bottom - top) * ty


def _gaussian_lattice(ix: torch.Tensor, iy: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    u1 = _hash01(ix, iy, seed, stream).clamp_min(1e-7)
    u2 = _hash01(ix, iy, seed, stream + 1)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


def _gaussian_noise(x: torch.Tensor, y: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    ix = torch.floor(x).to(torch.int64)
    iy = torch.floor(y).to(torch.int64)
    tx = _fade(x - ix.to(x.dtype))
    ty = _fade(y - iy.to(y.dtype))
    v00 = _gaussian_lattice(ix, iy, seed, stream)
    v10 = _gaussian_lattice(ix + 1, iy, seed, stream)
    v01 = _gaussian_lattice(ix, iy + 1, seed, stream)
    v11 = _gaussian_lattice(ix + 1, iy + 1, seed, stream)
    top = v00 + (v10 - v00) * tx
    bottom = v01 + (v11 - v01) * tx
    return top + (bottom - top) * ty


def _gradient_dot(ix: torch.Tensor, iy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    angle = _hash01(ix, iy, seed, stream) * (2.0 * math.pi)
    return torch.cos(angle) * dx + torch.sin(angle) * dy


def _gradient_noise(x: torch.Tensor, y: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    ix = torch.floor(x).to(torch.int64)
    iy = torch.floor(y).to(torch.int64)
    fx = x - ix.to(x.dtype)
    fy = y - iy.to(y.dtype)
    n00 = _gradient_dot(ix, iy, fx, fy, seed, stream)
    n10 = _gradient_dot(ix + 1, iy, fx - 1.0, fy, seed, stream)
    n01 = _gradient_dot(ix, iy + 1, fx, fy - 1.0, seed, stream)
    n11 = _gradient_dot(ix + 1, iy + 1, fx - 1.0, fy - 1.0, seed, stream)
    u, v = _fade(fx), _fade(fy)
    top = n00 + (n10 - n00) * u
    bottom = n01 + (n11 - n01) * u
    return (top + (bottom - top) * v) * 1.65


def _simplex_noise(x: torch.Tensor, y: torch.Tensor, seed: int, stream: int = 0) -> torch.Tensor:
    f2 = 0.3660254037844386
    g2 = 0.21132486540518713
    skew = (x + y) * f2
    i = torch.floor(x + skew).to(torch.int64)
    j = torch.floor(y + skew).to(torch.int64)
    unskew = (i + j).to(x.dtype) * g2
    x0 = x - (i.to(x.dtype) - unskew)
    y0 = y - (j.to(y.dtype) - unskew)
    i1 = (x0 > y0).to(torch.int64)
    j1 = 1 - i1
    x1 = x0 - i1.to(x.dtype) + g2
    y1 = y0 - j1.to(y.dtype) + g2
    x2 = x0 - 1.0 + 2.0 * g2
    y2 = y0 - 1.0 + 2.0 * g2

    def corner(cx, cy, ix, iy, corner_stream):
        attenuation = (0.5 - cx.square() - cy.square()).clamp_min(0.0)
        dot = _gradient_dot(ix, iy, cx, cy, seed, stream + corner_stream)
        return attenuation.pow(4) * dot

    n0 = corner(x0, y0, i, j, 0)
    n1 = corner(x1, y1, i + i1, j + j1, 1)
    n2 = corner(x2, y2, i + 1, j + 1, 2)
    return 70.0 * (n0 + n1 + n2)


def _fbm(x, y, seed: int, kernel, *, octaves: int = 5, stream: int = 0) -> torch.Tensor:
    total = torch.zeros_like(x, dtype=torch.float32)
    normalization = 0.0
    amplitude = 1.0
    frequency = 1.0
    for octave in range(octaves):
        sample = kernel(x * frequency, y * frequency, seed, stream + octave * 17)
        total = total + sample * amplitude
        normalization += amplitude
        frequency *= 2.0
        amplitude *= 0.5
    return total / max(normalization, 1e-8)


def _worley(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    ix = torch.floor(x).to(torch.int64)
    iy = torch.floor(y).to(torch.int64)
    nearest = torch.full_like(x, 2.0)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = ix + ox, iy + oy
            fx = cx.to(x.dtype) + _hash01(cx, cy, seed, 71)
            fy = cy.to(y.dtype) + _hash01(cx, cy, seed, 72)
            distance = torch.sqrt((x - fx).square() + (y - fy).square() + 1e-12)
            nearest = torch.minimum(nearest, distance)
    return nearest


def _coordinate_grid(height: int, width: int, device: torch.device, reference_dimensions, coordinate_origin, size: float):
    ref_h, ref_w = reference_dimensions or (height, width)
    long_edge = float(max(ref_h, ref_w))
    origin_x, origin_y = coordinate_origin
    yy = (torch.arange(height, device=device, dtype=torch.float32) + int(origin_y)).view(height, 1)
    xx = (torch.arange(width, device=device, dtype=torch.float32) + int(origin_x)).view(1, width)
    frequency = 32.0 / long_edge / size
    return xx.expand(height, width) * frequency, yy.expand(height, width) * frequency


def _film_scratches(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    across = x * 4.5
    column = torch.floor(across).to(torch.int64)
    within = across - column.to(across.dtype)
    center = 0.16 + 0.68 * _hash01(column, torch.zeros_like(column), seed, 701)
    drift = _value_noise(torch.zeros_like(x) + 2.3, y * 0.42, seed, 702) * 0.055
    offset = within - center - drift
    selected = (_hash01(column, torch.zeros_like(column), seed, 703) > 0.925).to(x.dtype)
    line = torch.exp(-((offset / 0.055).square())) * selected
    continuity = 0.24 + 0.76 * (0.5 + 0.5 * _value_noise(x * 0.035 + 4.1, y * 0.7, seed, 704))
    fine = _gaussian_noise(x * 9.0, y * 1.4, seed, 705) * 0.07
    return -line * continuity + fine


def _dust_and_fibers(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    gx, gy = x * 1.7, y * 1.7
    ix = torch.floor(gx).to(torch.int64)
    iy = torch.floor(gy).to(torch.int64)
    dots = torch.zeros_like(x)
    fibers = torch.zeros_like(x)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = ix + ox, iy + oy
            jx = cx.to(x.dtype) + 0.12 + 0.76 * _hash01(cx, cy, seed, 721)
            jy = cy.to(y.dtype) + 0.12 + 0.76 * _hash01(cx, cy, seed, 722)
            dx, dy = gx - jx, gy - jy
            active = (_hash01(cx, cy, seed, 723) > 0.925).to(x.dtype)
            radius = 0.035 + 0.055 * _hash01(cx, cy, seed, 724)
            dot = torch.exp(-(dx.square() + dy.square()) / (radius.square() + 1e-8)) * active
            dots = torch.maximum(dots, dot)

            fiber_active = (_hash01(cx, cy, seed, 725) > 0.982).to(x.dtype)
            angle = (_hash01(cx, cy, seed, 726) - 0.5) * 0.85
            ca, sa = torch.cos(angle), torch.sin(angle)
            along = dx * ca + dy * sa
            across = -dx * sa + dy * ca
            shape = (along / 0.34).square() + (across / 0.018).square()
            fibers = torch.maximum(fibers, torch.exp(-shape * 1.7) * fiber_active)
    base = _gaussian_noise(gx * 2.2, gy * 2.2, seed, 727) * 0.08
    return dots * 1.5 + fibers * 1.2 + base


def _paper_foxing_field(x: torch.Tensor, y: torch.Tensor, seed: int):
    """Return an irregular stain field and its warm discolouration mask."""
    cloud = _fbm(x * 0.48, y * 0.48, seed, _value_noise, octaves=5, stream=751)
    edge_irregularity = _fbm(x * 1.7 + 8.0, y * 1.7 - 5.0, seed, _gradient_noise, octaves=3, stream=761)
    stains = torch.sigmoid((cloud + edge_irregularity * 0.16 - 0.08) * 10.0)
    pores = _gaussian_noise(x * 11.0, y * 11.0, seed, 771) * 0.08
    field = -stains + pores + cloud * 0.18
    return field, stains.clamp(0.0, 1.0)


def _texture_field(
    mode: str, x: torch.Tensor, y: torch.Tensor, seed: int,
) -> torch.Tensor:
    if mode == "Gaussian":
        # 32 base cells × 32 gives approximately one texture cell per source pixel.
        return _gaussian_noise(x * 32.0, y * 32.0, seed, 88)
    if mode == "Value fBm":
        return _fbm(x, y, seed, _value_noise)
    if mode == "Gradient fBm":
        return _fbm(x, y, seed, _gradient_noise)
    if mode == "Simplex fBm":
        return _fbm(x, y, seed, _simplex_noise)
    if mode == "Ridged fBm":
        base = _fbm(x, y, seed, _gradient_noise)
        return (1.0 - torch.abs(base)).mul(2.0).sub(1.0)
    if mode == "Worley Microcells":
        return 1.0 - 2.0 * _worley(x * 1.85, y * 1.85, seed)
    if mode == "Domain Warp":
        warp_x = _fbm(x * 0.65 + 13.1, y * 0.65 - 4.7, seed, _gradient_noise, octaves=3, stream=101)
        warp_y = _fbm(x * 0.65 - 7.9, y * 0.65 + 19.3, seed, _gradient_noise, octaves=3, stream=211)
        return _fbm(x + warp_x * 0.30, y + warp_y * 0.30, seed, _simplex_noise, stream=307)
    if mode == "Silver Halide Clumps":
        clump = _fbm(x * 1.35, y * 1.35, seed, _simplex_noise, octaves=4, stream=401)
        carrier = _gaussian_noise(x * 9.0, y * 9.0, seed, 407)
        density = 0.28 + 1.05 * torch.sigmoid(clump * 3.2)
        return carrier * density + clump * 0.16
    if mode == "Emulsion Mottle":
        broad = _fbm(x * 0.52, y * 0.52, seed, _simplex_noise, octaves=4, stream=421)
        middle = _fbm(x * 2.1 + 5.7, y * 2.1 - 3.4, seed, _value_noise, octaves=4, stream=431)
        fine = _gaussian_noise(x * 5.5, y * 5.5, seed, 441)
        return broad * 0.78 + middle * 0.43 + fine * 0.12
    if mode == "Film Scratches":
        return _film_scratches(x, y, seed)
    if mode == "Dust & Fibers":
        return _dust_and_fibers(x, y, seed)
    if mode == "Paper Foxing":
        field, _stains = _paper_foxing_field(x, y, seed)
        return field
    return _fbm(x, y, seed, _value_noise)


def apply_noise_texture(
    rgb_srgb: torch.Tensor,
    *,
    texture: str = "Gaussian",
    amount: float = 0.0,
    seed: int = 73,
    scale: float = 1.0,
    blend_mode: str = "Additive",
    reference_dimensions: tuple[int, int] | None = None,
    coordinate_origin: tuple[int, int] = (0, 0),
    frame_index: int = 0,
) -> torch.Tensor:
    """Add a deterministic, monochrome photographic surface texture.

    ``scale`` is a feature-size multiplier: higher values create larger,
    broader marks; lower values create finer and more frequent detail. All
    kernels are anchored to absolute source pixels so native crops align.
    """
    try:
        strength = float(amount)
    except (TypeError, ValueError, OverflowError):
        strength = 0.0
    if not math.isfinite(strength):
        strength = 0.0
    # The expanded 0–2 control keeps the old 0–1 response intact while
    # allowing up to twice the previous procedural texture contribution.
    strength = min(max(strength, 0.0), 2.0)
    try:
        feature_size = float(scale)
    except (TypeError, ValueError, OverflowError):
        feature_size = 1.0
    if not math.isfinite(feature_size):
        feature_size = 1.0
    feature_size = min(max(feature_size, 0.0), 4.0)
    selected = normalize_noise_texture(texture)
    selected_blend = normalize_noise_blend_mode(blend_mode)
    if strength <= 0.0:
        return rgb_srgb
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC image with at least three channels")

    h, w = int(rgb_srgb.shape[0]), int(rgb_srgb.shape[1])
    # Scale 0 is a valid finest-detail setting; a small floor avoids division
    # by zero while preserving a useful near-pixel texture frequency.
    effective_size = max(feature_size, 0.05)
    x, y = _coordinate_grid(h, w, rgb_srgb.device, reference_dimensions, coordinate_origin, effective_size)
    frame_seed = (int(seed) + int(frame_index) * 7919) & 0x7FFFFFFF
    foxing_mask = None
    if selected == "Paper Foxing":
        field, foxing_mask = _paper_foxing_field(x, y, frame_seed)
    else:
        field = _texture_field(selected, x, y, frame_seed)
    field = field - field.mean()
    field = field / field.square().mean().sqrt().clamp_min(1e-5)

    from .color import luminance_rec709
    from .color import srgb_to_linear

    source = rgb_srgb[..., :3].to(torch.float32).clamp(0.0, 1.0)
    luminance = luminance_rec709(srgb_to_linear(source))
    tonal_weight = 0.42 + 0.58 * (4.0 * luminance * (1.0 - luminance)).clamp(0.0, 1.0)
    # A small signed carrier keeps every blend mode restrained around its
    # neutral point. Additive is the legacy 0.2.20 response and remains default.
    signal = field * (0.018 * strength) * tonal_weight
    signed_texture = signal[..., None]
    if foxing_mask is not None:
        # Fold the chromatic paper stain into the same compositing layer so
        # every texture blend mode controls the complete Paper Foxing effect.
        stain = foxing_mask * (0.012 * strength) * tonal_weight
        signed_texture = signed_texture + torch.stack((stain, stain * 0.22, -stain * 0.72), dim=-1)
    result = blend_signed_layer(source, signed_texture, selected_blend)
    return result
