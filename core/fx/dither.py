"""Dither Lab: error diffusion, ordered and noise dithering onto retro palettes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import FxContext, hash_noise, hex_to_rgb, luminance, ref_px, to_hwc, to_nchw

PALETTES = {
    "1-bit": ("#000000", "#FFFFFF"),
    "Game Boy": ("#0F380F", "#306230", "#8BAC0F", "#9BBC0F"),
    "Game Boy Pocket": ("#1E1E1E", "#4E4E4E", "#8C8C8C", "#C8C8C8"),
    "CGA": ("#000000", "#55FFFF", "#FF55FF", "#FFFFFF"),
    "CGA Warm": ("#000000", "#55FF55", "#FF5555", "#FFFF55"),
    "EGA": ("#000000", "#0000AA", "#00AA00", "#00AAAA", "#AA0000", "#AA00AA", "#AA5500", "#AAAAAA",
            "#555555", "#5555FF", "#55FF55", "#55FFFF", "#FF5555", "#FF55FF", "#FFFF55", "#FFFFFF"),
    "PICO-8": ("#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
               "#FF004D", "#FFA300", "#FFEC27", "#00E436", "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"),
    "C64": ("#000000", "#FFFFFF", "#68372B", "#70A4B2", "#6F3D86", "#588D43", "#352879", "#B8C76F",
            "#6F4F25", "#433900", "#9A6759", "#444444", "#6C6C6C", "#9AD284", "#6C5EB5", "#959595"),
    "Amber CRT": ("#120A00", "#7A4A00", "#FFB000"),
    "Green CRT": ("#001200", "#0F7A2E", "#33FF66"),
    "Riso": ("#F6F1E7", "#FF48B0", "#0078BF", "#1A1A1A"),
    "Sepia": ("#2B1D14", "#7A5638", "#C9A27A", "#F3E6CF"),
    "Thermal": ("#F4F1EA", "#9A968E", "#2B2B2B"),
}
PALETTE_CHOICES = tuple(PALETTES) + ("Original levels", "Custom")

# (dx, dy, weight) error kernels; divisor normalises the weights.
KERNELS = {
    "Floyd–Steinberg": (((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)), 16),
    "Atkinson": (((1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)), 8),
    "Jarvis-Judice-Ninke": (((1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                             (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)), 48),
    "Stucki": (((1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)), 42),
    "Burkes": (((1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)), 32),
    "Sierra": (((1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                (-1, 2, 2), (0, 2, 3), (1, 2, 2)), 32),
    "Sierra Two-Row": (((1, 0, 4), (2, 0, 3), (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)), 16),
    "Sierra Lite": (((1, 0, 2), (-1, 1, 1), (0, 1, 1)), 4),
}
ORDERED = ("Bayer 2×2", "Bayer 4×4", "Bayer 8×8", "Cluster dot", "Blue noise", "White noise", "Threshold")
ALGORITHMS = tuple(KERNELS) + ORDERED

_CLUSTER_4 = ((12, 5, 6, 13), (4, 0, 1, 7), (11, 3, 2, 8), (15, 10, 9, 14))
_WEIGHTS = (0.299, 0.587, 0.114)


def _bayer(n: int) -> list[list[int]]:
    matrix = [[0]]
    while len(matrix) < n:
        size = len(matrix)
        matrix = [[4 * matrix[y % size][x % size] + ((0, 2), (3, 1))[y // size][x // size] for x in range(2 * size)]
                  for y in range(2 * size)]
    return matrix


def _palette(p: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if p["palette"] == "Original levels":
        return None
    colours = PALETTES.get(p["palette"])
    if p["palette"] == "Custom":
        colours = [c for c in str(p["custom"]).replace(",", " ").split() if c.strip()] or ["#000000", "#FFFFFF"]
    return torch.stack([hex_to_rgb(c, device, dtype) for c in colours])


def _quantizer(palette: torch.Tensor | None, levels: int):
    if palette is None:
        steps = max(1, levels - 1)
        return lambda v: (v.clamp(0.0, 1.0) * steps).round() / steps
    weights = torch.tensor(_WEIGHTS, device=palette.device, dtype=palette.dtype)

    def nearest(v: torch.Tensor) -> torch.Tensor:
        distance = ((v.unsqueeze(-2) - palette).square() * weights).sum(-1)
        return palette[distance.argmin(-1)]
    return nearest


def _spread(palette: torch.Tensor | None, levels: int) -> float:
    """Typical distance between neighbouring palette colours: the ordered-dither amplitude."""
    if palette is None:
        return 1.0 / max(1, levels - 1)
    if palette.shape[0] < 2:
        return 0.5
    distance = torch.cdist(palette, palette)
    distance.fill_diagonal_(float("inf"))
    return float(distance.amin(-1).mean()) / math.sqrt(3.0) * 1.15


def _error_diffusion(image: torch.Tensor, quantize, kernel, divisor: int, strength: float) -> torch.Tensor:
    """Wavefront error diffusion: identical to a raster scan, vectorised along anti-diagonals.

    Pixel (x, y) is processed at step t = x + s*y. ``s`` is chosen so every
    kernel source lands on an earlier step than each pixel it feeds, which makes
    all pixels sharing a step independent of one another.
    """
    height, width = image.shape[:2]
    slope = 1
    for dx, dy, _ in kernel:
        if dy > 0 and dx < 0:
            slope = max(slope, math.floor(-dx / dy) + 1)
    flat = image.reshape(-1, 3).clone()
    rows = torch.arange(height, device=image.device)
    taps = torch.tensor([(dx, dy) for dx, dy, _ in kernel], device=image.device)
    tap_dx, tap_dy = taps[:, 0:1], taps[:, 1:2]
    tap_weight = torch.tensor([w for _, _, w in kernel], device=image.device, dtype=image.dtype).view(-1, 1, 1)
    for step in range(width + slope * (height - 1)):
        # Rows whose column step - slope*y lies inside the image, as plain ints (no device sync).
        first = max(0, -(-(step - width + 1) // slope))
        last = min(height - 1, step // slope)
        if first > last:
            continue
        y = rows[first:last + 1]
        x = step - slope * y
        index = y * width + x
        value = flat[index]
        chosen = quantize(value)
        flat[index] = chosen
        error = (value - chosen) * (strength / divisor)
        # All kernel taps in one scatter; out-of-image taps carry zero weight.
        nx, ny = x.unsqueeze(0) + tap_dx, y.unsqueeze(0) + tap_dy
        inside = ((nx >= 0) & (nx < width) & (ny < height)).unsqueeze(-1)
        target = ny.clamp(max=height - 1) * width + nx.clamp(0, width - 1)
        flat.index_add_(0, target.reshape(-1), (error.unsqueeze(0) * inside * tap_weight).reshape(-1, 3))
    return flat.view(height, width, 3)


def _threshold_map(name: str, height: int, width: int, device, dtype) -> torch.Tensor:
    if name == "Threshold":
        return torch.zeros(height, width, device=device, dtype=dtype)
    if name == "White noise":
        return hash_noise(height, width, 5, device, dtype)[..., 0] - 0.5
    if name == "Blue noise":
        # Interleaved gradient noise: a cheap, well-distributed blue-noise-like field.
        y = torch.arange(height, device=device, dtype=dtype).view(-1, 1)
        x = torch.arange(width, device=device, dtype=dtype).view(1, -1)
        return torch.frac(52.9829189 * torch.frac(0.06711056 * x + 0.00583715 * y)) - 0.5
    if name == "Cluster dot":
        matrix, n = _CLUSTER_4, 4
    else:
        n = int(name.split(" ")[1][0])
        matrix = _bayer(n)
    tile = (torch.tensor(matrix, device=device, dtype=dtype) + 0.5) / (n * n) - 0.5
    return tile.repeat(math.ceil(height / n), math.ceil(width / n))[:height, :width]


def dither(rgb: torch.Tensor, p: dict, ctx: FxContext) -> torch.Tensor:
    height, width = rgb.shape[:2]
    cell = max(1.0, ref_px(p["pixel_size"], rgb))
    work_h, work_w = max(1, round(height / cell)), max(1, round(width / cell))
    work = to_hwc(F.interpolate(to_nchw(rgb), size=(work_h, work_w), mode="area")) if cell > 1.0 else rgb
    work = ((work - 0.5) * (1.0 + p["contrast"]) + 0.5 + p["brightness"]).clamp(0.0, 1.0)
    if p["gamma"] != 1.0:
        work = work.pow(1.0 / p["gamma"])
    if p["mono"]:
        work = luminance(work).unsqueeze(-1).expand(-1, -1, 3).contiguous()
    levels = int(p["levels"])
    algorithm = p["algorithm"]
    if algorithm in KERNELS:
        # Thousands of tiny sequential steps run faster on CPU than as GPU launches.
        kernel, divisor = KERNELS[algorithm]
        cpu = torch.device("cpu")
        quantize = _quantizer(_palette(p, cpu, rgb.dtype), levels)
        out = _error_diffusion(work.to(cpu), quantize, kernel, divisor, p["strength"]).to(rgb.device)
    else:
        palette = _palette(p, rgb.device, rgb.dtype)
        quantize = _quantizer(palette, levels)
        threshold = _threshold_map(algorithm, work_h, work_w, rgb.device, rgb.dtype).unsqueeze(-1)
        out = quantize(work + threshold * _spread(palette, levels) * p["strength"] * 2.0)
    if cell > 1.0:
        out = to_hwc(F.interpolate(to_nchw(out), size=(height, width), mode="nearest"))
    return out.clamp(0.0, 1.0)
