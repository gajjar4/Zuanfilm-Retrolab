"""Optical radial exposure falloff for photographic vignette finishing."""

from __future__ import annotations

import torch

from .color import linear_to_srgb, srgb_to_linear


def apply_vignette(
    rgb_srgb: torch.Tensor,
    *,
    amount: float,
    feather: float,
) -> torch.Tensor:
    """Apply a centered, aspect-ratio-correct radial exposure falloff.

    ``amount`` is normalized to 0..2 and reaches a 3-stop reduction at the
    four corners. The previous 0..1 range is unchanged in response; the extended
    range provides twice the maximum corner exposure reduction. ``feather``
    controls how far the darkening transition begins from the frame center:
    higher values create a broader, more gradual falloff.
    The attenuation is applied in linear light so it behaves like an exposure
    reduction rather than an RGB overlay.
    """
    if amount <= 0.0:
        return rgb_srgb
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC RGB image")

    strength = min(max(float(amount), 0.0), 2.0)
    feather_value = min(max(float(feather), 0.0), 1.0)
    height, width = rgb_srgb.shape[:2]
    device = rgb_srgb.device

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radial = torch.sqrt(xx.square() + yy.square()).div(2.0 ** 0.5).clamp(0.0, 1.0)

    # Feather=0 confines the falloff to the outermost 10%; Feather=1 starts
    # the smooth transition at 0.30 radius for a broad, gradual vignette.
    start = 0.90 - 0.60 * feather_value
    t = ((radial - start) / max(1.0 - start, 1e-6)).clamp(0.0, 1.0)
    falloff = t.square() * (3.0 - 2.0 * t)
    exposure_stops = 1.5 * strength * falloff
    attenuation = torch.exp2(-exposure_stops).unsqueeze(-1)

    linear = srgb_to_linear(rgb_srgb[..., :3])
    result = linear_to_srgb(linear * attenuation)
    if rgb_srgb.shape[-1] > 3:
        return torch.cat((result, rgb_srgb[..., 3:]), dim=-1)
    return result
