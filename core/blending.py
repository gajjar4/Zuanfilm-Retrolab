"""Shared signed-carrier photographic blend modes for grain and texture."""

from __future__ import annotations

import torch

BLEND_MODES = (
    "Additive", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light",
    "Color Dodge", "Color Burn", "Lighten", "Darken", "Difference", "Exclusion",
)


def normalize_blend_mode(value: object) -> str:
    candidate = str(value or "Additive")
    return candidate if candidate in BLEND_MODES else "Additive"


def blend_signed_layer(source: torch.Tensor, signed_layer: torch.Tensor, mode: str = "Additive") -> torch.Tensor:
    """Composite a zero-centred signed photographic carrier over an sRGB source.

    The neutral point is zero for Additive/Multiply/Screen and 0.5 for
    Overlay/Soft Light/Hard Light. This is the established Procedural Texture
    compositing contract, shared with Film Grain so both controls behave
    consistently. All results are clamped to display-referred sRGB bounds.
    """
    selected = normalize_blend_mode(mode)
    if selected == "Additive":
        result = source + signed_layer
    elif selected == "Multiply":
        result = source * (1.0 + signed_layer)
    elif selected == "Screen":
        result = 1.0 - (1.0 - source) * (1.0 - signed_layer)
    elif selected == "Overlay":
        layer = 0.5 + signed_layer
        result = torch.where(
            source <= 0.5,
            2.0 * source * layer,
            1.0 - 2.0 * (1.0 - source) * (1.0 - layer),
        )
    elif selected == "Soft Light":
        layer = 0.5 + signed_layer
        d = torch.where(
            source <= 0.25,
            ((16.0 * source - 12.0) * source + 4.0) * source,
            torch.sqrt(source.clamp_min(0.0)),
        )
        result = torch.where(
            layer <= 0.5,
            source - (1.0 - 2.0 * layer) * source * (1.0 - source),
            source + (2.0 * layer - 1.0) * (d - source),
        )
    elif selected == "Hard Light":
        layer = 0.5 + signed_layer
        result = torch.where(
            layer <= 0.5,
            2.0 * source * layer,
            1.0 - 2.0 * (1.0 - source) * (1.0 - layer),
        )
    elif selected == "Color Dodge":
        result = source / (1.0 - signed_layer).clamp_min(1e-4)
    elif selected == "Color Burn":
        result = 1.0 - (1.0 - source) / (1.0 + signed_layer).clamp_min(1e-4)
    elif selected == "Lighten":
        result = torch.maximum(source, source + signed_layer)
    elif selected == "Darken":
        result = torch.minimum(source, source + signed_layer)
    elif selected == "Difference":
        result = torch.abs(source - signed_layer)
    elif selected == "Exclusion":
        result = source + signed_layer - 2.0 * source * signed_layer
    else:  # Defensive fallback; normalize_blend_mode already validates.
        result = source + signed_layer
    return result.clamp(0.0, 1.0)
