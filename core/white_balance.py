"""Resolve-inspired two-axis white-balance wheel for Zuanfilm Retrolab.

The wheel expresses a creative temperature/tint offset around a neutral center.
It is intentionally not calibrated to Kelvin or a camera sensor's RAW WB model.
"""

from __future__ import annotations

import math

import torch

from .color import linear_to_srgb, srgb_to_linear


def apply_white_balance(
    rgb_srgb: torch.Tensor,
    *,
    temperature: float = 0.0,
    tint: float = 0.0,
) -> torch.Tensor:
    """Apply a restrained linear-light temperature/tint gain-wheel adjustment.

    ``temperature`` ranges from cool (-1) to warm (+1). ``tint`` ranges from
    green (-1) to magenta (+1). A Rec.709-weighted gain normalization keeps a
    neutral gray's luminance stable while allowing its chromatic balance to move.
    """
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC image with at least three channels")

    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError):
        temperature = 0.0
    try:
        tint = float(tint)
    except (TypeError, ValueError, OverflowError):
        tint = 0.0
    if not math.isfinite(temperature):
        temperature = 0.0
    if not math.isfinite(tint):
        tint = 0.0
    temperature = min(max(temperature, -1.0), 1.0)
    tint = min(max(tint, -1.0), 1.0)

    if abs(temperature) < 1e-12 and abs(tint) < 1e-12:
        return rgb_srgb

    rgb_linear = srgb_to_linear(
        rgb_srgb[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    )

    # Temperature moves red/blue in opposite directions. Tint moves red and
    # blue together against green. Coefficients are deliberately moderate so
    # the full wheel travel stays useful for creative correction.
    r_ev = 0.38 * temperature + 0.22 * tint
    g_ev = -0.30 * tint
    b_ev = -0.38 * temperature + 0.22 * tint
    gains = torch.tensor(
        [2.0 ** r_ev, 2.0 ** g_ev, 2.0 ** b_ev],
        dtype=rgb_linear.dtype,
        device=rgb_linear.device,
    )

    # Normalize the wheel's gain so neutral gray retains Rec.709 luminance.
    neutral_luma_gain = (
        0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2]
    ).clamp_min(1e-8)
    gains = gains / neutral_luma_gain

    balanced = rgb_linear * gains.view(1, 1, 3)
    return linear_to_srgb(balanced)
