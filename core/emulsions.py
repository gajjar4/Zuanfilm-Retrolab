"""Public-facing creative color-film emulsion looks for Zuanfilm Retrolab.

These are independently authored RGB finishing profiles inspired by documented
film-family character. They are not sensitometric, chemical or scanner
reproductions of the named products. Film grain remains a separate engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch

from .color import linear_to_srgb, luminance_rec709, srgb_to_linear


@dataclass(frozen=True)
class EmulsionProfile:
    id: str
    name: str
    family: str
    exposure: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    toe: float = 0.0
    shoulder: float = 0.0
    channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _p(
    profile_id: str,
    name: str,
    family: str,
    *,
    exposure: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    toe: float = 0.0,
    shoulder: float = 0.0,
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> EmulsionProfile:
    return EmulsionProfile(
        profile_id, name, family, exposure, contrast, saturation, toe, shoulder,
        gains, shadows, highlights,
    )


# Exactly 50 creative profiles, grouped by photographic origin. Neutral is a
# separate identity entry so none of the 50 film looks has to double as bypass.
EMULSION_PROFILES: Final[tuple[EmulsionProfile, ...]] = (
    # ------------------------------------------------------------------ Cinema / motion-picture color negative (10)
    _p("vision3_50d", "Kodak VISION3 50D", "Cinema", contrast=0.96, saturation=0.96, toe=0.020, shoulder=0.16,
       gains=(1.000, 1.000, 1.006), shadows=(0.000, 0.002, 0.008), highlights=(0.006, 0.004, 0.000)),
    _p("vision3_250d", "Kodak VISION3 250D", "Cinema", exposure=0.015, contrast=1.015, saturation=1.025, toe=0.015, shoulder=0.13,
       gains=(1.012, 1.000, 0.988), shadows=(0.000, 0.002, 0.005), highlights=(0.008, 0.004, -0.002)),
    _p("vision3_200t", "Kodak VISION3 200T", "Cinema", contrast=0.99, saturation=0.98, toe=0.024, shoulder=0.17,
       gains=(0.990, 1.000, 1.018), shadows=(-0.002, 0.002, 0.010), highlights=(0.008, 0.004, -0.003)),
    _p("vision3_500t", "Kodak VISION3 500T", "Cinema", exposure=-0.01, contrast=1.025, saturation=1.00, toe=0.020, shoulder=0.15,
       gains=(1.000, 0.995, 1.018), shadows=(-0.002, 0.002, 0.009), highlights=(0.008, 0.004, -0.003)),
    _p("vision3_500t_push", "Kodak VISION3 500T · Push", "Cinema", exposure=0.035, contrast=1.16, saturation=1.07, toe=0.008, shoulder=0.09,
       gains=(1.008, 0.992, 1.020), shadows=(-0.006, 0.001, 0.010), highlights=(0.010, 0.003, -0.004)),
    _p("verita_200d", "Kodak VERITA 200D", "Cinema", exposure=0.015, contrast=1.12, saturation=1.17, toe=0.006, shoulder=0.08,
       gains=(1.025, 1.000, 0.974), shadows=(-0.006, 0.001, 0.004), highlights=(0.012, 0.006, -0.005)),
    _p("cinestill_50d", "CineStill 50D", "Cinema", contrast=1.075, saturation=1.04, toe=0.012, shoulder=0.12,
       gains=(1.015, 1.000, 0.990), shadows=(-0.002, 0.001, 0.006), highlights=(0.008, 0.004, -0.002)),
    _p("cinestill_400d", "CineStill 400D", "Cinema", exposure=0.015, contrast=1.06, saturation=1.08, toe=0.012, shoulder=0.12,
       gains=(1.018, 1.000, 0.985), shadows=(-0.003, 0.002, 0.006), highlights=(0.010, 0.004, -0.003)),
    _p("cinestill_800t", "CineStill 800T", "Cinema", exposure=0.02, contrast=1.08, saturation=1.04, toe=0.016, shoulder=0.10,
       gains=(1.000, 0.990, 1.025), shadows=(-0.004, 0.001, 0.012), highlights=(0.012, 0.005, -0.006)),
    _p("cinestill_800t_night", "CineStill 800T · Night", "Cinema", exposure=-0.025, contrast=1.19, saturation=1.12, toe=0.004, shoulder=0.075,
       gains=(1.010, 0.985, 1.035), shadows=(-0.009, 0.001, 0.018), highlights=(0.018, 0.006, -0.010)),

    # ------------------------------------------------------------------ Instant color films (10)
    _p("polaroid_sx70", "Polaroid SX-70 Color", "Instant", exposure=0.035, contrast=0.88, saturation=0.88, toe=0.055, shoulder=0.24,
       gains=(1.025, 1.000, 0.970), shadows=(0.006, 0.002, 0.008), highlights=(0.014, 0.006, -0.004)),
    _p("polaroid_600", "Polaroid 600 Color", "Instant", exposure=0.01, contrast=0.94, saturation=0.94, toe=0.045, shoulder=0.20,
       gains=(1.018, 1.000, 0.978), shadows=(0.004, 0.002, 0.006), highlights=(0.012, 0.006, -0.003)),
    _p("polaroid_itype", "Polaroid i-Type Color", "Instant", contrast=0.98, saturation=1.00, toe=0.030, shoulder=0.16,
       gains=(1.010, 1.000, 0.990), shadows=(0.002, 0.002, 0.004), highlights=(0.008, 0.004, -0.001)),
    _p("polaroid_669", "Polaroid 669 Packfilm", "Instant", exposure=0.015, contrast=1.06, saturation=1.02, toe=0.025, shoulder=0.14,
       gains=(1.025, 1.000, 0.965), shadows=(0.004, 0.002, 0.001), highlights=(0.012, 0.006, -0.006)),
    _p("polaroid_timezero", "Polaroid Time-Zero", "Instant", exposure=0.03, contrast=0.91, saturation=0.84, toe=0.07, shoulder=0.25,
       gains=(1.030, 1.000, 0.955), shadows=(0.010, 0.004, 0.006), highlights=(0.016, 0.008, -0.005)),
    _p("polaroid_690", "Polaroid 690 Color", "Instant", contrast=1.08, saturation=1.06, toe=0.025, shoulder=0.13,
       gains=(1.020, 1.000, 0.980), shadows=(0.001, 0.002, 0.004), highlights=(0.010, 0.006, -0.002)),
    _p("instax_mini", "Fujifilm instax mini", "Instant", exposure=0.015, contrast=1.06, saturation=1.08, toe=0.018, shoulder=0.12,
       gains=(1.010, 1.000, 0.985), shadows=(0.000, 0.001, 0.004), highlights=(0.006, 0.004, -0.002)),
    _p("instax_square", "Fujifilm instax SQUARE", "Instant", contrast=1.02, saturation=1.04, toe=0.025, shoulder=0.15,
       gains=(1.012, 1.000, 0.990), shadows=(0.001, 0.002, 0.004), highlights=(0.008, 0.005, -0.001)),
    _p("instax_wide", "Fujifilm instax WIDE", "Instant", exposure=0.01, contrast=1.00, saturation=1.02, toe=0.025, shoulder=0.16,
       gains=(1.015, 1.000, 0.985), shadows=(0.002, 0.001, 0.005), highlights=(0.009, 0.004, -0.002)),
    _p("fuji_fp100c", "Fujifilm FP-100C", "Instant", contrast=1.07, saturation=1.10, toe=0.022, shoulder=0.13,
       gains=(1.015, 1.000, 0.985), shadows=(0.002, 0.002, 0.006), highlights=(0.010, 0.005, -0.002)),

    # ------------------------------------------------------------------ Slide / reversal film (10)
    _p("ektachrome_e100", "Kodak Ektachrome E100", "Slide", contrast=1.09, saturation=1.06, toe=0.008, shoulder=0.06,
       gains=(1.000, 1.000, 1.006), shadows=(0.000, 0.001, 0.004), highlights=(0.003, 0.002, 0.002)),
    _p("ektachrome_100d", "Kodak Ektachrome 100D", "Slide", contrast=1.12, saturation=1.10, toe=0.006, shoulder=0.055,
       gains=(1.010, 1.000, 0.995), shadows=(-0.001, 0.001, 0.006), highlights=(0.006, 0.004, 0.000)),
    _p("velvia_50", "Fujichrome Velvia 50", "Slide", contrast=1.25, saturation=1.42, toe=0.002, shoulder=0.04,
       gains=(0.995, 1.005, 1.025), shadows=(-0.004, 0.001, 0.010), highlights=(0.002, 0.006, 0.008)),
    _p("velvia_100", "Fujichrome Velvia 100", "Slide", contrast=1.28, saturation=1.46, toe=0.001, shoulder=0.035,
       gains=(1.000, 1.010, 1.030), shadows=(-0.005, 0.001, 0.012), highlights=(0.003, 0.007, 0.010)),
    _p("provia_100f", "Fujichrome Provia 100F", "Slide", contrast=1.04, saturation=1.07, toe=0.008, shoulder=0.07,
       gains=(1.000, 1.000, 1.004), shadows=(0.000, 0.001, 0.004), highlights=(0.003, 0.003, 0.002)),
    _p("astia_100f", "Fujichrome Astia 100F", "Slide", contrast=0.91, saturation=0.91, toe=0.025, shoulder=0.16,
       gains=(1.015, 1.000, 0.985), shadows=(0.003, 0.002, 0.004), highlights=(0.010, 0.005, 0.000)),
    _p("sensia_100", "Fujichrome Sensia 100", "Slide", contrast=1.03, saturation=1.04, toe=0.012, shoulder=0.09,
       gains=(1.000, 1.000, 1.010), shadows=(0.000, 0.002, 0.006), highlights=(0.002, 0.004, 0.005)),
    _p("sensia_400", "Fujichrome Sensia 400", "Slide", contrast=1.07, saturation=1.08, toe=0.010, shoulder=0.08,
       gains=(1.008, 0.995, 1.012), shadows=(0.001, -0.002, 0.006), highlights=(0.006, 0.001, 0.003)),
    _p("kodachrome_64", "Kodachrome 64 · Archive", "Slide", exposure=-0.01, contrast=1.20, saturation=1.17, toe=0.004, shoulder=0.065,
       gains=(1.025, 1.000, 0.970), shadows=(-0.004, 0.001, 0.008), highlights=(0.012, 0.004, -0.005)),
    _p("agfachrome_rsx100", "Agfachrome RSX 100", "Slide", contrast=1.13, saturation=1.13, toe=0.006, shoulder=0.07,
       gains=(1.010, 1.000, 1.012), shadows=(-0.002, 0.002, 0.010), highlights=(0.004, 0.005, 0.005)),

    # ------------------------------------------------------------------ 35 mm color negative (20)
    _p("portra_160", "Kodak Portra 160", "35mm", exposure=0.015, contrast=0.91, saturation=0.88, toe=0.035, shoulder=0.19,
       gains=(1.020, 1.000, 0.985), shadows=(0.004, 0.002, 0.006), highlights=(0.010, 0.006, 0.001)),
    _p("portra_400", "Kodak Portra 400", "35mm", exposure=0.01, contrast=0.94, saturation=0.91, toe=0.030, shoulder=0.17,
       gains=(1.018, 1.000, 0.988), shadows=(0.003, 0.002, 0.006), highlights=(0.010, 0.006, 0.001)),
    _p("portra_800", "Kodak Portra 800", "35mm", exposure=0.02, contrast=0.98, saturation=0.93, toe=0.040, shoulder=0.16,
       gains=(1.020, 1.000, 0.978), shadows=(0.004, 0.002, 0.008), highlights=(0.012, 0.006, -0.001)),
    _p("ektar_100", "Kodak Ektar 100", "35mm", contrast=1.22, saturation=1.34, toe=0.003, shoulder=0.06,
       gains=(1.010, 1.000, 0.985), shadows=(-0.003, 0.001, 0.007), highlights=(0.004, 0.005, 0.000)),
    _p("gold_200", "Kodak Gold 200", "35mm", exposure=0.02, contrast=1.08, saturation=1.15, toe=0.015, shoulder=0.12,
       gains=(1.035, 1.000, 0.965), shadows=(0.003, 0.001, 0.000), highlights=(0.014, 0.008, -0.005)),
    _p("colorplus_200", "Kodak ColorPlus 200", "35mm", exposure=0.015, contrast=1.05, saturation=1.04, toe=0.020, shoulder=0.14,
       gains=(1.028, 1.000, 0.972), shadows=(0.003, 0.001, 0.003), highlights=(0.012, 0.006, -0.004)),
    _p("ultramax_400", "Kodak UltraMax 400", "35mm", contrast=1.12, saturation=1.17, toe=0.012, shoulder=0.10,
       gains=(1.025, 1.000, 0.970), shadows=(0.001, 0.001, 0.003), highlights=(0.010, 0.006, -0.004)),
    _p("kodacolor_100", "Kodak Kodacolor 100", "35mm", contrast=1.02, saturation=1.05, toe=0.020, shoulder=0.13,
       gains=(1.015, 1.000, 0.985), shadows=(0.002, 0.001, 0.004), highlights=(0.008, 0.005, 0.000)),
    _p("kodacolor_200", "Kodak Kodacolor 200", "35mm", contrast=1.06, saturation=1.10, toe=0.018, shoulder=0.12,
       gains=(1.020, 1.000, 0.978), shadows=(0.002, 0.001, 0.004), highlights=(0.010, 0.006, -0.002)),
    _p("fujicolor_400", "Fujicolor 400", "35mm", contrast=1.04, saturation=1.07, toe=0.020, shoulder=0.14,
       gains=(0.998, 1.005, 1.012), shadows=(0.000, 0.003, 0.007), highlights=(0.004, 0.006, 0.006)),
    _p("superia_400", "Fujicolor Superia X-TRA 400", "35mm", contrast=1.13, saturation=1.15, toe=0.010, shoulder=0.09,
       gains=(0.990, 1.012, 1.025), shadows=(-0.005, 0.004, 0.012), highlights=(0.003, 0.008, 0.010)),
    _p("superia_reala", "Fujicolor Superia Reala", "35mm", contrast=1.01, saturation=1.02, toe=0.020, shoulder=0.13,
       gains=(1.000, 1.005, 1.006), shadows=(0.001, 0.003, 0.005), highlights=(0.004, 0.005, 0.004)),
    _p("pro400h", "Fujicolor PRO 400H", "35mm", exposure=0.02, contrast=0.89, saturation=0.88, toe=0.040, shoulder=0.20,
       gains=(1.020, 1.005, 1.015), shadows=(0.002, 0.006, 0.010), highlights=(0.010, 0.009, 0.006)),
    _p("natura_1600", "Fujicolor Natura 1600", "35mm", exposure=0.015, contrast=1.10, saturation=1.02, toe=0.026, shoulder=0.12,
       gains=(1.012, 0.995, 1.018), shadows=(-0.003, 0.001, 0.010), highlights=(0.009, 0.003, 0.004)),
    _p("superia_800", "Fujicolor Superia 800", "35mm", contrast=1.15, saturation=1.12, toe=0.012, shoulder=0.09,
       gains=(0.995, 1.000, 1.025), shadows=(-0.005, 0.002, 0.013), highlights=(0.007, 0.005, 0.003)),
    _p("lomo_100", "Lomography Color Negative 100", "35mm", contrast=1.13, saturation=1.23, toe=0.009, shoulder=0.09,
       gains=(1.020, 1.000, 0.975), shadows=(0.001, 0.002, 0.006), highlights=(0.009, 0.005, -0.003)),
    _p("lomo_400", "Lomography Color Negative 400", "35mm", contrast=1.12, saturation=1.20, toe=0.014, shoulder=0.11,
       gains=(1.018, 1.000, 0.982), shadows=(0.002, 0.002, 0.006), highlights=(0.010, 0.006, -0.002)),
    _p("lomo_800", "Lomography Color Negative 800", "35mm", exposure=0.015, contrast=1.16, saturation=1.18, toe=0.018, shoulder=0.10,
       gains=(1.020, 0.995, 0.985), shadows=(-0.002, 0.001, 0.008), highlights=(0.012, 0.006, -0.002)),
    _p("ektacolor_pro_160", "Kodak Ektacolor Pro 160", "35mm", contrast=0.93, saturation=0.91, toe=0.028, shoulder=0.18,
       gains=(1.012, 1.000, 1.000), shadows=(0.002, 0.002, 0.004), highlights=(0.006, 0.005, 0.003)),
    _p("ektacolor_pro_400", "Kodak Ektacolor Pro 400", "35mm", contrast=0.98, saturation=0.97, toe=0.023, shoulder=0.16,
       gains=(1.008, 1.000, 1.006), shadows=(0.001, 0.002, 0.006), highlights=(0.005, 0.005, 0.004)),
)

EMULSION_BY_ID: Final[dict[str, EmulsionProfile]] = {profile.id: profile for profile in EMULSION_PROFILES}
EMULSION_CHOICES: Final[tuple[str, ...]] = ("neutral",) + tuple(profile.id for profile in EMULSION_PROFILES)
EMULSION_FAMILIES: Final[tuple[str, ...]] = ("Cinema", "Instant", "Slide", "35mm")
EMULSIONS_BY_FAMILY: Final[dict[str, tuple[EmulsionProfile, ...]]] = {
    family: tuple(profile for profile in EMULSION_PROFILES if profile.family == family)
    for family in EMULSION_FAMILIES
}


def normalize_emulsion_id(value: object) -> str:
    candidate = str(value or "neutral")
    return candidate if candidate == "neutral" or candidate in EMULSION_BY_ID else "neutral"


def normalize_emulsion_response(value: object) -> float:
    """Clamp Emulsion Response to its supported creative range."""
    try:
        response = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(response):
        return 1.0
    return min(max(response, 0.5), 3.0)


def apply_color_emulsion(
    rgb_srgb: torch.Tensor,
    emulsion_id: str = "neutral",
    response: float = 1.0,
) -> torch.Tensor:
    """Apply one independent creative color/tonal response profile.

    The transform works in linear RGB: exposure and a smooth luminance curve
    establish negative/reversal character, luminance-ratio recombination
    preserves the image's hue structure, and saturation/channel/tone biases
    establish the profile's color signature. Emulsion Response scales the
    complete profile deviation in linear light around the stage input. No grain
    is generated here. A response of 1.0 preserves the original profile output.
    """
    selected = EMULSION_BY_ID.get(normalize_emulsion_id(emulsion_id))
    if selected is None:
        return rgb_srgb

    response = normalize_emulsion_response(response)
    source_linear = srgb_to_linear(rgb_srgb[..., :3])
    linear = source_linear
    if selected.exposure:
        linear = linear * (2.0 ** selected.exposure)

    luminance = luminance_rec709(linear).clamp(0.0, 1.0)
    x = luminance
    # Gentle film-like S-curve with separate toe lift and shoulder compression.
    s_curve = (selected.contrast - 1.0) * 4.0 * x * (1.0 - x) * (x - 0.5)
    toe = selected.toe * 3.0 * x * (1.0 - x).square()
    shoulder = selected.shoulder * 3.0 * x.square() * (1.0 - x)
    mapped_luminance = (x + s_curve + toe - shoulder).clamp(0.0, 1.0)

    ratio = mapped_luminance / luminance.clamp_min(1e-6)
    ratio = ratio.clamp(0.0, 4.0)
    linear = linear * ratio[..., None]

    mapped_y = luminance_rec709(linear).clamp(0.0, 1.0)
    saturation = min(max(float(selected.saturation), 0.0), 2.0)
    linear = mapped_y[..., None] + (linear - mapped_y[..., None]) * saturation

    gains = torch.tensor(selected.channel_gains, dtype=linear.dtype, device=linear.device)
    linear = linear * gains
    shadow_weight = (1.0 - mapped_y).clamp(0.0, 1.0).square()
    highlight_weight = mapped_y.clamp(0.0, 1.0).square()
    shadow_tint = torch.tensor(selected.shadow_tint, dtype=linear.dtype, device=linear.device)
    highlight_tint = torch.tensor(selected.highlight_tint, dtype=linear.dtype, device=linear.device)
    linear = linear + shadow_weight[..., None] * shadow_tint + highlight_weight[..., None] * highlight_tint

    # Scale the complete tonal and color response around the exact stage input.
    # At 1.0 this branch is skipped, preserving the established output exactly.
    if response != 1.0:
        linear = source_linear + (linear - source_linear) * response
    return linear_to_srgb(linear)
