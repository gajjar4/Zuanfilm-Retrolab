"""Retrolab FX: a Photoshop-style stack of creative effects.

The catalog below is the single source of truth for every effect's controls.
The frontend builds its panels from ``catalog_payload()``, and ``normalize_stack``
re-validates browser- or workflow-supplied stacks against the same table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from . import adjust, blur, dither, glitch, light, print_fx, retro
from .common import BLEND_MODES, FxContext, composite


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # float | int | choice | color | bool | text
    default: Any
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    options: tuple[str, ...] = ()


def F(key, label, default, minimum, maximum, step=0.01):
    return Param(key, label, "float", float(default), float(minimum), float(maximum), float(step))


def I(key, label, default, minimum, maximum):
    return Param(key, label, "int", int(default), float(minimum), float(maximum), 1.0)


def C(key, label, default, options):
    return Param(key, label, "choice", default, options=tuple(options))


def H(key, label, default):
    return Param(key, label, "color", default)


def B(key, label, default):
    return Param(key, label, "bool", bool(default))


def T(key, label, default):
    return Param(key, label, "text", default)


CENTER = (F("cx", "Center X", 0.5, 0.0, 1.0), F("cy", "Center Y", 0.5, 0.0, 1.0))


@dataclass(frozen=True)
class Effect:
    name: str
    category: str
    run: Callable
    params: tuple[Param, ...]
    handle: str | None = None  # center | angle | band


_HSL_PARAMS = tuple(
    Param(f"{band}_{part}", f"{band.title()} {label}", "float", 0.0, -1.0, 1.0, 0.01)
    for band, _ in adjust.HSL_BANDS for part, label in (("hue", "hue"), ("sat", "saturation"), ("lum", "luminance"))
)

EFFECTS: dict[str, Effect] = {
    "directional_blur": Effect("Directional blur", "Blur", blur.directional_blur,
                               (F("angle", "Angle", 35, 0, 180, 1), F("distance", "Distance", 30, 0, 200, 0.5)) + CENTER, "angle"),
    "zoom_blur": Effect("Zoom blur", "Blur", blur.zoom_blur, (F("amount", "Amount", 0.12, 0, 0.6),) + CENTER, "center"),
    "spin_blur": Effect("Spin blur", "Blur", blur.spin_blur, (F("angle", "Angle", 6, 0, 45, 0.5),) + CENTER, "center"),
    "tilt_shift": Effect("Tilt-shift", "Blur", blur.tilt_shift,
                         (F("blur", "Blur", 8, 0, 40, 0.5), F("band", "Focus band", 0.22, 0.02, 0.9),
                          F("feather", "Feather", 0.18, 0.01, 0.6), F("angle", "Angle", 0, -90, 90, 1)) + CENTER, "band"),
    "lens_blur": Effect("Lens blur", "Blur", blur.lens_blur,
                        (F("radius", "Radius", 10, 0, 60, 0.5), C("shape", "Bokeh shape", "Circle", ("Circle", "Hexagon", "Cat eye")),
                         F("highlights", "Highlight bloom", 3.0, 0, 10, 0.1), F("focus", "Focus area", 0.0, 0, 1)) + CENTER, "center"),
    "anamorphic_streaks": Effect("Anamorphic streaks", "Light", light.anamorphic_streaks,
                                 (F("amount", "Amount", 0.6, 0, 2), F("threshold", "Threshold", 0.5, 0.2, 0.98),
                                  F("length", "Length", 260, 20, 1024, 1), H("tint", "Tint", "#5AA8FF"))),
    "god_rays": Effect("God rays", "Light", light.god_rays,
                       (F("amount", "Amount", 0.6, 0, 2), F("threshold", "Threshold", 0.42, 0.1, 0.98),
                        F("length", "Length", 0.35, 0, 0.9)) + (F("cx", "Center X", 0.5, 0, 1), F("cy", "Center Y", 0.25, 0, 1)), "center"),
    "prism": Effect("Prism", "Light", light.prism,
                    (F("offset", "Offset", 24, 0, 200, 0.5), F("angle", "Angle", 20, 0, 360, 1),
                     F("hue", "Hue shift", 40, -180, 180, 1), F("mix", "Mix", 0.45, 0, 1))),
    "chromatic_aberration": Effect("Chromatic aberration", "Retro", retro.chromatic_aberration,
                                   (F("amount", "Amount", 4, 0, 40, 0.5),) + CENTER, "center"),
    "vhs": Effect("VHS", "Retro", retro.vhs,
                  (F("bleed", "Color bleed", 4, 0, 24, 0.5), F("scanlines", "Scanlines", 0.4, 0, 1),
                   F("noise", "Noise", 0.25, 0, 1), F("tracking", "Tracking error", 0.3, 0, 1), I("seed", "Seed", 7, 0, 999))),
    "crt": Effect("CRT", "Retro", retro.crt,
                  (F("curvature", "Curvature", 0.4, 0, 1), F("scanlines", "Scanlines", 0.55, 0, 1),
                   F("mask", "Phosphor mask", 0.35, 0, 1), F("glow", "Glow", 0.4, 0, 1))),
    "film_frame": Effect("Film frame", "Retro", retro.film_frame,
                         (C("style", "Style", "35mm", ("35mm", "Polaroid", "Instax")), T("text", "Edge print / caption", "KODAK PORTRA 400  24A"))),
    "gate_weave": Effect("Gate weave", "Retro", retro.gate_weave,
                         (F("amount", "Weave", 2.5, 0, 20, 0.1), F("flicker", "Flicker", 0.5, 0, 1), I("seed", "Seed", 3, 0, 999))),
    "halftone": Effect("Halftone", "Print", print_fx.halftone,
                       (C("mode", "Mode", "Mono", ("Mono", "CMYK")), F("cell", "Cell size", 8, 2, 48, 0.5),
                        F("angle", "Angle", 45, 0, 90, 1), F("gain", "Dot gain", 0.1, -0.3, 0.6),
                        H("ink", "Ink", "#1B1B1B"), H("paper", "Paper", "#F4EFE6"))),
    "gradient_map": Effect("Gradient map", "Print", print_fx.gradient_map,
                           (H("shadows", "Shadows", "#1D2A6B"), H("midtones", "Midtones", "#C0496B"), H("highlights", "Highlights", "#FFC37A"),
                            B("use_midtones", "Use midtones", True), F("mix", "Mix", 1.0, 0, 1))),
    "risograph": Effect("Risograph", "Print", print_fx.risograph,
                        (H("ink_a", "Ink A", "#0078BF"), H("ink_b", "Ink B", "#FF48B0"), H("paper", "Paper", "#F6F1E7"),
                         F("cell", "Cell size", 6, 2, 32, 0.5), F("misregister", "Misregistration", 3, 0, 24, 0.5), F("grain", "Paper grain", 0.5, 0, 1))),
    "pixel_sort": Effect("Pixel sort", "Glitch", glitch.pixel_sort,
                         (F("low", "Low threshold", 0.35, 0, 1), F("high", "High threshold", 0.9, 0, 1),
                          C("direction", "Direction", "Vertical", ("Vertical", "Horizontal")), B("reverse", "Reverse", False))),
    "block_shift": Effect("Block shift", "Glitch", glitch.block_shift,
                          (F("amount", "Amount", 0.25, 0, 1), F("band", "Band height", 14, 2, 120, 0.5), F("shift", "Max shift", 60, 0, 400, 1),
                           F("split", "RGB split", 0.3, 0, 1), I("seed", "Seed", 21, 0, 999), B("animate", "Animate per frame", True))),
    "jpeg_crunch": Effect("JPEG crunch", "Glitch", glitch.jpeg_crunch, (I("quality", "Quality", 8, 1, 60), I("passes", "Passes", 2, 1, 6))),
    "curves": Effect("Curves", "Adjust", adjust.curves,
                     (F("contrast", "S-curve", 0.2, -1, 1), F("shadows", "Shadows", 0, -1, 1), F("darks", "Darks", 0, -1, 1),
                      F("lights", "Lights", 0, -1, 1), F("highlights", "Highlights", 0, -1, 1),
                      F("black", "Black point lift", 0, 0, 0.4), F("white", "White point drop", 0, 0, 0.4),
                      F("red", "Red balance", 0, -1, 1), F("green", "Green balance", 0, -1, 1), F("blue", "Blue balance", 0, -1, 1))),
    "hsl": Effect("HSL", "Adjust", adjust.hsl, _HSL_PARAMS),
    "color_wheels": Effect("Color wheels", "Adjust", adjust.color_wheels,
                           (F("lift_x", "Lift X", 0, -1, 1), F("lift_y", "Lift Y", 0, -1, 1), F("lift", "Lift", 0, -1, 1),
                            F("gamma_x", "Gamma X", 0, -1, 1), F("gamma_y", "Gamma Y", 0, -1, 1), F("gamma", "Gamma", 0, -1, 1),
                            F("gain_x", "Gain X", 0, -1, 1), F("gain_y", "Gain Y", 0, -1, 1), F("gain", "Gain", 0, -1, 1))),
    "clarity": Effect("Clarity", "Adjust", adjust.clarity, (F("amount", "Amount", 0.4, -1, 1), F("radius", "Radius", 24, 4, 120, 1))),
    "sharpen": Effect("Sharpen", "Adjust", adjust.sharpen,
                      (F("amount", "Amount", 0.6, 0, 3), F("radius", "Radius", 1.2, 0.3, 8, 0.1), F("threshold", "Threshold", 0.0, 0, 0.2, 0.005))),
    "dither": Effect("Dither Lab", "Dither", dither.dither,
                     (C("algorithm", "Algorithm", "Atkinson", dither.ALGORITHMS), C("palette", "Palette", "Game Boy", dither.PALETTE_CHOICES),
                      T("custom", "Custom colors", "#000000 #FFFFFF"), I("levels", "Levels per channel", 4, 2, 16),
                      F("pixel_size", "Pixel size", 3, 1, 32, 0.5), F("strength", "Error strength", 1.0, 0, 1.5),
                      F("contrast", "Contrast", 0.1, -0.8, 1.5), F("brightness", "Brightness", 0, -0.5, 0.5),
                      F("gamma", "Gamma", 1.0, 0.3, 3.0), B("mono", "Monochrome first", False))),
}


def catalog_payload() -> dict:
    """JSON-friendly catalog for the frontend."""
    effects = {}
    for key, effect in EFFECTS.items():
        effects[key] = {
            "name": effect.name, "category": effect.category, "handle": effect.handle,
            "params": [{"key": p.key, "label": p.label, "kind": p.kind, "default": p.default, "min": p.minimum,
                        "max": p.maximum, "step": p.step, "options": list(p.options)} for p in effect.params],
        }
    return {"effects": effects, "blend_modes": list(BLEND_MODES), "palettes": {k: list(v) for k, v in dither.PALETTES.items()}}


def _clean_param(param: Param, value: Any) -> Any:
    if param.kind in ("float", "int"):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = float(param.default)
        if not math.isfinite(number):
            number = float(param.default)
        number = min(max(number, param.minimum), param.maximum)
        return int(round(number)) if param.kind == "int" else number
    if param.kind == "choice":
        return value if value in param.options else param.default
    if param.kind == "bool":
        return bool(value) if isinstance(value, (bool, int)) else param.default
    if param.kind == "color":
        text = str(value or "").strip()
        body = text.lstrip("#")
        return f"#{body.upper()}" if len(body) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in body) else param.default
    return str(value if value is not None else param.default)[:200]


def normalize_stack(stack: Any) -> list[dict]:
    """Validate a stack (JSON text or list) against the catalog; unknown entries are dropped."""
    if isinstance(stack, str):
        try:
            stack = json.loads(stack or "[]")
        except json.JSONDecodeError:
            stack = []
    layers = []
    for entry in stack if isinstance(stack, list) else []:
        if not isinstance(entry, dict) or entry.get("type") not in EFFECTS:
            continue
        effect = EFFECTS[entry["type"]]
        raw = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        try:
            opacity = min(max(float(entry.get("opacity", 1.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            opacity = 1.0
        layers.append({
            "type": entry["type"],
            "on": entry.get("on", True) is not False,
            "opacity": opacity if math.isfinite(opacity) else 1.0,
            "blend": entry.get("blend") if entry.get("blend") in BLEND_MODES else "Normal",
            "params": {p.key: _clean_param(p, raw.get(p.key, p.default)) for p in effect.params},
        })
    return layers


def is_identity(layers: list[dict]) -> bool:
    return not any(layer["on"] and layer["opacity"] > 0.0 for layer in layers)


def apply_stack(rgb: torch.Tensor, layers: list[dict], ctx: FxContext) -> torch.Tensor:
    """Run a normalised stack over one HWC sRGB frame."""
    out = rgb[..., :3].to(torch.float32).clamp(0.0, 1.0)
    for layer in layers:
        if not layer["on"] or layer["opacity"] <= 0.0:
            continue
        result = EFFECTS[layer["type"]].run(out, layer["params"], ctx)
        out = composite(out, result.to(out.dtype), layer["blend"], layer["opacity"])
    return out
