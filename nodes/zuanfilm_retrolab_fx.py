"""ComfyUI node: Zuanfilm Retrolab FX, a layered stack of creative effects."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import fx
from ..core.fx.common import FxContext
from ..preview import create_preview_cache, create_video_preview_cache, register_preview_routes


def _processing_device(images: torch.Tensor) -> torch.device:
    if images.device.type == "cuda" or not torch.cuda.is_available():
        return images.device
    return torch.device("cuda")


class ZuanfilmRetroLabFX:
    """Blur, light, retro, print, glitch, adjustment and dither effects as reorderable layers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # The whole layer stack as JSON; the frontend edits it through the FX panel.
                "fx_stack": ("STRING", {"default": "[]", "multiline": False, "tooltip": "Layered effect stack (edited in the node's FX panel)."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Apply the stack only where the mask is white."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "Zuanfilm/RetroLab"

    def execute(self, image, fx_stack="[]", mask=None):
        layers = fx.normalize_stack(fx_stack)
        output = image if fx.is_identity(layers) else self._process(image, layers, mask)
        try:
            register_preview_routes()
            if image.shape[0] > 1:
                return {"ui": {"retrolab_video_preview_id": [create_video_preview_cache(image)]}, "result": (output,)}
            return {"ui": {"retrolab_preview_id": [create_preview_cache(image)]}, "result": (output,)}
        except Exception:
            # Preview failure must never break the actual image-processing node.
            return (output,)

    @staticmethod
    def _process(images: torch.Tensor, layers: list[dict], mask: torch.Tensor | None) -> torch.Tensor:
        device = _processing_device(images)
        output = torch.empty_like(images)
        height, width = images.shape[1:3]
        for index in range(images.shape[0]):
            frame = images[index]
            source = frame[..., :3].to(device=device, dtype=torch.float32)
            result = fx.apply_stack(source, layers, FxContext(frame_index=index))
            if mask is not None:
                weight = mask[min(index, mask.shape[0] - 1)].to(device=device, dtype=torch.float32)
                if weight.shape != (height, width):
                    weight = F.interpolate(weight.view(1, 1, *weight.shape), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
                result = torch.lerp(source, result, weight.clamp(0.0, 1.0).unsqueeze(-1))
            output[index, ..., :3] = result.to(device=images.device, dtype=images.dtype)
            if images.shape[-1] > 3:
                output[index, ..., 3:] = frame[..., 3:]
        return output
