"""ComfyUI node registrations."""

from .zuanfilm_retrolab import ZuanfilmRetroLab, ZuanfilmRetroLabVideo
from .zuanfilm_retrolab_fx import ZuanfilmRetroLabFX

NODE_CLASS_MAPPINGS = {
    "ZuanfilmRetroLab": ZuanfilmRetroLab,
    "ZuanfilmRetroLabVideo": ZuanfilmRetroLabVideo,
    "ZuanfilmRetroLabFX": ZuanfilmRetroLabFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZuanfilmRetroLab": "Zuanfilm Retrolab",
    "ZuanfilmRetroLabVideo": "Zuanfilm Retrolab Video",
    "ZuanfilmRetroLabFX": "Zuanfilm Retrolab FX",
}
