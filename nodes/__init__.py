"""ComfyUI node registrations."""

from .zuanfilm_retrolab import ZuanfilmRetroLab

NODE_CLASS_MAPPINGS = {
    "ZuanfilmRetroLab": ZuanfilmRetroLab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZuanfilmRetroLab": "Zuanfilm Retrolab",
}
