"""Zuanfilm Retrolab — ComfyUI post-decode photographic finishing."""

__version__ = "0.2.34-alpha"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .preview import register_preview_routes

register_preview_routes()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
