from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "zuanfilm_retrolab_test_package"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(PACKAGE_NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

from zuanfilm_retrolab_test_package import NODE_CLASS_MAPPINGS  # noqa: E402
from zuanfilm_retrolab_test_package.core import fx  # noqa: E402
from zuanfilm_retrolab_test_package.core.fx import dither  # noqa: E402
from zuanfilm_retrolab_test_package.core.fx.common import FxContext  # noqa: E402
from zuanfilm_retrolab_test_package.core.fx.glitch import pixel_sort  # noqa: E402


def _image(height=48, width=40, seed=3):
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(0.0, 1.0, height).view(-1, 1, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, -1, 1)
    base = torch.cat((x.expand(height, width, 1), y.expand(height, width, 1), (x * y).expand(height, width, 1)), dim=-1)
    return (base * 0.8 + torch.rand(height, width, 3, generator=generator) * 0.2).clamp(0.0, 1.0)


class FxCatalogTests(unittest.TestCase):
    def test_catalog_defaults_are_valid(self):
        payload = fx.catalog_payload()
        json.dumps(payload)
        self.assertGreaterEqual(len(payload["effects"]), 25)
        for key, effect in payload["effects"].items():
            for param in effect["params"]:
                if param["kind"] in ("float", "int"):
                    self.assertLessEqual(param["min"], param["default"], key)
                    self.assertLessEqual(param["default"], param["max"], key)
                if param["kind"] == "choice":
                    self.assertIn(param["default"], param["options"], key)
            if effect["handle"]:
                keys = {param["key"] for param in effect["params"]}
                self.assertTrue({"cx", "cy"} <= keys, key)

    def test_normalize_stack_clamps_and_drops_untrusted_values(self):
        layers = fx.normalize_stack(json.dumps([
            {"type": "zoom_blur", "opacity": 7, "blend": "Bogus", "params": {"amount": 99, "cx": "nan"}},
            {"type": "rm -rf"},
            {"type": "halftone", "params": {"ink": "red; drop", "mode": "Nope"}},
            "junk",
        ]))
        self.assertEqual([layer["type"] for layer in layers], ["zoom_blur", "halftone"])
        self.assertEqual(layers[0]["opacity"], 1.0)
        self.assertEqual(layers[0]["blend"], "Normal")
        self.assertEqual(layers[0]["params"]["amount"], 0.6)
        self.assertEqual(layers[0]["params"]["cx"], 0.5)
        self.assertEqual(layers[1]["params"]["ink"], "#1B1B1B")
        self.assertEqual(layers[1]["params"]["mode"], "Mono")
        self.assertEqual(fx.normalize_stack("{not json"), [])
        self.assertTrue(fx.is_identity([]))


class FxEffectTests(unittest.TestCase):
    def test_every_effect_is_deterministic_finite_and_in_range(self):
        image = _image()
        for key in fx.EFFECTS:
            with self.subTest(effect=key):
                layers = fx.normalize_stack([{"type": key}])
                first = fx.apply_stack(image, layers, FxContext(frame_index=2))
                second = fx.apply_stack(image, layers, FxContext(frame_index=2))
                self.assertEqual(first.shape, image.shape)
                self.assertTrue(torch.isfinite(first).all())
                self.assertGreaterEqual(float(first.min()), 0.0)
                self.assertLessEqual(float(first.max()), 1.0)
                self.assertTrue(torch.equal(first, second))

    def test_opacity_zero_and_hidden_layers_are_exact_identity(self):
        image = _image()
        stack = [{"type": "crt", "opacity": 0.0}, {"type": "vhs", "on": False}]
        self.assertTrue(torch.equal(fx.apply_stack(image, fx.normalize_stack(stack), FxContext()), image))

    def test_wavefront_error_diffusion_matches_raster_scan(self):
        image = _image(9, 11).to(torch.float64)
        palette = torch.stack([dither.hex_to_rgb(c, torch.device("cpu"), torch.float64) for c in dither.PALETTES["PICO-8"]])
        quantize = dither._quantizer(palette, 4)
        for name, (kernel, divisor) in dither.KERNELS.items():
            with self.subTest(kernel=name):
                expected = image.clone()
                height, width = expected.shape[:2]
                for y in range(height):
                    for x in range(width):
                        value = expected[y, x].clone()
                        chosen = quantize(value.view(1, 3))[0]
                        expected[y, x] = chosen
                        error = (value - chosen) / divisor
                        for dx, dy, weight in kernel:
                            if 0 <= x + dx < width and y + dy < height:
                                expected[y + dy, x + dx] += error * weight
                actual = dither._error_diffusion(image, quantize, kernel, divisor, 1.0)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-9))

    def test_dither_uses_only_palette_colours(self):
        layers = fx.normalize_stack([{"type": "dither", "params": {"algorithm": "Floyd–Steinberg", "palette": "Game Boy", "pixel_size": 1}}])
        out = fx.apply_stack(_image(), layers, FxContext())
        palette = {tuple(round(v * 255) for v in dither.hex_to_rgb(c, torch.device("cpu"), torch.float32).tolist())
                   for c in dither.PALETTES["Game Boy"]}
        colours = {tuple(round(v * 255) for v in pixel) for pixel in out.reshape(-1, 3).tolist()}
        self.assertTrue(colours <= palette)

    def test_pixel_sort_only_reorders_in_range_runs(self):
        column = torch.tensor([0.95, 0.2, 0.7, 0.4, 0.99, 0.6, 0.3]).view(-1, 1, 1).expand(-1, 1, 3).contiguous()
        out = pixel_sort(column, {"low": 0.1, "high": 0.9, "direction": "Vertical", "reverse": False}, FxContext())
        values = out[:, 0, 0].tolist()
        self.assertAlmostEqual(values[0], 0.95)
        self.assertAlmostEqual(values[4], 0.99)
        self.assertEqual([round(v, 2) for v in values[1:4]], [0.2, 0.4, 0.7])
        self.assertEqual([round(v, 2) for v in values[5:]], [0.3, 0.6])


class FxNodeTests(unittest.TestCase):
    def test_node_identity_mask_and_batch_preview(self):
        node = NODE_CLASS_MAPPINGS["ZuanfilmRetroLabFX"]()
        image = _image().unsqueeze(0)
        result = node.execute(image, "[]")
        self.assertTrue(torch.equal(result["result"][0], image))
        self.assertIn("retrolab_preview_id", result["ui"])

        stack = json.dumps([{"type": "gradient_map"}])
        masked = node.execute(image, stack, torch.zeros(1, 48, 40))["result"][0]
        self.assertTrue(torch.allclose(masked, image, atol=1e-6))
        changed = node.execute(image, stack, torch.ones(1, 48, 40))["result"][0]
        self.assertGreater(float((changed - image).abs().mean()), 0.01)

        batch = node.execute(image.expand(3, -1, -1, -1).contiguous(), stack)
        self.assertIn("retrolab_video_preview_id", batch["ui"])
        self.assertEqual(batch["result"][0].shape, (3, 48, 40, 3))


if __name__ == "__main__":
    unittest.main()
