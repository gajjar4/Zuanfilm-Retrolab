from __future__ import annotations

import unittest

import torch

from core.anti_ai import apply_anti_ai
from core.exposure import apply_exposure
from core.bloom import apply_bloom
from core.color import linear_to_srgb, srgb_to_linear
from core.grain import apply_film_grain
from core.blending import BLEND_MODES, normalize_blend_mode
from core.halation import (
    DEFAULT_HALATION_COLOR, _halation_channel_weights, apply_halation,
    build_halation_kernel, normalize_halation_color,
)
from core.light_leak import LEAK_MODES, POSITIONS, apply_light_leak, normalize_hex_color
from core.emulsions import EMULSION_PROFILES, EMULSION_BY_ID, EMULSION_CHOICES, apply_color_emulsion, normalize_emulsion_id, normalize_emulsion_response
from core.pipeline import RetroLabSettings, is_identity, process_batch
from core.sabattier import _solarization_lut, apply_sabattier
from core.soft_focus import apply_soft_focus
from core.vignette import apply_vignette
from core.white_balance import apply_white_balance
from core.noise_texture import NOISE_TEXTURES, NOISE_BLEND_MODES, apply_noise_texture, normalize_noise_texture, normalize_noise_blend_mode


class RetroLabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        h, w = 96, 128
        y = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
        x = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
        base = torch.stack((x, y, 0.25 + 0.5 * x * y), dim=-1)
        self.image = base.clamp(0.0, 1.0)

    def test_identity_bypass_returns_original_tensor(self):
        batch = self.image.unsqueeze(0)
        settings = RetroLabSettings()
        self.assertTrue(is_identity(settings))
        output = process_batch(batch, settings)
        self.assertIs(output, batch)
        self.assertTrue(torch.equal(output, batch))

    def test_default_node_settings_are_exact_identity(self):
        batch = torch.rand((2, 48, 64, 3), dtype=torch.float32)
        output = process_batch(batch, RetroLabSettings())
        self.assertTrue(torch.equal(output, batch))

    def test_batch_preserves_shape_dtype_device_and_alpha(self):
        alpha = torch.rand((2, 96, 128, 1))
        batch = torch.cat((self.image.unsqueeze(0).repeat(2, 1, 1, 1), alpha), dim=-1)
        settings = RetroLabSettings(grain_amount=0.4, leak_seed=77)
        output = process_batch(batch, settings)
        self.assertEqual(output.shape, batch.shape)
        self.assertEqual(output.dtype, batch.dtype)
        self.assertEqual(output.device, batch.device)
        self.assertTrue(torch.equal(output[..., 3:], batch[..., 3:]))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(float((output[..., :3] - batch[..., :3]).abs().mean()), 0.0)
        # Re-running produces the same evolving grain sequence; adjacent frames
        # have distinct grain fields rather than one frozen pattern.
        output_again = process_batch(batch, settings)
        self.assertTrue(torch.equal(output, output_again))
        self.assertFalse(torch.equal(output[0, ..., :3], output[1, ..., :3]))



    def test_exposure_zero_is_exact_identity(self):
        self.assertIs(apply_exposure(self.image, amount=0), self.image)
        self.assertTrue(torch.equal(apply_exposure(self.image, amount=0.0), self.image))
        batch = self.image.unsqueeze(0)
        self.assertTrue(is_identity(RetroLabSettings(exposure=0.0)))
        self.assertIs(process_batch(batch, RetroLabSettings(exposure=0.0)), batch)

    def test_exposure_negative_endpoint_moves_linear_light_to_five_percent(self):
        gray = torch.full((12, 16, 3), 0.5, dtype=torch.float32)
        dark = apply_exposure(gray, amount=-100)
        source_linear = srgb_to_linear(gray)[..., 0]
        result_linear = srgb_to_linear(dark)[..., 0]
        self.assertTrue(torch.allclose(result_linear, source_linear * 0.05, atol=2e-6, rtol=1e-5))
        self.assertTrue(torch.isfinite(dark).all())

    def test_exposure_positive_endpoint_moves_ninety_five_percent_toward_white(self):
        gray = torch.full((12, 16, 3), 0.5, dtype=torch.float32)
        bright = apply_exposure(gray, amount=100)
        source_linear = srgb_to_linear(gray)[..., 0]
        result_linear = srgb_to_linear(bright)[..., 0]
        expected = source_linear + (1.0 - source_linear) * 0.95
        self.assertTrue(torch.allclose(result_linear, expected, atol=2e-6, rtol=1e-5))
        self.assertTrue(torch.isfinite(bright).all())
        self.assertGreater(float(bright.mean()), float(gray.mean()))

    def test_exposure_positive_midrange_uses_25_power_ease_in(self):
        gray = torch.full((12, 16, 3), 0.5, dtype=torch.float32)
        eased = apply_exposure(gray, amount=50)
        source_linear = srgb_to_linear(gray)[..., 0]
        result_linear = srgb_to_linear(eased)[..., 0]
        expected_progress = 0.95 * (0.5 ** 2.5)
        expected = source_linear + (1.0 - source_linear) * expected_progress
        self.assertTrue(torch.allclose(result_linear, expected, atol=2e-6, rtol=1e-5))
        # The eased midpoint must be gentler than both the quadratic and former linear curves.
        former_quadratic = source_linear + (1.0 - source_linear) * (0.5 ** 2 * 0.95)
        former_linear = source_linear + (1.0 - source_linear) * (0.5 * 0.95)
        self.assertTrue(torch.all(result_linear < former_quadratic))
        self.assertTrue(torch.all(result_linear < former_linear))
        self.assertTrue(torch.isfinite(eased).all())

    def test_exposure_is_monotonic_bounded_and_pipeline_integrated(self):
        darker = apply_exposure(self.image, amount=-50)
        neutral = apply_exposure(self.image, amount=0)
        brighter = apply_exposure(self.image, amount=50)
        self.assertTrue(torch.all(darker <= neutral + 1e-7))
        self.assertTrue(torch.all(brighter >= neutral - 1e-7))
        for output in (darker, brighter):
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)
        batch = self.image.unsqueeze(0)
        self.assertFalse(is_identity(RetroLabSettings(exposure=50)))
        processed = process_batch(batch, RetroLabSettings(exposure=50))
        self.assertEqual(processed.shape, batch.shape)
        self.assertFalse(torch.equal(processed, batch))

    def test_white_balance_neutral_is_exact_identity(self):
        self.assertIs(apply_white_balance(self.image), self.image)
        self.assertTrue(torch.equal(apply_white_balance(self.image, temperature=0.0, tint=0.0), self.image))

    def test_white_balance_wheel_axes_and_luminance_normalization(self):
        gray = torch.full((24, 32, 3), 0.5, dtype=torch.float32)
        warm = apply_white_balance(gray, temperature=0.7, tint=0.0)
        cool = apply_white_balance(gray, temperature=-0.7, tint=0.0)
        magenta = apply_white_balance(gray, temperature=0.0, tint=0.7)
        green = apply_white_balance(gray, temperature=0.0, tint=-0.7)
        self.assertGreater(float(warm[..., 0].mean()), float(warm[..., 2].mean()))
        self.assertGreater(float(cool[..., 2].mean()), float(cool[..., 0].mean()))
        self.assertGreater(float(magenta[..., 0].mean()), float(magenta[..., 1].mean()))
        self.assertGreater(float(magenta[..., 2].mean()), float(magenta[..., 1].mean()))
        self.assertGreater(float(green[..., 1].mean()), float(green[..., 0].mean()))
        for balanced in (warm, cool, magenta, green):
            self.assertTrue(torch.isfinite(balanced).all())
            self.assertGreaterEqual(float(balanced.min()), 0.0)
            self.assertLessEqual(float(balanced.max()), 1.0)
            y = 0.2126 * srgb_to_linear(balanced)[..., 0] + 0.7152 * srgb_to_linear(balanced)[..., 1] + 0.0722 * srgb_to_linear(balanced)[..., 2]
            self.assertAlmostEqual(float(y.mean()), float(srgb_to_linear(gray)[..., 0].mean()), places=5)

    def test_white_balance_settings_are_part_of_identity_and_pipeline(self):
        batch = self.image.unsqueeze(0)
        self.assertFalse(is_identity(RetroLabSettings(white_balance_temperature=0.5)))
        warm = process_batch(batch, RetroLabSettings(white_balance_temperature=0.5))
        cool = process_batch(batch, RetroLabSettings(white_balance_temperature=-0.5))
        self.assertFalse(torch.equal(warm, batch))
        self.assertFalse(torch.equal(cool, batch))
        self.assertFalse(torch.equal(warm, cool))

    def test_emulsion_atlas_has_fifty_unique_profiles_in_four_families(self):
        self.assertEqual(len(EMULSION_PROFILES), 50)
        self.assertEqual(len(EMULSION_BY_ID), 50)
        self.assertEqual(len(EMULSION_CHOICES), 51)
        self.assertEqual(EMULSION_CHOICES[0], "neutral")
        self.assertEqual(
            {family: sum(1 for profile in EMULSION_PROFILES if profile.family == family)
             for family in ("Cinema", "Instant", "Slide", "35mm")},
            {"Cinema": 10, "Instant": 10, "Slide": 10, "35mm": 20},
        )
        self.assertEqual(normalize_emulsion_id("not-a-profile"), "neutral")
        self.assertEqual(normalize_emulsion_id("velvia_50"), "velvia_50")

    def test_neutral_emulsion_is_an_exact_identity(self):
        self.assertIs(apply_color_emulsion(self.image, "neutral"), self.image)
        self.assertTrue(torch.equal(apply_color_emulsion(self.image, "invalid"), self.image))
        batch = self.image.unsqueeze(0)
        output = process_batch(batch, RetroLabSettings(color_emulsion="neutral"))
        self.assertIs(output, batch)

    def test_all_fifty_emulsion_profiles_render_finite_bounded_and_distinct(self):
        outputs = []
        for profile in EMULSION_PROFILES:
            rendered = apply_color_emulsion(self.image, profile.id)
            self.assertEqual(rendered.shape, self.image.shape, profile.id)
            self.assertEqual(rendered.dtype, self.image.dtype, profile.id)
            self.assertEqual(rendered.device, self.image.device, profile.id)
            self.assertTrue(torch.isfinite(rendered).all(), profile.id)
            self.assertGreaterEqual(float(rendered.min()), 0.0, profile.id)
            self.assertLessEqual(float(rendered.max()), 1.0, profile.id)
            self.assertGreater(float((rendered - self.image).abs().mean()), 1e-5, profile.id)
            expressive = apply_color_emulsion(self.image, profile.id, response=3.0)
            self.assertTrue(torch.isfinite(expressive).all(), f"{profile.id} at 3.0x")
            self.assertGreaterEqual(float(expressive.min()), 0.0, f"{profile.id} at 3.0x")
            self.assertLessEqual(float(expressive.max()), 1.0, f"{profile.id} at 3.0x")
            outputs.append(rendered.flatten())

        # Every named profile should be a recognizable, distinct creative
        # response on the same diagnostic image, not a duplicate preset.
        stacked = torch.stack(outputs)
        distances = torch.cdist(stacked, stacked)
        distances.fill_diagonal_(float('inf'))
        self.assertGreater(float(distances.min()), 1e-4)

    def test_emulsion_response_one_preserves_original_profile_output_exactly(self):
        for profile_id in ("portra_400", "velvia_50", "polaroid_timezero", "vision3_500t"):
            original = apply_color_emulsion(self.image, profile_id)
            response_one = apply_color_emulsion(self.image, profile_id, response=1.0)
            self.assertTrue(torch.equal(original, response_one), profile_id)

    def test_emulsion_response_scales_full_profile_deviation_in_linear_light(self):
        baseline = self.image
        source_linear = srgb_to_linear(baseline)
        standard = apply_color_emulsion(baseline, "portra_400", response=1.0)
        profile_linear = srgb_to_linear(standard)
        for response in (0.5, 1.5, 3.0):
            rendered = apply_color_emulsion(baseline, "portra_400", response=response)
            actual_linear = srgb_to_linear(rendered)
            expected_linear = (source_linear + (profile_linear - source_linear) * response).clamp(0.0, 1.0)
            self.assertTrue(torch.allclose(actual_linear, expected_linear, atol=2e-6, rtol=1e-5))
            self.assertTrue(torch.isfinite(rendered).all())
            self.assertGreaterEqual(float(rendered.min()), 0.0)
            self.assertLessEqual(float(rendered.max()), 1.0)

    def test_emulsion_response_is_clamped_and_neutral_remains_exact_identity(self):
        self.assertEqual(normalize_emulsion_response(-5.0), 0.5)
        self.assertEqual(normalize_emulsion_response(9.0), 3.0)
        self.assertEqual(normalize_emulsion_response(float("nan")), 1.0)
        self.assertEqual(normalize_emulsion_response("invalid"), 1.0)
        self.assertIs(apply_color_emulsion(self.image, "neutral", response=0.5), self.image)
        self.assertIs(apply_color_emulsion(self.image, "neutral", response=3.0), self.image)

    def test_emulsion_response_is_applied_through_pipeline_settings(self):
        batch = self.image.unsqueeze(0)
        half = process_batch(batch, RetroLabSettings(color_emulsion="portra_400", emulsion_response=0.5))
        full = process_batch(batch, RetroLabSettings(color_emulsion="portra_400", emulsion_response=1.0))
        triple = process_batch(batch, RetroLabSettings(color_emulsion="portra_400", emulsion_response=3.0))
        self.assertFalse(torch.equal(half, full))
        self.assertFalse(torch.equal(full, triple))
        self.assertTrue(torch.equal(full, process_batch(batch, RetroLabSettings(color_emulsion="portra_400"))))

    def test_emulsion_profiles_produce_distinct_finite_color_responses(self):
        neutral = self.image
        portra = apply_color_emulsion(neutral, "portra_400")
        velvia = apply_color_emulsion(neutral, "velvia_50")
        timezero = apply_color_emulsion(neutral, "polaroid_timezero")
        vision3 = apply_color_emulsion(neutral, "vision3_500t")
        for output in (portra, velvia, timezero, vision3):
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)
            self.assertGreater(float((output - neutral).abs().mean()), 1e-4)
        self.assertFalse(torch.allclose(portra, velvia, atol=1e-4, rtol=0.0))
        self.assertFalse(torch.allclose(timezero, vision3, atol=1e-4, rtol=0.0))
        self.assertTrue(torch.equal(velvia, apply_color_emulsion(neutral, "velvia_50")))

    def test_emulsion_is_independent_from_film_grain_and_leak_seed(self):
        settings_a = RetroLabSettings(color_emulsion="ektar_100", grain_amount=0.0, leak_seed=27)
        settings_b = RetroLabSettings(color_emulsion="ektar_100", grain_amount=0.0, leak_seed=991)
        output_a = process_batch(self.image.unsqueeze(0), settings_a)
        output_b = process_batch(self.image.unsqueeze(0), settings_b)
        self.assertTrue(torch.equal(output_a, output_b))
        with_grain = process_batch(self.image.unsqueeze(0), RetroLabSettings(color_emulsion="ektar_100", grain_amount=0.7))
        self.assertFalse(torch.equal(output_a, with_grain))

    def test_leak_seed_does_not_change_film_grain_pattern(self):
        settings_a = RetroLabSettings(grain_amount=0.8, leak_seed=27)
        settings_b = RetroLabSettings(grain_amount=0.8, leak_seed=901)
        output_a = process_batch(self.image.unsqueeze(0), settings_a)
        output_b = process_batch(self.image.unsqueeze(0), settings_b)
        self.assertTrue(torch.equal(output_a, output_b))

    def test_grain_is_deterministic_and_seed_sensitive(self):
        a = apply_film_grain(self.image, amount=0.8, grain_size=1.4, color_grain=0.35, seed=42)
        b = apply_film_grain(self.image, amount=0.8, grain_size=1.4, color_grain=0.35, seed=42)
        c = apply_film_grain(self.image, amount=0.8, grain_size=1.4, color_grain=0.35, seed=43)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(float((a - self.image).abs().mean()), 0.0)

    def test_film_grain_blend_modes_share_texture_compositing_contract(self):
        source = torch.full((96, 128, 3), 0.48, dtype=torch.float32)
        outputs = []
        for mode in BLEND_MODES:
            first = apply_film_grain(source, amount=1.4, grain_size=0.9, color_grain=0.35, seed=31, blend_mode=mode)
            second = apply_film_grain(source, amount=1.4, grain_size=0.9, color_grain=0.35, seed=31, blend_mode=mode)
            self.assertTrue(torch.equal(first, second), mode)
            self.assertTrue(torch.isfinite(first).all(), mode)
            self.assertGreaterEqual(float(first.min()), 0.0, mode)
            self.assertLessEqual(float(first.max()), 1.0, mode)
            outputs.append(first.flatten())
        distances = torch.cdist(torch.stack(outputs), torch.stack(outputs))
        distances.fill_diagonal_(float("inf"))
        self.assertGreater(float(distances.min()), 1e-8)

    def test_film_grain_additive_remains_legacy_default_and_invalid_mode_fallback(self):
        default = apply_film_grain(self.image, amount=0.8, grain_size=1.2, color_grain=0.2, seed=17)
        explicit = apply_film_grain(self.image, amount=0.8, grain_size=1.2, color_grain=0.2, seed=17, blend_mode="Additive")
        invalid = apply_film_grain(self.image, amount=0.8, grain_size=1.2, color_grain=0.2, seed=17, blend_mode="invalid")
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(default, invalid))
        self.assertEqual(normalize_blend_mode("Screen"), "Screen")
        self.assertEqual(normalize_blend_mode("invalid"), "Additive")

    def test_procedural_texture_expanded_amount_doubles_previous_maximum(self):
        source = torch.full((96, 128, 3), 0.5, dtype=torch.float32)
        at_one = apply_noise_texture(source, texture="Gaussian", amount=1.0, seed=81, scale=1.0)
        at_two = apply_noise_texture(source, texture="Gaussian", amount=2.0, seed=81, scale=1.0)
        above_two = apply_noise_texture(source, texture="Gaussian", amount=9.0, seed=81, scale=1.0)
        self.assertGreater(float((at_two - source).abs().mean()), float((at_one - source).abs().mean()))
        self.assertTrue(torch.equal(at_two, above_two))
        self.assertTrue(torch.isfinite(at_two).all())

    def test_film_grain_blend_mode_is_applied_through_pipeline_settings(self):
        base = torch.full((1, 64, 80, 3), 0.48, dtype=torch.float32)
        additive = process_batch(base, RetroLabSettings(grain_amount=1.2, grain_blend_mode="Additive"))
        multiply = process_batch(base, RetroLabSettings(grain_amount=1.2, grain_blend_mode="Multiply"))
        self.assertFalse(torch.equal(additive, multiply))

    def test_fine_grain_size_floor_produces_pixel_scale_grain(self):
        image = torch.full((128, 160, 3), 0.5, dtype=torch.float32)
        fine = apply_film_grain(image, amount=1.0, grain_size=0.1, color_grain=0.0, seed=123, reference_dimensions=(2048, 2048))
        coarse = apply_film_grain(image, amount=1.0, grain_size=0.7, color_grain=0.0, seed=123, reference_dimensions=(2048, 2048))
        fine_noise = (fine[..., 0] - image[..., 0])
        coarse_noise = (coarse[..., 0] - image[..., 0])
        fine_corr = torch.mean(fine_noise[:, :-1] * fine_noise[:, 1:]) / fine_noise.square().mean().clamp_min(1e-8)
        coarse_corr = torch.mean(coarse_noise[:, :-1] * coarse_noise[:, 1:]) / coarse_noise.square().mean().clamp_min(1e-8)
        self.assertLess(float(fine_corr), float(coarse_corr))

    def test_grain_amount_three_is_stronger_than_one_and_clamps(self):
        midtone = torch.full((96, 128, 3), 0.5, dtype=torch.float32)
        one = apply_film_grain(midtone, amount=1.0, grain_size=1.2, color_grain=0.25, seed=91)
        three = apply_film_grain(midtone, amount=3.0, grain_size=1.2, color_grain=0.25, seed=91)
        overflow = apply_film_grain(midtone, amount=9.0, grain_size=1.2, color_grain=0.25, seed=91)
        self.assertGreater(float((three - midtone).abs().std()), float((one - midtone).abs().std()))
        self.assertTrue(torch.equal(three, overflow))
        self.assertTrue(torch.isfinite(three).all())

    def test_halation_kernel_is_normalized_and_annular(self):
        kernel = build_halation_kernel(8.0, 40.0, 0.65, device=torch.device("cpu"))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)
        centre = float(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2])
        radial_max = float(kernel.max())
        self.assertGreater(radial_max, centre)


    def test_halation_default_color_preserves_original_channel_response(self):
        weights = _halation_channel_weights(DEFAULT_HALATION_COLOR, torch.device("cpu"))
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 0.32, 0.10]), atol=1e-7, rtol=0.0))
        self.assertEqual(normalize_halation_color("#ff9959"), DEFAULT_HALATION_COLOR)
        self.assertEqual(normalize_halation_color("invalid"), DEFAULT_HALATION_COLOR)

    def test_halation_color_picker_changes_emission_spectrum(self):
        red = _halation_channel_weights("#FF0000", torch.device("cpu"))
        green = _halation_channel_weights("#00FF00", torch.device("cpu"))
        neutral = _halation_channel_weights("#FFFFFF", torch.device("cpu"))
        black = _halation_channel_weights("#000000", torch.device("cpu"))
        self.assertEqual(red.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(green.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(neutral.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(black.tolist(), [0.0, 0.0, 0.0])
        image = torch.zeros((96, 128, 3), dtype=torch.float32)
        image[48, 64] = 1.0
        green_halo = apply_halation(image, amount=1.0, ring_radius=8.0, color="#00FF00")
        sample = green_halo[48, 74]
        self.assertEqual(float(sample[0]), 0.0)
        self.assertGreater(float(sample[1]), 0.0)
        self.assertEqual(float(sample[2]), 0.0)

    def test_halation_has_full_response_and_doubled_amount_headroom(self):
        image = torch.zeros((96, 128, 3), dtype=torch.float32)
        image[48, 64] = 1.0
        amount_one = apply_halation(image, amount=1.0, ring_radius=8.0)
        amount_two = apply_halation(image, amount=2.0, ring_radius=8.0)
        self.assertTrue(torch.isfinite(amount_two).all())
        delta_one = (srgb_to_linear(amount_one) - srgb_to_linear(image)).sum()
        delta_two = (srgb_to_linear(amount_two) - srgb_to_linear(image)).sum()
        self.assertGreater(float(delta_one), 0.0)
        self.assertGreater(float(delta_two), float(delta_one))
        # The halo should be red-dominant away from the white source.
        sample = amount_two[48, 74]
        self.assertGreaterEqual(float(sample[0]), float(sample[1]))
        self.assertGreaterEqual(float(sample[1]), float(sample[2]))

    def test_bloom_is_deterministic_finite_and_localized_to_highlights(self):
        image = torch.zeros((96, 128, 3), dtype=torch.float32)
        image[48, 64] = 1.0
        out1 = apply_bloom(image, amount=0.8, threshold=0.65, radius=8.0)
        out2 = apply_bloom(image, amount=0.8, threshold=0.65, radius=8.0)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.isfinite(out1).all())
        self.assertGreater(float(out1[48, 70].max()), 0.0)
        out_at_max = apply_bloom(image, amount=2.0, threshold=0.65, radius=8.0)
        self.assertGreater(
            float((srgb_to_linear(out_at_max) - srgb_to_linear(image)).sum()),
            float((srgb_to_linear(out1) - srgb_to_linear(image)).sum()),
        )

    def test_bloom_supports_three_point_amount_maximum(self):
        image = torch.zeros((96, 128, 3), dtype=torch.float32)
        image[48, 64] = 1.0
        amount_two = apply_bloom(image, amount=2.0, threshold=0.0, radius=8.0)
        amount_three = apply_bloom(image, amount=3.0, threshold=0.0, radius=8.0)
        amount_overflow = apply_bloom(image, amount=9.0, threshold=0.0, radius=8.0)
        delta_two = (srgb_to_linear(amount_two) - srgb_to_linear(image)).sum()
        delta_three = (srgb_to_linear(amount_three) - srgb_to_linear(image)).sum()
        self.assertTrue(torch.isfinite(amount_three).all())
        self.assertGreater(float(delta_three), float(delta_two))
        self.assertTrue(torch.equal(amount_three, amount_overflow))

    def test_zero_bloom_threshold_does_not_create_light_from_black(self):
        black = torch.zeros((24, 32, 3), dtype=torch.float32)
        output = apply_bloom(black, amount=3.0, threshold=0.0, radius=8.0)
        self.assertTrue(torch.equal(output, black))

    def test_sabattier_reverses_highlights_and_preserves_hue(self):
        x = torch.linspace(0.0, 1.0, 256)
        curve = _solarization_lut(x, beta=1.4, shield=2.0)
        self.assertGreater(int(torch.argmax(curve)), 0)
        self.assertLess(int(torch.argmax(curve)), 255)
        self.assertLess(float(curve[-1]), float(curve.max()))

        rgb = torch.stack((x, x * 0.6, x * 0.25), dim=-1).view(1, 256, 3)
        out = apply_sabattier(rgb, solarization=0.6, highlight_confinement=0.4, mackie_lines=0.0)
        self.assertTrue(torch.isfinite(out).all())
        # All channels are scaled by one luminance ratio (up to clipping).
        lin_in = srgb_to_linear(rgb)
        lin_out = srgb_to_linear(out)
        ratios = lin_out[..., 0] / lin_in[..., 0].clamp_min(1e-6)
        lin_out = srgb_to_linear(out)
        channel_ratios = lin_out / lin_in.clamp_min(1e-6)
        # At every non-black pixel, the RGB channels receive the same ratio.
        valid = lin_in[..., 0] > 1e-3
        spread = channel_ratios[valid].amax(dim=-1) - channel_ratios[valid].amin(dim=-1)
        self.assertLess(float(spread.max()), 1e-4)

    def test_mackie_only_mode_changes_edges(self):
        image = torch.zeros((96, 128, 3), dtype=torch.float32)
        image[:, 64:, :] = 0.8
        out = apply_sabattier(image, solarization=0.0, highlight_confinement=0.5, mackie_lines=0.8)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(float((out - image).abs().sum()), 0.0)

    def test_light_leak_modes_are_deterministic_and_seeded(self):
        for character in LEAK_MODES:
            a = apply_light_leak(self.image, mode=character, position="Top Left", exposure=0.8, seed=19)
            b = apply_light_leak(self.image, mode=character, position="Top Left", exposure=0.8, seed=19)
            self.assertTrue(torch.equal(a, b), character)
            self.assertTrue(torch.isfinite(a).all(), character)
            self.assertGreater(float((a - self.image).abs().mean()), 0.0, character)

    def test_light_leak_seed_changes_organic_pattern(self):
        a = apply_light_leak(self.image, mode="Soft Edge", position="Left", exposure=0.9, seed=1)
        b = apply_light_leak(self.image, mode="Soft Edge", position="Left", exposure=0.9, seed=2)
        self.assertFalse(torch.equal(a, b))

    def test_light_leak_red_shift_increases_with_path_length(self):
        black = torch.zeros((128, 160, 3), dtype=torch.float32)
        edge = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=0.8, seed=33)
        edge_linear = srgb_to_linear(edge)
        near = edge_linear[0, 80, 0] / edge_linear[0, 80, 2].clamp_min(1e-6)
        far = edge_linear[80, 80, 0] / edge_linear[80, 80, 2].clamp_min(1e-6)
        self.assertGreater(float(far), float(near))

    def test_light_leak_exposure_range_supports_tenfold_strength(self):
        # A dark incident color avoids output clipping, so the exposure ratio
        # remains an exact test of the public control range and linear mapping.
        black = torch.zeros((128, 160, 3), dtype=torch.float32)
        at_one = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=1.0, seed=42, color="#303030")
        at_ten = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=10.0, seed=42, color="#303030")
        above_ten = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=100.0, seed=42, color="#303030")
        energy_one = srgb_to_linear(at_one).sum()
        energy_ten = srgb_to_linear(at_ten).sum()
        self.assertGreater(float(energy_one), 0.0)
        self.assertAlmostEqual(float(energy_ten / energy_one), 10.0, places=4)
        self.assertTrue(torch.equal(at_ten, above_ten), "Engine must clamp exposure at the public maximum of 10")

    def test_all_five_leak_modes_are_supported_and_repeatable(self):
        black = torch.zeros((160, 224, 3), dtype=torch.float32)
        self.assertEqual(len(LEAK_MODES), 5)
        self.assertEqual(
            LEAK_MODES,
            ("Soft Edge", "First-Frame Burn", "120 Fat Roll", "Light Piping", "Holga 135"),
        )
        outputs = {}
        for character in LEAK_MODES:
            a = apply_light_leak(black, mode=character, position="Top", exposure=1.0, seed=2468)
            b = apply_light_leak(black, mode=character, position="Top", exposure=1.0, seed=2468)
            self.assertTrue(torch.equal(a, b), character)
            self.assertTrue(torch.isfinite(a).all(), character)
            self.assertGreater(float(a.sum()), 0.0, character)
            outputs[character] = a
        # Each procedural geometry should produce a recognizably different field.
        for first, second in (("First-Frame Burn", "120 Fat Roll"), ("Light Piping", "Holga 135"), ("Soft Edge", "120 Fat Roll")):
            self.assertFalse(torch.equal(outputs[first], outputs[second]))

    def test_all_five_leak_modes_use_the_full_exposure_ten_range(self):
        black = torch.zeros((128, 192, 3), dtype=torch.float32)
        for character in LEAK_MODES:
            with self.subTest(mode=character):
                at_two = apply_light_leak(black, mode=character, position="Top", exposure=2.0, seed=31415, color="#303030")
                at_ten = apply_light_leak(black, mode=character, position="Top", exposure=10.0, seed=31415, color="#303030")
                energy_two = srgb_to_linear(at_two).sum()
                energy_ten = srgb_to_linear(at_ten).sum()
                self.assertGreater(float(energy_two), 0.0)
                self.assertTrue(torch.isfinite(at_ten).all())
                if character == "First-Frame Burn":
                    # Its near-white hot core reaches display clipping first.
                    self.assertGreater(float(energy_ten), float(energy_two))
                else:
                    self.assertAlmostEqual(float(energy_ten / energy_two), 5.0, places=4)

    def test_leak_radius_one_preserves_previous_default_footprint(self):
        implicit = apply_light_leak(
            self.image, mode="120 Fat Roll", position="Top", exposure=1.7, seed=712, color="#E58A42"
        )
        explicit = apply_light_leak(
            self.image, mode="120 Fat Roll", position="Top", exposure=1.7, seed=712, color="#E58A42",
            influence_radius=1.0,
        )
        self.assertTrue(torch.equal(implicit, explicit))

    def test_leak_radius_expands_effect_toward_opposite_border_for_all_characters(self):
        # Use a dark substrate and dark incident color to keep the full response
        # away from display clipping while measuring the far side of the frame.
        black = torch.zeros((192, 256, 3), dtype=torch.float32)
        far_band = slice(160, 192)
        for character in LEAK_MODES:
            with self.subTest(mode=character):
                compact = apply_light_leak(
                    black, mode=character, position="Top", exposure=1.0, seed=2468,
                    color="#303030", influence_radius=1.0,
                )
                expanded = apply_light_leak(
                    black, mode=character, position="Top", exposure=1.0, seed=2468,
                    color="#303030", influence_radius=10.0,
                )
                compact_far = srgb_to_linear(compact[far_band]).mean()
                expanded_far = srgb_to_linear(expanded[far_band]).mean()
                self.assertGreater(float(expanded_far), float(compact_far), character)
                self.assertTrue(torch.isfinite(expanded).all())

    def test_leak_radius_is_deterministic_and_clamped_to_ten(self):
        black = torch.zeros((96, 128, 3), dtype=torch.float32)
        a = apply_light_leak(black, mode="Soft Edge", position="Right", exposure=1.0, seed=13, influence_radius=10.0)
        b = apply_light_leak(black, mode="Soft Edge", position="Right", exposure=1.0, seed=13, influence_radius=10.0)
        over = apply_light_leak(black, mode="Soft Edge", position="Right", exposure=1.0, seed=13, influence_radius=99.0)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, over))

    def test_leak_color_picker_changes_incident_spectrum(self):
        black = torch.zeros((96, 128, 3), dtype=torch.float32)
        amber = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=1.0, seed=42, color="#E58A42")
        cyan = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=1.0, seed=42, color="#42D7E5")
        self.assertFalse(torch.equal(amber, cyan))
        amber_linear = srgb_to_linear(amber)
        cyan_linear = srgb_to_linear(cyan)
        self.assertGreater(float(amber_linear[0, 64, 0]), float(amber_linear[0, 64, 2]))
        self.assertGreater(float(cyan_linear[0, 64, 2]), float(cyan_linear[0, 64, 0]))

    def test_leak_color_hex_normalization_is_safe(self):
        self.assertEqual(normalize_hex_color("#e58a42"), "#E58A42")
        self.assertEqual(normalize_hex_color("abc"), "#AABBCC")
        self.assertEqual(normalize_hex_color("not-a-color"), "#C2AF8E")
        self.assertEqual(normalize_hex_color(None), "#C2AF8E")

    def test_first_frame_burn_is_stronger_than_soft_edge_at_same_exposure(self):
        black = torch.zeros((128, 192, 3), dtype=torch.float32)
        soft = apply_light_leak(black, mode="Soft Edge", position="Top", exposure=1.0, seed=22)
        burn = apply_light_leak(black, mode="First-Frame Burn", position="Top", exposure=1.0, seed=22)
        self.assertGreater(float(srgb_to_linear(burn).sum()), float(srgb_to_linear(soft).sum()))

    def test_first_frame_burn_has_near_white_entry_core(self):
        black = torch.zeros((128, 192, 3), dtype=torch.float32)
        burn = apply_light_leak(black, mode="First-Frame Burn", position="Top", exposure=5.0, seed=42, color="#E58A42")
        linear = srgb_to_linear(burn)
        index = linear.mean(dim=-1).argmax()
        y = int(index // linear.shape[1])
        x = int(index % linear.shape[1])
        core = linear[y, x]
        self.assertGreater(float(core.min()), 0.80)
        self.assertLess(float(core.max() - core.min()), 0.18)

    def test_legacy_135_sprocket_name_maps_to_holga_135(self):
        old_name = apply_light_leak(self.image, mode="135 Sprocket", position="Bottom", exposure=0.8, seed=19)
        new_name = apply_light_leak(self.image, mode="Holga 135", position="Bottom", exposure=0.8, seed=19)
        self.assertTrue(torch.equal(old_name, new_name))

    def test_legacy_removed_characters_fall_back_to_soft_edge(self):
        for legacy_character in ("Pinhole", "Mechanical Marks"):
            with self.subTest(mode=legacy_character):
                a = apply_light_leak(self.image, mode=legacy_character, position="Top Left", exposure=0.8, seed=19)
                b = apply_light_leak(self.image, mode="Soft Edge", position="Top Left", exposure=0.8, seed=19)
                self.assertTrue(torch.equal(a, b))

    def test_all_supported_positions(self):
        for position in POSITIONS:
            output = apply_light_leak(self.image, mode="Soft Edge", position=position, exposure=0.5, seed=42)
            self.assertEqual(output.shape, self.image.shape)
            self.assertTrue(torch.isfinite(output).all())


    def test_vignette_preserves_center_and_reduces_corner_exposure(self):
        image = torch.full((129, 161, 3), 0.6, dtype=torch.float32)
        output = apply_vignette(image, amount=1.0, feather=0.5)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[64, 80], image[64, 80]))
        self.assertLess(float(output[0, 0].mean()), float(image[0, 0].mean()))
        self.assertGreater(float(output[0, 80].mean()), float(output[0, 0].mean()))
        self.assertTrue(torch.allclose(output[0, 0], output[0, -1], atol=1e-6))
        self.assertTrue(torch.allclose(output[-1, 0], output[0, 0], atol=1e-6))

    def test_vignette_amount_two_reaches_three_stops_and_clamps(self):
        image = torch.ones((21, 21, 3), dtype=torch.float32)
        one = apply_vignette(image, amount=1.0, feather=1.0)
        two = apply_vignette(image, amount=2.0, feather=1.0)
        overflow = apply_vignette(image, amount=9.0, feather=1.0)
        self.assertAlmostEqual(float(two[0, 0, 0]), float(0.38857286), places=4)
        self.assertLess(float(two[0, 0, 0]), float(one[0, 0, 0]))
        self.assertTrue(torch.equal(two, overflow))
        self.assertTrue(torch.allclose(two[10, 10], torch.ones(3), atol=1e-6))

    def test_vignette_feather_controls_transition_and_zero_is_identity(self):
        image = torch.full((129, 161, 3), 0.6, dtype=torch.float32)
        self.assertIs(apply_vignette(image, amount=0.0, feather=0.5), image)
        confined = apply_vignette(image, amount=1.0, feather=0.0)
        broad = apply_vignette(image, amount=1.0, feather=1.0)
        # A mid-edge point is darkened more when the transition is feathered
        # farther inward instead of being confined to the outer corners.
        self.assertLess(float(broad[0, 80, 0]), float(confined[0, 80, 0]))

    def test_soft_focus_preserves_center_and_blurs_corners(self):
        yy, xx = torch.meshgrid(torch.arange(129), torch.arange(129), indexing="ij")
        checker = ((xx + yy) % 2).to(torch.float32) * 0.8 + 0.1
        image = checker.unsqueeze(-1).repeat(1, 1, 3)
        output = apply_soft_focus(image, amount=1.0, radius=24.0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output[64, 64], image[64, 64]))
        source_corner_contrast = float((image[0, 0] - image[0, 1]).abs().mean())
        output_corner_contrast = float((output[0, 0] - output[0, 1]).abs().mean())
        self.assertLess(output_corner_contrast, source_corner_contrast)
        # The center patch lies inside the protected sharp area.
        self.assertTrue(torch.equal(output[58:71, 58:71], image[58:71, 58:71]))

    def test_soft_focus_amount_and_radius_are_deterministic_and_clamped(self):
        image = torch.rand((96, 128, 3), dtype=torch.float32)
        weak = apply_soft_focus(image, amount=0.35, radius=24.0)
        strong = apply_soft_focus(image, amount=1.0, radius=24.0)
        wide = apply_soft_focus(image, amount=1.0, radius=48.0)
        max_output = apply_soft_focus(image, amount=1.0, radius=80.0)
        overflow = apply_soft_focus(image, amount=10.0, radius=100.0)
        self.assertTrue(torch.isfinite(strong).all())
        self.assertGreater(float((strong - image).abs().mean()), float((weak - image).abs().mean()))
        self.assertGreater(float((wide - image).abs().mean()), float((strong - image).abs().mean()))
        self.assertTrue(torch.equal(max_output, overflow))
        self.assertIs(apply_soft_focus(image, amount=0.0, radius=24.0), image)

    def test_pipeline_combines_soft_focus_and_vignette_and_identity_bypass(self):
        image = torch.rand((1, 96, 128, 4), dtype=torch.float32)
        default = process_batch(image, RetroLabSettings())
        self.assertIs(default, image)
        settings = RetroLabSettings(
            soft_focus_amount=0.7, soft_focus_radius=28.0,
            vignette_amount=0.8, vignette_feather=0.6,
        )
        output = process_batch(image, settings)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, image.dtype)
        self.assertEqual(output.device, image.device)
        self.assertTrue(torch.equal(output[..., 3:], image[..., 3:]))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(float((output[..., :3] - image[..., :3]).abs().mean()), 0.0)


    def test_anti_ai_zero_is_exact_identity_and_full_range_is_bounded(self):
        self.assertIs(apply_anti_ai(self.image, amount=0.0), self.image)
        self.assertTrue(torch.equal(apply_anti_ai(self.image, amount=0.0), self.image))
        corrected = apply_anti_ai(self.image, amount=100.0)
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertGreaterEqual(float(corrected.min()), 0.0)
        self.assertLessEqual(float(corrected.max()), 1.0)
        self.assertFalse(torch.equal(corrected, self.image))
        self.assertTrue(torch.equal(corrected, apply_anti_ai(self.image, amount=100.0)))
        self.assertTrue(torch.equal(apply_anti_ai(self.image, amount=150.0), corrected))

    def test_anti_ai_restrains_red_orange_and_softens_tonal_extremes(self):
        red_orange = torch.tensor([0.95, 0.24, 0.08], dtype=torch.float32).view(1, 1, 3).expand(24, 32, 3).clone()
        corrected_color = apply_anti_ai(red_orange, amount=100.0)
        source_chroma = float(red_orange.amax(dim=-1).mean() - red_orange.amin(dim=-1).mean())
        corrected_chroma = float(corrected_color.amax(dim=-1).mean() - corrected_color.amin(dim=-1).mean())
        self.assertLess(corrected_chroma, source_chroma)

        ramp = torch.linspace(0.0, 1.0, 256, dtype=torch.float32).view(1, 256, 1).expand(8, 256, 3).clone()
        softened = apply_anti_ai(ramp, amount=100.0)
        self.assertGreater(float(softened[:, 8, 0].mean()), float(ramp[:, 8, 0].mean()),
                           "The lower toe should be gently lifted")
        self.assertGreater(float(softened[:, 118, 0].mean()), float(ramp[:, 118, 0].mean()),
                           "The midtone region should receive a restrained lift")
        self.assertLess(float(softened[:, -1, 0].mean()), float(ramp[:, -1, 0].mean()),
                        "The highlight shoulder should roll off softly")

    def test_procedural_noise_catalog_is_deterministic_seeded_and_independent(self):
        base = torch.full((64, 80, 3), 0.5, dtype=torch.float32)
        self.assertEqual(len(NOISE_TEXTURES), 12)
        self.assertNotIn("Gabor Texture", NOISE_TEXTURES)
        self.assertEqual(normalize_noise_texture("Gabor Texture"), "Emulsion Mottle")
        self.assertEqual(normalize_noise_texture("not-a-noise"), "Gaussian")
        self.assertIs(apply_noise_texture(base, texture="Gaussian", amount=0.0, seed=73), base)

        outputs = []
        for texture in NOISE_TEXTURES:
            rendered = apply_noise_texture(base, texture=texture, amount=0.8, seed=73)
            repeated = apply_noise_texture(base, texture=texture, amount=0.8, seed=73)
            different_seed = apply_noise_texture(base, texture=texture, amount=0.8, seed=74)
            self.assertEqual(rendered.shape, base.shape, texture)
            self.assertTrue(torch.isfinite(rendered).all(), texture)
            self.assertGreaterEqual(float(rendered.min()), 0.0, texture)
            self.assertLessEqual(float(rendered.max()), 1.0, texture)
            self.assertTrue(torch.equal(rendered, repeated), texture)
            self.assertFalse(torch.equal(rendered, different_seed), texture)
            outputs.append(rendered)
        signatures = {
            tuple(round(float(output[y, x, 0]), 6) for y, x in ((0, 0), (2, 4), (10, 20), (20, 40), (33, 12), (63, 79)))
            for output in outputs
        }
        self.assertEqual(len(signatures), len(NOISE_TEXTURES), "The noise families should produce distinct deterministic patterns")

    def test_paper_foxing_uses_warm_discolouration_not_only_monochrome_grain(self):
        base = torch.full((128, 160, 3), 0.5, dtype=torch.float32)
        aged = apply_noise_texture(base, texture="Paper Foxing", amount=0.9, seed=23)
        self.assertGreater(float((aged[..., 0] - aged[..., 2]).mean()), 0.002)
        self.assertTrue(torch.isfinite(aged).all())
        self.assertGreaterEqual(float(aged.min()), 0.0)
        self.assertLessEqual(float(aged.max()), 1.0)

    def test_procedural_texture_scale_changes_feature_structure_and_is_bounded(self):
        base = torch.full((128, 160, 3), 0.5, dtype=torch.float32)
        fine = apply_noise_texture(base, texture="Emulsion Mottle", amount=0.8, seed=73, scale=0.5)
        standard = apply_noise_texture(base, texture="Emulsion Mottle", amount=0.8, seed=73, scale=1.0)
        broad = apply_noise_texture(base, texture="Emulsion Mottle", amount=0.8, seed=73, scale=3.0)
        self.assertFalse(torch.equal(fine, standard))
        self.assertFalse(torch.equal(standard, broad))
        self.assertTrue(torch.isfinite(broad).all())
        self.assertIs(apply_noise_texture(base, texture="Emulsion Mottle", amount=0.0, scale=4.0), base)

    def test_procedural_noise_is_independent_from_leak_seed_and_film_grain_controls(self):
        source = self.image.unsqueeze(0)
        a = process_batch(source, RetroLabSettings(noise_texture="Value fBm", noise_amount=0.8, noise_seed=73, leak_seed=27))
        b = process_batch(source, RetroLabSettings(noise_texture="Value fBm", noise_amount=0.8, noise_seed=73, leak_seed=999))
        self.assertTrue(torch.equal(a, b), "Leak Seed must not alter Procedural Texture")
        c = process_batch(source, RetroLabSettings(noise_texture="Value fBm", noise_amount=0.8, noise_seed=74, leak_seed=27))
        self.assertFalse(torch.equal(a, c), "Noise Seed must alter the procedural texture")
        self.assertFalse(is_identity(RetroLabSettings(noise_amount=0.2)))
        self.assertFalse(is_identity(RetroLabSettings(anti_ai=1.0)))
        self.assertTrue(is_identity(RetroLabSettings(noise_amount=0.0, anti_ai=0.0)))

    def test_all_twelve_texture_blend_modes_are_deterministic_distinct_and_bounded(self):
        outputs = []
        for mode in NOISE_BLEND_MODES:
            first = apply_noise_texture(self.image, texture="Emulsion Mottle", amount=1.0, seed=33, scale=0.0, blend_mode=mode)
            second = apply_noise_texture(self.image, texture="Emulsion Mottle", amount=1.0, seed=33, scale=0.0, blend_mode=mode)
            self.assertTrue(torch.equal(first, second), mode)
            self.assertTrue(torch.isfinite(first).all(), mode)
            self.assertGreaterEqual(float(first.min()), 0.0, mode)
            self.assertLessEqual(float(first.max()), 1.0, mode)
            self.assertGreater(float((first - self.image).abs().mean()), 1e-7, mode)
            outputs.append(first.flatten())
        distances = torch.cdist(torch.stack(outputs), torch.stack(outputs))
        distances.fill_diagonal_(float("inf"))
        self.assertGreater(float(distances.min()), 1e-7)

    def test_additive_is_legacy_default_and_invalid_blend_falls_back(self):
        default = apply_noise_texture(self.image, texture="Value fBm", amount=0.8, seed=17)
        explicit = apply_noise_texture(self.image, texture="Value fBm", amount=0.8, seed=17, blend_mode="Additive")
        invalid = apply_noise_texture(self.image, texture="Value fBm", amount=0.8, seed=17, blend_mode="not-a-mode")
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(default, invalid))
        self.assertEqual(normalize_noise_blend_mode("Screen"), "Screen")
        self.assertEqual(normalize_noise_blend_mode("invalid"), "Additive")

    def test_texture_scale_zero_is_safe_repeatable_and_finest(self):
        zero_a = apply_noise_texture(self.image, texture="Emulsion Mottle", amount=0.8, seed=73, scale=0.0)
        zero_b = apply_noise_texture(self.image, texture="Emulsion Mottle", amount=0.8, seed=73, scale=0.0)
        standard = apply_noise_texture(self.image, texture="Emulsion Mottle", amount=0.8, seed=73, scale=1.0)
        self.assertTrue(torch.equal(zero_a, zero_b))
        self.assertTrue(torch.isfinite(zero_a).all())
        self.assertFalse(torch.equal(zero_a, standard))

    def test_blend_mode_composites_complete_paper_foxing_effect(self):
        additive_default = apply_noise_texture(self.image, texture="Paper Foxing", amount=0.9, seed=47, scale=0.0)
        additive_explicit = apply_noise_texture(self.image, texture="Paper Foxing", amount=0.9, seed=47, scale=0.0, blend_mode="Additive")
        multiply = apply_noise_texture(self.image, texture="Paper Foxing", amount=0.9, seed=47, scale=0.0, blend_mode="Multiply")
        self.assertTrue(torch.equal(additive_default, additive_explicit))
        self.assertFalse(torch.equal(additive_default, multiply))
        self.assertTrue(torch.isfinite(multiply).all())
        self.assertGreaterEqual(float(multiply.min()), 0.0)
        self.assertLessEqual(float(multiply.max()), 1.0)

    def test_combined_pipeline_is_repeatable_and_bounded(self):
        settings = RetroLabSettings(
            grain_amount=0.35,
            halation_amount=0.25,
            bloom_amount=0.30,
            solarization=0.18,
            mackie_lines=0.20,
            leak_mode="Holga 135",
            leak_position="Top",
            leak_exposure=0.25,
            leak_color="#D56A3A",
            leak_seed=90210,
            leak_radius=2.5,
            anti_ai=42.0, noise_texture="Gradient fBm", noise_amount=0.65, noise_seed=991,
        )
        source = self.image.unsqueeze(0)
        out_a = process_batch(source, settings)
        out_b = process_batch(source, settings)
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertTrue(torch.isfinite(out_a).all())
        self.assertGreaterEqual(float(out_a.min()), 0.0)
        self.assertLessEqual(float(out_a.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
