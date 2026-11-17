"""Tests for invisible CPU/CUDA execution planning and recovery."""

from __future__ import annotations

import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from core import pipeline
from core import execution
from core.execution import (
    ExecutionPlan,
    estimate_cuda_working_set_bytes,
    select_execution_plan,
)
from core.pipeline import RetroLabSettings


_GIB = 1024 ** 3


class AutomaticExecutionTests(unittest.TestCase):
    def test_cpu_only_pytorch_selects_cpu_without_cuda_memory_queries(self):
        images = torch.rand((1, 3000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=0.8)
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.cuda.mem_get_info"
        ) as memory_info:
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.output_device.type, "cpu")
        self.assertEqual(plan.reason, "cuda-unavailable")
        memory_info.assert_not_called()

    def test_cuda_availability_probe_error_falls_back_to_cpu(self):
        images = torch.rand((1, 3000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=0.8)
        with mock.patch("torch.cuda.is_available", side_effect=RuntimeError("CUDA driver mismatch")):
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.reason, "cuda-unavailable")

    def test_small_simple_correction_stays_on_cpu_to_avoid_transfer_overhead(self):
        images = torch.rand((1, 1200, 1600, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=18.0)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.mem_get_info"
        ) as memory_info:
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.reason, "workload-below-transfer-threshold")
        memory_info.assert_not_called()

    def test_large_spatial_workload_selects_cuda_when_memory_is_available(self):
        images = torch.rand((1, 3000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=0.8, halation_amount=0.5)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ), mock.patch("torch.cuda.mem_get_info", return_value=(12 * _GIB, 16 * _GIB)), mock.patch(
            "core.execution._cuda_compatibility_preflight_safely",
            return_value=(True, None, False),
        ):
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cuda")
        self.assertEqual(plan.processing_device.index, 0)
        self.assertEqual(plan.output_device.type, "cpu")
        self.assertEqual(plan.reason, "cuda-workload-and-memory-approved")
        self.assertGreater(plan.estimated_working_bytes, 0)
        self.assertTrue(plan.cuda_compatibility_checked)
        self.assertTrue(plan.cuda_compatibility_ok)

    def test_high_resolution_simple_correction_can_use_cuda(self):
        images = torch.rand((1, 6000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=35.0)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ), mock.patch("torch.cuda.mem_get_info", return_value=(12 * _GIB, 16 * _GIB)), mock.patch(
            "core.execution._cuda_compatibility_preflight_safely",
            return_value=(True, None, False),
        ):
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cuda")
        self.assertEqual(plan.output_device.type, "cpu")

    def test_cuda_preflight_failure_falls_back_to_cpu(self):
        images = torch.rand((1, 3000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=0.8)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", side_effect=RuntimeError("CUDA context unavailable")
        ):
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.reason, "cuda-preflight-failed")

    def test_cuda_operation_compatibility_failure_falls_back_before_image_transfer(self):
        images = torch.rand((1, 3000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=0.8)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ), mock.patch("core.execution._cuda_compatibility_preflight_safely", return_value=(
            False, "RuntimeError: unsupported convolution kernel", False
        )), mock.patch("torch.cuda.mem_get_info") as memory_info:
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.output_device.type, "cpu")
        self.assertEqual(plan.reason, "cuda-compatibility-preflight-failed")
        self.assertTrue(plan.cuda_compatibility_checked)
        self.assertFalse(plan.cuda_compatibility_ok)
        self.assertIn("unsupported convolution", plan.cuda_compatibility_error)
        memory_info.assert_not_called()

    def test_cuda_compatibility_preflight_caches_success_per_runtime_device(self):
        device = torch.device("cuda:0")
        properties = SimpleNamespace(name="Mock NVIDIA", major=8, minor=6, total_memory=16 * _GIB)
        execution._CUDA_COMPATIBILITY_CACHE.clear()
        try:
            with mock.patch("torch.cuda.get_device_properties", return_value=properties), mock.patch.object(
                execution, "_run_cuda_compatibility_probe"
            ) as probe:
                first = execution._cuda_compatibility_preflight_safely(device)
                second = execution._cuda_compatibility_preflight_safely(device)
            self.assertEqual(first, (True, None, False))
            self.assertEqual(second, (True, None, True))
            probe.assert_called_once_with(device)
        finally:
            execution._CUDA_COMPATIBILITY_CACHE.clear()

    def test_cuda_compatibility_probe_failure_is_not_cached(self):
        device = torch.device("cuda:0")
        properties = SimpleNamespace(name="Mock NVIDIA", major=8, minor=6, total_memory=16 * _GIB)
        execution._CUDA_COMPATIBILITY_CACHE.clear()
        try:
            with mock.patch("torch.cuda.get_device_properties", return_value=properties), mock.patch.object(
                execution, "_run_cuda_compatibility_probe", side_effect=RuntimeError("kernel unsupported")
            ) as probe:
                first = execution._cuda_compatibility_preflight_safely(device)
                second = execution._cuda_compatibility_preflight_safely(device)
            self.assertFalse(first[0])
            self.assertFalse(second[0])
            self.assertIn("kernel unsupported", first[1])
            self.assertEqual(first[2], False)
            self.assertEqual(second[2], False)
            self.assertEqual(probe.call_count, 2)
        finally:
            execution._CUDA_COMPATIBILITY_CACHE.clear()

    def test_insufficient_free_vram_selects_cpu_before_transfer(self):
        images = torch.rand((1, 6000, 4000, 3), dtype=torch.float32)
        settings = RetroLabSettings(bloom_amount=1.0, halation_amount=1.0, grain_amount=0.8)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=0
        ), mock.patch("torch.cuda.mem_get_info", return_value=(2 * _GIB, 16 * _GIB)), mock.patch(
            "core.execution._cuda_compatibility_preflight_safely",
            return_value=(True, None, False),
        ):
            plan = select_execution_plan(images, settings)
        self.assertEqual(plan.processing_device.type, "cpu")
        self.assertEqual(plan.reason, "insufficient-free-vram")

    def test_memory_estimate_scales_with_image_area_and_active_effects(self):
        settings = RetroLabSettings(bloom_amount=1.0, halation_amount=1.0)
        small = estimate_cuda_working_set_bytes(2000, 2000, settings)
        large = estimate_cuda_working_set_bytes(4000, 4000, settings)
        fixed_workspace = 128 * 1024 ** 2
        self.assertAlmostEqual(large - fixed_workspace, (small - fixed_workspace) * 4, delta=1)

    def test_cuda_oom_retries_entire_batch_on_cpu(self):
        source = torch.rand((2, 32, 48, 4), dtype=torch.float32)
        settings = RetroLabSettings(exposure=25.0)
        original_worker = pipeline._process_batch_on_device
        calls = []

        def simulated_worker(images, selected_settings, *, processing_device, output_device):
            calls.append((processing_device.type, output_device.type))
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory. simulated allocation failure")
            return original_worker(
                images, selected_settings,
                processing_device=processing_device,
                output_device=output_device,
            )

        cuda_plan = ExecutionPlan(
            torch.device("cuda:0"), torch.device("cpu"), "test-cuda-route"
        )
        with mock.patch.object(pipeline, "select_execution_plan", return_value=cuda_plan), mock.patch.object(
            pipeline, "_process_batch_on_device", side_effect=simulated_worker
        ):
            output = pipeline.process_batch(source, settings)

        self.assertEqual(calls, [("cuda", "cpu"), ("cpu", "cpu")])
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, source.dtype)
        self.assertTrue(torch.equal(output[..., 3:], source[..., 3:]))
        self.assertFalse(torch.equal(output[..., :3], source[..., :3]))

    def test_non_oom_cuda_errors_are_not_silently_retried(self):
        source = torch.rand((1, 32, 48, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=25.0)
        cuda_plan = ExecutionPlan(
            torch.device("cuda:0"), torch.device("cpu"), "test-cuda-route"
        )
        with mock.patch.object(pipeline, "select_execution_plan", return_value=cuda_plan), mock.patch.object(
            pipeline, "_process_batch_on_device", side_effect=RuntimeError("invalid CUDA kernel launch")
        ) as worker:
            with self.assertRaisesRegex(RuntimeError, "invalid CUDA kernel launch"):
                pipeline.process_batch(source, settings)
        worker.assert_called_once()


    def test_execution_report_captures_cpu_route_shape_settings_and_timing(self):
        import json
        source = torch.rand((1, 48, 64, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=25.0, grain_amount=0.2, color_emulsion="portra_400", emulsion_response=2.35)
        with mock.patch("torch.cuda.is_available", return_value=False):
            output, report_text = pipeline.process_batch_with_report(source, settings)
        report = json.loads(report_text)
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(report["schema"], "zuanfilm.retrolab.execution-report.v1")
        self.assertEqual(report["image"]["input_shape_bhwc"], [1, 48, 64, 3])
        self.assertEqual(report["image"]["input_dtype"], "torch.float32")
        self.assertEqual(report["image"]["internal_processing_dtype"], "torch.float32")
        self.assertEqual(report["image"]["input_device"], "cpu")
        self.assertEqual(report["execution"]["route"], "cpu")
        self.assertEqual(report["execution"]["planner_reason"], "cuda-unavailable")
        self.assertGreaterEqual(report["execution"]["processing_elapsed_seconds"], 0.0)
        self.assertIn("exposure", report["workload"]["active_stages_in_order"])
        self.assertIn("film_grain", report["workload"]["active_stages_in_order"])
        self.assertEqual(report["settings"]["exposure"], 25.0)
        self.assertEqual(report["settings"]["emulsion_response"], 2.35)
        self.assertEqual(report["settings"]["color_emulsion"], "portra_400")
        self.assertGreater(report["execution"]["throughput_megapixels_per_second"], 0.0)
        self.assertFalse(report["cuda"]["available"])
        self.assertEqual(
            report["cuda"]["compatibility_preflight"]["operation_families"],
            [
                "float32_tensor_math",
                "int64_deterministic_indexing",
                "convolution",
                "max_pooling",
                "replicate_padding",
                "fft2",
                "bilinear_interpolation",
                "erf",
            ],
        )

    def test_execution_report_records_cuda_oom_cpu_retry(self):
        import json
        source = torch.rand((2, 24, 32, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=25.0)
        cuda_plan = ExecutionPlan(
            torch.device("cuda:0"), torch.device("cpu"), "test-cuda-route",
            estimated_working_bytes=123456, free_vram_bytes=8 * _GIB,
            total_vram_bytes=16 * _GIB, cuda_available=True,
            cuda_device_name="Mock NVIDIA GPU", reserved_vram_bytes=_GIB,
            usable_vram_bytes=7 * _GIB,
        )
        original_worker = pipeline._process_batch_on_device
        calls = []

        def simulated_worker(images, selected_settings, *, processing_device, output_device):
            calls.append(processing_device.type)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory. simulated allocation failure")
            return original_worker(
                images, selected_settings,
                processing_device=processing_device,
                output_device=output_device,
            )

        with mock.patch.object(pipeline, "select_execution_plan", return_value=cuda_plan), mock.patch.object(
            pipeline, "_process_batch_on_device", side_effect=simulated_worker
        ):
            output, report_text = pipeline.process_batch_with_report(source, settings)

        report = json.loads(report_text)
        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(report["execution"]["planned_processing_device"], "cuda:0")
        self.assertEqual(report["execution"]["actual_processing_device"], "cpu")
        self.assertEqual(report["execution"]["route"], "cpu")
        self.assertTrue(report["execution"]["fallback_occurred"])
        self.assertIn("CUDA out of memory", report["execution"]["fallback_reason"])
        self.assertEqual(report["cuda"]["device_name"], "Mock NVIDIA GPU")
        self.assertEqual(report["cuda"]["estimated_one_frame_working_set_bytes"], 123456)


    def test_report_telemetry_failure_does_not_discard_processed_image(self):
        import json
        source = torch.rand((1, 24, 32, 3), dtype=torch.float32)
        settings = RetroLabSettings(exposure=10.0)
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch.object(
            pipeline, "_make_execution_report", side_effect=RuntimeError("diagnostic formatting failure")
        ):
            output, report_text = pipeline.process_batch_with_report(source, settings)
        report = json.loads(report_text)
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(report["result"], "completed")
        self.assertIn("report_generation_error", report)

    def test_identity_bypass_does_not_query_execution_environment(self):
        source = torch.rand((1, 32, 48, 3), dtype=torch.float32)
        with mock.patch.object(pipeline, "select_execution_plan") as planner:
            output = pipeline.process_batch(source, RetroLabSettings())
        self.assertIs(output, source)
        planner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
