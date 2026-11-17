"""Single-image processing orchestration for Zuanfilm Retrolab."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
import platform
import sys
from time import perf_counter

import torch

from .execution import (
    ExecutionPlan,
    is_cuda_out_of_memory_error,
    release_unused_cuda_cache,
    select_execution_plan,
)

from .anti_ai import apply_anti_ai
from .bloom import apply_bloom
from .grain import apply_film_grain
from .halation import DEFAULT_HALATION_COLOR, apply_halation
from .emulsions import apply_color_emulsion, normalize_emulsion_id, normalize_emulsion_response
from .exposure import apply_exposure
from .light_leak import apply_light_leak
from .sabattier import apply_sabattier
from .soft_focus import apply_soft_focus
from .vignette import apply_vignette
from .white_balance import apply_white_balance
from .noise_texture import apply_noise_texture, normalize_noise_texture, normalize_noise_blend_mode


# Grain remains reproducible and independent of the user-facing Leak Seed.
DEFAULT_GRAIN_SEED = 42


@dataclass(frozen=True)
class RetroLabSettings:
    grain_amount: float = 0.0
    grain_size: float = 0.3
    color_grain: float = 0.09
    halation_amount: float = 0.0
    ring_radius: float = 8.0
    bloom_amount: float = 0.0
    bloom_threshold: float = 0.1
    bloom_radius: float = 8.0
    solarization: float = 0.0
    highlight_confinement: float = 0.09
    mackie_lines: float = 0.0
    leak_mode: str = "Soft Edge"
    leak_position: str = "Bottom"
    leak_exposure: float = 0.0
    leak_seed: int = 27
    leak_color: str = "#C2AF8E"
    leak_radius: float = 1.0
    halation_color: str = DEFAULT_HALATION_COLOR
    soft_focus_amount: float = 0.0
    soft_focus_radius: float = 24.0
    vignette_amount: float = 0.0
    vignette_feather: float = 0.65
    color_emulsion: str = "neutral"
    emulsion_response: float = 1.0
    white_balance_temperature: float = 0.0
    white_balance_tint: float = 0.0
    anti_ai: float = 0.0
    noise_texture: str = "Gaussian"
    noise_amount: float = 0.0
    noise_seed: int = 73
    noise_scale: float = 1.0
    noise_blend_mode: str = "Additive"
    exposure: float = 0.0
    grain_blend_mode: str = "Additive"


def is_identity(settings: RetroLabSettings, *, include_grain: bool = True, include_texture: bool = True) -> bool:
    """True when every processing engine is at its exact bypass setting."""
    return (
        (not include_grain or settings.grain_amount <= 0.0)
        and (not include_texture or settings.noise_amount <= 0.0)
        and settings.halation_amount <= 0.0
        and settings.bloom_amount <= 0.0
        and settings.solarization <= 0.0
        and settings.mackie_lines <= 0.0
        and settings.leak_exposure <= 0.0
        and settings.soft_focus_amount <= 0.0
        and settings.vignette_amount <= 0.0
        and normalize_emulsion_id(settings.color_emulsion) == "neutral"
        and settings.white_balance_temperature == 0.0
        and settings.white_balance_tint == 0.0
        and settings.anti_ai <= 0.0
        and settings.exposure == 0.0
    )


def process_single_image(
    rgb_srgb: torch.Tensor, settings: RetroLabSettings, *, frame_index: int = 0, include_grain: bool = True, include_texture: bool = True
) -> torch.Tensor:
    """Process one HWC RGB image in the fixed finishing order.

    Order: White Balance → Anti-AI → Exposure → Bloom → Halation → Light Leak → Color Film Emulsion → Sabattier → Soft Focus → Vignette → Procedural Texture → Film Grain.
    Composition previews bypass both Film Grain and Procedural Texture so the
    reduced proxy stays clean; the native-resolution detail crop renders both.
    """
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] < 3:
        raise ValueError("Expected one HWC image with at least three channels")
    if is_identity(settings, include_grain=include_grain, include_texture=include_texture):
        return rgb_srgb

    rgb = rgb_srgb[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    # White balance is a source correction: apply it before the creative light
    # and film-emulsion stages, like a primary temperature/tint adjustment.
    if settings.white_balance_temperature != 0.0 or settings.white_balance_tint != 0.0:
        rgb = apply_white_balance(
            rgb,
            temperature=settings.white_balance_temperature,
            tint=settings.white_balance_tint,
        )
    if settings.anti_ai > 0.0:
        rgb = apply_anti_ai(rgb, amount=settings.anti_ai)
    if settings.exposure != 0.0:
        rgb = apply_exposure(rgb, amount=settings.exposure)
    if settings.bloom_amount > 0.0:
        rgb = apply_bloom(
            rgb,
            amount=settings.bloom_amount,
            threshold=settings.bloom_threshold,
            radius=settings.bloom_radius,
        )
    if settings.halation_amount > 0.0:
        rgb = apply_halation(
            rgb,
            amount=settings.halation_amount,
            ring_radius=settings.ring_radius,
            color=settings.halation_color,
        )
    if settings.leak_exposure > 0.0:
        rgb = apply_light_leak(
            rgb,
            mode=settings.leak_mode,
            position=settings.leak_position,
            exposure=settings.leak_exposure,
            seed=settings.leak_seed,
            color=settings.leak_color,
            influence_radius=settings.leak_radius,
        )
    if normalize_emulsion_id(settings.color_emulsion) != "neutral":
        rgb = apply_color_emulsion(rgb, settings.color_emulsion, response=settings.emulsion_response)
    if settings.solarization > 0.0 or settings.mackie_lines > 0.0:
        rgb = apply_sabattier(
            rgb,
            solarization=settings.solarization,
            highlight_confinement=settings.highlight_confinement,
            mackie_lines=settings.mackie_lines,
        )
    if settings.soft_focus_amount > 0.0:
        rgb = apply_soft_focus(
            rgb, amount=settings.soft_focus_amount, radius=settings.soft_focus_radius
        )
    if settings.vignette_amount > 0.0:
        rgb = apply_vignette(
            rgb, amount=settings.vignette_amount, feather=settings.vignette_feather
        )
    if include_texture and settings.noise_amount > 0.0:
        rgb = apply_noise_texture(
            rgb, texture=normalize_noise_texture(settings.noise_texture),
            amount=settings.noise_amount, seed=settings.noise_seed, scale=settings.noise_scale,
            blend_mode=normalize_noise_blend_mode(settings.noise_blend_mode), frame_index=frame_index,
        )
    if include_grain and settings.grain_amount > 0.0:
        rgb = apply_film_grain(
            rgb,
            amount=settings.grain_amount,
            grain_size=settings.grain_size,
            color_grain=settings.color_grain,
            # Grain evolves per batch frame without making the leak geometry flicker.
            seed=(DEFAULT_GRAIN_SEED + int(frame_index) * 7919) & 0x7FFFFFFF,
            blend_mode=settings.grain_blend_mode,
        )
    return rgb


def _process_batch_on_device(
    images: torch.Tensor,
    settings: RetroLabSettings,
    *,
    processing_device: torch.device,
    output_device: torch.device,
) -> torch.Tensor:
    """Process sequential frames on one device and write into one output batch.

    Only RGB is transferred to the working device. Additional channels are
    copied directly from the source to the output, avoiding needless GPU use.
    """
    output = torch.empty(
        tuple(images.shape), dtype=images.dtype, device=output_device
    )
    for frame_index in range(images.shape[0]):
        source_frame = images[frame_index]
        source_rgb = source_frame[..., :3]
        working_rgb = source_rgb.to(device=processing_device)
        processed_rgb = process_single_image(
            working_rgb, settings, frame_index=frame_index
        )
        output[frame_index, ..., :3].copy_(
            processed_rgb.to(device=output_device, dtype=images.dtype)
        )
        if images.shape[-1] > 3:
            output[frame_index, ..., 3:].copy_(
                source_frame[..., 3:].to(device=output_device, dtype=images.dtype)
            )
    return output


def _active_stage_names(settings: RetroLabSettings) -> list[str]:
    """Return creative stages that are active for this execution, in pipeline order."""
    active = []
    if settings.white_balance_temperature != 0.0 or settings.white_balance_tint != 0.0:
        active.append("white_balance")
    if settings.anti_ai > 0.0:
        active.append("anti_ai")
    if settings.exposure != 0.0:
        active.append("exposure")
    if settings.bloom_amount > 0.0:
        active.append("bloom")
    if settings.halation_amount > 0.0:
        active.append("halation")
    if settings.leak_exposure > 0.0:
        active.append("light_leak")
    if normalize_emulsion_id(settings.color_emulsion) != "neutral":
        active.append("color_emulsion")
    if settings.solarization > 0.0 or settings.mackie_lines > 0.0:
        active.append("sabattier")
    if settings.soft_focus_amount > 0.0:
        active.append("soft_focus")
    if settings.vignette_amount > 0.0:
        active.append("vignette")
    if settings.noise_amount > 0.0:
        active.append("procedural_texture")
    if settings.grain_amount > 0.0:
        active.append("film_grain")
    return active


def _cuda_snapshot(device: torch.device | None) -> dict:
    """Collect best-effort CUDA allocator/global-memory snapshots."""
    if device is None or device.type != "cuda":
        return {}
    snapshot = {}
    queries = (
        ("allocated_bytes", torch.cuda.memory_allocated),
        ("reserved_bytes", torch.cuda.memory_reserved),
    )
    for key, query in queries:
        try:
            snapshot[key] = int(query(device))
        except Exception as exc:
            snapshot[f"{key}_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot["free_bytes"] = int(free_bytes)
        snapshot["total_bytes"] = int(total_bytes)
    except Exception as exc:
        snapshot["global_memory_query_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return snapshot


def _make_execution_report_safely(*args, **kwargs) -> str:
    """Keep optional diagnostics from interrupting a completed image result."""
    try:
        return _make_execution_report(*args, **kwargs)
    except Exception as exc:
        images = args[0] if args else kwargs.get("images")
        output = args[1] if len(args) > 1 else kwargs.get("output")
        minimal = {
            "schema": "zuanfilm.retrolab.execution-report.v1",
            "result": "completed",
            "report_generation_error": f"{type(exc).__name__}: {exc}"[:500],
        }
        if isinstance(images, torch.Tensor):
            minimal["input_shape_bhwc"] = [int(v) for v in images.shape]
            minimal["input_device"] = str(images.device)
        if isinstance(output, torch.Tensor):
            minimal["output_shape_bhwc"] = [int(v) for v in output.shape]
            minimal["output_device"] = str(output.device)
        return json.dumps(minimal, indent=2, ensure_ascii=False)


def _runtime_package_version() -> str:
    """Resolve the root custom-node package version under ComfyUI's loader."""
    root_package_name = __package__.rsplit(".core", 1)[0]
    root_package = sys.modules.get(root_package_name)
    return str(getattr(root_package, "__version__", "unknown"))


def _make_execution_report(
    images: torch.Tensor,
    output: torch.Tensor,
    settings: RetroLabSettings,
    *,
    elapsed_seconds: float,
    plan: ExecutionPlan | None,
    actual_processing_device: torch.device | None,
    cuda_before: dict | None = None,
    cuda_after: dict | None = None,
    fallback_error: str | None = None,
    telemetry_errors: list[str] | None = None,
) -> str:
    """Create the JSON report returned by Retrolab's text output socket."""
    active_stages = _active_stage_names(settings)
    identity_bypass = plan is None
    input_device = images.device
    output_device = output.device
    processing_device = actual_processing_device
    cuda_device = None
    if plan is not None and plan.processing_device.type == "cuda":
        cuda_device = plan.processing_device
    elif input_device.type == "cuda":
        cuda_device = input_device

    if identity_bypass:
        route = "identity_bypass"
        reason = "all_effects_neutral"
        planned_device = "not_required"
    else:
        route = processing_device.type if processing_device is not None else "unknown"
        reason = plan.reason
        planned_device = str(plan.processing_device)

    if plan is not None and plan.cuda_available is not None:
        cuda_available = bool(plan.cuda_available)
    elif input_device.type == "cuda":
        cuda_available = True
    else:
        cuda_available = None

    free_before = plan.free_vram_bytes if plan is not None else 0
    total_vram = plan.total_vram_bytes if plan is not None else 0
    report = {
        "schema": "zuanfilm.retrolab.execution-report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product": {
            "name": "Zuanfilm Retrolab",
            "version": _runtime_package_version(),
        },
        "result": "completed",
        "timing_scope": "Full-resolution Retrolab image pipeline only; excludes preview-cache construction, frontend rendering and upstream/downstream graph nodes.",
        "image": {
            "input_shape_bhwc": [int(v) for v in images.shape],
            "output_shape_bhwc": [int(v) for v in output.shape],
            "height": int(images.shape[1]),
            "width": int(images.shape[2]),
            "megapixels": round((int(images.shape[1]) * int(images.shape[2])) / 1_000_000, 3),
            "batch_frames": int(images.shape[0]),
            "channels": int(images.shape[-1]),
            "input_dtype": str(images.dtype),
            "output_dtype": str(output.dtype),
            "internal_processing_dtype": "torch.float32",
            "input_device": str(input_device),
            "output_device": str(output_device),
            "input_contiguous": bool(images.is_contiguous()),
            "input_stride": [int(v) for v in images.stride()],
        },
        "execution": {
            "route": route,
            "planned_processing_device": planned_device,
            "actual_processing_device": str(processing_device) if processing_device is not None else "none",
            "output_device": str(output_device),
            "planner_reason": reason,
            "frame_strategy": "sequential_one_frame_at_a_time",
            "identity_bypass": identity_bypass,
            "fallback_occurred": fallback_error is not None,
            "fallback_reason": fallback_error,
            "processing_elapsed_seconds": round(float(elapsed_seconds), 6),
            "average_seconds_per_frame": round(float(elapsed_seconds) / max(1, int(images.shape[0])), 6),
            "throughput_megapixels_per_second": round(
                (int(images.shape[0]) * int(images.shape[1]) * int(images.shape[2]) / 1_000_000)
                / max(float(elapsed_seconds), 1e-12), 3
            ),
        },
        "workload": {
            "active_stages_in_order": active_stages,
            "active_stage_count": len(active_stages),
        },
        "settings": asdict(settings),
        "cuda": {
            "available": cuda_available,
            "device_name": plan.cuda_device_name if plan is not None else None,
            "compatibility_preflight": {
                "checked": bool(plan.cuda_compatibility_checked) if plan is not None else False,
                "compatible": plan.cuda_compatibility_ok if plan is not None else None,
                "cached_success": bool(plan.cuda_compatibility_cached) if plan is not None else False,
                "error": plan.cuda_compatibility_error if plan is not None else None,
                "operation_families": [
                    "float32_tensor_math",
                    "int64_deterministic_indexing",
                    "convolution",
                    "max_pooling",
                    "replicate_padding",
                    "fft2",
                    "bilinear_interpolation",
                    "erf",
                ],
            },
            "device_query_error": plan.device_query_error if plan is not None else None,
            "availability_probe_error": plan.cuda_probe_error if plan is not None else None,
            "memory_query_error": plan.memory_query_error if plan is not None else None,
            "runtime_version": getattr(torch.version, "cuda", None),
            "total_vram_bytes": int(total_vram) if total_vram else None,
            "free_vram_before_bytes": int(free_before) if free_before else None,
            "estimated_one_frame_working_set_bytes": int(plan.estimated_working_bytes) if plan is not None and plan.estimated_working_bytes else None,
            "reserved_headroom_bytes": int(plan.reserved_vram_bytes) if plan is not None and plan.reserved_vram_bytes else None,
            "usable_vram_after_reserve_bytes": int(plan.usable_vram_bytes) if plan is not None and plan.total_vram_bytes else None,
            "allocator_before": cuda_before or {},
            "allocator_after": cuda_after or {},
            "memory_metric_note": "Working-set size is a conservative preflight estimate. Allocator and global-memory values are before/after snapshots, not a measured per-node peak.",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": str(torch.__version__),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "pytorch_intraop_threads": torch.get_num_threads(),
            "pytorch_interop_threads": torch.get_num_interop_threads(),
        },
        "telemetry_errors": telemetry_errors or [],
    }
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False)


@torch.no_grad()
def _run_batch(
    images: torch.Tensor,
    settings: RetroLabSettings,
    *,
    include_report: bool,
) -> tuple[torch.Tensor, str | None]:
    """Shared batch executor; diagnostics are collected only when requested."""
    started = perf_counter() if include_report else None
    if not isinstance(images, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Expected ComfyUI IMAGE tensor with shape (B,H,W,C), C >= 3")

    if is_identity(settings):
        if not include_report:
            return images, None
        elapsed = perf_counter() - started
        report = _make_execution_report_safely(
            images, images, settings,
            elapsed_seconds=elapsed,
            plan=None,
            actual_processing_device=None,
        )
        return images, report

    plan = select_execution_plan(images, settings)
    cuda_device = plan.processing_device if plan.uses_cuda else (
        images.device if images.device.type == "cuda" else None
    )
    if (
        cuda_device is None
        and plan.cuda_available
        and not (plan.cuda_compatibility_checked and plan.cuda_compatibility_ok is False)
        and (plan.free_vram_bytes > 0 or plan.cuda_device_name or plan.memory_query_error)
    ):
        try:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        except Exception:
            cuda_device = None
    cuda_before = _cuda_snapshot(cuda_device) if include_report else {}
    telemetry_errors = []
    fallback_error = None
    actual_device = plan.processing_device

    try:
        output = _process_batch_on_device(
            images, settings,
            processing_device=plan.processing_device,
            output_device=plan.output_device,
        )
    except Exception as exc:
        if not plan.uses_cuda or not is_cuda_out_of_memory_error(exc):
            raise
        fallback_error = f"{type(exc).__name__}: {exc}"[:1000]
        logging.getLogger("zuanfilm.retrolab.execution").warning(
            "CUDA memory was insufficient during Retrolab processing; retrying the complete image batch on CPU."
        )
        # The exception traceback can retain failed worker CUDA tensors.
        exc.__traceback__ = None
        del exc
        release_unused_cuda_cache()
        cpu_device = torch.device("cpu")
        cpu_plan = ExecutionPlan(cpu_device, cpu_device, "cuda-oom-cpu-retry")
        output = _process_batch_on_device(
            images, settings,
            processing_device=cpu_plan.processing_device,
            output_device=cpu_plan.output_device,
        )
        actual_device = cpu_device

    if include_report and plan.uses_cuda:
        try:
            torch.cuda.synchronize(plan.processing_device)
        except Exception as exc:
            telemetry_errors.append(f"CUDA synchronize: {type(exc).__name__}: {exc}"[:300])

    if not include_report:
        return output, None

    cuda_after = _cuda_snapshot(cuda_device)
    elapsed = perf_counter() - started
    report = _make_execution_report_safely(
        images, output, settings,
        elapsed_seconds=elapsed,
        plan=plan,
        actual_processing_device=actual_device,
        cuda_before=cuda_before,
        cuda_after=cuda_after,
        fallback_error=fallback_error,
        telemetry_errors=telemetry_errors,
    )
    return output, report


@torch.no_grad()
def process_batch_with_report(
    images: torch.Tensor,
    settings: RetroLabSettings,
) -> tuple[torch.Tensor, str]:
    """Process a ComfyUI IMAGE batch and return its machine-readable report."""
    output, report = _run_batch(images, settings, include_report=True)
    return output, report or "{}"


@torch.no_grad()
def process_batch(images: torch.Tensor, settings: RetroLabSettings) -> torch.Tensor:
    """Backward-compatible tensor-only processing API without telemetry overhead."""
    output, _report = _run_batch(images, settings, include_report=False)
    return output
