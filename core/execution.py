"""Automatic, invisible execution planning for Zuanfilm Retrolab.

The module deliberately exposes no user-facing device controls. It chooses a
processing device from the incoming tensor, workload size and currently free
CUDA memory. CPU remains the universal fallback and the photographic engines
remain device-agnostic PyTorch implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

import torch
import torch.nn.functional as F


_LOGGER = logging.getLogger("zuanfilm.retrolab.execution")
_GIB = 1024 ** 3
_MIB = 1024 ** 2

# Small jobs stay on CPU to avoid paying for host/device transfers. Spatial
# effects cross the threshold earlier because their filtering/FFT workloads
# benefit from parallel execution sooner than simple pointwise corrections.
_MIN_CUDA_PIXELS_SIMPLE = 8_000_000
_MIN_CUDA_PIXELS_SPATIAL = 2_500_000

# Retain room for ComfyUI's active models and other GPU users. This is a
# reservation, not a claim about exact allocator consumption.
_MINIMUM_VRAM_RESERVE_BYTES = 1 * _GIB
_VRAM_RESERVE_FRACTION = 0.10
_FIXED_WORKSPACE_BYTES = 128 * _MIB

# A very small operation probe catches CUDA runtimes which report availability
# but cannot execute one or more of the operation families used by Retrolab.
# Successful probes are cached per PyTorch/runtime/device signature; failures
# are deliberately not cached so a transient initialization problem can be
# retried on a later graph execution.
_CUDA_COMPATIBILITY_CACHE: set[tuple] = set()
_CUDA_COMPATIBILITY_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExecutionPlan:
    """Internal processing route; never serialized into the node UI."""

    processing_device: torch.device
    output_device: torch.device
    reason: str
    estimated_working_bytes: int = 0
    free_vram_bytes: int = 0
    total_vram_bytes: int = 0
    cuda_available: bool | None = None
    cuda_probe_error: str | None = None
    cuda_device_name: str | None = None
    device_query_error: str | None = None
    memory_query_error: str | None = None
    reserved_vram_bytes: int = 0
    usable_vram_bytes: int = 0
    cuda_compatibility_checked: bool = False
    cuda_compatibility_ok: bool | None = None
    cuda_compatibility_cached: bool = False
    cuda_compatibility_error: str | None = None

    @property
    def uses_cuda(self) -> bool:
        return self.processing_device.type == "cuda"


def _enabled(settings: object, name: str, *, positive: bool = True) -> bool:
    value = getattr(settings, name, None)
    if isinstance(value, str):
        return bool(value) and value.lower() not in {"neutral", "none", "off"}
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return bool(value)
    return number > 0.0 if positive else number != 0.0


def _has_spatial_workload(settings: object) -> bool:
    return any((
        _enabled(settings, "bloom_amount"),
        _enabled(settings, "halation_amount"),
        _enabled(settings, "leak_exposure"),
        _enabled(settings, "soft_focus_amount"),
        _enabled(settings, "vignette_amount"),
        _enabled(settings, "noise_amount"),
        _enabled(settings, "grain_amount"),
    ))


def _has_meaningful_workload(settings: object) -> bool:
    return any((
        _has_spatial_workload(settings),
        _enabled(settings, "white_balance_temperature", positive=False),
        _enabled(settings, "white_balance_tint", positive=False),
        _enabled(settings, "anti_ai"),
        _enabled(settings, "exposure", positive=False),
        _enabled(settings, "solarization"),
        _enabled(settings, "mackie_lines"),
        _enabled(settings, "color_emulsion"),
    ))


def _estimate_workspace_multiplier(settings: object) -> float:
    """Conservative peak-working-set estimate in full RGB float32 buffers.

    Effects are executed sequentially, so costs are not summed as though every
    engine retained all of its intermediates. The estimate adds a bounded
    allowance for the largest spatial stages and active pointwise transforms.
    The runtime OOM fallback remains the final safety net.
    """
    multiplier = 5.5
    if _enabled(settings, "bloom_amount"):
        multiplier += 1.5
    if _enabled(settings, "halation_amount"):
        multiplier += 2.0
    if _enabled(settings, "soft_focus_amount"):
        multiplier += 1.25
    if _enabled(settings, "noise_amount"):
        multiplier += 1.25
    if _enabled(settings, "grain_amount"):
        multiplier += 0.75
    if _enabled(settings, "leak_exposure"):
        multiplier += 0.5
    if _enabled(settings, "vignette_amount"):
        multiplier += 0.5
    if _enabled(settings, "anti_ai"):
        multiplier += 0.5
    if _enabled(settings, "color_emulsion"):
        multiplier += 0.5
    if _enabled(settings, "solarization") or _enabled(settings, "mackie_lines"):
        multiplier += 0.5
    if _enabled(settings, "white_balance_temperature", positive=False) or _enabled(settings, "white_balance_tint", positive=False):
        multiplier += 0.25
    if _enabled(settings, "exposure", positive=False):
        multiplier += 0.25
    return min(multiplier, 14.0)


def estimate_cuda_working_set_bytes(height: int, width: int, settings: object) -> int:
    """Estimate one-frame CUDA working memory in bytes, including headroom."""
    pixels = max(0, int(height)) * max(0, int(width))
    rgb_float32_bytes = pixels * 3 * 4
    estimated = int(rgb_float32_bytes * _estimate_workspace_multiplier(settings))
    return estimated + _FIXED_WORKSPACE_BYTES


def _cuda_is_worthwhile(height: int, width: int, settings: object) -> bool:
    pixels = max(0, int(height)) * max(0, int(width))
    if not _has_meaningful_workload(settings):
        return False
    threshold = (
        _MIN_CUDA_PIXELS_SPATIAL
        if _has_spatial_workload(settings)
        else _MIN_CUDA_PIXELS_SIMPLE
    )
    return pixels >= threshold


def _cuda_available_safely() -> tuple[bool, str | None]:
    """Probe CUDA without allowing a broken runtime to break CPU execution."""
    try:
        return bool(torch.cuda.is_available()), None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:500]
        _LOGGER.debug("CUDA availability probe failed; using CPU: %s", message)
        return False, message


def _cuda_device_name_safely(device: torch.device) -> tuple[str | None, str | None]:
    try:
        return str(torch.cuda.get_device_name(device)), None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:500]
        _LOGGER.debug("CUDA device-name query failed: %s", message)
        return None, message


def _cuda_memory_safely(device: torch.device) -> tuple[int, int, str | None]:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes), int(total_bytes), None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:500]
        _LOGGER.debug("CUDA memory query failed: %s", message)
        return 0, 0, message


def _cuda_compatibility_key(device: torch.device) -> tuple:
    """Return a key that invalidates the cached probe across runtime/device changes."""
    properties = torch.cuda.get_device_properties(device)
    return (
        str(torch.__version__),
        str(getattr(torch.version, "cuda", None)),
        int(device.index if device.index is not None else torch.cuda.current_device()),
        str(getattr(properties, "name", "unknown")),
        int(getattr(properties, "major", -1)),
        int(getattr(properties, "minor", -1)),
        int(getattr(properties, "total_memory", 0)),
    )


@torch.no_grad()
def _run_cuda_compatibility_probe(device: torch.device) -> None:
    """Exercise small representative kernels required by Retrolab.

    This is intentionally tiny and non-photographic. It validates float32
    tensor math, int64 deterministic-index arithmetic, convolution, max-pool,
    replicate padding, FFT and interpolation without touching user images or changing global precision,
    allocator settings or ComfyUI device state.
    """
    sample = torch.linspace(
        -1.0, 1.0, 64, device=device, dtype=torch.float32
    ).reshape(1, 1, 8, 8)
    kernel = torch.full(
        (1, 1, 3, 3), 1.0 / 9.0, device=device, dtype=torch.float32
    )
    convolved = F.conv2d(sample, kernel, padding=1)
    pooled = F.max_pool2d(convolved, kernel_size=2, stride=2)
    padded = F.pad(pooled, (1, 1, 1, 1), mode="replicate")
    spectrum = torch.fft.rfft2(padded, dim=(-2, -1))
    restored = torch.fft.irfft2(spectrum, s=padded.shape[-2:], dim=(-2, -1))
    resized = F.interpolate(
        restored, size=(16, 16), mode="bilinear", align_corners=False
    )
    indices = torch.arange(64, device=device, dtype=torch.int64)
    deterministic_indices = torch.remainder(
        indices * 1103515245 + 12345, 2147483647
    )
    tonal = torch.erf(torch.sqrt(torch.clamp(resized.abs(), min=0.0)))
    checksum = tonal.mean() + deterministic_indices.float().mean() * 1.0e-8
    if not bool(torch.isfinite(checksum).item()):
        raise RuntimeError("CUDA compatibility probe produced a non-finite result")
    torch.cuda.synchronize(device)


def _cuda_compatibility_preflight_safely(
    device: torch.device,
) -> tuple[bool, str | None, bool]:
    """Verify core CUDA operation support, caching successful device probes.

    Returns ``(compatible, error, cache_hit)``. Every failure is fail-closed to
    CPU by the planner; no image tensor is involved in the preflight.
    """
    try:
        key = _cuda_compatibility_key(device)
        with _CUDA_COMPATIBILITY_LOCK:
            if key in _CUDA_COMPATIBILITY_CACHE:
                return True, None, True
            _run_cuda_compatibility_probe(device)
            _CUDA_COMPATIBILITY_CACHE.add(key)
        return True, None, False
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:700]
        _LOGGER.warning(
            "CUDA compatibility preflight failed; Retrolab will use CPU: %s",
            message,
        )
        return False, message, False


def select_execution_plan(images: torch.Tensor, settings: object) -> ExecutionPlan:
    """Choose CUDA only when it is available, useful and conservatively funded.

    CPU-origin ComfyUI images are returned on CPU even when processed on CUDA.
    Images already resident on CUDA stay there unless the processing attempt
    encounters an out-of-memory failure (handled by the caller).
    """
    input_device = images.device
    height, width = int(images.shape[1]), int(images.shape[2])

    if input_device.type == "cuda":
        name, name_error = _cuda_device_name_safely(input_device)
        compatibility_ok, compatibility_error, compatibility_cached = (
            _cuda_compatibility_preflight_safely(input_device)
        )
        if not compatibility_ok:
            return ExecutionPlan(
                torch.device("cpu"), torch.device("cpu"),
                "cuda-compatibility-preflight-failed",
                estimate_cuda_working_set_bytes(height, width, settings),
                cuda_available=True,
                cuda_device_name=name,
                device_query_error=name_error,
                cuda_compatibility_checked=True,
                cuda_compatibility_ok=False,
                cuda_compatibility_error=compatibility_error,
            )
        free_bytes, total_bytes, memory_error = _cuda_memory_safely(input_device)
        return ExecutionPlan(
            input_device, input_device, "input-already-on-cuda",
            estimate_cuda_working_set_bytes(height, width, settings),
            free_bytes, total_bytes, True, None, name,
            name_error, memory_error, 0, max(0, free_bytes),
            True, True, compatibility_cached, None,
        )

    if input_device.type != "cpu":
        return ExecutionPlan(input_device, input_device, "preserve-input-device")

    cuda_available, probe_error = _cuda_available_safely()
    if not cuda_available:
        return ExecutionPlan(
            torch.device("cpu"), input_device,
            "cuda-unavailable", cuda_available=False,
            cuda_probe_error=probe_error,
        )

    required_bytes = estimate_cuda_working_set_bytes(height, width, settings)

    # Avoid CUDA context/device queries for workloads already known to be too
    # small to benefit from a host/device transfer. Availability alone is enough
    # to explain why this execution deliberately stayed on CPU.
    if not _cuda_is_worthwhile(height, width, settings):
        return ExecutionPlan(
            torch.device("cpu"), input_device,
            "workload-below-transfer-threshold",
            estimated_working_bytes=required_bytes,
            cuda_available=True,
        )

    try:
        device = torch.device("cuda", torch.cuda.current_device())
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:500]
        _LOGGER.debug("CUDA device selection failed; using CPU: %s", message)
        return ExecutionPlan(
            torch.device("cpu"), input_device, "cuda-preflight-failed",
            estimated_working_bytes=required_bytes,
            cuda_available=True, memory_query_error=message,
        )

    compatibility_ok, compatibility_error, compatibility_cached = (
        _cuda_compatibility_preflight_safely(device)
    )
    if not compatibility_ok:
        return ExecutionPlan(
            torch.device("cpu"), input_device, "cuda-compatibility-preflight-failed",
            estimated_working_bytes=required_bytes,
            cuda_available=True,
            cuda_device_name=_cuda_device_name_safely(device)[0],
            cuda_compatibility_checked=True,
            cuda_compatibility_ok=False,
            cuda_compatibility_cached=False,
            cuda_compatibility_error=compatibility_error,
        )

    device_name, name_error = _cuda_device_name_safely(device)
    free_bytes, total_bytes, memory_error = _cuda_memory_safely(device)
    reserve_bytes = max(
        _MINIMUM_VRAM_RESERVE_BYTES,
        int(total_bytes * _VRAM_RESERVE_FRACTION),
    ) if total_bytes else _MINIMUM_VRAM_RESERVE_BYTES
    usable_bytes = max(0, free_bytes - reserve_bytes)

    common = {
        "estimated_working_bytes": required_bytes,
        "free_vram_bytes": free_bytes,
        "total_vram_bytes": total_bytes,
        "cuda_available": True,
        "cuda_probe_error": None,
        "cuda_device_name": device_name,
        "device_query_error": name_error,
        "memory_query_error": memory_error,
        "reserved_vram_bytes": reserve_bytes,
        "usable_vram_bytes": usable_bytes,
        "cuda_compatibility_checked": True,
        "cuda_compatibility_ok": True,
        "cuda_compatibility_cached": compatibility_cached,
        "cuda_compatibility_error": None,
    }

    if memory_error:
        return ExecutionPlan(
            torch.device("cpu"), input_device, "cuda-preflight-failed", **common
        )

    if required_bytes > usable_bytes:
        return ExecutionPlan(
            torch.device("cpu"), input_device,
            "insufficient-free-vram", **common,
        )

    return ExecutionPlan(
        device, input_device, "cuda-workload-and-memory-approved", **common
    )


def is_cuda_out_of_memory_error(error: BaseException) -> bool:
    """Recognize allocator, cuFFT and cuDNN allocation failures without
    swallowing unrelated CUDA/programming errors.
    """
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(cuda_oom_type, type) and isinstance(error, cuda_oom_type):
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "out of memory",
        "cuda_error_out_of_memory",
        "cufft_alloc_failed",
        "cudnn_status_alloc_failed",
    ))


def release_unused_cuda_cache() -> None:
    """Release unused allocator cache after a failed CUDA attempt."""
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Cleanup must never mask the original processing failure.
        _LOGGER.debug("CUDA cache cleanup did not complete", exc_info=True)
