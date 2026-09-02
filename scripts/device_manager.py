"""
Central GPU/CPU device management for dataset_anonymizer.

Policy:
  1. Try GPU first (CUDA probe + model warm on GPU backends).
  2. On any GPU-related failure (CUDA, DLL, OOM, model load, inference):
     automatically switch to CPU fallback — PaddleOCR + OpenCV blur/inpaint.
  3. Disable targeted GAN inpainting (DeepPrivacy2 + IOPaint/LaMa GPU).
  4. No user interaction; audit JSON sidecars and reports unchanged.

Used by main_pipeline, batch_processor, app.py, gpu_runtime, setup_environment.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Mapping, Optional, Tuple

from .security import redact_secrets_obj
from .utils import get_logger

logger = get_logger(__name__)

# User-facing notice (structlog + console + Flask UI)
FALLBACK_MESSAGE_TEMPLATE = (
    "⚠️ GPU configuration failed: {reason}. "
    "Running on CPU with basic anonymization. Targeted inpainting disabled."
)

# Back-compat alias
GPU_FALLBACK_USER_MESSAGE = FALLBACK_MESSAGE_TEMPLATE.format(
    reason="GPU unavailable at startup"
)

_COMPUTE_PROFILE: Dict[str, Any] = {}
_FALLBACK_EVENTS: list[Dict[str, Any]] = []


class GpuFallbackRequired(Exception):
    """Raise to signal an automatic downgrade to CPU fallback."""

    def __init__(self, reason: str, *, component: str = "unknown") -> None:
        self.reason = reason
        self.component = component
        super().__init__(reason)


def format_user_message(reason: str) -> str:
    r = (reason or "unknown GPU error").strip()
    if len(r) > 500:
        r = r[:497] + "..."
    return FALLBACK_MESSAGE_TEMPLATE.format(reason=r)


def get_compute_profile() -> Dict[str, Any]:
    return dict(_COMPUTE_PROFILE)


def is_cpu_fallback_mode() -> bool:
    return bool(_COMPUTE_PROFILE.get("cpu_fallback"))


def is_gan_inpainting_enabled() -> bool:
    return bool(_COMPUTE_PROFILE.get("gan_inpainting_enabled")) and not is_cpu_fallback_mode()


def get_fallback_events() -> list[Dict[str, Any]]:
    return list(_FALLBACK_EVENTS)


def is_gpu_related_error(exc: BaseException) -> bool:
    """True for CUDA/DLL/OOM/GPU provider failures (not missing vendor repos)."""
    if exc is None:
        return False
    try:
        import torch

        if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)):
            return True
    except Exception:  # noqa: BLE001
        pass

    markers = (
        "cuda",
        "cudnn",
        "cublas",
        "out of memory",
        "oom",
        "gpu",
        "dll",
        "onnxruntime",
        "cudart",
        "device-side assert",
        "no kernel image",
        "c10::",
        "nvidia",
        "could not load",
        "failed to load",
        "cudaexecutionprovider",
    )
    text = f"{type(exc).__name__} {exc}".lower()
    if any(m in text for m in markers):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_gpu_related_error(cause)
    return False


def announce_fallback(reason: str, *, component: str = "startup") -> str:
    """structlog warning + console print; returns formatted user message."""
    msg = format_user_message(reason)
    logger.warning(
        "gpu_configuration_failed_cpu_fallback",
        reason=reason,
        component=component,
        user_message=msg,
    )
    try:
        print(msg, flush=True)
    except OSError:
        pass
    return msg


def _apply_cpu_cfg_mutations(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("gpu", {})
    cfg["gpu"]["device"] = "cpu"
    cfg["device"] = "cpu"
    cfg["gpu_id"] = -1
    lama = cfg.get("lama")
    if isinstance(lama, dict):
        lama = dict(lama)
        lama["device"] = "cpu"
        lama["backend"] = "cpu_redaction"
        cfg["lama"] = lama
    ic = cfg.get("insightface")
    if isinstance(ic, dict):
        ic = dict(ic)
        ic["ctx_id"] = -1
        cfg["insightface"] = ic


def activate_cpu_fallback(
    cfg: Dict[str, Any],
    reason: str,
    *,
    component: str = "startup",
    announce: bool = True,
) -> Dict[str, Any]:
    """
    Switch process to CPU fallback mode (mutates ``cfg`` and global profile).
    """
    global _COMPUTE_PROFILE, _FALLBACK_EVENTS
    from .gpu_runtime import SharedCudaContext, sync_cfg_device_fields

    msg = announce_fallback(reason, component=component) if announce else format_user_message(reason)
    _FALLBACK_EVENTS.append(
        redact_secrets_obj({"component": component, "reason": reason, "user_message": msg})  # type: ignore[arg-type]
    )
    _apply_cpu_cfg_mutations(cfg)
    sync_cfg_device_fields(cfg)

    profile: Dict[str, Any] = {
        "requested_device": str((cfg.get("gpu") or {}).get("device", "cuda")),
        "cpu_fallback": True,
        "resolved_device": "cpu",
        "user_message": msg,
        "gan_inpainting_enabled": False,
        "fallback_reason": reason,
        "fallback_component": component,
        "fallback_events": list(_FALLBACK_EVENTS),
        "mode": "cpu_basic_anonymization",
    }
    _COMPUTE_PROFILE = redact_secrets_obj(profile)  # type: ignore[assignment]

    try:
        from .shared_models import clear_shared_models

        clear_shared_models()
    except Exception as exc:  # noqa: BLE001
        logger.debug("clear_shared_models_skipped", error=str(exc))

    try:
        SharedCudaContext.reset_for_tests()
        SharedCudaContext.configure(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shared_cuda_reconfigure_after_fallback", error=str(exc))

    return dict(_COMPUTE_PROFILE)


def probe_torch_cuda(*, gpu_id: int = 0) -> Tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        if torch.cuda.device_count() < 1:
            return False, "no_cuda_devices"
        idx = int(gpu_id)
        if idx < 0 or idx >= torch.cuda.device_count():
            idx = 0
        dev = torch.device(f"cuda:{idx}")
        x = torch.zeros(1, device=dev)
        _ = x + 1
        torch.cuda.synchronize(dev)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def initialize_compute(cfg: Dict[str, Any], *, force_gpu: bool = False) -> Dict[str, Any]:
    """
    Startup GPU probe. Tries GPU when ``gpu.device`` requests CUDA; else CPU mode.
    """
    global _COMPUTE_PROFILE, _FALLBACK_EVENTS
    from .gpu_runtime import resolve_gpu_config, sync_cfg_device_fields

    _FALLBACK_EVENTS = []
    g = resolve_gpu_config(cfg)
    gpu_block = dict(cfg.get("gpu") or {})
    force = bool(force_gpu or gpu_block.get("require_cuda"))
    requested = str(g.device).lower()
    cuda_ok, cuda_reason = probe_torch_cuda(gpu_id=g.gpu_id) if g.wants_cuda else (False, "cpu_requested")

    if force and not cuda_ok:
        raise RuntimeError(
            f"GPU validation failed: --gpu requires CUDA but probe failed ({cuda_reason})"
        )

    if g.wants_cuda and cuda_ok:
        profile: Dict[str, Any] = {
            "requested_device": requested,
            "cuda_probe_ok": True,
            "cuda_probe_reason": cuda_reason,
            "cpu_fallback": False,
            "resolved_device": f"cuda:{g.gpu_id}",
            "user_message": None,
            "gan_inpainting_enabled": True,
            "mode": "gpu_full",
            "fallback_events": [],
        }
        _COMPUTE_PROFILE = redact_secrets_obj(profile)  # type: ignore[assignment]
        sync_cfg_device_fields(cfg)
        logger.info(
            "compute_profile_gpu",
            resolved_device=profile["resolved_device"],
            cuda_probe=cuda_reason,
        )
        return dict(_COMPUTE_PROFILE)

    reason = cuda_reason if g.wants_cuda else "gpu.device=cpu in config"
    if not g.wants_cuda:
        activate_cpu_fallback(cfg, reason, component="config", announce=False)
        prof = get_compute_profile()
        prof["user_message"] = (
            "Running on CPU (gpu.device=cpu). Basic anonymization only; targeted inpainting disabled."
        )
        _COMPUTE_PROFILE = redact_secrets_obj(prof)  # type: ignore[assignment]
        logger.info("compute_profile_cpu_explicit", device="cpu")
        return dict(_COMPUTE_PROFILE)

    return activate_cpu_fallback(cfg, cuda_reason, component="cuda_probe")


def trigger_runtime_gpu_fallback(
    cfg: Dict[str, Any],
    exc: BaseException,
    *,
    component: str,
) -> Dict[str, Any]:
    """Downgrade after GPU model warm / inference failure mid-run."""
    reason = f"{component}: {exc}"
    return activate_cpu_fallback(cfg, reason, component=component, announce=True)


def handle_gpu_exception(
    cfg: Dict[str, Any],
    exc: BaseException,
    *,
    component: str,
    force_fallback: bool = False,
) -> bool:
    """
    If ``exc`` is GPU-related, activate CPU fallback and return True.
    """
    if force_fallback or is_gpu_related_error(exc):
        trigger_runtime_gpu_fallback(cfg, exc, component=component)
        return True
    return False


def gpu_readiness_check(*, gpu_id: int = 0) -> Dict[str, Any]:
    """
    Full readiness report for setup_environment.py (no secrets, no image data).
    """
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "cuda_probe": {},
        "torch": {},
        "onnxruntime": {},
        "paddle": {},
        "recommendation": "",
    }
    cuda_ok, cuda_reason = probe_torch_cuda(gpu_id=gpu_id)
    report["cuda_probe"] = {"ok": cuda_ok, "reason": cuda_reason}

    try:
        import torch

        report["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if cuda_ok:
            report["torch"]["device_name"] = torch.cuda.get_device_name(int(gpu_id))
    except Exception as exc:  # noqa: BLE001
        report["torch"] = {"error": str(exc)}

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        report["onnxruntime"] = {
            "version": getattr(ort, "__version__", "unknown"),
            "providers": providers,
            "cuda_provider": "CUDAExecutionProvider" in providers,
        }
    except Exception as exc:  # noqa: BLE001
        report["onnxruntime"] = {"error": str(exc)}

    try:
        import paddle

        report["paddle"] = {"version": getattr(paddle, "__version__", "unknown")}
    except Exception as exc:  # noqa: BLE001
        report["paddle"] = {"error": str(exc)}

    if cuda_ok:
        report["recommendation"] = "GPU path available — pipeline will try GPU models first."
    else:
        report["recommendation"] = (
            f"GPU not ready ({cuda_reason}). Pipeline will auto-fallback to CPU at runtime."
        )
    return redact_secrets_obj(report)  # type: ignore[return-value]


def reset_for_tests() -> None:
    global _COMPUTE_PROFILE, _FALLBACK_EVENTS
    _COMPUTE_PROFILE = {}
    _FALLBACK_EVENTS = []
