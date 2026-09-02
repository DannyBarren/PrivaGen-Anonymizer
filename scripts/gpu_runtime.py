"""
Centralized CUDA/GPU configuration for dataset_anonymizer.

All Torch-backed models (DeepPrivacy2/tops, IOPaint/LaMa, InsightFace ONNX CUDA)
share a single ``torch.device`` via ``SharedCudaContext`` to limit fragmentation.

Security: GPU memory snapshots and validation logs use ``redact_secrets_obj``;
no image bytes or embeddings are logged. After secure temp wipe, call
``empty_cuda_cache_after_batch`` so allocator caches do not retain stale buffers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .device_manager import (
    GPU_FALLBACK_USER_MESSAGE,
    get_compute_profile,
    initialize_compute,
    is_cpu_fallback_mode,
    probe_torch_cuda,
)
from .security import redact_secrets_obj
from .utils import get_logger

logger = get_logger(__name__)

# Re-export for backward compatibility
apply_compute_profile = initialize_compute

# Process-wide singleton device (set once at first ``SharedCudaContext.configure``)
_SHARED_DEVICE: Optional[Any] = None  # torch.device
_GPU_VALIDATION: Dict[str, Any] = {}


@dataclass(frozen=True)
class GpuConfig:
    device: str = "cuda"
    gpu_id: int = 0
    memory_efficient: bool = True
    torch_compile: bool = False
    allow_tf32: bool = True
    adaptive_batch: bool = True
    empty_cache_between_batches: bool = True
    use_fp16_inpaint: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2

    @property
    def wants_cuda(self) -> bool:
        return str(self.device).lower().startswith("cuda")

    @property
    def resolved_device_str(self) -> str:
        if self.wants_cuda:
            return f"cuda:{self.gpu_id}"
        return "cpu"


def resolve_gpu_config(cfg: Mapping[str, Any]) -> GpuConfig:
    """Merge ``gpu:`` block with legacy top-level ``device`` / ``gpu_id`` / ``pin_memory``."""
    gpu = dict(cfg.get("gpu") or {})
    legacy_device = str(cfg.get("device", gpu.get("device", "cuda")))
    legacy_gpu_id = int(cfg.get("gpu_id", gpu.get("gpu_id", 0)))
    empty_cache = bool(
        gpu.get(
            "empty_cache_between_batches",
            cfg.get("cuda_empty_cache_between_batches", True),
        )
    )
    return GpuConfig(
        device=str(gpu.get("device", legacy_device)),
        gpu_id=int(gpu.get("gpu_id", legacy_gpu_id)),
        memory_efficient=bool(gpu.get("memory_efficient", True)),
        torch_compile=bool(gpu.get("torch_compile", False)),
        allow_tf32=bool(gpu.get("allow_tf32", True)),
        adaptive_batch=bool(gpu.get("adaptive_batch", True)),
        empty_cache_between_batches=empty_cache,
        use_fp16_inpaint=bool(gpu.get("use_fp16_inpaint", gpu.get("memory_efficient", True))),
        pin_memory=bool(gpu.get("pin_memory", cfg.get("pin_memory", True))),
        prefetch_factor=int(gpu.get("prefetch_factor", 2)),
    )


def sync_cfg_device_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Write resolved device/gpu_id back for components that read top-level keys."""
    g = resolve_gpu_config(cfg)
    if g.wants_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                cfg["device"] = "cpu"
                cfg["gpu_id"] = -1
            else:
                cfg["device"] = "cuda"
                cfg["gpu_id"] = g.gpu_id
        except Exception:  # noqa: BLE001
            cfg["device"] = "cpu"
            cfg["gpu_id"] = -1
    else:
        cfg["device"] = "cpu"
        cfg["gpu_id"] = -1
    cfg["cuda_empty_cache_between_batches"] = g.empty_cache_between_batches
    if "insightface" in cfg and isinstance(cfg["insightface"], dict):
        ic = dict(cfg["insightface"])
        if g.wants_cuda and cfg.get("device") == "cuda":
            ic["ctx_id"] = g.gpu_id
        elif not g.wants_cuda or cfg.get("device") == "cpu":
            ic["ctx_id"] = -1
        cfg["insightface"] = ic
    lama = cfg.get("lama")
    if isinstance(lama, dict) and lama.get("device") is None:
        lama = dict(lama)
        lama["device"] = cfg["device"]
        cfg["lama"] = lama
    return cfg


class SharedCudaContext:
    """Single shared CUDA device for all Torch operations in this process."""

    @classmethod
    def configure(cls, cfg: Mapping[str, Any]) -> Any:
        global _SHARED_DEVICE
        import torch

        g = resolve_gpu_config(cfg)
        cuda_ok, _ = probe_torch_cuda(gpu_id=g.gpu_id) if g.wants_cuda else (False, "cpu")
        if g.wants_cuda and cuda_ok:
            try:
                idx = int(g.gpu_id)
                if idx < 0 or idx >= torch.cuda.device_count():
                    logger.warning(
                        "gpu_id_out_of_range",
                        gpu_id=idx,
                        device_count=int(torch.cuda.device_count()),
                        fallback=0,
                    )
                    idx = 0
                torch.cuda.set_device(idx)
                _SHARED_DEVICE = torch.device(f"cuda:{idx}")
                if g.allow_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                if g.memory_efficient:
                    try:
                        torch.backends.cudnn.benchmark = True
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("cuda_configure_failed", error=str(exc), fallback="cpu")
                _SHARED_DEVICE = torch.device("cpu")
        else:
            if g.wants_cuda:
                logger.warning("cuda_requested_but_unavailable", fallback="cpu")
            _SHARED_DEVICE = torch.device("cpu")
        return _SHARED_DEVICE

    @classmethod
    def device(cls, cfg: Optional[Mapping[str, Any]] = None) -> Any:
        global _SHARED_DEVICE
        if _SHARED_DEVICE is None:
            if cfg is None:
                import torch

                return torch.device("cpu")
            return cls.configure(cfg)
        return _SHARED_DEVICE

    @classmethod
    def reset_for_tests(cls) -> None:
        global _SHARED_DEVICE, _GPU_VALIDATION
        from .device_manager import reset_for_tests as reset_device_profile

        _SHARED_DEVICE = None
        _GPU_VALIDATION = {}
        reset_device_profile()


def cuda_memory_snapshot() -> Dict[str, Any]:
    """Free/total GPU memory (``mem_get_info``) plus Torch allocator stats — safe for logs."""
    out: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        dev = SharedCudaContext.device()
        idx = dev.index if dev.type == "cuda" else torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        out.update(
            {
                "cuda_available": True,
                "device_index": int(idx),
                "free_mb": round(free_b / (1024**2), 2),
                "total_mb": round(total_b / (1024**2), 2),
                "used_mb": round((total_b - free_b) / (1024**2), 2),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024**2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024**2), 2),
            }
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return redact_secrets_obj(out)  # type: ignore[return-value]


def validate_gpu_at_startup(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Check CUDA availability, log GPU name and memory, apply global settings.
    Returns a redacted dict stored for reports.
    """
    global _GPU_VALIDATION
    import torch

    g = resolve_gpu_config(cfg)
    SharedCudaContext.configure(cfg)
    profile = get_compute_profile()
    report: Dict[str, Any] = {
        "requested_device": g.device,
        "gpu_id": g.gpu_id,
        "memory_efficient": g.memory_efficient,
        "allow_tf32": g.allow_tf32,
        "torch_compile": g.torch_compile,
        "adaptive_batch": g.adaptive_batch,
        "resolved_device": str(SharedCudaContext.device(cfg)),
        "cuda_available": bool(torch.cuda.is_available()),
        "compute_profile": profile,
        "cpu_fallback": bool(profile.get("cpu_fallback")),
        "user_message": profile.get("user_message"),
    }
    if torch.cuda.is_available() and SharedCudaContext.device(cfg).type == "cuda":
        idx = SharedCudaContext.device(cfg).index or 0
        props = torch.cuda.get_device_properties(idx)
        report["gpu_name"] = props.name
        report["compute_capability"] = f"{props.major}.{props.minor}"
        report["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        report.update(cuda_memory_snapshot())
    else:
        report["fallback"] = "cpu"
    _GPU_VALIDATION = redact_secrets_obj(report)  # type: ignore[assignment]
    logger.info("gpu_startup_validation", **{k: v for k, v in report.items() if k != "error"})
    return report


def get_gpu_validation_report() -> Dict[str, Any]:
    return dict(_GPU_VALIDATION)


def empty_cuda_cache_after_batch(cfg: Mapping[str, Any], *, label: str = "") -> None:
    g = resolve_gpu_config(cfg)
    if not g.empty_cache_between_batches:
        return
    try:
        import torch

        if torch.cuda.is_available() and SharedCudaContext.device(cfg).type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if label:
                logger.debug("cuda_cache_cleared", label=label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cuda_empty_cache_failed", error=str(exc))


def empty_cuda_cache_after_secure_wipe(cfg: Mapping[str, Any]) -> None:
    """Call after ``secure_wipe_dir`` on temp batches so wiped buffers are not retained in VRAM."""
    empty_cuda_cache_after_batch(cfg, label="post_secure_wipe")


def resolve_gpu_adaptive_batch_size(
    cfg: Mapping[str, Any],
    pending_count: int,
    *,
    base_batch: Optional[int] = None,
) -> int:
    """
    Combine queue-based scaling (``performance``) with VRAM headroom from ``mem_get_info``.
    """
    from .performance import resolve_queue_adaptive_batch_size

    perf = dict(cfg.get("performance") or {})
    g = resolve_gpu_config(cfg)
    queue_batch = int(base_batch) if base_batch is not None else resolve_queue_adaptive_batch_size(cfg, pending_count)

    if not g.adaptive_batch or not g.wants_cuda:
        return queue_batch

    try:
        import torch

        if not torch.cuda.is_available() or SharedCudaContext.device(cfg).type != "cuda":
            return queue_batch

        idx = SharedCudaContext.device(cfg).index or 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        free_mb = free_b / (1024**2)
        max_batch = int(perf.get("max_batch_size", 64))
        min_batch = max(1, int(cfg.get("batch_size", 32)))

        # Heuristic: ~180 MB per image path (DP2 + OCR + LaMa peak) on 24GB class GPUs
        mb_per_image = 200.0 if g.memory_efficient else 280.0
        vram_cap = max(min_batch, int(free_mb / mb_per_image))
        tuned = min(queue_batch, vram_cap, max_batch)
        if tuned < queue_batch:
            logger.info(
                "gpu_adaptive_batch_clamped",
                queue_batch=int(queue_batch),
                vram_batch=int(tuned),
                free_mb=round(free_mb, 1),
            )
        return max(min_batch, tuned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpu_adaptive_batch_fallback", error=str(exc))
        return queue_batch


def dataloader_cuda_kwargs(cfg: Mapping[str, Any], *, num_workers: int = 0) -> Dict[str, Any]:
    """Extra DataLoader kwargs when CUDA is active (``pin_memory``; prefetch only if workers > 0)."""
    g = resolve_gpu_config(cfg)
    import torch

    if not g.wants_cuda or not torch.cuda.is_available():
        return {"pin_memory": False}
    out: Dict[str, Any] = {"pin_memory": bool(g.pin_memory)}
    if num_workers > 0 and g.prefetch_factor:
        out["prefetch_factor"] = max(2, int(g.prefetch_factor))
    return out


def configure_paddle_gpu(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Set Paddle device before PaddleOCR construction; returns status meta."""
    g = resolve_gpu_config(cfg)
    meta: Dict[str, Any] = {"paddle_device": "cpu", "use_gpu": False}
    if not g.wants_cuda:
        return meta
    try:
        import torch

        if not torch.cuda.is_available():
            return meta
        import paddle

        dev = f"gpu:{g.gpu_id}"
        paddle.set_device(dev)
        meta.update({"paddle_device": dev, "use_gpu": True})
        logger.info("paddle_gpu_configured", device=dev)
    except ImportError:
        logger.warning("paddle_not_installed_for_gpu")
    except Exception as exc:  # noqa: BLE001
        logger.warning("paddle_gpu_configure_failed", error=str(exc))
    return meta


def paddle_ocr_device_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Kwargs for PaddleOCR when GPU backend is requested."""
    g = resolve_gpu_config(cfg)
    if not g.wants_cuda:
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
    except Exception:  # noqa: BLE001
        return {}
    dev = f"gpu:{g.gpu_id}"
    for kw in (
        {"device": dev},
        {"use_gpu": True},
        {"use_gpu": True, "gpu_id": g.gpu_id},
    ):
        return dict(kw)
    return {}


def insightface_ctx_id(cfg: Mapping[str, Any]) -> int:
    g = resolve_gpu_config(cfg)
    if not bool((cfg.get("insightface") or {}).get("enabled", False)):
        return -1
    if not g.wants_cuda:
        return -1
    try:
        import torch

        if not torch.cuda.is_available():
            return -1
        return int(g.gpu_id)
    except Exception:  # noqa: BLE001
        return -1


def inpaint_autocast_enabled(cfg: Mapping[str, Any]) -> bool:
    g = resolve_gpu_config(cfg)
    if not g.use_fp16_inpaint or not g.memory_efficient:
        return False
    try:
        import torch

        return g.wants_cuda and torch.cuda.is_available() and SharedCudaContext.device(cfg).type == "cuda"
    except Exception:  # noqa: BLE001
        return False


def maybe_torch_compile(module: Any, cfg: Mapping[str, Any]) -> Any:
    g = resolve_gpu_config(cfg)
    if not g.torch_compile:
        return module
    try:
        import torch

        if hasattr(torch, "compile"):
            return torch.compile(module, mode="reduce-overhead")
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch_compile_skipped", error=str(exc))
    return module


def warm_gpu_dummy_forward(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Tiny CUDA op to warm allocator/driver (no image data logged)."""
    try:
        import torch

        dev = SharedCudaContext.device(cfg)
        if dev.type != "cuda":
            return {"warmed": False, "reason": "cpu"}
        x = torch.zeros(1, device=dev)
        _ = x + 1
        torch.cuda.synchronize()
        empty_cuda_cache_after_batch(cfg, label="post_warmup")
        return {"warmed": True, **cuda_memory_snapshot()}
    except Exception as exc:  # noqa: BLE001
        return {"warmed": False, "error": str(exc)}


def merge_warm_meta(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in parts:
        out.update(dict(p))
    return redact_secrets_obj(out)  # type: ignore[return-value]


GPU_READINESS_REPORT = "gpu_readiness_report.md"


def write_gpu_readiness_report(
    project_root: Path,
    cfg: Mapping[str, Any],
    *,
    gpu_validation: Optional[Mapping[str, Any]] = None,
    warm_meta: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    model_stats: Optional[Mapping[str, Any]] = None,
    elapsed_sec: Optional[float] = None,
    benchmark_compare: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``reports/gpu_readiness_report.md`` with CUDA tuning notes and run metrics."""
    from datetime import datetime, timezone

    project_root = Path(project_root).resolve()
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / GPU_READINESS_REPORT
    g = resolve_gpu_config(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    metrics = metrics or {}
    ips = float(metrics.get("images_per_sec") or 0.0)
    if not ips and elapsed_sec and metrics.get("total_routed"):
        ips = float(metrics["total_routed"]) / float(elapsed_sec)

    lines = [
        "# GPU Readiness Report",
        "",
        f"Generated: {ts}",
        "",
        "## CUDA configuration",
        "",
        f"| Setting | Value |",
        f"|---------|-------|",
        f"| device | `{g.device}` |",
        f"| gpu_id | `{g.gpu_id}` |",
        f"| memory_efficient | `{g.memory_efficient}` |",
        f"| torch_compile | `{g.torch_compile}` |",
        f"| allow_tf32 | `{g.allow_tf32}` |",
        f"| adaptive_batch | `{g.adaptive_batch}` |",
        f"| empty_cache_between_batches | `{g.empty_cache_between_batches}` |",
        f"| use_fp16_inpaint | `{g.use_fp16_inpaint}` |",
        f"| pin_memory (DataLoader) | `{g.pin_memory}` |",
        "",
        "## Startup validation",
        "",
    ]
    val = gpu_validation or get_gpu_validation_report()
    for k, v in sorted(val.items()):
        lines.append(f"- **{k}**: `{v}`")

    lines.extend(
        [
            "",
            "## Model warm-up",
            "",
            "```json",
            json.dumps(redact_secrets_obj(dict(warm_meta or {})), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Run throughput",
            "",
            f"- Images/sec: **{ips:.3f}**",
            f"- Avg sec/image: **{float(metrics.get('avg_sec_per_image', 0)):.3f}**",
            f"- Total routed: **{metrics.get('total_routed', 0)}**",
            f"- Pipeline elapsed: **{float(elapsed_sec or 0):.1f}s**",
            "",
        ]
    )

    if model_stats:
        lines.extend(
            [
                "## Shared models",
                "",
                f"- OCR inference calls: **{model_stats.get('ocr_inference_calls', 0)}**",
                f"- OCR QA cache hits: **{model_stats.get('ocr_cache_hits', 0)}**",
                f"- InsightFace calls: **{model_stats.get('insightface_calls', 0)}**",
                "",
            ]
        )

    snap = cuda_memory_snapshot()
    if snap.get("cuda_available"):
        lines.extend(
            [
                "## VRAM snapshot (post-run)",
                "",
                f"- Free: **{snap.get('free_mb')} MB** / Total: **{snap.get('total_mb')} MB**",
                f"- Torch allocated: **{snap.get('allocated_mb')} MB**",
                f"- Torch reserved: **{snap.get('reserved_mb')} MB**",
                "",
            ]
        )

    lines.extend(
        [
            "## CoreWeave instance guidance",
            "",
            "| GPU | VRAM | Suggested batch_size | Notes |",
            "|-----|------|---------------------|-------|",
            "| RTX 4090 | 24 GB | 24–48 | DP2 + LaMa + OCR; enable `memory_efficient` |",
            "| A100 40GB | 40 GB | 48–64 | Production 37k runs; `adaptive_batch: true` |",
            "| A100 80GB | 80 GB | 64+ | Raise `performance.max_batch_size` |",
            "| H100 | 80 GB | 64+ | TF32 on by default; monitor thermals |",
            "",
            "- Use **one process per GPU** (`CUDA_VISIBLE_DEVICES` + `gpu.gpu_id: 0`).",
            "- Set `empty_cache_between_batches: true` when mixing large portraits and text-heavy frames.",
            "- For CPU fallback, set `gpu.device: cpu` — PaddleOCR and InsightFace follow automatically.",
            "",
            "## Security",
            "",
            "- Logs redact secrets; GPU snapshots contain **memory stats only** (no tensors/embeddings).",
            "- `secure_wipe` on temp dirs triggers `torch.cuda.empty_cache()` after wipe.",
            "",
            "## Optimizations enabled",
            "",
            "- Single `SharedCudaContext` for DP2, IOPaint/LaMa, and Torch ops",
            "- Shared PaddleOCR + InsightFace singletons (processing + QA)",
            "- VRAM-aware batch clamp via `torch.cuda.mem_get_info()`",
            "- Optional fp16 autocast for IOPaint when `use_fp16_inpaint: true`",
            "- `pin_memory` on path DataLoader when CUDA active",
            "",
        ]
    )

    if benchmark_compare:
        lines.extend(
            [
                "## Benchmark comparison",
                "",
                "| Mode | images/sec | avg sec/img | peak allocated MB |",
                "|------|------------|-------------|-------------------|",
            ]
        )
        for mode, row in benchmark_compare.items():
            if row.get("skipped"):
                lines.append(f"| {mode} | — | — | skipped: `{row.get('reason', '')}` |")
                continue
            lines.append(
                f"| {mode} | {float(row.get('images_per_sec', 0)):.3f} | "
                f"{float(row.get('avg_sec_per_image', 0)):.3f} | {row.get('peak_allocated_mb', '—')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## How to re-run benchmarks",
            "",
            "```bash",
            "python -m scripts.run_gpu_roundtrip_test --count 200",
            "python -m scripts.main_pipeline   # uses config.yaml gpu: block",
            "```",
            "",
            "On CoreWeave, install CUDA-matched `torch`, `paddlepaddle-gpu`, and ONNX Runtime GPU for InsightFace.",
            "Re-run the roundtrip test; `reports/gpu_benchmark.json` will include a `gpu` row with VRAM peaks.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("gpu_readiness_report_written", path=str(path))
    return path
