"""Runtime metrics helpers for pipeline test mode and observability."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)


def log_gpu_memory(label: str) -> Dict[str, Any]:
    """Log CUDA free/total + allocator stats (no image/tensor payload)."""
    from .gpu_runtime import cuda_memory_snapshot

    snapshot: Dict[str, Any] = {"label": label, **cuda_memory_snapshot()}
    logger.info("gpu_memory", **snapshot)
    return snapshot


def log_system_resources(label: str) -> Dict[str, Any]:
    """Log CPU%, RSS memory, and disk via psutil."""
    snapshot: Dict[str, Any] = {"label": label}
    try:
        import psutil

        proc = psutil.Process()
        mem = proc.memory_info()
        snapshot.update(
            {
                "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
                "rss_mb": float(mem.rss / (1024**2)),
                "vms_mb": float(mem.vms / (1024**2)),
                "system_memory_percent": float(psutil.virtual_memory().percent),
            }
        )
        logger.info("system_resources", **snapshot)
    except Exception as exc:  # noqa: BLE001
        snapshot["error"] = str(exc)
        logger.warning("system_resources_unavailable", **snapshot)
    return snapshot


class PipelineMetricsCollector:
    """Collect per-batch timing, GPU/CPU/RAM snapshots, routing counters, ETA, and bottlenecks."""

    def __init__(self, *, resource_monitoring: bool = True) -> None:
        self.resource_monitoring = bool(resource_monitoring)
        self.batch_timings: List[Dict[str, Any]] = []
        self.gpu_snapshots: List[Dict[str, Any]] = []
        self.resource_snapshots: List[Dict[str, Any]] = []
        self._batch_t0: Optional[float] = None
        self._batch_index: Optional[int] = None
        self._batch_n: int = 0
        self._run_t0: float = time.monotonic()
        self.totals = {"pass": 0, "quarantine": 0, "manual_review": 0, "fail": 0}
        self._processed_cumulative: int = 0
        self._total_hint: int = 0

    def set_total_hint(self, n: int) -> None:
        self._total_hint = max(0, int(n))

    def on_batch_start(self, batch_index: int, n: int) -> None:
        self._batch_t0 = time.monotonic()
        self._batch_index = int(batch_index)
        self._batch_n = int(n)
        self.gpu_snapshots.append(log_gpu_memory(f"batch_{batch_index:05d}_start"))
        if self.resource_monitoring:
            self.resource_snapshots.append(log_system_resources(f"batch_{batch_index:05d}_start"))

    def on_batch_complete(
        self,
        batch_index: int,
        *,
        counters: Dict[str, int],
        processed_this_run: int,
    ) -> Dict[str, Any]:
        elapsed = (time.monotonic() - self._batch_t0) if self._batch_t0 is not None else 0.0
        n = max(1, self._batch_n)
        per_image_sec = elapsed / n
        self._processed_cumulative = int(processed_this_run)
        run_elapsed = max(1e-6, time.monotonic() - self._run_t0)
        ips = self._processed_cumulative / run_elapsed if self._processed_cumulative else 0.0
        remaining = max(0, self._total_hint - self._processed_cumulative)
        eta_sec = (remaining / ips) if ips > 0 else None
        self.gpu_snapshots.append(log_gpu_memory(f"batch_{batch_index:05d}_end"))
        resource_end: Optional[Dict[str, Any]] = None
        if self.resource_monitoring:
            resource_end = log_system_resources(f"batch_{batch_index:05d}_end")
            self.resource_snapshots.append(resource_end)

        for key in ("pass", "quarantine", "manual_review", "fail"):
            self.totals[key] = int(self.totals.get(key, 0)) + int(counters.get(key, 0))

        row = {
            "batch_index": int(batch_index),
            "images": int(n),
            "elapsed_sec": float(elapsed),
            "sec_per_image": float(per_image_sec),
            "images_per_sec": float(1.0 / per_image_sec) if per_image_sec > 0 else 0.0,
            "processed_cumulative": int(processed_this_run),
            "eta_sec": float(eta_sec) if eta_sec is not None else None,
            "counters": dict(counters),
            "resources_end": resource_end,
        }
        self.batch_timings.append(row)
        routed_batch = sum(int(counters.get(k, 0)) for k in ("pass", "quarantine", "manual_review", "fail"))
        q_rate = (
            (self.totals["quarantine"] / max(1, sum(self.totals.values())))
            if sum(self.totals.values())
            else 0.0
        )
        gpu_end = self.gpu_snapshots[-1] if self.gpu_snapshots else {}
        logger.info(
            "batch_timing",
            batch_index=int(batch_index),
            images=int(n),
            sec_per_image=float(per_image_sec),
            images_per_sec=row["images_per_sec"],
            eta_sec=row["eta_sec"],
            quarantine=int(counters.get("quarantine", 0)),
            quarantine_rate_cumulative=float(q_rate),
        )
        try:
            from .monitoring import get_monitoring

            mon = get_monitoring()
            if mon is not None:
                mon.record_batch(
                    batch_index=int(batch_index),
                    images=int(n),
                    elapsed_sec=float(elapsed),
                    sec_per_image=float(per_image_sec),
                    images_per_sec=float(row["images_per_sec"]),
                    eta_sec=row.get("eta_sec"),
                    counters=dict(counters),
                    gpu_snapshot=gpu_end,
                    resource_snapshot=resource_end,
                    quarantine_rate_cumulative=float(q_rate),
                )
        except Exception:  # noqa: BLE001
            pass
        row["quarantine_rate_cumulative"] = float(q_rate)
        row["routed_batch"] = int(routed_batch)
        return row

    def summary(self, *, total_input: int) -> Dict[str, Any]:
        routed = sum(self.totals.values())
        quarantine_rate = (self.totals["quarantine"] / routed) if routed else 0.0
        manual_rate = (self.totals["manual_review"] / routed) if routed else 0.0
        pass_rate = (self.totals["pass"] / routed) if routed else 0.0
        total_elapsed = sum(r["elapsed_sec"] for r in self.batch_timings)
        avg_per_image = (total_elapsed / routed) if routed else 0.0

        peak_gpu_mb = 0.0
        for snap in self.gpu_snapshots:
            try:
                peak_gpu_mb = max(peak_gpu_mb, float(snap.get("allocated_mb") or 0))
            except (TypeError, ValueError):
                continue

        out = {
            "total_input_images": int(total_input),
            "total_routed": int(routed),
            "pass_count": int(self.totals["pass"]),
            "quarantine_count": int(self.totals["quarantine"]),
            "manual_review_count": int(self.totals["manual_review"]),
            "pass_rate": float(pass_rate),
            "quarantine_rate": float(quarantine_rate),
            "manual_review_rate": float(manual_rate),
            "total_elapsed_sec": float(total_elapsed),
            "avg_sec_per_image": float(avg_per_image),
            "images_per_sec": float(routed / total_elapsed) if total_elapsed > 0 else 0.0,
            "peak_gpu_allocated_mb": float(peak_gpu_mb),
            "batch_timings": list(self.batch_timings),
            "gpu_snapshots": list(self.gpu_snapshots),
            "resource_snapshots": list(self.resource_snapshots),
        }
        logger.info(
            "pipeline_metrics_summary",
            **{k: v for k, v in out.items() if k not in ("batch_timings", "gpu_snapshots", "resource_snapshots")},
        )
        return out
