"""
Structured monitoring: append JSON lines to ``logs/monitoring.jsonl`` for analysis.

All records are redacted (no secrets, paths to credentials, or image payloads).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .security import redact_secrets_obj
from .utils import get_logger, utc_now_iso

logger = get_logger(__name__)

_LOCK = threading.Lock()
_MONITOR: Optional["MonitoringLogger"] = None


class MonitoringLogger:
    """Append-only JSONL metrics sink + structlog mirror."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.peak_gpu_allocated_mb: float = 0.0
        self.security_events: list[Dict[str, Any]] = []
        self.transfer_events: list[Dict[str, Any]] = []

    def record(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "type": event_type,
        }
        safe = redact_secrets_obj(payload)
        if isinstance(safe, dict):
            row.update(safe)
        else:
            row["payload"] = safe

        with _LOCK:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(f"monitor_{event_type}", **(safe if isinstance(safe, dict) else {"data": safe}))

        if event_type.startswith("security_"):
            self.security_events.append(row)
        if event_type.startswith("transfer_") or event_type.startswith("b2_"):
            self.transfer_events.append(row)

        alloc = row.get("allocated_mb")
        if alloc is not None:
            try:
                self.peak_gpu_allocated_mb = max(self.peak_gpu_allocated_mb, float(alloc))
            except (TypeError, ValueError):
                pass
        return row

    def record_batch(
        self,
        *,
        batch_index: int,
        images: int,
        elapsed_sec: float,
        sec_per_image: float,
        images_per_sec: float,
        eta_sec: Optional[float],
        counters: Mapping[str, int],
        gpu_snapshot: Optional[Mapping[str, Any]] = None,
        resource_snapshot: Optional[Mapping[str, Any]] = None,
        quarantine_rate_cumulative: float = 0.0,
    ) -> None:
        self.record(
            "batch_complete",
            batch_index=int(batch_index),
            images=int(images),
            elapsed_sec=float(elapsed_sec),
            sec_per_image=float(sec_per_image),
            images_per_sec=float(images_per_sec),
            eta_sec=float(eta_sec) if eta_sec is not None else None,
            counters=dict(counters),
            quarantine_rate_cumulative=float(quarantine_rate_cumulative),
            gpu=dict(gpu_snapshot or {}),
            resources=dict(resource_snapshot or {}),
        )

    def record_pipeline_start(self, **kwargs: Any) -> None:
        self.record("pipeline_start", **kwargs)

    def record_pipeline_complete(self, **kwargs: Any) -> None:
        self.record("pipeline_complete", **kwargs)

    def record_security(self, name: str, **kwargs: Any) -> None:
        self.record(f"security_{name}", **kwargs)

    def record_transfer(self, phase: str, **kwargs: Any) -> None:
        self.record(f"transfer_{phase}", **kwargs)

    def summary(self) -> Dict[str, Any]:
        return {
            "log_path": str(self.log_path),
            "peak_gpu_allocated_mb": float(self.peak_gpu_allocated_mb),
            "security_event_count": len(self.security_events),
            "transfer_event_count": len(self.transfer_events),
        }


def init_monitoring(logs_dir: Path) -> MonitoringLogger:
    global _MONITOR
    path = Path(logs_dir) / "monitoring.jsonl"
    _MONITOR = MonitoringLogger(path)
    return _MONITOR


def get_monitoring() -> Optional[MonitoringLogger]:
    return _MONITOR


def attach_security_context(monitor: MonitoringLogger, security_ctx: Any) -> None:
    """Mirror future security_ctx events into monitoring (hook after pipeline constructs ctx)."""

    original_log = security_ctx.log_event

    def _wrapped(name: str, **payload: Any) -> None:
        original_log(name, **payload)
        monitor.record_security(name, **payload)

    security_ctx.log_event = _wrapped  # type: ignore[method-assign]
