"""
Bridge pipeline progress callbacks to Web UI Socket.IO events.

Safe when Flask is not loaded — callbacks no-op if emit is None.
Does not log secrets or full filesystem paths (uses counts and batch indices only).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Optional

EmitFn = Callable[[str, Dict[str, Any]], None]

_state_lock = threading.Lock()
_live: Dict[str, Any] = {
    "status": "idle",
    "total_detected": 0,
    "total_target": 0,
    "processed": 0,
    "progress_pct": 0.0,
    "current_batch": 0,
    "total_batches_estimate": 0,
    "current_batch_size": 0,
    "compute_mode": "unknown",
    "cpu_fallback": False,
    "mode_explanation": "",
    "images_per_sec": 0.0,
    "eta_sec": None,
    "success_rate": None,
    "quarantine_rate": None,
    "wave": 0,
    "updated_at": None,
}


def reset_live_state(*, total_detected: int = 0) -> None:
    with _state_lock:
        _live.update(
            {
                "status": "idle",
                "total_detected": int(total_detected),
                "total_target": 0,
                "processed": 0,
                "progress_pct": 0.0,
                "current_batch": 0,
                "total_batches_estimate": 0,
                "current_batch_size": 0,
                "compute_mode": "unknown",
                "cpu_fallback": False,
                "mode_explanation": "",
                "images_per_sec": 0.0,
                "eta_sec": None,
                "success_rate": None,
                "quarantine_rate": None,
                "wave": 0,
                "updated_at": time.time(),
            }
        )


def get_live_state() -> Dict[str, Any]:
    with _state_lock:
        return dict(_live)


def set_live_field(**kwargs: Any) -> None:
    with _state_lock:
        _live.update(kwargs)
        _live["updated_at"] = time.time()
        proc = int(_live.get("processed") or 0)
        tgt = int(_live.get("total_target") or 0)
        if tgt > 0:
            _live["progress_pct"] = round(100.0 * min(proc, tgt) / tgt, 2)
        elif proc > 0:
            _live["progress_pct"] = 100.0


def apply_compute_profile(profile: Optional[Dict[str, Any]]) -> None:
    if not profile:
        return
    cpu_fb = bool(profile.get("cpu_fallback"))
    if cpu_fb:
        mode = "CPU fallback"
        expl = profile.get("user_message") or (
            "Basic anonymization: PaddleOCR text redaction + OpenCV face blur. "
            "Targeted GAN inpainting disabled."
        )
    elif profile.get("mode") == "gpu_full" or profile.get("gan_inpainting_enabled"):
        mode = "GPU"
        expl = "Full pipeline: DeepPrivacy2 (when vendored), IOPaint/LaMa, GPU OCR."
    else:
        mode = "CPU" if str(profile.get("resolved_device", "")).startswith("cpu") else "GPU"
        expl = profile.get("user_message") or ""
    set_live_field(
        compute_mode=mode,
        cpu_fallback=cpu_fb,
        mode_explanation=str(expl)[:500] if expl else "",
    )


def _merge_event(event: Dict[str, Any]) -> None:
    t = str(event.get("type", ""))
    if t == "pipeline_start":
        set_live_field(status="running", processed=0, current_batch=0)
        prof = event.get("compute_profile") or {}
        apply_compute_profile(prof)
    elif t == "wave_start":
        pending = int(event.get("pending") or 0)
        cur_tgt = int(get_live_state().get("total_target") or 0)
        set_live_field(
            status="running",
            wave=int(event.get("wave") or 0),
            total_target=max(cur_tgt, pending),
        )
    elif t == "batch_start":
        n = int(event.get("n") or 0)
        bi = int(event.get("batch_index") or 0)
        tbe = int(event.get("total_batches_estimate") or _live.get("total_batches_estimate") or 0)
        set_live_field(
            status="running",
            current_batch=bi,
            current_batch_size=n,
            total_batches_estimate=tbe or bi,
        )
    elif t == "batch_complete":
        proc = int(event.get("processed_this_run") or 0)
        tgt = int(event.get("total_hint") or event.get("total_target") or 0)
        set_live_field(
            processed=proc,
            total_target=tgt,
            current_batch=int(event.get("batch_index") or 0),
            images_per_sec=float(event.get("images_per_sec") or 0),
            eta_sec=event.get("eta_sec"),
            success_rate=event.get("success_rate"),
            quarantine_rate=event.get("quarantine_rate"),
        )
    elif t == "progress_tick":
        if "processed" in event:
            set_live_field(processed=int(event["processed"]))
        if "total_target" in event:
            set_live_field(total_target=int(event["total_target"]))
        if "images_per_sec" in event:
            set_live_field(images_per_sec=float(event["images_per_sec"]))
        if "eta_sec" in event:
            set_live_field(eta_sec=event.get("eta_sec"))
    elif t == "pipeline_complete":
        set_live_field(status="idle", progress_pct=100.0)
    elif t == "pipeline_error":
        set_live_field(status="error")


def wrap_progress_callback(
    emit: Optional[EmitFn],
    base: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Return a callback that updates live state and forwards to ``emit`` + ``base``."""

    def _forward(event: Dict[str, Any]) -> None:
        payload = dict(event)
        _merge_event(payload)
        if base is not None:
            try:
                base(payload)
            except Exception:  # noqa: BLE001
                pass
        if emit is None:
            return
        t = str(payload.get("type", ""))
        try:
            emit("pipeline_event", payload)
            status = get_live_state()
            emit("pipeline_status_update", status)
            if t in ("batch_start", "batch_complete", "progress_tick", "pipeline_complete"):
                emit(t, {**payload, **status})
        except Exception:  # noqa: BLE001
            pass

    return _forward


def estimate_total_batches(pending_count: int, batch_size: int) -> int:
    bs = max(1, int(batch_size))
    return max(1, int(math.ceil(max(0, pending_count) / bs)))


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "—"
    s = int(max(0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"
