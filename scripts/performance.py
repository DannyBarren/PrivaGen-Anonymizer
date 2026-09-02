"""
High-volume performance helpers: adaptive batching, ingest screening, reports.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils import get_logger, utc_now_iso

logger = get_logger(__name__)

PERFORMANCE_REPORT_NAME = "performance_report.md"


def resolve_num_workers(cfg: Mapping[str, Any]) -> int:
    """Tune joblib decode parallelism from config + CPU count."""
    perf = dict(cfg.get("performance") or {})
    configured = int(cfg.get("num_workers", 4) or 4)
    if not bool(perf.get("auto_num_workers", True)):
        return max(1, configured)

    cpu = os.cpu_count() or 4
    max_w = int(perf.get("max_num_workers", 8))
    reserve = int(perf.get("cpu_reserve", 2))
    auto = max(1, min(configured, cpu - reserve, max_w))
    return auto


def resolve_queue_adaptive_batch_size(cfg: Mapping[str, Any], pending_count: int) -> int:
    """Scale batch size from pending queue depth only (no VRAM probe)."""
    perf = dict(cfg.get("performance") or {})
    base = max(1, int(cfg.get("batch_size", 32)))
    if not bool(perf.get("adaptive_batch_size", True)):
        return base

    max_batch = int(perf.get("max_batch_size", 64))
    large_at = int(perf.get("large_dataset_threshold", 5000))
    medium_at = int(perf.get("medium_dataset_threshold", 1000))

    if pending_count >= large_at:
        return min(max_batch, max(base, int(base * 2)))
    if pending_count >= medium_at:
        return min(max_batch, max(base, int(base * 1.5)))
    return base


def resolve_adaptive_batch_size(cfg: Mapping[str, Any], pending_count: int) -> int:
    """Queue-based scaling, then optional GPU VRAM clamp (``gpu.adaptive_batch``)."""
    from .gpu_runtime import resolve_gpu_adaptive_batch_size, resolve_gpu_config

    batch = resolve_queue_adaptive_batch_size(cfg, pending_count)
    g = resolve_gpu_config(cfg)
    if g.adaptive_batch and g.wants_cuda:
        return resolve_gpu_adaptive_batch_size(cfg, pending_count, base_batch=batch)
    return batch


def _image_pixel_count(path: Path) -> Optional[int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w) * int(h)
    except Exception:
        return None


def screen_input_images(
    paths: Sequence[Path],
    cfg: Mapping[str, Any],
    *,
    quarantine_dir: Path,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Early quarantine for corrupted, oversize, or extreme-resolution images on ingest.

    Returns (accepted_paths, quarantine_records).
    """
    perf = dict(cfg.get("performance") or {})
    if not bool(perf.get("ingest_screening", True)):
        return list(paths), []

    max_bytes = int(perf.get("max_ingest_bytes", 50 * 1024 * 1024))
    max_pixels = int(perf.get("max_ingest_pixels", 50_000_000))
    min_pixels = int(perf.get("min_ingest_pixels", 64 * 64))

    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    accepted: List[Path] = []
    rejected: List[Dict[str, Any]] = []

    for path in paths:
        path = Path(path)
        reason: Optional[str] = None
        try:
            size = path.stat().st_size
            if size > max_bytes:
                reason = f"file_too_large:{size}>{max_bytes}"
            elif size == 0:
                reason = "empty_file"
            else:
                pixels = _image_pixel_count(path)
                if pixels is None:
                    reason = "corrupted_or_unreadable"
                elif pixels > max_pixels:
                    reason = f"resolution_too_high:{pixels}>{max_pixels}"
                elif pixels < min_pixels:
                    reason = f"resolution_too_low:{pixels}<{min_pixels}"
        except OSError as exc:
            reason = f"stat_failed:{exc}"

        if reason:
            dst = quarantine_dir / path.name
            try:
                if path.resolve().parent != quarantine_dir.resolve():
                    shutil.copy2(path, dst)
            except OSError as exc:
                logger.warning("ingest_screen_copy_failed", src=str(path), error=str(exc))
            rejected.append(
                {
                    "path": str(path),
                    "reason": reason,
                    "timestamp": utc_now_iso(),
                }
            )
            logger.info("ingest_screen_quarantine", path=str(path), reason=reason)
        else:
            accepted.append(path)

    if rejected:
        log_path = quarantine_dir / "ingest_screen_rejects.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            for rec in rejected:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return accepted, rejected


def detect_bottlenecks(batch_timings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Flag slow batches and high per-image latency outliers."""
    if not batch_timings:
        return []

    spi_values = [float(r.get("sec_per_image") or 0.0) for r in batch_timings]
    avg_spi = sum(spi_values) / len(spi_values) if spi_values else 0.0
    bottlenecks: List[Dict[str, Any]] = []

    for row in batch_timings:
        spi = float(row.get("sec_per_image") or 0.0)
        if avg_spi > 0 and spi >= avg_spi * 1.75:
            bottlenecks.append(
                {
                    "batch_index": row.get("batch_index"),
                    "sec_per_image": spi,
                    "avg_sec_per_image": avg_spi,
                    "reason": "slow_batch_outlier",
                }
            )

    if avg_spi > 5.0:
        bottlenecks.append(
            {
                "reason": "high_avg_latency",
                "avg_sec_per_image": avg_spi,
                "hint": "Consider GPU, larger batch_size, or disabling InsightFace on CPU runs",
            }
        )
    return bottlenecks


def build_recommendations(
    metrics: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    model_stats: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Human-readable tuning suggestions from a run."""
    recs: List[str] = []
    perf = dict(cfg.get("performance") or {})
    avg_spi = float(metrics.get("avg_sec_per_image") or 0.0)
    q_rate = float(metrics.get("quarantine_rate") or 0.0)
    pass_rate = float(metrics.get("pass_rate") or 0.0)

    if avg_spi > 3.0:
        recs.append(
            f"Average {avg_spi:.2f}s/image is high — enable CUDA, increase batch_size "
            f"(current max {perf.get('max_batch_size', 64)}), or set insightface.enabled: false for bulk CPU runs."
        )
    elif avg_spi < 1.0 and int(cfg.get("batch_size", 32)) < int(perf.get("max_batch_size", 64)):
        recs.append("Throughput headroom detected — adaptive batch_size can raise batch size on large queues.")

    if q_rate > 0.05:
        recs.append(
            f"Quarantine rate {q_rate:.1%} exceeds 5% — run calibrate_qa_thresholds on 200 images "
            "and review artifact_ssim_min (typical range 0.72–0.80)."
        )

    if pass_rate < 0.95:
        recs.append(f"Pass rate {pass_rate:.1%} below 95% — review QA thresholds and ingest screening rejects.")

    if model_stats:
        cache_rate = float(model_stats.get("ocr_cache_hit_rate") or 0.0)
        if cache_rate < 0.5:
            recs.append(
                "OCR cache hit rate low — ensure qa.reuse_processing_ocr: true and processing writes qa_cache.output_ocr_boxes."
            )
        elif cache_rate >= 0.5:
            recs.append(
                f"OCR cache hit rate {cache_rate:.0%} — shared PaddleOCR avoiding redundant inference on QA."
            )

    if not recs:
        recs.append("Metrics within expected bounds for current hardware profile.")
    return recs


def write_performance_report(
    project_root: Path,
    metrics: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    model_stats: Optional[Mapping[str, Any]] = None,
    ingest_screen: Optional[Mapping[str, Any]] = None,
    elapsed_sec: Optional[float] = None,
) -> Path:
    """Write ``reports/performance_report.md`` with metrics and recommendations."""
    project_root = Path(project_root).resolve()
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / PERFORMANCE_REPORT_NAME
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    bottlenecks = detect_bottlenecks(metrics.get("batch_timings") or [])
    recommendations = build_recommendations(metrics, cfg, model_stats=model_stats)

    total_routed = int(metrics.get("total_routed") or 0)
    total_elapsed = float(elapsed_sec or metrics.get("total_elapsed_sec") or 0.0)
    ips = (total_routed / total_elapsed) if total_elapsed > 0 else 0.0

    lines = [
        "# Performance Report",
        "",
        f"Generated: {ts}",
        "",
        "## Summary",
        "",
        f"- Total input images: **{metrics.get('total_input_images', 0)}**",
        f"- Routed: **{total_routed}**",
        f"- Pass rate: **{100 * float(metrics.get('pass_rate', 0)):.1f}%**",
        f"- Quarantine rate: **{100 * float(metrics.get('quarantine_rate', 0)):.1f}%**",
        f"- Avg sec/image: **{float(metrics.get('avg_sec_per_image', 0)):.3f}**",
        f"- Throughput: **{ips:.2f} images/sec**",
        f"- Total elapsed: **{total_elapsed:.1f}s**",
        "",
        "## Configuration",
        "",
        f"- batch_size: `{cfg.get('batch_size')}` (adaptive: `{((cfg.get('performance') or {}).get('adaptive_batch_size', True))}`)",
        f"- num_workers: `{resolve_num_workers(cfg)}` (configured: `{cfg.get('num_workers')}`)",
        f"- device: `{cfg.get('device')}` (gpu: `{((cfg.get('gpu') or {}).get('device'))}`)",
        f"- qa.reuse_processing_ocr: `{((cfg.get('qa') or {}).get('reuse_processing_ocr', True))}`",
        "",
    ]

    if model_stats:
        lines.extend(
            [
                "## Shared Model Usage",
                "",
                f"- PaddleOCR inference calls: **{model_stats.get('ocr_inference_calls', 0)}**",
                f"- QA OCR cache hits: **{model_stats.get('ocr_cache_hits', 0)}**",
                f"- OCR cache hit rate: **{100 * float(model_stats.get('ocr_cache_hit_rate', 0)):.1f}%**",
                f"- InsightFace inference calls: **{model_stats.get('insightface_calls', 0)}**",
                "",
            ]
        )

    if ingest_screen:
        lines.extend(
            [
                "## Ingest Screening",
                "",
                f"- Accepted: **{ingest_screen.get('accepted', 0)}**",
                f"- Quarantined: **{ingest_screen.get('rejected', 0)}**",
                "",
            ]
        )

    if bottlenecks:
        lines.append("## Bottlenecks")
        lines.append("")
        for b in bottlenecks[:10]:
            lines.append(f"- `{json.dumps(b, ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for r in recommendations:
        lines.append(f"- {r}")
    lines.append("")

    # Per-batch table (last 10)
    timings = list(metrics.get("batch_timings") or [])
    if timings:
        lines.extend(["## Recent Batch Timings", "", "| Batch | Images | sec/img | ETA hint |", "|------:|-------:|--------:|---------:|"])
        for row in timings[-10:]:
            eta = row.get("eta_sec")
            eta_s = f"{float(eta):.0f}s" if eta is not None else "—"
            lines.append(
                f"| {row.get('batch_index')} | {row.get('images')} | "
                f"{float(row.get('sec_per_image', 0)):.3f} | {eta_s} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("performance_report_written", path=str(path), images_per_sec=ips)
    return path
