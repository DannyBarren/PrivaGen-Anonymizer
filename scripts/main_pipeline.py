"""
Main orchestrator for the production-grade image anonymization pipeline.

Workflow (exact sequence):
1) One-time folder scaffolding + logging setup
2) Model warm-up (DeepPrivacy2 + PaddleOCR + LaMa + optional InsightFace probe)
3) Outer “wave” loop:
   - Discover pending work from `input_raw/` (not yet present in `final_clean/`) and `quarantine/`
     (automatic retries under max_retries)
   - For each batch:
       a) `batch_processor.process_batch` → `temp_processed/batch_XXXXX/`
       b) `agentic_qa_crew.run_qa_on_batch_directory` (CrewAI optional; deterministic QA always)
       c) Route each image to `final_clean/`, `quarantine/`, or `manual_review/`
4) Emit per-batch CSV summaries + buyer-facing master CSV/PDF + JSON audit array

Run (from project root):
    python -m scripts.main_pipeline
"""

from __future__ import annotations

import argparse
import copy
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

# NOTE: ``agentic_qa_crew`` and ``batch_processor`` are intentionally NOT imported at
# module top-level. They pull in the heavy GPU/ML stack (torch, cv2, scikit-image,
# paddleocr) which is only required for an actual processing run. Importing them lazily
# inside the functions that need them keeps lightweight operations — most importantly
# ``run_pipeline(dry_run=True)`` — usable as a pre-flight health check on a plain CPU
# host (e.g. before renting a GPU instance) without the full dependency set installed.
from .component_check import run_component_activation_check
from .device_manager import get_compute_profile, initialize_compute
from .gpu_runtime import (
    cuda_memory_snapshot,
    get_gpu_validation_report,
    sync_cfg_device_fields,
    validate_gpu_at_startup,
    write_gpu_readiness_report,
)
from .performance import (
    resolve_adaptive_batch_size,
    screen_input_images,
    write_performance_report,
)
from .pipeline_metrics import PipelineMetricsCollector, log_gpu_memory, log_system_resources
from .processing_locks import active_locked_stems, discover_resume_batch_dirs, mark_processing_complete
from .shared_models import get_model_usage_stats
from .rclone_integration import (
    default_export_command,
    default_ingest_command,
    export_to_backblaze,
    default_rclone_commands,
    ingest_from_backblaze,
    load_b2_config,
    write_rclone_config,
)
from .security import resolve_under
from .monitoring import attach_security_context, get_monitoring, init_monitoring
from .security_hardening import (
    SecurityContext,
    backup_critical_artifacts,
    load_security_hardening,
    stems_fully_processed,
)
from .utils import (
    append_master_audit,
    compute_master_statistics,
    configure_structlog,
    deep_update,
    discover_images,
    ensure_model_assets,
    get_logger,
    load_audit_json,
    load_config,
    resolve_log_artifacts,
    resolve_pipeline_paths,
    resolve_project_root,
    resolve_report_artifact,
    save_audit_json,
    set_input_raw_guard,
    setup_project_folders,
    safe_move,
    write_batch_summary_csv,
    write_master_summary_csv_and_pdf,
)

logger = get_logger(__name__)

# Safe rclone commands, hard-set as the pipeline defaults. Ingest keeps --dry-run until
# an operator removes it for the real copy; export uses --checksum --fast-list
# --transfers 16.
DEFAULT_INGEST_COMMAND = default_ingest_command(dry_run=True)
DEFAULT_INGEST_COMMAND_REAL = default_ingest_command(dry_run=False)
DEFAULT_EXPORT_COMMAND = default_export_command()

# --- Processing scope: IMAGES ONLY (video support deferred indefinitely) --------
# This is a hard, intentional design constraint. The pipeline only ever discovers and
# processes image files (see utils.discover_images); video files are excluded at ingest.
# There is no video decode / frame-extraction / re-encode / audio path in this codebase.
PROCESSING_MODE = "images_only"
VIDEO_SUPPORT = "deferred"
SCOPE_BANNER = "Current Scope: Images Only (Video support deferred)"


def _write_run_scope_marker(reports_dir: Path, *, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Stamp every run with an Image-Only scope marker in the audit trail
    (``reports/run_scope.json``), so each job is explicitly recorded as Image-Only.
    Best-effort: never raises (auditing must not block a run).
    """
    try:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "processing_mode": PROCESSING_MODE,
            "video_support": VIDEO_SUPPORT,
            "banner": SCOPE_BANNER,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            payload.update(extra)
        (reports_dir / "run_scope.json").write_text(
            __import__("json").dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 - auditing is best-effort
        logger.warning("run_scope_marker_failed", error=str(exc))


def _final_clean_has_stem(final_clean: Path, stem: str) -> bool:
    """
    True if sale-ready output for ``stem`` already exists in ``final_clean/``.

    Processing always writes ``{stem}.jpg``; also accept other image extensions for back-compat.
    """
    if (final_clean / f"{stem}.jpg").is_file():
        return True
    for ext in (".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        if (final_clean / f"{stem}{ext}").is_file():
            return True
    return False


def discover_pending(
    project_root: Path,
    cfg: Dict[str, Any],
    *,
    security_ctx: Optional[SecurityContext] = None,
) -> List[Path]:
    """
    Build the work queue:
    - Fresh uploads in `input_raw/` that are not already sale-finalized in `final_clean/`
    - Quarantined images that still have retry budget
    - Skips stems with active (non-stale) ``.processing.lock`` files
    """
    exts: Sequence[str] = cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])
    max_retries = int(cfg.get("max_retries", 3))
    lock_stale = int((cfg.get("resume", {}) or {}).get("lock_stale_sec", 3600))

    paths = resolve_pipeline_paths(project_root, cfg)
    input_raw = paths["input_raw"]
    final_clean = paths["final_clean"]
    quarantine = paths["quarantine"]
    manual_review = paths["manual_review"]
    temp_processed = paths["temp_processed"]

    locked_stems = active_locked_stems(temp_processed, stale_sec=lock_stale)
    manifest_skip: set[str] = set()
    if security_ctx is not None:
        manifest_skip = stems_fully_processed(
            security_ctx.processed_manifest_path(),
            final_clean=final_clean,
        )

    pending: List[Path] = []
    seen_stems: set[str] = set()

    for p in discover_images(quarantine, exts):
        audit = load_audit_json(p)
        if int(audit.get("retry_count", 0)) >= max_retries:
            continue
        if p.stem in locked_stems:
            continue
        if p.stem in manifest_skip:
            continue
        if not _final_clean_has_stem(final_clean, p.stem) and p.stem not in seen_stems:
            pending.append(p)
            seen_stems.add(p.stem)

    for p in discover_images(input_raw, exts):
        if p.stem in locked_stems:
            logger.debug("discover_pending_skip_locked", stem=p.stem)
            continue
        if p.stem in manifest_skip:
            continue
        if not _final_clean_has_stem(final_clean, p.stem) and p.stem not in seen_stems:
            pending.append(p)
            seen_stems.add(p.stem)

    _ = manual_review
    return pending


def route_batch_outputs(
    batch_dir: Path,
    *,
    project_root: Path,
    cfg: Dict[str, Any],
    audit_json_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Move each batch output to its terminal folder based on QA sidecars.
    Returns rows suitable for master reporting + small counters.
    """
    from .agentic_qa_crew import (
        build_failure_reason_text,
        evaluate_image_qa,
        propose_recycle_overrides,
    )

    project_root = Path(project_root)
    paths = resolve_pipeline_paths(project_root, cfg)
    final_clean = paths["final_clean"]
    quarantine = paths["quarantine"]
    manual_review = paths["manual_review"]

    max_retries = int(cfg.get("max_retries", 3))
    rows: List[Dict[str, Any]] = []
    counters = {"pass": 0, "fail": 0, "quarantine": 0, "manual_review": 0}

    def _enrich_master_row(base: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        sm = audit.get("success_metrics") or {}
        det = audit.get("detections") or {}
        base["text_regions_detected"] = det.get("paddleocr_count")
        base["text_regions_inpainted"] = sm.get("text_regions_inpainted")
        base["original_face_count"] = sm.get("original_face_count")
        base["face_gan_applied"] = sm.get("face_gan_applied")
        base["processing_status"] = sm.get("processing_status")
        return base

    for image_path in sorted(batch_dir.glob("*.jpg")):
        audit = load_audit_json(image_path)
        decision = (audit.get("qa") or {}).get("final_decision") or evaluate_image_qa(
            image_path, cfg, project_root=project_root
        )["final_decision"]

        if decision == "pass":
            audit["failure_reason"] = None
            audit.setdefault("qa", {})["failure_reason"] = None
            save_audit_json(image_path, audit)
            safe_move(image_path, final_clean / image_path.name)
            counters["pass"] += 1
            row = _enrich_master_row({"image": image_path.name, "final_decision": "pass", "route": "final_clean"}, audit)
            rows.append(row)
            append_master_audit(project_root, {"image": image_path.name, "decision": "pass"}, audit_json_path)
            continue

        # FAIL path — persist explicit buyer-facing failure text + recycle parameters for the next wave
        counters["fail"] += 1
        dec_full = audit.get("qa", {}).get("deterministic") or evaluate_image_qa(image_path, cfg, project_root=project_root)
        fail_text = build_failure_reason_text(dec_full)
        audit["failure_reason"] = fail_text
        audit.setdefault("qa", {})
        audit["qa"]["failure_reason_detail"] = fail_text
        audit["qa"]["routing_decision"] = "fail"

        prev_retry = int(audit.get("retry_count", 0))
        new_retry = prev_retry + 1
        audit["retry_count"] = int(new_retry)
        audit["next_pass_overrides"] = propose_recycle_overrides(new_retry, cfg)
        audit["qa"]["failure_retry_assigned"] = int(new_retry)
        save_audit_json(image_path, audit)

        if new_retry >= max_retries:
            safe_move(image_path, manual_review / image_path.name)
            counters["manual_review"] += 1
            rows.append(
                _enrich_master_row(
                    {
                        "image": image_path.name,
                        "final_decision": "fail",
                        "route": "manual_review",
                        "retry_count": new_retry,
                    },
                    audit,
                )
            )
        else:
            safe_move(image_path, quarantine / image_path.name)
            counters["quarantine"] += 1
            rows.append(
                _enrich_master_row(
                    {
                        "image": image_path.name,
                        "final_decision": "fail",
                        "route": "quarantine",
                        "retry_count": new_retry,
                    },
                    audit,
                )
            )

        append_master_audit(
            project_root,
            {"image": image_path.name, "decision": "fail", "retry_count": new_retry},
            audit_json_path,
        )

    mark_processing_complete(batch_dir)
    return rows, counters


def _quarantine_corrupted_sources(
    batch_dir: Path,
    *,
    quarantine: Path,
    input_raw: Path,
    batch_paths: Sequence[Path],
) -> int:
    """Copy/move failed sources into quarantine; never delete or move from ``input_raw/``."""
    count = 0
    stems_failed = {p.stem for p in batch_dir.glob("*.error_audit.json")}
    for src in batch_paths:
        if src.stem not in stems_failed:
            continue
        err = batch_dir / f"{src.stem}.error_audit.json"
        dst = quarantine / src.name
        if resolve_under(input_raw, src):
            shutil.copy2(src, dst)
        else:
            safe_move(src, dst)
        if err.is_file():
            shutil.copy2(err, quarantine / err.name)
        count += 1
    return count


def _resume_incomplete_batches(
    *,
    project_root: Path,
    cfg: Dict[str, Any],
    batch_summary_dir: Path,
    audit_json: Path,
    metrics: Optional[PipelineMetricsCollector],
    master_rows: List[Dict[str, Any]],
    batch_index: int,
) -> int:
    """QA + route batch folders left from a prior interrupted run."""
    from .agentic_qa_crew import run_qa_on_batch_directory

    paths = resolve_pipeline_paths(project_root, cfg)
    lock_stale = int((cfg.get("resume", {}) or {}).get("lock_stale_sec", 3600))
    resume_dirs = discover_resume_batch_dirs(paths["temp_processed"], stale_sec=lock_stale)
    if not resume_dirs:
        return batch_index

    for batch_dir in resume_dirs:
        jpgs = list(batch_dir.glob("*.jpg"))
        if not jpgs:
            continue
        batch_index += 1
        logger.info("resume_batch", batch_dir=str(batch_dir))
        if metrics is not None:
            metrics.on_batch_start(int(batch_index), int(len(jpgs)))
        qa_rows = run_qa_on_batch_directory(batch_dir, cfg, project_root=project_root)
        write_batch_summary_csv(
            batch_dir,
            qa_rows,
            batch_summary_dir / f"batch_{batch_index:05d}_resume.csv",
        )
        routed, batch_counters = route_batch_outputs(
            batch_dir,
            project_root=project_root,
            cfg=cfg,
            audit_json_path=audit_json,
        )
        master_rows.extend(routed)
        if metrics is not None:
            metrics.on_batch_complete(int(batch_index), counters=dict(batch_counters), processed_this_run=0)
    return batch_index


def _emit_progress(
    cb: Optional[Callable[[Dict[str, Any]], None]],
    payload: Dict[str, Any],
) -> None:
    if cb is None:
        return
    try:
        cb(dict(payload))
    except Exception:  # noqa: BLE001
        pass


def _verify_pending_integrity(pending: Sequence[Path]) -> Dict[str, Any]:
    """
    Lightweight, read-only integrity pass over the discovered work queue.

    For each pending source we confirm the file is present, readable, and decodes
    as a valid image (Pillow header + verify). Used by ``run_pipeline(dry_run=True,
    verify_integrity=True)`` so operators can validate inputs before committing GPU
    time. Never mutates or moves anything; returns aggregate counts plus per-file
    problems (capped) for reporting.
    """
    import hashlib

    from PIL import Image

    verified = 0
    corrupt: List[Dict[str, str]] = []
    for path in pending:
        try:
            if not path.is_file():
                corrupt.append({"image": path.name, "error": "missing_or_not_a_file"})
                continue
            with path.open("rb") as fh:
                hashlib.sha256(fh.read()).hexdigest()
            with Image.open(path) as img:
                img.verify()
            verified += 1
        except Exception as exc:  # noqa: BLE001 - any decode/read error means unusable input
            corrupt.append({"image": path.name, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "checked": len(pending),
        "verified": verified,
        "corrupt": len(corrupt),
        "ok": len(corrupt) == 0,
        "problems": corrupt[:50],
    }


def run_pipeline(
    config_path: Path | None = None,
    *,
    project_root: Path | None = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    stop_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ingest_from_b2: Optional[str] = None,
    export_to_b2: Optional[str] = None,
    ingest_dry_run: bool = False,
    export_dry_run: bool = False,
    test_mode: bool = False,
    security_level: Optional[str] = None,
    verify_after_export: Optional[bool] = None,
    enable_crypt: bool = False,
    backup_manifests: bool = False,
    force_gpu_validation: bool = False,
    gpu_device: Optional[str] = None,
    max_images: Optional[int] = None,
    dry_run: bool = False,
    verify_integrity: bool = False,
) -> Dict[str, Any]:
    """
    Run the full anonymization orchestration.

    ``config_overrides`` is deep-merged into the loaded YAML (e.g. ``batch_size``, ``paths``).

    ``stop_event``: when set, the loop exits gracefully **after** the current batch finishes.

    ``progress_callback`` receives small dict events for UIs (e.g. Flask / Socket.IO).

    Returns a summary dict including aggregate ``stats`` and ``stopped_early``.
    """
    project_root = Path(project_root).resolve() if project_root is not None else resolve_project_root()
    cfg_path = Path(config_path or (project_root / "config.yaml"))
    cfg = copy.deepcopy(load_config(cfg_path))
    if config_overrides:
        deep_update(cfg, copy.deepcopy(config_overrides))
    if enable_crypt:
        cfg.setdefault("security", {})
        cfg["security"]["crypt_enabled"] = True
    if gpu_device:
        cfg.setdefault("gpu", {})
        cfg["gpu"]["device"] = str(gpu_device).strip().lower()

    compute_profile = initialize_compute(cfg, force_gpu=force_gpu_validation)
    sync_cfg_device_fields(cfg)

    try:
        from .ui_bridge import apply_compute_profile

        apply_compute_profile(compute_profile)
    except ImportError:
        pass

    if test_mode:
        deep_update(
            cfg,
            {
                "batch_size": min(int(cfg.get("batch_size", 32)), 8),
                "max_qa_waves": min(int(cfg.get("max_qa_waves", 200)), 5),
                "test_mode": True,
            },
        )
        cfg.setdefault("logging", {})
        cfg["logging"]["level"] = "DEBUG"

    setup_project_folders(project_root, cfg)
    paths = resolve_pipeline_paths(project_root, cfg)
    # Stamp the run as Image-Only + pilot/full in the audit trail (video deferred by design).
    _is_pilot = max_images is not None and int(max_images) > 0
    _run_type = "pilot" if _is_pilot else "full"
    logger.info(
        "processing_scope",
        processing_mode=PROCESSING_MODE,
        video_support=VIDEO_SUPPORT,
        run_type=_run_type,
        batch_size=int(cfg.get("batch_size", 32)),
        max_images=int(max_images) if _is_pilot else None,
    )
    _write_run_scope_marker(
        paths["reports"],
        extra={
            "run_type": _run_type,
            "batch_size": int(cfg.get("batch_size", 32)),
            "max_images": int(max_images) if _is_pilot else None,
            "dry_run": bool(dry_run),
        },
    )
    sec_cfg = load_security_hardening(cfg, security_level=security_level)
    security_ctx = SecurityContext(project_root=project_root, config=sec_cfg)
    security_ctx.apply_startup_hardening(paths)
    set_input_raw_guard(paths["input_raw"])
    log_file, batch_summary_dir = resolve_log_artifacts(project_root, paths["logs"], cfg)
    log_level = "DEBUG" if test_mode else str(cfg.get("logging", {}).get("level", "INFO"))
    configure_structlog(log_level, str(log_file))
    monitor = init_monitoring(paths["logs"])

    if backup_manifests:
        backup_dir = backup_critical_artifacts(
            project_root,
            backup_root=project_root / str((cfg.get("security") or {}).get("backups_dir", "backups")),
        )
        monitor.record_security("backup_manifests", backup_dir=str(backup_dir))
        logger.info("backup_manifests_complete", backup_dir=str(backup_dir))

    b2_cfg = None
    ingest_record = None
    if ingest_from_b2 is not None or export_to_b2 is not None:
        b2_cfg = load_b2_config(
            yaml_cfg=cfg.get("backblaze") or {},
            security_cfg=cfg.get("security") or {},
        )
        write_rclone_config(b2_cfg)

    if ingest_from_b2 is not None:
        remote_path = ingest_from_b2 if ingest_from_b2 else b2_cfg.ingest_remote_path
        security_ctx.pre_ingest(
            remote_path=remote_path,
            local_input=paths["input_raw"],
            dry_run=bool(ingest_dry_run),
        )
        logger.info(
            "b2_ingest_start",
            remote_path=remote_path,
            local_input=str(paths["input_raw"]),
            dry_run=bool(ingest_dry_run),
        )
        ingest_record = ingest_from_backblaze(
            remote_path,
            paths["input_raw"],
            batch_size=b2_cfg.transfer_batch_size,
            cfg=b2_cfg,
            dry_run=ingest_dry_run,
            confirm=not ingest_dry_run,
            project_root=project_root,
            quarantine_dir=paths["quarantine"] / "b2_transfer_failures",
            enforce_readonly_ingest=sec_cfg.enforce_readonly_ingest,
            verify_checksums=sec_cfg.verify_ingest_checksums,
        )
        security_ctx.post_ingest(
            local_input=paths["input_raw"],
            ingest_record=ingest_record,
            dry_run=bool(ingest_dry_run),
        )
        if ingest_record:
            monitor.record_transfer(
                "ingest_complete",
                dry_run=bool(ingest_dry_run),
                files_copied=int(ingest_record.get("files_copied") or 0),
                bytes_transferred=int(ingest_record.get("bytes_transferred") or 0),
                elapsed_sec=float(ingest_record.get("elapsed_sec") or 0),
            )

    if dry_run:
        pending_preview = discover_pending(project_root, cfg, security_ctx=security_ctx)
        if max_images is not None and max_images > 0:
            pending_preview = pending_preview[: int(max_images)]
        plan = {
            "dry_run": True,
            "processing_mode": PROCESSING_MODE,
            "video_support": VIDEO_SUPPORT,
            "run_type": _run_type,
            "pending_images": len(pending_preview),
            "max_images": max_images,
            "security_level": sec_cfg.level,
            "crypt_enabled": bool(sec_cfg.crypt_enabled),
            "gpu_device": str((cfg.get("gpu") or {}).get("device", cfg.get("device"))),
            "ingest_from_b2": ingest_from_b2 is not None,
            "export_to_b2": export_to_b2 is not None,
            "batch_size": int(cfg.get("batch_size", 32)),
            "rclone_ingest_command": DEFAULT_INGEST_COMMAND,
            "rclone_export_command": DEFAULT_EXPORT_COMMAND,
        }
        integrity_report: Optional[Dict[str, Any]] = None
        if verify_integrity:
            integrity_report = _verify_pending_integrity(pending_preview)
            plan["verify_integrity"] = True
            plan["integrity_verified"] = integrity_report["verified"]
            plan["integrity_corrupt"] = integrity_report["corrupt"]
            plan["integrity_ok"] = integrity_report["ok"]
        monitor.record_pipeline_start(**plan)
        logger.info("pipeline_dry_run_plan", **plan)
        _emit_progress(progress_callback, {"type": "dry_run_plan", **plan})
        result: Dict[str, Any] = {
            "dry_run": True,
            "plan": plan,
            "pending_sample": [str(p.name) for p in pending_preview[:20]],
            "monitoring": monitor.summary(),
            "rclone_defaults": default_rclone_commands(),
        }
        if integrity_report is not None:
            result["integrity"] = integrity_report
        return result

    # Heavy GPU/ML stack is only needed for a real processing run (never for --dry-run).
    from .agentic_qa_crew import run_qa_on_batch_directory
    from .batch_processor import AnonymizationEngine, process_batch

    dl_meta = ensure_model_assets(project_root, cfg)
    logger.info("model_assets", meta=dl_meta)

    logger.info(
        "pipeline_start",
        root=str(project_root),
        test_mode=bool(test_mode),
        compute_profile=compute_profile,
    )
    if compute_profile.get("cpu_fallback") and compute_profile.get("user_message"):
        logger.warning("pipeline_cpu_fallback", message=compute_profile["user_message"])

    gpu_validation = validate_gpu_at_startup(cfg)
    gpu_validation["compute_profile"] = get_compute_profile()
    attach_security_context(monitor, security_ctx)
    monitor.record_pipeline_start(
        test_mode=bool(test_mode),
        security_level=sec_cfg.level,
        gpu_validation=gpu_validation,
        compute_profile=compute_profile,
    )
    if test_mode or bool((cfg.get("monitoring") or {}).get("resource_monitoring", False)):
        log_gpu_memory("pipeline_start")
    _emit_progress(
        progress_callback,
        {
            "type": "pipeline_start",
            "project_root": str(project_root),
            "gpu": gpu_validation,
            "compute_profile": compute_profile,
            "cpu_fallback": bool(compute_profile.get("cpu_fallback")),
            "user_message": compute_profile.get("user_message"),
        },
    )

    engine = AnonymizationEngine(project_root, cfg)
    warm_meta = engine.warm_models()
    components = run_component_activation_check(cfg, project_root)
    warm_meta["component_activation"] = components
    logger.info("models_warmed", meta=warm_meta)
    _emit_progress(
        progress_callback,
        {
            "type": "models_warmed",
            "meta": warm_meta,
            "components": components,
            "cpu_fallback": bool(get_compute_profile().get("cpu_fallback")),
            "user_message": get_compute_profile().get("user_message"),
            "compute_profile": get_compute_profile(),
        },
    )

    reports = cfg.get("reports", {}) or {}
    master_csv = resolve_report_artifact(
        project_root,
        paths["reports"],
        str(reports.get("master_csv", "reports/master_summary.csv")),
        "reports/master_summary.csv",
    )
    master_pdf = resolve_report_artifact(
        project_root,
        paths["reports"],
        str(reports.get("master_pdf", "reports/master_summary.pdf")),
        "reports/master_summary.pdf",
    )
    audit_json = resolve_report_artifact(
        project_root,
        paths["reports"],
        str(reports.get("audit_json", "reports/anonymization_audit.json")),
        "reports/anonymization_audit.json",
    )

    master_rows: List[Dict[str, Any]] = []
    batch_index = 0
    final_clean = paths["final_clean"]

    exts = cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])
    total_hint = len(discover_images(paths["input_raw"], exts))
    try:
        from .ui_bridge import set_live_field

        set_live_field(total_detected=int(total_hint), total_target=int(total_hint))
    except ImportError:
        pass
    perf_cfg = dict(cfg.get("performance") or {})
    monitoring = cfg.get("monitoring", {}) or {}
    resource_monitoring = bool(
        test_mode
        or monitoring.get("resource_monitoring", False)
        or perf_cfg.get("always_monitor", False)
        or total_hint >= int(perf_cfg.get("monitor_threshold", 500))
    )
    metrics = PipelineMetricsCollector(resource_monitoring=resource_monitoring)
    metrics.set_total_hint(total_hint)
    use_pbar = progress_callback is None
    global_pbar = tqdm(
        total=total_hint or None,
        desc="dataset_anonymizer",
        unit="img",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not use_pbar,
    )

    stopped_early = False
    processed_this_run = 0
    t0: Optional[float] = None
    ingest_screen_stats: Dict[str, Any] = {"accepted": 0, "rejected": 0}
    pipeline_t0 = time.monotonic()

    try:
        max_waves = int(cfg.get("max_qa_waves", 200))
        batch_index = _resume_incomplete_batches(
            project_root=project_root,
            cfg=cfg,
            batch_summary_dir=batch_summary_dir,
            audit_json=audit_json,
            metrics=metrics,
            master_rows=master_rows,
            batch_index=batch_index,
        )
        # Pilot cap: lock the first N distinct images on the first wave and only ever
        # (re)process THAT set on later waves — never pull in image N+1. This makes a
        # "pilot 1000" a hard TOTAL limit, not a per-wave limit.
        pilot_stems: Optional[set] = None
        for wave in range(max_waves):
            if stop_event is not None and stop_event.is_set():
                logger.info("pipeline_stop_requested_between_waves", wave=int(wave))
                stopped_early = True
                break

            pending = discover_pending(project_root, cfg, security_ctx=security_ctx)
            if max_images is not None and max_images > 0:
                if pilot_stems is None:
                    pilot_stems = {p.stem for p in pending[: int(max_images)]}
                pending = [p for p in pending if p.stem in pilot_stems]
            if not pending:
                logger.info("no_pending_work", wave=int(wave))
                break

            if wave == 0 and bool(perf_cfg.get("ingest_screening", True)):
                accepted, rejected = screen_input_images(
                    pending,
                    cfg,
                    quarantine_dir=paths["quarantine"] / "ingest_rejects",
                )
                if rejected:
                    logger.info("ingest_screen_complete", rejected=len(rejected), accepted=len(accepted))
                pending = accepted
                ingest_screen_stats = {"accepted": len(accepted), "rejected": len(rejected)}
                if not pending:
                    break

            logger.info("wave_scheduled", wave=int(wave), pending=int(len(pending)))
            _emit_progress(
                progress_callback,
                {"type": "wave_start", "wave": int(wave), "pending": int(len(pending))},
            )

            batch_size = resolve_adaptive_batch_size(cfg, len(pending))
            try:
                from .ui_bridge import estimate_total_batches

                total_batches_est = estimate_total_batches(len(pending), batch_size)
            except ImportError:
                total_batches_est = max(1, (len(pending) + batch_size - 1) // max(1, batch_size))
            batch_num_in_wave = 0
            for i in range(0, len(pending), batch_size):
                batch_num_in_wave += 1
                batch_index += 1
                slice_ = pending[i : i + batch_size]
                batch_paths = [p for p in slice_ if not _final_clean_has_stem(final_clean, p.stem)]
                if not batch_paths:
                    continue

                if stop_event is not None and stop_event.is_set():
                    logger.info("pipeline_stop_requested_before_batch_start", batch_index=int(batch_index))
                    stopped_early = True
                    break

                if t0 is None:
                    t0 = time.monotonic()

                logger.info("batch_start", batch_index=int(batch_index), n=int(len(batch_paths)))
                if metrics is not None:
                    metrics.on_batch_start(int(batch_index), int(len(batch_paths)))
                _emit_progress(
                    progress_callback,
                    {
                        "type": "batch_start",
                        "batch_index": int(batch_index),
                        "batch_in_wave": int(batch_num_in_wave),
                        "total_batches_estimate": int(total_batches_est),
                        "n": int(len(batch_paths)),
                        "current_batch_size": int(len(batch_paths)),
                        "images": [p.name for p in batch_paths],
                        "wave": int(wave),
                    },
                )

                bp = process_batch(
                    batch_paths,
                    batch_index=batch_index,
                    engine=engine,
                    cfg=cfg,
                    project_root=project_root,
                )

                qa_rows = run_qa_on_batch_directory(bp.batch_dir, cfg, project_root=project_root)
                global_pbar.update(int(len(batch_paths)))
                processed_this_run += int(len(batch_paths))

                write_batch_summary_csv(
                    bp.batch_dir,
                    qa_rows,
                    batch_summary_dir / f"batch_{batch_index:05d}.csv",
                )

                routed, batch_counters = route_batch_outputs(
                    bp.batch_dir,
                    project_root=project_root,
                    cfg=cfg,
                    audit_json_path=audit_json,
                )
                security_ctx.post_qa_batch(batch_dir=bp.batch_dir, routed_count=len(routed))
                corrupted_n = _quarantine_corrupted_sources(
                    bp.batch_dir,
                    quarantine=paths["quarantine"],
                    input_raw=paths["input_raw"],
                    batch_paths=batch_paths,
                )
                if corrupted_n:
                    batch_counters["quarantine"] = int(batch_counters.get("quarantine", 0)) + corrupted_n
                    logger.warning("corrupted_sources_quarantined", n=int(corrupted_n))
                master_rows.extend(routed)

                elapsed = (time.monotonic() - t0) if t0 is not None else 0.0
                rate = (processed_this_run / elapsed) if elapsed > 0 else 0.0
                remaining_by_queue = max(0, total_hint - processed_this_run)
                eta_s = (remaining_by_queue / rate) if rate > 0 else None

                df_partial = pd.DataFrame(master_rows)
                if len(master_rows) and "final_decision" in df_partial.columns:
                    passed_so_far = int((df_partial["final_decision"] == "pass").sum())
                else:
                    passed_so_far = 0
                success_rate = (passed_so_far / len(master_rows)) if master_rows else 0.0

                logger.info("batch_complete", batch_index=int(batch_index), routed=int(len(routed)))
                batch_metrics_row: Dict[str, Any] = {}
                if metrics is not None:
                    batch_metrics_row = metrics.on_batch_complete(
                        int(batch_index),
                        counters=dict(batch_counters),
                        processed_this_run=int(processed_this_run),
                    )
                q_cum = float(batch_metrics_row.get("quarantine_rate_cumulative", 0))
                gpu_snap = cuda_memory_snapshot()
                _emit_progress(
                    progress_callback,
                    {
                        "type": "batch_complete",
                        "batch_index": int(batch_index),
                        "routed": int(len(routed)),
                        "counters": dict(batch_counters),
                        "qa_rows": int(len(qa_rows)),
                        "processed_this_run": int(processed_this_run),
                        "total_hint": int(total_hint),
                        "total_target": int(total_hint),
                        "success_rate": float(success_rate),
                        "elapsed_sec": float(elapsed),
                        "eta_sec": float(eta_s) if eta_s is not None else None,
                        "images_per_sec": float(rate),
                        "quarantine_rate": float(q_cum),
                        "gpu": gpu_snap,
                    },
                )
                _emit_progress(
                    progress_callback,
                    {
                        "type": "progress_tick",
                        "processed": int(processed_this_run),
                        "total_target": int(total_hint),
                        "images_per_sec": float(rate),
                        "eta_sec": float(eta_s) if eta_s is not None else None,
                        "batch_index": int(batch_index),
                    },
                )

                if stop_event is not None and stop_event.is_set():
                    logger.info("pipeline_stop_requested_after_batch", batch_index=int(batch_index))
                    stopped_early = True
                    break

            if stopped_early:
                break
    finally:
        global_pbar.close()

    # Buyer-facing master artifacts (CSV + PDF with aggregate KPIs)
    if master_rows:
        df = pd.DataFrame(master_rows)
        if "image" in df.columns:
            df = df.drop_duplicates(subset=["image"], keep="last")
        master_records = df.to_dict(orient="records")
    else:
        master_records = []

    stats = compute_master_statistics(master_records, target_success_rate=float(cfg.get("min_success_rate", 0.99)))
    pdf_engine = str((cfg.get("reports", {}) or {}).get("pdf_engine", "reportlab"))
    write_master_summary_csv_and_pdf(
        project_root,
        master_records,
        master_csv,
        master_pdf,
        pdf_engine=pdf_engine,
        aggregate_stats=stats,
    )

    logger.info(
        "pipeline_complete",
        master_csv=str(master_csv),
        master_pdf=str(master_pdf),
        audit_json=str(audit_json),
        stats=stats,
        stopped_early=bool(stopped_early),
    )
    _emit_progress(
        progress_callback,
        {"type": "pipeline_complete", "stats": stats, "stopped_early": bool(stopped_early)},
    )

    export_record = None
    if export_to_b2 is not None and not stopped_early:
        remote_dest = export_to_b2 if export_to_b2 else b2_cfg.export_remote_path
        security_ctx.pre_export(
            local_final=paths["final_clean"],
            remote_dest=remote_dest,
            dry_run=bool(export_dry_run),
        )
        logger.info(
            "b2_export_start",
            remote_dest=remote_dest,
            local_final=str(paths["final_clean"]),
            dry_run=bool(export_dry_run),
        )
        export_record = export_to_backblaze(
            paths["final_clean"],
            remote_dest,
            cfg=b2_cfg,
            dry_run=export_dry_run,
            confirm=not export_dry_run,
            project_root=project_root,
            quarantine_dir=paths["quarantine"] / "b2_transfer_failures",
            verify_after_export=(
                verify_after_export
                if verify_after_export is not None
                else (sec_cfg.is_full or bool((cfg.get("backblaze") or {}).get("verify_after_export", True)))
            ),
        )
        security_ctx.log_event(
            "security_post_export",
            verify_after_export=export_record.get("post_export_check") is not None,
            crypt_enabled=bool(b2_cfg.crypt_enabled),
        )
        if export_record:
            mon = get_monitoring()
            if mon is not None:
                mon.record_transfer(
                    "export_complete",
                    dry_run=bool(export_dry_run),
                    files_uploaded=int(export_record.get("files_uploaded") or 0),
                    bytes_transferred=int(export_record.get("bytes_transferred") or 0),
                    elapsed_sec=float(export_record.get("elapsed_sec") or 0),
                    verify_ok=bool((export_record.get("post_export_check") or {}).get("ok", True)),
                )

    security_ctx.pre_cleanup(temp_dir=paths["temp_processed"], reports_dir=paths["reports"])
    security_ctx.cleanup_temp(paths["temp_processed"], pipeline_cfg=cfg)

    security_report_path = security_ctx.write_report(
        summary={
            "stopped_early": bool(stopped_early),
            "stats": stats,
            "ingest_record": bool(ingest_record),
            "export_record": bool(export_record),
            "security_level": sec_cfg.level,
            "rclone": (
                {
                    "performance": {
                        "transfers": b2_cfg.transfers,
                        "checkers": b2_cfg.checkers,
                        "upload_concurrency": b2_cfg.upload_concurrency,
                        "chunk_size": b2_cfg.chunk_size,
                        "bwlimit": b2_cfg.bwlimit,
                    },
                    "verify_after_export": (
                        verify_after_export
                        if verify_after_export is not None
                        else (sec_cfg.is_full or bool((cfg.get("backblaze") or {}).get("verify_after_export", True)))
                    ),
                    "crypt_enabled": bool(b2_cfg.crypt_enabled),
                    "crypt_decryption_doc": (export_record or {}).get("crypt_decryption_doc"),
                    "post_export_check": (export_record or {}).get("post_export_check"),
                }
                if b2_cfg
                else None
            ),
        }
    )

    metrics_summary = metrics.summary(total_input=total_hint)
    pipeline_elapsed = time.monotonic() - pipeline_t0
    model_stats = get_model_usage_stats()
    performance_report_path = write_performance_report(
        project_root,
        metrics_summary,
        cfg,
        model_stats=model_stats,
        ingest_screen=ingest_screen_stats if ingest_screen_stats.get("rejected") else None,
        elapsed_sec=pipeline_elapsed,
    )
    if test_mode or resource_monitoring:
        log_gpu_memory("pipeline_complete")
        log_system_resources("pipeline_complete")

    gpu_report_path = write_gpu_readiness_report(
        project_root,
        cfg,
        gpu_validation=get_gpu_validation_report(),
        warm_meta=warm_meta,
        metrics=metrics_summary,
        model_stats=model_stats,
        elapsed_sec=pipeline_elapsed,
    )

    mon_summary = monitor.summary() if monitor is not None else {}
    monitor.record_pipeline_complete(
        stopped_early=bool(stopped_early),
        pass_rate=float(metrics_summary.get("pass_rate", 0)),
        quarantine_rate=float(metrics_summary.get("quarantine_rate", 0)),
        images_per_sec=float(metrics_summary.get("images_per_sec", 0)),
        peak_gpu_allocated_mb=float(metrics_summary.get("peak_gpu_allocated_mb", 0)),
        elapsed_sec=float(pipeline_elapsed),
    )

    return {
        "stopped_early": bool(stopped_early),
        "stats": stats,
        "master_csv": str(master_csv),
        "master_pdf": str(master_pdf),
        "audit_json": str(audit_json),
        "log_file": str(log_file),
        "images_in_master": int(stats.get("images_in_master", 0)),
        "b2_export": export_record,
        "b2_ingest": ingest_record,
        "security_report": str(security_report_path),
        "security_level": sec_cfg.level,
        "performance_report": str(performance_report_path),
        "model_usage": model_stats,
        "test_mode": bool(test_mode),
        "metrics": metrics_summary,
        "gpu_validation": gpu_validation,
        "compute_profile": get_compute_profile(),
        "gpu_readiness_report": str(gpu_report_path),
        "monitoring": mon_summary,
        "monitoring_log": str(paths["logs"] / "monitoring.jsonl"),
    }


def main() -> None:
    # Load encrypted/locked secrets into the environment before any B2 config is read,
    # then (optionally) confirm B2 bucket connectivity so failures surface immediately.
    try:
        from .secrets_manager import load_secrets, confirm_bucket_access

        load_secrets()
        confirm_bucket_access()
    except Exception as exc:  # noqa: BLE001
        # Do not leak values; a run may still rely on platform-injected env vars.
        print(f"[secrets] startup secrets/bucket step issue ({exc}); using process environment.")

    parser = argparse.ArgumentParser(description="GPU-accelerated image anonymization pipeline")
    parser.add_argument(
        "--ingest-from-b2",
        nargs="?",
        const="",
        default=None,
        metavar="REMOTE_PATH",
        help=(
            "Copy originals from read-only B2 remote into input_raw/ before processing. "
            "Omit the value to use the configured default path "
            f"({default_ingest_command(dry_run=True)!r})"
        ),
    )
    parser.add_argument(
        "--export-to-b2",
        nargs="?",
        const="",
        default=None,
        metavar="REMOTE_DEST",
        help=(
            "Upload final_clean/ to write B2 remote after successful routing. "
            "Omit the value to use the configured default path "
            f"({default_export_command()!r})"
        ),
    )
    parser.add_argument(
        "--ingest-dry-run",
        action="store_true",
        help="Plan B2 ingest with rclone --dry-run (no files copied)",
    )
    parser.add_argument(
        "--export-dry-run",
        action="store_true",
        help="Plan B2 export with rclone --dry-run (no files uploaded)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Small batches, DEBUG logging, GPU memory + timing metrics",
    )
    parser.add_argument(
        "--security-level",
        choices=["standard", "full"],
        default=None,
        help="Security hardening level (full enables secure wipe, mandatory backups, strict umask)",
    )
    parser.add_argument(
        "--verify-after-export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run rclone check --checksum after B2 export (default: true when --security-level full)",
    )
    parser.add_argument(
        "--enable-crypt",
        action="store_true",
        help="Enable rclone crypt wrapper on B2 export (requires RCLONE_CRYPT_PASSWORD in env)",
    )
    parser.add_argument(
        "--backup-manifests",
        action="store_true",
        help="Backup reports/manifests to backups/ before processing",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Fail fast if CUDA is not available (production GPU hosts)",
    )
    parser.add_argument(
        "--gpu-device",
        metavar="DEVICE",
        default=None,
        help='Override gpu.device (e.g. "cuda", "cuda:1", "cpu")',
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Cap work queue size per run (testing / staged rollouts)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="Alias for --max-images: cap the work queue to the first N images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log execution plan only (no model warm, no batch processing)",
    )
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="During --dry-run, read + decode each pending image to confirm inputs are intact",
    )
    args = parser.parse_args()
    max_images = args.max_images if args.max_images is not None else args.count
    run_pipeline(
        ingest_from_b2=args.ingest_from_b2,
        export_to_b2=args.export_to_b2,
        ingest_dry_run=bool(args.ingest_dry_run) or bool(args.dry_run),
        export_dry_run=bool(args.export_dry_run) or bool(args.dry_run),
        test_mode=bool(args.test_mode),
        security_level=args.security_level,
        verify_after_export=args.verify_after_export,
        enable_crypt=bool(args.enable_crypt),
        backup_manifests=bool(args.backup_manifests),
        force_gpu_validation=bool(args.gpu),
        gpu_device=args.gpu_device,
        max_images=max_images,
        dry_run=bool(args.dry_run),
        verify_integrity=bool(args.verify_integrity),
    )


if __name__ == "__main__":
    # Support both ``python -m scripts.main_pipeline`` (package) and ``python scripts/main_pipeline.py`` (script).
    if __package__:
        main()
    else:
        import importlib.util
        import sys
        import types

        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        if "scripts" not in sys.modules:
            pkg = types.ModuleType("scripts")
            pkg.__path__ = [str(_root / "scripts")]  # type: ignore[attr-defined]
            sys.modules["scripts"] = pkg
        spec = importlib.util.spec_from_file_location("scripts.main_pipeline", Path(__file__))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        mod.__package__ = "scripts"
        mod.__name__ = "scripts.main_pipeline"
        sys.modules["scripts.main_pipeline"] = mod
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.main()
