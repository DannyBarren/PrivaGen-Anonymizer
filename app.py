"""
Flask + Flask-SocketIO control plane for dataset_anonymizer.

Launches with minimal deps (flask + flask-socketio). Full pipeline packages
can be installed from the dashboard via "Install All Dependencies Now".

Run from project root:
    pip install -r requirements-ui.txt
    python app.py

Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading

# Safe before any ML stack import during *deep* environment re-check (Windows/Anaconda).
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flask import Flask, jsonify, render_template, request, send_file
from flask_socketio import SocketIO

# ---------------------------------------------------------------------------
# Project root (no scripts.* imports at module load — keeps UI boot minimal)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Dual-mode host/port binding ------------------------------------------------
# Default (no special env vars): bind to 127.0.0.1:5000 — the SECURE posture used by
# local dev, Lambda, and SSH port-forwarding. This is unchanged.
# Cloud preview (e.g. Render): if $PORT is injected by the platform, or BIND_HOST is
# explicitly set to 0.0.0.0, listen on 0.0.0.0:$PORT so external traffic can be routed.
_CLOUD_PREVIEW = bool(os.environ.get("PORT")) or os.environ.get("BIND_HOST") == "0.0.0.0"
_BIND_HOST = "0.0.0.0" if _CLOUD_PREVIEW else os.environ.get("BIND_HOST", "127.0.0.1")
_BIND_PORT = int(os.environ.get("PORT") or os.environ.get("BIND_PORT") or "5000")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = __import__("secrets").token_hex(32)
# Local/Lambda: pin CORS to localhost (unchanged). Cloud preview is served behind the
# platform's proxy (its own https origin), so allow any origin only in that mode.
_CORS_ORIGINS = "*" if _CLOUD_PREVIEW else [
    f"http://{_BIND_HOST}:{_BIND_PORT}",
    f"http://localhost:{_BIND_PORT}",
]
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins=_CORS_ORIGINS,
)

_state_lock = threading.Lock()
_pipeline_state: Dict[str, Any] = {
    "status": "idle",
    "stop_requested": False,
    "stop_event": None,
    "worker": None,
    "last_error": None,
    "last_result": None,
    "last_overrides": None,
    "live_metrics": {},
}

_env_lock = threading.Lock()
_env_status_cache: Optional[Dict[str, Any]] = None

_install_job: Any = None
_live_ticker_stop = threading.Event()
_live_ticker_thread: Optional[threading.Thread] = None


def _get_install_job() -> Any:
    global _install_job
    if _install_job is None:
        from scripts.environment_installer import InstallJob

        _install_job = InstallJob()
    return _install_job


# ---------------------------------------------------------------------------
# Lazy pipeline imports (only when requirements_ok)
# ---------------------------------------------------------------------------
_pipeline_modules: Optional[Dict[str, Any]] = None


def _load_pipeline_modules() -> Dict[str, Any]:
    global _pipeline_modules
    if _pipeline_modules is not None:
        return _pipeline_modules
    from scripts.device_manager import get_compute_profile, initialize_compute
    from scripts.gpu_runtime import cuda_memory_snapshot
    from scripts.main_pipeline import run_pipeline
    from scripts.utils import (
        deep_update,
        discover_images,
        load_config,
        resolve_pipeline_paths,
        resolve_project_root,
    )

    _pipeline_modules = {
        "get_compute_profile": get_compute_profile,
        "initialize_compute": initialize_compute,
        "cuda_memory_snapshot": cuda_memory_snapshot,
        "run_pipeline": run_pipeline,
        "deep_update": deep_update,
        "discover_images": discover_images,
        "load_config": load_config,
        "resolve_pipeline_paths": resolve_pipeline_paths,
        "resolve_project_root": resolve_project_root,
    }
    return _pipeline_modules


def _project_root() -> Path:
    """Project root without loading the ML pipeline (keeps UI boot minimal)."""
    return _PROJECT_ROOT


def _load_yaml_light() -> Dict[str, Any]:
    """Load config.yaml without importing the ML stack."""
    path = _project_root() / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def _default_image_exts(cfg: Optional[Dict[str, Any]] = None) -> tuple[str, ...]:
    data = cfg or _load_yaml_light()
    raw = data.get("image_extensions") or [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    ]
    return tuple(str(e).lower() if str(e).startswith(".") else f".{e}".lower() for e in raw)


def _light_resolve_paths(path_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    """Resolve pipeline folders from config + query overrides (no torch/utils)."""
    root = _project_root()
    cfg = _load_yaml_light()
    paths_cfg = dict(cfg.get("paths") or {})
    if path_overrides:
        paths_cfg.update({k: v for k, v in path_overrides.items() if v})
    keys = (
        "input_raw",
        "final_clean",
        "quarantine",
        "manual_review",
        "temp_processed",
        "logs",
        "reports",
    )
    out: Dict[str, Path] = {}
    for key in keys:
        rel = str(paths_cfg.get(key, key))
        p = Path(rel)
        out[key] = p.resolve() if p.is_absolute() else (root / rel).resolve()
    return out


def _light_count_images(folder: Path, exts: tuple[str, ...]) -> int:
    if not folder.is_dir():
        return 0
    allowed = {e.lower() for e in exts}
    n = 0
    try:
        for entry in folder.rglob("*"):
            if entry.is_file() and entry.suffix.lower() in allowed:
                n += 1
    except OSError:
        return 0
    return n


def _folder_counts_payload(path_overrides: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    paths = _light_resolve_paths(path_overrides)
    exts = _default_image_exts()
    return {k: _light_count_images(p, exts) for k, p in paths.items() if k != "logs" and k != "reports"}


def _refresh_environment(*, use_cache: bool = True, deep: bool = False) -> Dict[str, Any]:
    global _env_status_cache
    from scripts.environment_checker import check_environment

    with _env_lock:
        if use_cache and not deep and _env_status_cache is not None:
            return dict(_env_status_cache)
        try:
            status = check_environment(_PROJECT_ROOT, deep=deep)
        except Exception as exc:  # noqa: BLE001
            status = {
                "readiness": "not_ready",
                "readiness_label": f"Checker error: {exc}",
                "requirements_ok": False,
                "ui_minimal_ok": True,
                "warnings": [str(exc)],
            }
        _env_status_cache = status
        return dict(status)


def _pipeline_ready() -> bool:
    env = _refresh_environment(use_cache=True)
    return env.get("readiness") in ("ready_gpu", "ready_cpu")


def _emit_install(event: str, data: Dict[str, Any]) -> None:
    with app.app_context():
        socketio.emit(event, data, namespace="/")


def _socket_emit(event_name: str, data: Dict[str, Any]) -> None:
    with app.app_context():
        socketio.emit(event_name, data, namespace="/")


def _progress_base(event: Dict[str, Any]) -> None:
    if event.get("type") == "batch_complete":
        with _state_lock:
            _pipeline_state["live_metrics"] = {
                "processed": event.get("processed_this_run"),
                "success_rate": event.get("success_rate"),
                "eta_sec": event.get("eta_sec"),
                "images_per_sec": event.get("images_per_sec"),
                "gpu": event.get("gpu"),
                "quarantine_rate": event.get("quarantine_rate"),
            }


def _emit_pipeline(event: Dict[str, Any]) -> None:
    """Legacy direct emit (hello/connect); pipeline runs use wrapped callback."""
    _socket_emit("pipeline_event", event)
    _progress_base(event)


def _make_progress_callback() -> Callable[[Dict[str, Any]], None]:
    from scripts.ui_bridge import get_live_state, reset_live_state, wrap_progress_callback

    try:
        from scripts.dataset_scanner import load_dataset_config

        ds = load_dataset_config(_project_root())
        scan = (ds.get("last_scan") or {}) if isinstance(ds.get("last_scan"), dict) else {}
        detected = int(scan.get("image_count") or 0)
    except Exception:  # noqa: BLE001
        detected = 0
    reset_live_state(total_detected=detected)
    cb = wrap_progress_callback(_socket_emit, _progress_base)
    assert cb is not None
    return cb


def _start_live_ticker() -> None:
    global _live_ticker_thread
    _live_ticker_stop.clear()

    def _loop() -> None:
        from scripts.ui_bridge import get_live_state

        while not _live_ticker_stop.wait(1.5):
            with _state_lock:
                if _pipeline_state.get("status") != "running":
                    continue
            _socket_emit("pipeline_status_update", get_live_state())

    _live_ticker_thread = threading.Thread(target=_loop, name="live-pipeline-ticker", daemon=True)
    _live_ticker_thread.start()


def _stop_live_ticker() -> None:
    _live_ticker_stop.set()


def _base_cfg() -> Dict[str, Any]:
    mods = _load_pipeline_modules()
    return copy.deepcopy(mods["load_config"](_project_root() / "config.yaml"))


def _paths_from_request_args() -> Dict[str, str]:
    keys = ("input_raw", "final_clean", "quarantine", "manual_review", "temp_processed", "logs", "reports")
    out: Dict[str, str] = {}
    for k in keys:
        v = (request.args.get(k) or "").strip()
        if v:
            out[k] = v
    return out


def _cfg_for_paths(path_kwargs: Dict[str, str]) -> Dict[str, Any]:
    mods = _load_pipeline_modules()
    cfg = _base_cfg()
    if path_kwargs:
        cfg.setdefault("paths", {})
        mods["deep_update"](cfg["paths"], path_kwargs)  # type: ignore[arg-type]
    return cfg


def _build_pipeline_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if data.get("security_level"):
        kw["security_level"] = str(data["security_level"])
    if data.get("enable_crypt"):
        kw["enable_crypt"] = True
    if data.get("verify_after_export") is not None:
        kw["verify_after_export"] = bool(data["verify_after_export"])
    if data.get("backup_manifests"):
        kw["backup_manifests"] = True
    if data.get("force_gpu"):
        kw["force_gpu_validation"] = True
    if data.get("gpu_device"):
        kw["gpu_device"] = str(data["gpu_device"])
    if data.get("max_images") is not None:
        kw["max_images"] = int(data["max_images"])
    if data.get("test_mode"):
        kw["test_mode"] = True
    if data.get("dry_run"):
        kw["dry_run"] = True
    if data.get("ingest_from_b2") is not None:
        kw["ingest_from_b2"] = data.get("ingest_from_b2", "")
    if data.get("export_to_b2") is not None:
        kw["export_to_b2"] = data.get("export_to_b2", "")
    if data.get("ingest_dry_run"):
        kw["ingest_dry_run"] = True
    if data.get("export_dry_run"):
        kw["export_dry_run"] = True
    return kw


def _merge_dataset_overrides(
    overrides: Dict[str, Any],
    pipeline_kw: Dict[str, Any],
    *,
    inline_dataset: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    from scripts.dataset_scanner import apply_dataset_to_pipeline_overrides, load_dataset_config

    ds = load_dataset_config(_project_root())
    if inline_dataset and isinstance(inline_dataset, dict):
        ds.update(inline_dataset)
    applied = apply_dataset_to_pipeline_overrides(ds, _project_root())
    mods = _load_pipeline_modules()
    ov = copy.deepcopy(overrides)
    pk = copy.deepcopy(pipeline_kw)
    mods["deep_update"](ov, applied.get("overrides") or {})
    pk.update(applied.get("pipeline") or {})
    return ov, pk


def _pipeline_worker(overrides: Dict[str, Any], pipeline_kw: Dict[str, Any]) -> None:
    mods = _load_pipeline_modules()
    run_pipeline = mods["run_pipeline"]
    stop_ev = threading.Event()
    with _state_lock:
        _pipeline_state["stop_event"] = stop_ev
        _pipeline_state["status"] = "running"
        _pipeline_state["stop_requested"] = False
        _pipeline_state["last_error"] = None
        _pipeline_state["last_result"] = None
        _pipeline_state["live_metrics"] = {}

    progress_cb = _make_progress_callback()
    _start_live_ticker()

    try:
        result = run_pipeline(
            config_path=None,
            project_root=_project_root(),
            config_overrides=overrides,
            stop_event=stop_ev,
            progress_callback=progress_cb,
            **pipeline_kw,
        )
        with _state_lock:
            _pipeline_state["last_result"] = result
        _emit_pipeline({"type": "pipeline_complete", **{k: v for k, v in result.items() if k != "metrics"}})
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _pipeline_state["last_error"] = str(exc)
        _emit_pipeline({"type": "pipeline_error", "message": str(exc)})
    finally:
        _stop_live_ticker()
        try:
            from scripts.ui_bridge import get_live_state

            _socket_emit("pipeline_status_update", get_live_state())
        except Exception:  # noqa: BLE001
            pass
        with _state_lock:
            _pipeline_state["status"] = "idle"
            _pipeline_state["stop_requested"] = False
            _pipeline_state["stop_event"] = None
            _pipeline_state["worker"] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/environment")
def api_environment() -> Any:
    env = _refresh_environment(use_cache=True, deep=False)
    job = _get_install_job()
    return jsonify({
        "environment": env,
        "install_running": job.running,
        "pipeline_can_start": env.get("readiness") in ("ready_gpu", "ready_cpu"),
        "bind": f"{_BIND_HOST}:{_BIND_PORT}",
    })


@app.post("/api/environment/check")
def api_environment_check() -> Any:
    global _env_status_cache
    with _env_lock:
        _env_status_cache = None
    env = _refresh_environment(use_cache=False, deep=True)
    return jsonify({"ok": True, "environment": env, "check_mode": env.get("check_mode", "deep")})


@app.post("/api/environment/install")
def api_environment_install() -> Any:
    job = _get_install_job()
    if job.running:
        return jsonify({"ok": False, "error": "Install already in progress."}), 409
    if not (_PROJECT_ROOT / "requirements.txt").is_file():
        return jsonify({"ok": False, "error": "requirements.txt not found."}), 404

    global _env_status_cache, _pipeline_modules
    with _env_lock:
        _env_status_cache = None
    _pipeline_modules = None

    started = job.start(_PROJECT_ROOT, _emit_install)
    if not started:
        return jsonify({"ok": False, "error": "Could not start install job."}), 409
    _emit_install("install_status", {"message": "Install started…", "phase": "start"})
    return jsonify({"ok": True, "message": "Install started. Watch the setup terminal for live output."})


@app.get("/api/status")
def api_status() -> Any:
    from scripts.ui_bridge import get_live_state

    with _state_lock:
        body = {
            "status": _pipeline_state["status"],
            "stop_requested": bool(_pipeline_state.get("stop_requested")),
            "last_error": _pipeline_state["last_error"],
            "last_result": _pipeline_state.get("last_result"),
            "live_metrics": _pipeline_state.get("live_metrics") or {},
            "pipeline_ready": _pipeline_ready(),
            "live": get_live_state(),
        }
    return jsonify(body)


@app.get("/api/live/status")
def api_live_status() -> Any:
    from scripts.ui_bridge import get_live_state

    return jsonify({"live": get_live_state()})


@app.get("/api/dataset/config")
def api_dataset_config_get() -> Any:
    from scripts.dataset_scanner import load_dataset_config

    return jsonify({"ok": True, "config": load_dataset_config(_project_root())})


@app.post("/api/dataset/config")
def api_dataset_config_post() -> Any:
    from scripts.dataset_scanner import load_dataset_config, save_dataset_config

    data = request.get_json(silent=True) or {}
    cfg = load_dataset_config(_project_root())
    for key in (
        "source_mode",
        "local_path",
        "b2_remote_path",
        "b2_ingest_on_start",
        "b2_export_remote_path",
        "b2_export_on_complete",
        "b2_transfer_batch_size",
        "sync_to_input_raw",
    ):
        if key in data:
            cfg[key] = data[key]
    path = save_dataset_config(_project_root(), cfg)
    if data.get("persist_yaml"):
        from scripts.dataset_scanner import persist_ui_section_yaml

        persist_ui_section_yaml(_project_root(), cfg)
    return jsonify({"ok": True, "path": str(path), "config": cfg})


@app.post("/api/dataset/scan")
def api_dataset_scan() -> Any:
    from scripts.dataset_scanner import load_dataset_config, run_scan
    from scripts.utils import load_config

    data = request.get_json(silent=True) or {}
    cfg = load_dataset_config(_project_root())
    cfg.update({k: data[k] for k in data if k in cfg or k in (
        "source_mode",
        "local_path",
        "b2_remote_path",
        "b2_ingest_on_start",
        "b2_export_remote_path",
        "b2_export_on_complete",
        "b2_transfer_batch_size",
        "sync_to_input_raw",
    )})
    yaml_cfg = load_config(_project_root() / "config.yaml")
    scan = run_scan(cfg, _project_root(), yaml_cfg=yaml_cfg)
    return jsonify({"ok": bool(scan.get("ok", scan.get("image_count", 0) >= 0)), "scan": scan, "config": cfg})


@app.get("/api/b2/overview")
def api_b2_overview() -> Any:
    from scripts.dataset_scanner import load_dataset_config
    from scripts.rclone_integration import try_load_b2_config_for_ui
    from scripts.utils import load_config

    root = _project_root()
    yaml_cfg = load_config(root / "config.yaml")
    overview = try_load_b2_config_for_ui(root, yaml_cfg=yaml_cfg.get("backblaze") or {})
    ui_cfg = load_dataset_config(root)
    return jsonify({
        "ok": True,
        "b2": overview,
        "ui": {
            "b2_remote_path": ui_cfg.get("b2_remote_path"),
            "b2_ingest_on_start": bool(ui_cfg.get("b2_ingest_on_start")),
            "b2_export_remote_path": ui_cfg.get("b2_export_remote_path"),
            "b2_export_on_complete": bool(ui_cfg.get("b2_export_on_complete")),
            "b2_transfer_batch_size": ui_cfg.get("b2_transfer_batch_size"),
        },
    })


@app.get("/api/b2/commands")
def api_b2_commands() -> Any:
    """Live rclone preview — always returns commands (placeholders when paths/buckets unset)."""
    from scripts.dataset_scanner import load_dataset_config
    from scripts.rclone_integration import (
        build_rclone_command_reference_for_ui,
        try_load_b2_config_for_ui,
    )
    from scripts.utils import load_config

    root = _project_root()
    yaml_full = load_config(root / "config.yaml")
    bz = yaml_full.get("backblaze") or {}
    sec = yaml_full.get("security") or {}
    ui_cfg = load_dataset_config(root)
    overview = try_load_b2_config_for_ui(root, yaml_cfg=bz)

    def _arg(name: str, fallback: str = "") -> str:
        return (request.args.get(name) or "").strip() or fallback

    ingest_path = _arg("ingest_path", str(ui_cfg.get("b2_remote_path") or ""))
    export_path = _arg("export_path", str(ui_cfg.get("b2_export_remote_path") or ""))
    ro_bucket = _arg("readonly_bucket", str(overview.get("readonly_bucket") or ""))
    wr_bucket = _arg("write_bucket", str(overview.get("write_bucket") or ""))
    batch_raw = request.args.get("batch_size") or ui_cfg.get("b2_transfer_batch_size") or bz.get("transfer_batch_size") or 32
    try:
        batch_size = max(1, int(batch_raw))
    except (TypeError, ValueError):
        batch_size = 32

    crypt = request.args.get("crypt_enabled", "").lower() in ("1", "true", "yes")
    if request.args.get("crypt_enabled") is None:
        crypt = bool(sec.get("crypt_enabled", False))

    paths = yaml_full.get("paths") or {}
    ref = build_rclone_command_reference_for_ui(
        readonly_bucket=ro_bucket,
        write_bucket=wr_bucket,
        ingest_remote_path=ingest_path,
        export_remote_path=export_path,
        local_input_dir=_arg("local_input", str(paths.get("input_raw") or "input_raw")),
        local_final_dir=_arg("local_final", str(paths.get("final_clean") or "final_clean")),
        batch_size=batch_size,
        crypt_enabled=crypt,
        rclone_config=str(overview.get("rclone_config") or bz.get("rclone_config") or ""),
        yaml_backblaze=bz,
        dry_run=bool((request.args.get("dry_run") or "").lower() in ("1", "true", "yes")),
        use_placeholders=True,
    )
    return jsonify({
        "ok": True,
        "reference": ref,
        "overview": overview,
        "configured": bool(overview.get("configured")),
    })


@app.get("/api/stats")
def api_stats() -> Any:
    env = _refresh_environment(use_cache=True)
    path_kwargs = _paths_from_request_args()
    paths = _light_resolve_paths(path_kwargs)
    counts = _folder_counts_payload(path_kwargs)
    pipeline_ready = env.get("readiness") in ("ready_gpu", "ready_cpu")
    cfg = _load_yaml_light()
    sec = cfg.get("security") or {}
    gpu = cfg.get("gpu") or {}

    body: Dict[str, Any] = {
        "environment": env,
        "pipeline_ready": pipeline_ready,
        "paths": {k: str(v) for k, v in paths.items()},
        "counts": {
            "input_raw": counts.get("input_raw", 0),
            "quarantine": counts.get("quarantine", 0),
            "final_clean": counts.get("final_clean", 0),
            "manual_review": counts.get("manual_review", 0),
        },
        "security": {
            "level": sec.get("level", "standard"),
            "crypt_enabled": bool(sec.get("crypt_enabled", False)),
            "verify_ingest_checksums": bool(sec.get("verify_ingest_checksums", True)),
            "secure_wipe": bool(sec.get("secure_wipe", False)),
        },
        "gpu": {
            "device": gpu.get("device", cfg.get("device", "cuda")),
            "adaptive_batch": bool(gpu.get("adaptive_batch", True)),
            "snapshot": None,
        },
        "compute_profile": None,
        "cpu_fallback": None,
        "user_message": env.get("compute_message"),
        "manifest": _manifest_summary(paths["reports"]),
        "bind": f"{_BIND_HOST}:{_BIND_PORT}",
        "stats_mode": "light",
    }
    if not pipeline_ready:
        body["message"] = (
            env.get("readiness_label")
            or "Install dependencies from Setup Environment to enable processing."
        )

    if env.get("requirements_ok"):
        try:
            mods = _load_pipeline_modules()
            full_cfg = _cfg_for_paths(path_kwargs)
            paths = mods["resolve_pipeline_paths"](_project_root(), full_cfg)
            exts = full_cfg.get("image_extensions", _default_image_exts())
            sec = full_cfg.get("security") or sec
            gpu = full_cfg.get("gpu") or gpu

            def count(folder: Path) -> int:
                return len(mods["discover_images"](folder, exts)) if folder.is_dir() else 0

            compute_profile = mods["get_compute_profile"]()
            if not compute_profile:
                compute_profile = mods["initialize_compute"](copy.deepcopy(full_cfg), force_gpu=False)
            body.update(
                {
                    "paths": {k: str(v) for k, v in paths.items()},
                    "counts": {
                        "input_raw": count(paths["input_raw"]),
                        "quarantine": count(paths["quarantine"]),
                        "final_clean": count(paths["final_clean"]),
                        "manual_review": count(paths["manual_review"]),
                    },
                    "security": {
                        "level": sec.get("level", "standard"),
                        "crypt_enabled": bool(sec.get("crypt_enabled", False)),
                        "verify_ingest_checksums": bool(sec.get("verify_ingest_checksums", True)),
                        "secure_wipe": bool(sec.get("secure_wipe", False)),
                    },
                    "gpu": {
                        "device": gpu.get("device", full_cfg.get("device")),
                        "adaptive_batch": bool(gpu.get("adaptive_batch", True)),
                        "snapshot": mods["cuda_memory_snapshot"](),
                    },
                    "compute_profile": compute_profile,
                    "cpu_fallback": bool(compute_profile.get("cpu_fallback")),
                    "user_message": compute_profile.get("user_message"),
                    "mode": compute_profile.get("mode"),
                    "gan_inpainting_enabled": bool(compute_profile.get("gan_inpainting_enabled")),
                    "fallback_events": compute_profile.get("fallback_events", []),
                    "manifest": _manifest_summary(paths["reports"]),
                    "stats_mode": "full",
                    "message": None if pipeline_ready else body.get("message"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            body["stats_warning"] = f"Full stats unavailable: {exc}"

    return jsonify(body)


def _manifest_summary(reports_dir: Path) -> Dict[str, Any]:
    manifest = reports_dir / "processed_manifest.json"
    if not manifest.is_file():
        return {"exists": False, "entries": 0}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entries = data.get("entries") or {}
        return {
            "exists": True,
            "entries": len(entries) if isinstance(entries, dict) else 0,
            "updated_at": data.get("updated_at"),
            "schema": data.get("schema"),
        }
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "error": "unreadable"}


@app.get("/api/monitoring")
def api_monitoring() -> Any:
    paths = _light_resolve_paths(_paths_from_request_args())
    events = _read_monitoring_tail(paths["logs"], n=50)
    payload: Dict[str, Any] = {
        "events": events,
        "security_events": [e for e in events if str(e.get("type", "")).startswith("security_")][-15:],
        "transfer_events": [e for e in events if str(e.get("type", "")).startswith("transfer_")][-10:],
        "recent_batches": [e for e in events if e.get("type") == "batch_complete"][-10:],
        "gpu_snapshot": None,
    }
    if not events:
        payload["message"] = "No monitoring events yet — run the pipeline or check logs/monitoring.jsonl path."
    if _pipeline_ready():
        try:
            mods = _load_pipeline_modules()
            payload["gpu_snapshot"] = mods["cuda_memory_snapshot"]()
        except Exception:  # noqa: BLE001
            pass
    return jsonify(payload)


def _read_monitoring_tail(logs_dir: Path, n: int = 40) -> list[Dict[str, Any]]:
    path = logs_dir / "monitoring.jsonl"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[Dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.post("/api/start")
def api_start() -> Any:
    env = _refresh_environment(use_cache=True)
    if env.get("readiness") not in ("ready_gpu", "ready_cpu"):
        return jsonify({
            "ok": False,
            "error": "Environment not ready. Install dependencies from Setup Environment first.",
            "environment": env,
        }), 503

    data = request.get_json(silent=True) or {}

    # Scope enforcement: Image-Only mode must be explicitly confirmed by the operator
    # (video support is deferred). Defense-in-depth beyond the UI gating.
    if str(data.get("processing_mode") or "").strip().lower() != "images_only":
        return jsonify({
            "ok": False,
            "error": (
                "Image-Only mode must be explicitly selected and confirmed before "
                "starting. Videos are excluded by design (video support deferred)."
            ),
        }), 400

    batch_size = max(8, min(128, int(data.get("batch_size", 32))))

    path_keys = ("input_raw", "final_clean", "quarantine", "manual_review", "temp_processed", "logs", "reports")
    paths_override = {k: str((data.get("paths") or {}).get(k, "")).strip() for k in path_keys}
    paths_override = {k: v for k, v in paths_override.items() if v}

    overrides: Dict[str, Any] = {
        "batch_size": batch_size,
        "monitoring": {"resource_monitoring": True},
        "performance": {"always_monitor": True},
        "scope": {
            "processing_mode": "images_only",
            "video_support": "deferred",
            "confirmed_via": "ui",
        },
    }
    if paths_override:
        overrides["paths"] = paths_override
    if data.get("security_level"):
        overrides.setdefault("security", {})["level"] = str(data["security_level"])
    if data.get("enable_crypt"):
        overrides.setdefault("security", {})["crypt_enabled"] = True
    if data.get("gpu_device"):
        overrides.setdefault("gpu", {})["device"] = str(data["gpu_device"])

    pipeline_kw = _build_pipeline_kwargs(data)

    # Record run scope (pilot vs full) for the audit trail / config log.
    _max_images = int(data.get("max_images") or 0)
    overrides["scope"]["run_type"] = "pilot" if _max_images > 0 else "full"
    overrides["scope"]["batch_size"] = batch_size
    if _max_images > 0:
        overrides["scope"]["max_images"] = _max_images

    inline_dataset = data.get("dataset") if isinstance(data.get("dataset"), dict) else None
    try:
        overrides, pipeline_kw = _merge_dataset_overrides(
            overrides, pipeline_kw, inline_dataset=inline_dataset
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Dataset config merge failed: {exc}"}), 400
    if paths_override:
        overrides.setdefault("paths", {})
        mods = _load_pipeline_modules()
        mods["deep_update"](overrides["paths"], paths_override)

    with _state_lock:
        if _pipeline_state["status"] == "running":
            return jsonify({"ok": False, "error": "Pipeline is already running."}), 409
        t = threading.Thread(
            target=_pipeline_worker,
            args=(overrides, pipeline_kw),
            name="anonymizer-pipeline",
            daemon=True,
        )
        _pipeline_state["worker"] = t
        _pipeline_state["last_overrides"] = {**copy.deepcopy(overrides), "_pipeline": pipeline_kw}
        t.start()

    return jsonify({"ok": True, "overrides": overrides, "pipeline": pipeline_kw})


@app.post("/api/stop")
def api_stop() -> Any:
    with _state_lock:
        ev = _pipeline_state.get("stop_event")
        if ev is not None:
            ev.set()
            _pipeline_state["stop_requested"] = True
    return jsonify({"ok": True, "message": "Stop requested; will finish after the current batch."})


@app.get("/api/reports/<name>")
def api_reports(name: str) -> Any:
    allowed = {
        "master_summary.csv",
        "master_summary.pdf",
        "anonymization_audit.json",
        "gpu_readiness_report.md",
        "readiness_summary.md",
        "performance_report.md",
        "security_report.md",
        "ui_environment_status.json",
    }
    if name not in allowed:
        return jsonify({"error": "unknown report"}), 404
    paths = _light_resolve_paths(_paths_from_request_args())
    path = paths["reports"] / name
    if not path.is_file():
        return jsonify({"error": "file not found", "path": str(path)}), 404
    return send_file(path, as_attachment=False, download_name=name)


@app.get("/api/logs/latest")
def api_logs_latest() -> Any:
    paths = _light_resolve_paths(_paths_from_request_args())
    cfg = _load_yaml_light()
    logs_cfg = cfg.get("logs") or {}
    log_name = str(logs_cfg.get("processing_log", "processing.log"))
    log_path = paths["logs"] / log_name
    if not log_path.is_file():
        return jsonify({
            "lines": [],
            "path": str(log_path),
            "message": "Log file not created yet — start a pipeline run first.",
        })
    raw = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return jsonify({"path": str(log_path), "lines": raw[-400:]})


@socketio.on("connect")
def _on_connect() -> None:
    from scripts.ui_bridge import get_live_state

    env = _refresh_environment(use_cache=True)
    socketio.emit(
        "environment_status",
        {"environment": env, "install_running": _get_install_job().running},
        namespace="/",
    )
    socketio.emit("pipeline_status_update", get_live_state(), namespace="/")
    _emit_pipeline({"type": "hello", "message": "connected", "bind": f"{_BIND_HOST}:{_BIND_PORT}"})


def main() -> None:
    print("PrivaGen™ · a Barren Business Development Product")
    print("  Barren Business Development — Web control panel (dataset_anonymizer)")
    _access = "localhost only — secure" if _BIND_HOST == "127.0.0.1" else "network-exposed (cloud preview mode)"
    print(f"  UI URL: http://{_BIND_HOST}:{_BIND_PORT} ({_access})")
    try:
        from scripts.secrets_manager import load_secrets

        applied = load_secrets(_PROJECT_ROOT)
        if applied:
            print(f"  Secrets: loaded {len(applied)} value(s) from encrypted/locked env file.")
    except Exception as exc:  # noqa: BLE001
        # Never print secret values; surface only the reason so boot isn't silent.
        print(f"  Secrets: not loaded ({exc}). Falling back to process environment.")
    try:
        from scripts.environment_checker import check_environment_light, load_cached_status

        env = load_cached_status(_PROJECT_ROOT) or check_environment_light(_PROJECT_ROOT)
        print(f"  Readiness: {env.get('readiness_label', 'unknown')}")
        if env.get("check_mode") == "light" and env.get("readiness") == "not_ready":
            if env.get("requirements_ok"):
                print("  Click Re-check environment in the dashboard to verify GPU/CPU.")
            else:
                print("  Install full dependencies from the dashboard (Setup Environment).")
                print("  Minimal UI deps: pip install -r requirements-ui.txt")
        elif env.get("compute_message"):
            print(f"  {env['compute_message']}")
        elif not env.get("requirements_ok"):
            print("  Install full dependencies from the dashboard (Setup Environment).")
            print("  Minimal UI deps: pip install -r requirements-ui.txt")
    except Exception as exc:  # noqa: BLE001
        print(f"  Environment check skipped at startup: {exc}")
    print("  Starting server (safe mode — no torch/paddle import at boot).")
    socketio.run(
        app,
        host=_BIND_HOST,
        port=_BIND_PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
