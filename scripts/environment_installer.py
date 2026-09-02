"""
Safe background pip install for the Web UI (no shell=True).

Streams stdout/stderr lines to a callback for Flask-SocketIO.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

EmitFn = Callable[[str, Dict[str, Any]], None]

ROOT = Path(__file__).resolve().parent.parent


def _emit_line(emit: EmitFn, line: str, *, stream: str = "stdout") -> None:
    if line is None:
        return
    text = line.rstrip("\r\n")
    if text:
        emit("install_output", {"line": text, "stream": stream})


def _emit_status(emit: EmitFn, message: str, *, phase: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {"message": message}
    if phase:
        payload["phase"] = phase
    emit("install_status", payload)


def _log_install_event(phase: str, message: str) -> None:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.utils import get_logger

        get_logger(__name__).info("ui_environment_install", phase=phase, message=message)
    except Exception:  # noqa: BLE001
        pass


def run_pip_install(
    project_root: Path,
    *,
    emit: EmitFn,
    python_executable: Optional[str] = None,
    requirements_name: str = "requirements.txt",
) -> Dict[str, Any]:
    """
    Run ``python -m pip install -r requirements.txt`` and stream output.

    Returns ``{ok, returncode, error}``.
    """
    project_root = Path(project_root).resolve()
    req = project_root / requirements_name
    if not req.is_file():
        err = f"Missing {req}"
        emit("install_error", {"error": err})
        return {"ok": False, "returncode": -1, "error": err}

    py = python_executable or sys.executable
    conda = _conda_label()

    _emit_status(emit, f"Starting install with {py}", phase="start")
    _log_install_event("start", f"pip install from {req}")
    if conda:
        _emit_status(emit, f"Conda environment: {conda}", phase="conda")
    _emit_status(emit, "Upgrading pip, wheel, setuptools…", phase="bootstrap")
    _run_pip_command(
        [py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
        project_root,
        emit,
    )
    _emit_status(emit, "Installing Torch + CUDA wheels (may take several minutes)…", phase="torch")
    _emit_status(emit, "Resolving Pillow 9.5.0 for IOPaint compatibility…", phase="pillow")

    cmd: List[str] = [py, "-m", "pip", "install", "-r", str(req)]
    _emit_status(emit, f"Running: {' '.join(cmd)}", phase="pip_install")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
    except OSError as exc:
        emit("install_error", {"error": str(exc)})
        return {"ok": False, "returncode": -1, "error": str(exc)}

    assert proc.stdout is not None
    for line in proc.stdout:
        _emit_line(emit, line)

    code = int(proc.wait())
    ok = code == 0
    if ok:
        _emit_status(emit, "Install finished — running environment check…", phase="verify")
    else:
        emit("install_error", {"error": f"pip exited with code {code}"})

    return {"ok": ok, "returncode": code, "error": None if ok else f"exit {code}"}


def _run_pip_command(cmd: List[str], cwd: Path, emit: EmitFn) -> int:
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _emit_line(emit, line)
        return int(proc.wait())
    except OSError as exc:
        emit("install_error", {"error": str(exc)})
        return -1


def _conda_label() -> Optional[str]:
    import os

    name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    if name:
        return name
    if prefix:
        return Path(prefix).name
    return None


def run_post_install_check(project_root: Path, *, emit: EmitFn) -> Dict[str, Any]:
    """Re-import checker after install (fresh subprocess avoids stale modules)."""
    _emit_status(emit, "Testing GPU / CPU readiness…", phase="gpu_test")
    py = sys.executable
    cmd = [py, "-m", "scripts.environment_checker", "--json", "--root", str(project_root)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
            shell=False,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            emit("install_error", {"error": "Environment check produced no output"})
            return {"ok": False}
        import json

        status = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        emit("install_error", {"error": f"Post-install check failed: {exc}"})
        return {"ok": False, "error": str(exc)}

    readiness = status.get("readiness", "not_ready")
    mode = status.get("fallback_mode", "CPU")
    label = status.get("readiness_label", "")

    if readiness == "ready_gpu":
        msg = "✅ Environment ready. You can now start processing on GPU."
        detail = status.get("warnings", [""])[0] if status.get("warnings") else (
            "GPU configuration successful → Full pipeline enabled"
        )
    elif readiness == "ready_cpu":
        msg = "✅ Environment ready. You can now start processing on CPU."
        detail = status.get("compute_message") or (
            "⚠️ GPU failed → Running in CPU fallback mode (text anonymization + basic blurring only). "
            "No user action needed."
        )
    else:
        msg = "Install completed but pipeline is not fully ready — see log."
        detail = label

    emit("install_complete", {
        "ok": readiness != "not_ready",
        "message": msg,
        "detail": detail,
        "readiness": readiness,
        "fallback_mode": mode,
        "status": status,
    })
    return status


class InstallJob:
    """Background install thread manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._last_result: Optional[Dict[str, Any]] = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, project_root: Path, emit: EmitFn) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._last_result = None

        def _worker() -> None:
            result: Dict[str, Any] = {"ok": False}
            try:
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                pip_result = run_pip_install(project_root, emit=emit)
                result.update(pip_result)
                if pip_result.get("ok"):
                    check = run_post_install_check(project_root, emit=emit)
                    result["environment"] = check
                else:
                    emit("install_complete", {
                        "ok": False,
                        "message": "Install failed — see output above.",
                        "detail": pip_result.get("error"),
                    })
            except Exception as exc:  # noqa: BLE001
                emit("install_error", {"error": str(exc)})
                emit("install_complete", {"ok": False, "message": str(exc)})
                result = {"ok": False, "error": str(exc)}
            finally:
                with self._lock:
                    self._running = False
                    self._last_result = result

        threading.Thread(target=_worker, name="env-install", daemon=True).start()
        return True
