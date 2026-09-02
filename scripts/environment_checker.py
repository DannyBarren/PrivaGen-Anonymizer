"""
Environment readiness checker for dataset_anonymizer.

Safe to import with minimal dependencies (stdlib only for core checks).
Heavy imports (torch, paddleocr, etc.) are wrapped in try/except.

CLI:
    python -m scripts.environment_checker
    python -m scripts.environment_checker --json
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Map PyPI distribution names → import names
_IMPORT_ALIASES: Dict[str, str] = {
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "scikit-image": "skimage",
    "python-dateutil": "dateutil",
    "flask-socketio": "flask_socketio",
    "python-engineio": "engineio",
    "python-socketio": "socketio",
    "paddlepaddle-gpu": "paddle",
    "paddlepaddle": "paddle",
    "onnxruntime-gpu": "onnxruntime",
    "simple-lama-inpainting": "simple_lama_inpainting",
    "markupsafe": "markupsafe",
    "lazy_loader": "lazy_loader",
}

# Critical packages for pipeline (subset of requirements.txt)
_CRITICAL_PACKAGES: Tuple[str, ...] = (
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "Pillow",
    "paddleocr",
    "structlog",
    "flask",
    "flask-socketio",
    "reportlab",
    "insightface",
    "onnxruntime-gpu",
    "iopaint",
    "simple-lama-inpainting",
    "tqdm",
    "PyYAML",
    "pandas",
    "scipy",
    "scikit-image",
)

_UI_MINIMAL_PACKAGES: Tuple[str, ...] = ("flask", "flask-socketio")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalize_pkg_name(line: str) -> Optional[str]:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return None
    name = re.split(r"[<>=!;\[]", line, maxsplit=1)[0].strip()
    return name or None


def _parse_requirements(path: Path) -> List[str]:
    if not path.is_file():
        return list(_CRITICAL_PACKAGES)
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        pkg = _normalize_pkg_name(raw)
        if pkg:
            names.append(pkg)
    return names


def _import_name(pkg: str) -> str:
    return _IMPORT_ALIASES.get(pkg.lower(), pkg.lower().replace("-", "_"))


def _check_import(pkg: str) -> Tuple[bool, Optional[str]]:
    mod = _import_name(pkg)
    try:
        if importlib.util.find_spec(mod) is None:
            return False, f"module {mod} not found"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _conda_info() -> Dict[str, Any]:
    prefix = os.environ.get("CONDA_PREFIX", "")
    default_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    return {
        "active": bool(prefix),
        "prefix": prefix or None,
        "env_name": default_env or None,
        "python": sys.executable,
    }


def _check_torch_cuda() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "version": None,
        "device_name": None,
        "error": None,
    }
    try:
        import torch

        out["torch_available"] = True
        out["version"] = getattr(torch, "__version__", "unknown")
        if torch.cuda.is_available():
            out["cuda_available"] = True
            try:
                out["device_name"] = torch.cuda.get_device_name(0)
                x = torch.zeros(1, device="cuda:0")
                _ = x + 1
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                out["cuda_available"] = False
                out["error"] = f"cuda_probe_failed: {exc}"
        else:
            out["error"] = "torch.cuda.is_available() is False"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def _check_deep_privacy2(project_root: Path) -> Dict[str, Any]:
    vendor = project_root / "vendor" / "deep_privacy2"
    cfg_path = project_root / "config.yaml"
    repo_rel = "vendor/deep_privacy2"
    if cfg_path.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            repo_rel = (cfg.get("deep_privacy2") or {}).get("repo_root") or repo_rel
        except Exception:  # noqa: BLE001
            pass
    repo = Path(repo_rel)
    if not repo.is_absolute():
        repo = (project_root / repo).resolve()
    ok = repo.is_dir() and any(repo.iterdir())
    return {
        "ok": ok,
        "repo_path": str(repo),
        "status": "installed" if ok else "missing_clone",
        "note": "Clone https://github.com/hukkelas/deep_privacy2 into vendor/deep_privacy2 for face GAN",
    }


def _check_iopaint() -> Dict[str, Any]:
    try:
        import iopaint  # noqa: F401

        return {"ok": True, "status": "import_ok"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": "import_failed", "error": str(exc)}


def _check_paddleocr() -> Dict[str, Any]:
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        return {"ok": True, "status": "import_ok"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": "import_failed", "error": str(exc)}


def _enable_omp_duplicate_ok() -> None:
    """Windows/Anaconda: allow torch + paddle in one process during *deep* checks only."""
    if sys.platform == "win32":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _check_component_light(pkg: str, label: Optional[str] = None) -> Dict[str, Any]:
    """Presence check via importlib only — does not import torch/paddle/paddleocr."""
    mod = _import_name(pkg)
    ok = importlib.util.find_spec(mod) is not None
    return {
        "ok": ok,
        "status": "installed" if ok else "missing",
        "module": mod,
        "label": label or pkg,
        "note": "Full import probe runs on Re-check environment",
    }


def _check_pillow_version() -> Dict[str, Any]:
    try:
        from PIL import Image

        ver = getattr(Image, "__version__", "unknown")
        ok = ver.startswith("9.5")
        return {
            "ok": ok,
            "version": ver,
            "expected": "9.5.x (required by iopaint==1.6.0)",
            "warning": None if ok else "Pillow should be 9.5.0 for IOPaint compatibility",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "version": None, "error": str(exc)}


def _merge_cached_probe_fields(status: Dict[str, Any], cached: Dict[str, Any]) -> None:
    """Reuse GPU/CPU probe results from last deep check (safe for UI boot)."""
    for key in (
        "readiness",
        "readiness_label",
        "torch_available",
        "cuda_available",
        "gpu_mode_possible",
        "cpu_pipeline_ok",
        "fallback_mode",
        "compute_message",
    ):
        if key in cached:
            status[key] = cached[key]
    cached_components = cached.get("components")
    if isinstance(cached_components, dict):
        status.setdefault("components", {})
        for comp in ("torch", "paddleocr", "iopaint", "pillow"):
            if comp in cached_components:
                status["components"][comp] = cached_components[comp]


def check_environment_light(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Fast, crash-safe check for Web UI startup and polling.

    Uses importlib.find_spec only (no torch/paddle import). Merges GPU/CPU
    readiness from ``reports/ui_environment_status.json`` when present.
    """
    project_root = Path(project_root or ROOT).resolve()
    req_path = project_root / "requirements.txt"
    req_names = _parse_requirements(req_path)

    missing: List[str] = []
    installed: List[str] = []
    for pkg in req_names:
        ok, _ = _check_import(pkg)
        if ok:
            installed.append(pkg)
        else:
            missing.append(pkg)

    ui_ok = all(_check_import(p)[0] for p in _UI_MINIMAL_PACKAGES)
    critical_missing = [p for p in _CRITICAL_PACKAGES if not _check_import(p)[0]]
    requirements_ok = len(critical_missing) == 0

    torch_info = {
        "torch_available": _check_import("torch")[0],
        "cuda_available": None,
        "version": None,
        "device_name": None,
        "error": None,
        "probe": "light",
    }
    paddle = _check_component_light("paddleocr", "paddleocr")
    iopaint = _check_component_light("iopaint", "iopaint")
    pillow = _check_component_light("Pillow", "pillow")
    dp2 = _check_deep_privacy2(project_root)

    torch_available = bool(torch_info.get("torch_available"))
    cuda_available = False
    gpu_mode_possible = False
    cpu_pipeline_ok = requirements_ok and paddle.get("ok")

    warnings: List[str] = []
    if requirements_ok and not dp2.get("ok"):
        warnings.append("DeepPrivacy2 vendor clone missing — GPU face GAN disabled; OCR/blur still work")

    cached = load_cached_status(project_root)
    if cached:
        torch_available = bool(cached.get("torch_available", torch_available))
        cuda_available = bool(cached.get("cuda_available", False))
        gpu_mode_possible = bool(cached.get("gpu_mode_possible", False))
        cpu_pipeline_ok = bool(cached.get("cpu_pipeline_ok", cpu_pipeline_ok))

    if not requirements_ok:
        readiness = "not_ready"
        readiness_label = "Not ready — install dependencies from Setup Environment"
    elif cached and cached.get("readiness") in ("ready_gpu", "ready_cpu"):
        readiness = str(cached["readiness"])
        readiness_label = str(
            cached.get("readiness_label") or "Ready (cached — click Re-check to refresh GPU status)"
        )
    elif requirements_ok:
        readiness = "not_ready"
        readiness_label = (
            "Packages detected — click Re-check environment to verify GPU/CPU before processing"
        )
        warnings.append(
            "Lightweight scan only (UI safe mode). Click Re-check environment for full GPU/Paddle probe."
        )
    else:
        readiness = "not_ready"
        readiness_label = "Not ready — core packages missing"

    status: Dict[str, Any] = {
        "checked_at": _utc_now(),
        "project_root": str(project_root),
        "requirements_ok": requirements_ok,
        "ui_minimal_ok": ui_ok,
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "gpu_mode_possible": gpu_mode_possible,
        "cpu_pipeline_ok": cpu_pipeline_ok,
        "fallback_mode": cached.get("fallback_mode", "CPU") if cached else "CPU",
        "readiness": readiness,
        "readiness_label": readiness_label,
        "missing_packages": missing,
        "critical_missing": critical_missing,
        "installed_count": len(installed),
        "requirements_total": len(req_names),
        "warnings": warnings,
        "compute_message": cached.get("compute_message") if cached else None,
        "conda": _conda_info(),
        "components": {
            "torch": torch_info,
            "pillow": pillow,
            "paddleocr": paddle,
            "iopaint": iopaint,
            "deep_privacy2": dp2,
        },
        "requirements_path": str(req_path),
        "python_executable": sys.executable,
        "check_mode": "light",
    }
    if cached:
        _merge_cached_probe_fields(status, cached)
    return status


def check_environment(project_root: Optional[Path] = None, *, deep: bool = True) -> Dict[str, Any]:
    """
    Environment status dict — single source of truth for UI and CLI.

    ``deep=False`` (default for Web UI boot): safe importlib-only scan.
    ``deep=True``: imports torch/paddle and probes CUDA (use Re-check in UI).
    """
    if not deep:
        return check_environment_light(project_root)

    _enable_omp_duplicate_ok()
    project_root = Path(project_root or ROOT).resolve()
    req_path = project_root / "requirements.txt"
    req_names = _parse_requirements(req_path)

    missing: List[str] = []
    installed: List[str] = []
    for pkg in req_names:
        ok, _ = _check_import(pkg)
        if ok:
            installed.append(pkg)
        else:
            missing.append(pkg)

    ui_ok = all(_check_import(p)[0] for p in _UI_MINIMAL_PACKAGES)
    critical_missing = [p for p in _CRITICAL_PACKAGES if not _check_import(p)[0]]
    requirements_ok = len(critical_missing) == 0

    torch_info = _check_torch_cuda()
    torch_available = bool(torch_info.get("torch_available"))
    cuda_available = bool(torch_info.get("cuda_available"))

    dp2 = _check_deep_privacy2(project_root)
    iopaint = _check_iopaint()
    paddle = _check_paddleocr()
    pillow = _check_pillow_version()

    gpu_mode_possible = requirements_ok and cuda_available and dp2.get("ok")
    cpu_pipeline_ok = requirements_ok and paddle.get("ok")

    warnings: List[str] = []
    if pillow.get("warning"):
        warnings.append(str(pillow["warning"]))
    if torch_info.get("error") and torch_available:
        warnings.append(f"CUDA: {torch_info['error']}")
    if requirements_ok and not dp2.get("ok"):
        warnings.append("DeepPrivacy2 vendor clone missing — GPU face GAN disabled; OCR/blur still work")
    if requirements_ok and not iopaint.get("ok"):
        warnings.append("IOPaint not importable — LaMa may use simple_lama or CPU OpenCV redaction")

    if not requirements_ok:
        readiness = "not_ready"
        readiness_label = "Not ready — install dependencies"
    elif gpu_mode_possible and iopaint.get("ok"):
        readiness = "ready_gpu"
        readiness_label = "Ready for GPU — full pipeline (DeepPrivacy2 + targeted inpainting)"
    elif cpu_pipeline_ok:
        readiness = "ready_cpu"
        readiness_label = "Ready for CPU fallback — text anonymization + basic blurring"
    else:
        readiness = "not_ready"
        readiness_label = "Not ready — core packages missing"

    fallback_mode = "GPU" if (cuda_available and requirements_ok) else "CPU"

    compute_message: Optional[str] = None
    if requirements_ok:
        try:
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from scripts.device_manager import format_user_message, initialize_compute
            from scripts.utils import load_config

            cfg = load_config(project_root / "config.yaml")
            profile = initialize_compute(cfg, force_gpu=False)
            fallback_mode = "CPU" if profile.get("cpu_fallback") else "GPU"
            compute_message = profile.get("user_message")
            if profile.get("cpu_fallback") and compute_message:
                warnings.append(compute_message)
            elif not profile.get("cpu_fallback") and cuda_available:
                warnings.append(
                    "GPU configuration successful → Full pipeline (DeepPrivacy2 + targeted inpainting) enabled"
                    if dp2.get("ok")
                    else "GPU available — clone DeepPrivacy2 for face GAN; OCR/inpaint active"
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Compute profile probe skipped: {exc}")

    status: Dict[str, Any] = {
        "checked_at": _utc_now(),
        "project_root": str(project_root),
        "requirements_ok": requirements_ok,
        "ui_minimal_ok": ui_ok,
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "gpu_mode_possible": gpu_mode_possible,
        "cpu_pipeline_ok": cpu_pipeline_ok,
        "fallback_mode": fallback_mode,
        "readiness": readiness,
        "readiness_label": readiness_label,
        "missing_packages": missing,
        "critical_missing": critical_missing,
        "installed_count": len(installed),
        "requirements_total": len(req_names),
        "warnings": warnings,
        "compute_message": compute_message,
        "conda": _conda_info(),
        "components": {
            "torch": torch_info,
            "pillow": pillow,
            "paddleocr": paddle,
            "iopaint": iopaint,
            "deep_privacy2": dp2,
        },
        "requirements_path": str(req_path),
        "python_executable": sys.executable,
        "check_mode": "deep",
    }

    _write_status_cache(project_root, status)
    return status


def _write_status_cache(project_root: Path, status: Dict[str, Any]) -> Path:
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "ui_environment_status.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_cached_status(project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    project_root = Path(project_root or ROOT).resolve()
    path = project_root / "reports" / "ui_environment_status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check dataset_anonymizer environment")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    status = check_environment(args.root)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Readiness: {status['readiness_label']}")
        print(f"Requirements OK: {status['requirements_ok']}")
        print(f"Torch: {status['torch_available']}  CUDA: {status['cuda_available']}")
        if status["critical_missing"]:
            print(f"Missing critical: {', '.join(status['critical_missing'])}")
        for w in status.get("warnings") or []:
            print(f"  ⚠ {w}")
    sys.exit(0 if status["readiness"] != "not_ready" else 1)


if __name__ == "__main__":
    main()
