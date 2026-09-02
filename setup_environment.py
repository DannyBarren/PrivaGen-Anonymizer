#!/usr/bin/env python3
"""
Bootstrap dataset_anonymizer (CLI companion to the Web UI Setup Environment).

Usage:
    python setup_environment.py
    python setup_environment.py --skip-conda --skip-install
    python setup_environment.py --install-only
    python setup_environment.py --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_UI = ROOT / "requirements-ui.txt"
DEFAULT_ENV = "dataset_anonymizer"
PYTHON_VERSION = "3.10"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"\n>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd or ROOT), check=check, text=True)


def _conda_available() -> bool:
    return shutil.which("conda") is not None


def _in_target_env(env_name: str) -> bool:
    prefix = os.environ.get("CONDA_PREFIX", "")
    return env_name in Path(prefix).name or os.environ.get("CONDA_DEFAULT_ENV") == env_name


def ensure_conda_env(env_name: str, *, skip_conda: bool) -> None:
    if skip_conda:
        print(f"Using: {sys.executable}")
        return
    if not _conda_available():
        print("conda not on PATH — using active Python.")
        return
    if _in_target_env(env_name):
        print(f"Conda env '{env_name}' active.")
        return
    result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=False)
    if env_name not in (result.stdout or ""):
        _run(["conda", "create", "-n", env_name, f"python={PYTHON_VERSION}", "-y"])
    print(f"Activate: conda activate {env_name}")


def pip_install_ui() -> None:
    if REQUIREMENTS_UI.is_file():
        _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_UI)])


def pip_install_full(emit_print: bool = True) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.environment_installer import run_pip_install

    def emit(event: str, data: dict) -> None:
        if event == "install_output" and emit_print:
            print(data.get("line", ""))
        elif event == "install_status" and emit_print:
            print(f"[{data.get('phase', 'status')}] {data.get('message', '')}")
        elif event == "install_error" and emit_print:
            print(f"ERROR: {data.get('error')}", file=sys.stderr)

    return run_pip_install(ROOT, emit=emit)


def run_check() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.environment_checker import check_environment, gpu_readiness_check

    status = check_environment(ROOT)
    readiness = gpu_readiness_check()
    status["gpu_readiness"] = readiness
    return status


def print_status(status: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(status.get("readiness_label", "Unknown"))
    print(sep)
    print(f"Requirements OK: {status.get('requirements_ok')}")
    print(f"Torch: {status.get('torch_available')}  CUDA: {status.get('cuda_available')}")
    print(f"Fallback mode: {status.get('fallback_mode')}")
    if status.get("critical_missing"):
        print(f"Missing: {', '.join(status['critical_missing'][:12])}")
    for w in status.get("warnings") or []:
        print(f"  ⚠ {w}")
    gr = status.get("gpu_readiness") or {}
    if gr.get("recommendation"):
        print(f"\n{gr['recommendation']}")
    print(sep)
    print("Web UI:  python app.py  →  http://127.0.0.1:5000")
    print("         Use Setup Environment for one-click install + live log.")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap dataset_anonymizer")
    parser.add_argument("--env-name", default=DEFAULT_ENV)
    parser.add_argument("--skip-conda", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--install-ui", action="store_true", help="Install flask only (launch dashboard)")
    args = parser.parse_args()

    ensure_conda_env(args.env_name, skip_conda=args.skip_conda)

    if args.check_only:
        status = run_check()
        print_status(status)
        sys.exit(0 if status.get("readiness") != "not_ready" else 1)

    if args.install_ui:
        pip_install_ui()
        print("UI deps installed. Run: python app.py")
        return

    if not args.skip_install:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], check=False)
        result = pip_install_full()
        if not result.get("ok"):
            print("pip install failed.", file=sys.stderr)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.environment_installer import run_post_install_check

        def emit(e: str, d: dict) -> None:
            if e == "install_status":
                print(d.get("message", ""))
            elif e == "install_complete":
                print(d.get("message", ""))
                if d.get("detail"):
                    print(d.get("detail"))

        if result.get("ok"):
            run_post_install_check(ROOT, emit=emit)

    if args.install_only:
        return

    status = run_check()
    print_status(status)
    sys.exit(0 if status.get("readiness") != "not_ready" else 1)


if __name__ == "__main__":
    main()
