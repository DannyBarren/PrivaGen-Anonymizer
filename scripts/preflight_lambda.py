"""
Strict pre-flight readiness gate for the IMAGE-ONLY pipeline on a Lambda.ai GPU host.

Unlike the broad ``scripts.health_check`` (which reports and advises), this script is a
HARD GATE: it exits non-zero if any critical requirement for a real sensitive-data run
is missing, so it can be wired into launch scripts / CI before the 36k-image run.

Checks (each PASS / FAIL / WARN):
  * Python interpreter is the supported 3.10.x (pins target 3.10; 3.12 has wheel gaps)
  * GPU + CUDA available (nvidia-smi + torch.cuda.is_available + torch.version.cuda)
  * DeepPrivacy2 present in vendor/ (config deep_privacy2.repo_root)
  * Model weights present (models/ populated, or a configured/known weights dir)
  * rclone on PATH (B2 ingest/export)
  * exiftool on PATH (strongest metadata stripping)
  * security.level == full (secure wipe of temp files for sensitive data)
  * Scope is images-only (video support deferred) — informational reinforcement

Run from project root:
    python -m scripts.preflight_lambda                 # strict: exit 1 on any FAIL
    python -m scripts.preflight_lambda --json          # machine-readable
    python -m scripts.preflight_lambda --allow-cpu     # downgrade GPU FAIL to WARN (dev only)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PY = (3, 10)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _torch_cuda() -> Tuple[bool, str]:
    try:
        import torch  # type: ignore

        avail = bool(torch.cuda.is_available())
        cu = getattr(torch.version, "cuda", None)
        if avail:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                name = "cuda:0"
            return True, f"torch {torch.__version__}, CUDA {cu}, device {name}"
        return False, f"torch {torch.__version__} present but torch.cuda.is_available()=False (CUDA {cu})"
    except Exception as exc:  # noqa: BLE001
        return False, f"torch not importable: {exc}"


class Check:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []
        self.failed = 0
        self.warned = 0

    def add(self, name: str, status: str, detail: str, remediation: str = "") -> None:
        self.rows.append(
            {"check": name, "status": status, "detail": detail, "remediation": remediation}
        )
        if status == "FAIL":
            self.failed += 1
        elif status == "WARN":
            self.warned += 1


def run_preflight(*, allow_cpu: bool = False, project_root: Optional[Path] = None) -> Check:
    root = Path(project_root or ROOT).resolve()
    cfg = _load_yaml_config(root / "config.yaml")
    chk = Check()

    # 1) Python version
    v = sys.version_info
    if (v.major, v.minor) == REQUIRED_PY:
        chk.add("python_version", "PASS", f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        chk.add(
            "python_version",
            "FAIL",
            f"Python {v.major}.{v.minor}.{v.micro} (pins target 3.10)",
            "Create the supported env: conda create -n privagen python=3.10 -y && conda activate privagen",
        )

    # 2) GPU + CUDA
    smi = shutil.which("nvidia-smi")
    cuda_ok, cuda_detail = _torch_cuda()
    if smi and cuda_ok:
        chk.add("gpu_cuda", "PASS", cuda_detail)
    else:
        status = "WARN" if allow_cpu else "FAIL"
        detail = cuda_detail if smi else "nvidia-smi not found (no GPU driver)"
        chk.add(
            "gpu_cuda",
            status,
            detail,
            "Use a GPU instance; install NVIDIA driver + CUDA 12.1 and torch==2.4.1+cu121.",
        )

    # 3) DeepPrivacy2 in vendor/
    dp_rel = str(((cfg.get("deep_privacy2") or {}).get("repo_root")) or "vendor/deep_privacy2").strip()
    dp_path = (root / dp_rel) if not os.path.isabs(dp_rel) else Path(dp_rel)
    dp_populated = dp_path.is_dir() and any(dp_path.iterdir())
    if dp_populated:
        chk.add("deep_privacy2", "PASS", f"present at {dp_path}")
    else:
        chk.add(
            "deep_privacy2",
            "FAIL",
            f"missing/empty: {dp_path}",
            "git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2",
        )

    # 4) Model weights present
    models_dir = root / "models"
    weights = []
    if models_dir.is_dir():
        weights = [p for p in models_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    extra_weight_dir = os.environ.get("IOPAINT_MODEL_DIR") or os.environ.get("TORCH_HOME")
    if weights:
        chk.add("model_weights", "PASS", f"{len(weights)} weight file(s) under models/")
    elif extra_weight_dir and Path(extra_weight_dir).expanduser().is_dir():
        chk.add("model_weights", "WARN", f"models/ empty; using cache dir {extra_weight_dir} (verify on first load)")
    else:
        chk.add(
            "model_weights",
            "FAIL",
            "models/ is empty and no weights cache configured",
            "Place DeepPrivacy2 checkpoint + let IOPaint/LaMa download to a persistent volume "
            "(set TORCH_HOME / IOPAINT_MODEL_DIR to the mounted disk on ephemeral Lambda hosts).",
        )

    # 5) rclone
    if shutil.which("rclone"):
        chk.add("rclone", "PASS", shutil.which("rclone") or "on PATH")
    else:
        chk.add("rclone", "FAIL", "not found on PATH", "sudo apt-get install -y rclone")

    # 6) exiftool
    if shutil.which("exiftool"):
        chk.add("exiftool", "PASS", shutil.which("exiftool") or "on PATH")
    else:
        chk.add(
            "exiftool",
            "FAIL",
            "not found on PATH (metadata stripping degrades to Pillow fallback)",
            "sudo apt-get install -y libimage-exiftool-perl",
        )

    # 7) security.level == full  (env override wins, mirroring the CLI --security-level)
    level = (os.environ.get("SECURITY_LEVEL") or (cfg.get("security") or {}).get("level") or "standard")
    level = str(level).strip().lower()
    if level == "full":
        chk.add("security_level_full", "PASS", "security.level=full (secure_wipe enabled)")
    else:
        chk.add(
            "security_level_full",
            "FAIL",
            f"security.level={level!r} (secure temp wipe NOT guaranteed)",
            "Run with --security-level full (CLI) or set security.level: full in config.yaml.",
        )

    # 8) Scope reinforcement (informational)
    scope = (cfg.get("scope") or {})
    mode = str(scope.get("processing_mode") or "images_only")
    if mode == "images_only":
        chk.add("scope_images_only", "PASS", "Images Only (video support deferred)")
    else:
        chk.add(
            "scope_images_only",
            "FAIL",
            f"unexpected scope.processing_mode={mode!r}",
            "This build is Image-Only; do not enable video. Set scope.processing_mode: images_only.",
        )

    return chk


def _print_human(chk: Check) -> None:
    icons = {"PASS": "\u2705", "FAIL": "\u274c", "WARN": "\u26a0\ufe0f"}
    print("\n===== Lambda.ai IMAGE-ONLY pre-flight gate =====")
    for r in chk.rows:
        line = f"{icons.get(r['status'], r['status'])} [{r['status']}] {r['check']}: {r['detail']}"
        print(line)
        if r["status"] in ("FAIL", "WARN") and r["remediation"]:
            print(f"      remediation: {r['remediation']}")
    verdict = "GO" if chk.failed == 0 else "NO-GO"
    print(f"\nResult: {verdict}  (FAIL={chk.failed}, WARN={chk.warned})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict Image-Only Lambda.ai pre-flight readiness gate.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a human report")
    ap.add_argument("--allow-cpu", action="store_true", help="Downgrade GPU/CUDA FAIL to WARN (dev only)")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    args = ap.parse_args()

    chk = run_preflight(allow_cpu=bool(args.allow_cpu), project_root=args.project_root)
    if args.json:
        print(json.dumps({"failed": chk.failed, "warned": chk.warned, "checks": chk.rows}, indent=2))
    else:
        _print_human(chk)
    raise SystemExit(1 if chk.failed else 0)


if __name__ == "__main__":
    main()
