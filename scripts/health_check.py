"""
Comprehensive read-only health check for dataset_anonymizer production readiness.

Verifies folder structure, critical files, configuration, dependencies, model warm-up,
GPU/rclone status, and runs an ultra-light pipeline dry-run.

Run from project root:
    python -m scripts.health_check
    python -m scripts.health_check --skip-model-warm
    python -m scripts.health_check --no-color
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.component_check import run_component_activation_check
from scripts.gpu_runtime import get_gpu_validation_report, validate_gpu_at_startup
from scripts.utils import (
    discover_images,
    load_config,
    resolve_log_artifacts,
    resolve_pipeline_paths,
    setup_project_folders,
)

# ---------------------------------------------------------------------------
# Terminal colors (ANSI; disabled on Windows unless NO_COLOR / --no-color)
# ---------------------------------------------------------------------------

_USE_COLOR = True


class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel.GetConsoleMode(handle, ctypes.byref(mode))
        kernel.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001
        pass


def _init_color(no_color: bool) -> None:
    global _USE_COLOR
    if no_color or os.environ.get("NO_COLOR"):
        _USE_COLOR = False
        return
    if sys.platform == "win32":
        _enable_windows_ansi()
    if not sys.stdout.isatty():
        _USE_COLOR = False


def _c(text: str, color: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{color}{text}{C.RESET}"


def ok(msg: str) -> str:
    return _c(f"[PASSED] {msg}", C.GREEN)


def fail(msg: str) -> str:
    return _c(f"[FAILED] {msg}", C.RED)


def warn(msg: str) -> str:
    return _c(f"[WARN]   {msg}", C.YELLOW)


def info(msg: str) -> str:
    return _c(msg, C.CYAN)


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

REQUIRED_FOLDERS: Tuple[Tuple[str, bool], ...] = (
    ("input_raw", True),
    ("final_clean", True),
    ("quarantine", True),
    ("manual_review", True),
    ("temp_processed", True),
    ("logs", True),
    ("reports", True),
    ("backups", False),
)

CRITICAL_FILES: Tuple[str, ...] = (
    "config.yaml",
    "requirements.txt",
    "scripts/main_pipeline.py",
    "scripts/batch_processor.py",
    "scripts/agentic_qa_crew.py",
    "scripts/rclone_integration.py",
    "scripts/security_hardening.py",
    "scripts/utils.py",
    "app.py",
    "README.md",
)

CONFIG_SECTIONS: Tuple[str, ...] = (
    "backblaze",
    "gpu",
    "security",
    "qa",
    "deep_privacy2",
    "paddleocr",
    "lama",
)

IMPORT_MODULES: Tuple[str, ...] = (
    "scripts.main_pipeline",
    "scripts.batch_processor",
    "scripts.rclone_integration",
    "scripts.security_hardening",
    "scripts.agentic_qa_crew",
    "scripts.utils",
)


@dataclass
class CategoryResult:
    name: str
    passed: bool = True
    messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


@dataclass
class HealthReport:
    categories: List[CategoryResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def add(self, cat: CategoryResult) -> None:
        self.categories.append(cat)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.categories)

    @property
    def blockers(self) -> List[str]:
        out: List[str] = []
        for c in self.categories:
            out.extend(c.blockers)
        return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_folders(project_root: Path, cfg: dict) -> CategoryResult:
    cat = CategoryResult(name="Folder Structure")
    if not cfg:
        cfg = load_config(project_root / "config.yaml") if (project_root / "config.yaml").is_file() else {}
    setup_project_folders(project_root, cfg if cfg else None)
    paths = resolve_pipeline_paths(project_root, cfg)
    _, batch_dir = resolve_log_artifacts(project_root, paths["logs"], cfg)
    batch_dir.mkdir(parents=True, exist_ok=True)

    for rel, required in REQUIRED_FOLDERS:
        p = project_root / rel
        if p.is_dir():
            cat.messages.append(ok(f"{rel}/ exists"))
        elif required:
            cat.passed = False
            cat.blockers.append(f"Missing required folder: {rel}/")
            cat.messages.append(fail(f"{rel}/ missing (could not create)"))
        else:
            cat.warnings.append(f"Optional folder missing: {rel}/")
            cat.messages.append(warn(f"{rel}/ missing (optional)"))

    resolved = resolve_pipeline_paths(project_root, cfg)
    for key, path in resolved.items():
        cat.messages.append(info(f"  paths.{key} -> {path}"))

    cat.passed = cat.passed and not any(
        not (project_root / rel).is_dir() for rel, req in REQUIRED_FOLDERS if req
    )
    return cat


def check_critical_files(project_root: Path) -> CategoryResult:
    cat = CategoryResult(name="Critical Files")
    for rel in CRITICAL_FILES:
        p = project_root / rel
        if p.is_file() and p.stat().st_size >= 0:
            cat.messages.append(ok(rel))
        else:
            cat.passed = False
            cat.blockers.append(f"Missing critical file: {rel}")
            cat.messages.append(fail(f"{rel} missing or unreadable"))
    return cat


def check_configuration(project_root: Path) -> Tuple[CategoryResult, dict]:
    cat = CategoryResult(name="Configuration Validation")
    cfg_path = project_root / "config.yaml"
    try:
        cfg = load_config(cfg_path)
        cat.messages.append(ok("config.yaml loads successfully"))
    except Exception as exc:  # noqa: BLE001
        cat.passed = False
        cat.blockers.append(f"config.yaml failed to load: {exc}")
        cat.messages.append(fail(f"config.yaml load error: {exc}"))
        return cat, {}

    for section in CONFIG_SECTIONS:
        if section in cfg and isinstance(cfg.get(section), dict):
            cat.messages.append(ok(f"section '{section}' present"))
        else:
            cat.passed = False
            cat.blockers.append(f"config.yaml missing section: {section}")
            cat.messages.append(fail(f"section '{section}' missing or invalid"))

    paths = resolve_pipeline_paths(project_root, cfg)
    for key, path in paths.items():
        if path.exists():
            cat.messages.append(ok(f"resolved path '{key}' -> {path.name}/"))
        else:
            cat.warnings.append(f"Resolved path does not exist yet: {key}={path}")
            cat.messages.append(warn(f"resolved path '{key}' not on disk yet"))

    b2 = cfg.get("backblaze") or {}
    src = str(b2.get("source_bucket") or "").strip()
    dest = str(b2.get("dest_bucket") or "").strip()
    if not src:
        cat.warnings.append("backblaze.source_bucket is empty")
        cat.messages.append(warn("backblaze.source_bucket empty (set in config or B2_READONLY_BUCKET)"))
    else:
        cat.messages.append(ok(f"backblaze.source_bucket = {src!r}"))
    if not dest:
        cat.warnings.append("backblaze.dest_bucket is empty")
        cat.messages.append(warn("backblaze.dest_bucket empty (set in config or B2_WRITE_BUCKET)"))
    else:
        cat.messages.append(ok(f"backblaze.dest_bucket = {dest!r}"))

    return cat, cfg


def check_imports() -> CategoryResult:
    cat = CategoryResult(name="Module Imports")
    for mod_name in IMPORT_MODULES:
        try:
            importlib.import_module(mod_name)
            cat.messages.append(ok(mod_name))
        except Exception as exc:  # noqa: BLE001
            cat.passed = False
            cat.blockers.append(f"Import failed: {mod_name}: {exc}")
            cat.messages.append(fail(f"{mod_name}: {exc}"))
    return cat


def check_gpu(cfg: dict) -> CategoryResult:
    cat = CategoryResult(name="GPU / CUDA")
    try:
        validation = validate_gpu_at_startup(cfg)
        report = get_gpu_validation_report()
        cuda = bool(validation.get("cuda_available"))
        device = str(validation.get("resolved_device", validation.get("requested_device", "?")))

        if cuda:
            cat.messages.append(ok(f"CUDA available (device={device})"))
        else:
            cat.warnings.append("CUDA not available - production 37k runs need a GPU host")
            cat.messages.append(warn(f"CUDA unavailable (resolved_device={device}, cpu fallback)"))

        for k in ("requested_device", "resolved_device", "cuda_available", "fallback"):
            if k in validation:
                cat.messages.append(info(f"  {k}: {validation[k]}"))

        if report:
            cat.messages.append(info(f"  torch: {report.get('torch_version', 'n/a')}"))
    except Exception as exc:  # noqa: BLE001
        cat.warnings.append(f"GPU check error: {exc}")
        cat.messages.append(warn(f"GPU check error: {exc}"))
    return cat


def check_model_warm(project_root: Path, cfg: dict) -> CategoryResult:
    cat = CategoryResult(name="Model Warm-up")
    try:
        from scripts.batch_processor import AnonymizationEngine

        engine = AnonymizationEngine(project_root, cfg)
        meta = engine.warm_models()

        checks = [
            ("paddleocr", lambda m: m.get("status") == "ok"),
            ("deep_privacy2", lambda m: m.get("status") != "skipped"),
            ("lama", lambda m: m.get("status") == "ok" or m.get("backend")),
            ("insightface_probe", lambda m: m.get("enabled") is not False),
        ]
        for key, is_ok in checks:
            part = meta.get(key) or {}
            if isinstance(part, dict):
                status = part.get("status", part.get("backend", "unknown"))
                if key == "deep_privacy2" and part.get("status") == "skipped":
                    cat.warnings.append(
                        f"DeepPrivacy2 not ready: {part.get('reason', 'vendor/deep_privacy2 missing')}"
                    )
                    cat.messages.append(warn(f"deep_privacy2 skipped ({part.get('reason', 'not installed')})"))
                    cat.blockers.append("DeepPrivacy2 required for face anonymization in production")
                elif key == "paddleocr" and part.get("status") != "ok":
                    cat.passed = False
                    cat.blockers.append(f"PaddleOCR failed: {part}")
                    cat.messages.append(fail(f"paddleocr: {part}"))
                elif key == "lama":
                    backend = part.get("backend", "?")
                    if part.get("status") == "ok" or backend:
                        cat.messages.append(ok(f"lama backend={backend} status={status}"))
                    else:
                        cat.warnings.append(f"LaMa warm issue: {part}")
                        cat.messages.append(warn(f"lama: {part}"))
                else:
                    cat.messages.append(ok(f"{key}: {json.dumps(part, default=str)[:120]}"))
            else:
                cat.messages.append(info(f"{key}: {part}"))

        gpu_part = meta.get("gpu") or {}
        cat.messages.append(info(f"  gpu warm: {json.dumps(gpu_part, default=str)[:100]}"))

    except Exception as exc:  # noqa: BLE001
        cat.passed = False
        cat.blockers.append(f"Model warm-up failed: {exc}")
        cat.messages.append(fail(f"warm_models() error: {exc}"))
    return cat


def check_rclone(cfg: dict) -> CategoryResult:
    cat = CategoryResult(name="Rclone / B2")
    binary = (os.environ.get("RCLONE_BINARY") or "rclone").strip() or "rclone"

    if shutil.which(binary):
        cat.messages.append(ok(f"rclone binary found: {binary}"))
    else:
        cat.warnings.append("rclone not on PATH")
        cat.messages.append(warn(f"rclone binary not found ({binary!r})"))
        return cat

    try:
        proc = subprocess.run(
            [binary, "version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        version_line = (proc.stdout or proc.stderr or "").splitlines()[0] if proc.stdout or proc.stderr else "unknown"
        cat.messages.append(info(f"  {version_line.strip()}"))
    except Exception as exc:  # noqa: BLE001
        cat.warnings.append(f"rclone version check failed: {exc}")
        cat.messages.append(warn(f"rclone version: {exc}"))

    try:
        from scripts.rclone_integration import load_b2_config, write_rclone_config

        sec = cfg.get("security") or {}
        b2_cfg = load_b2_config(yaml_cfg=cfg.get("backblaze") or {}, security_cfg=sec)
        cat.messages.append(ok("B2 credentials and bucket names loaded"))
        conf_path = write_rclone_config(b2_cfg)
        cat.messages.append(info(f"  rclone config: {conf_path}"))

        proc = subprocess.run(
            [binary, "--config", str(conf_path), "listremotes"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode == 0:
            remotes = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            cat.messages.append(ok(f"listremotes: {', '.join(remotes) or '(none)'}"))
            for remote in ("b2-readonly:", "b2-write:"):
                if any(r.startswith(remote.rstrip(":")) for r in remotes):
                    lsf = subprocess.run(
                        [
                            binary,
                            "--config",
                            str(conf_path),
                            "lsf",
                            remote,
                            "--max-depth",
                            "1",
                            "--dry-run",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if lsf.returncode == 0:
                        cat.messages.append(ok(f"dry-run lsf {remote} OK"))
                    else:
                        err = (lsf.stderr or lsf.stdout or "")[:200]
                        cat.warnings.append(f"dry-run lsf {remote} failed: {err}")
                        cat.messages.append(warn(f"dry-run lsf {remote}: {err}"))
        else:
            err = (proc.stderr or proc.stdout or "")[:200]
            cat.warnings.append(f"listremotes failed: {err}")
            cat.messages.append(warn(f"listremotes: {err}"))

    except Exception as exc:  # noqa: BLE001
        cat.warnings.append(str(exc))
        cat.messages.append(warn(f"B2/rclone config: {str(exc)[:200]}"))

    return cat


def check_input_queue(project_root: Path, cfg: dict) -> CategoryResult:
    cat = CategoryResult(name="Input Queue")
    paths = resolve_pipeline_paths(project_root, cfg)
    exts = cfg.get("image_extensions") or [".jpg", ".jpeg", ".png"]
    images = discover_images(paths["input_raw"], exts)
    pending = [p for p in images if p.name != ".gitkeep"]
    fc = len(list(paths["final_clean"].glob("*.jpg")))
    qz = len(discover_images(paths["quarantine"], exts))
    mr = len(discover_images(paths["manual_review"], exts))

    cat.messages.append(info(f"  input_raw images: {len(pending)}"))
    cat.messages.append(info(f"  final_clean: {fc}  quarantine: {qz}  manual_review: {mr}"))

    if len(pending) == 0:
        cat.warnings.append("input_raw/ is empty — add production images before running the pipeline")
        cat.messages.append(warn("input_raw/ has no images ready to process"))
    else:
        cat.messages.append(ok(f"{len(pending)} image(s) waiting in input_raw/"))
    return cat


def check_pipeline_dry_run(project_root: Path) -> CategoryResult:
    cat = CategoryResult(name="Pipeline Dry-run")
    try:
        from scripts.main_pipeline import run_pipeline

        result = run_pipeline(
            project_root=project_root,
            config_path=project_root / "config.yaml",
            dry_run=True,
            max_images=3,
        )
        if not result.get("dry_run"):
            cat.passed = False
            cat.blockers.append("Dry-run did not return dry_run=True")
            cat.messages.append(fail("Unexpected dry-run response"))
            return cat

        plan = result.get("plan") or {}
        cat.messages.append(ok("pipeline dry-run completed without errors"))
        cat.messages.append(info(f"  pending_images: {plan.get('pending_images')}"))
        cat.messages.append(info(f"  batch_size: {plan.get('batch_size')}"))
        cat.messages.append(info(f"  gpu_device: {plan.get('gpu_device')}"))
        cat.messages.append(info(f"  security_level: {plan.get('security_level')}"))
        sample = result.get("pending_sample") or []
        if sample:
            cat.messages.append(info(f"  pending_sample: {sample[:3]}"))
    except Exception as exc:  # noqa: BLE001
        cat.passed = False
        cat.blockers.append(f"Pipeline dry-run failed: {exc}")
        cat.messages.append(fail(f"dry-run error: {exc}"))
    return cat


def build_recommendations(report: HealthReport, project_root: Path, cfg: dict) -> List[str]:
    recs: List[str] = []
    paths = resolve_pipeline_paths(project_root, cfg)
    n_input = len(discover_images(paths["input_raw"], cfg.get("image_extensions") or [".jpg"]))

    if n_input == 0:
        recs.append("Add production images to input_raw/ before starting the 37k run.")
    b2 = cfg.get("backblaze") or {}
    if not str(b2.get("source_bucket") or "").strip():
        recs.append("Set backblaze.source_bucket in config.yaml or B2_READONLY_BUCKET in .env.")
    if not str(b2.get("dest_bucket") or "").strip():
        recs.append("Set backblaze.dest_bucket in config.yaml or B2_WRITE_BUCKET in .env.")
    if any("DeepPrivacy2" in b for b in report.blockers):
        recs.append("Clone DeepPrivacy2: git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2")
    if any("CUDA" in w for c in report.categories for w in c.warnings):
        recs.append("Deploy on a CUDA GPU host (CoreWeave) for production throughput.")
    if shutil.which("exiftool") is None:
        recs.append("Install exiftool on PATH for strongest metadata stripping (currently Pillow fallback).")
    if not recs:
        recs.append("All checks passed - proceed with ingest and python -m scripts.main_pipeline.")
    return recs


def print_report(report: HealthReport, project_root: Path) -> int:
    width = 72
    print()
    print(_c("=" * width, C.BOLD))
    print(_c(" dataset_anonymizer - Production Health Check", C.BOLD))
    print(_c("=" * width, C.BOLD))
    print(info(f" Project: {project_root}"))
    print(info(f" Started: {report.started_at}"))
    print(info(f" Finished: {report.finished_at}"))
    print()

    for cat in report.categories:
        status = ok(cat.name) if cat.passed else fail(cat.name)
        print(_c(f"\n--- {cat.name} ---", C.BOLD))
        print(status)
        for line in cat.messages:
            print(f"  {line}")
        for w in cat.warnings:
            if w not in [x.replace("[WARN]   ", "") for x in cat.messages if "[WARN]" in x]:
                print(f"  {warn(w)}")

    print()
    print(_c("--- Recommendations ---", C.BOLD))
    for rec in report.recommendations:
        print(f"  - {rec}")

    print()
    print(_c("--- Overall Status ---", C.BOLD))
    blockers = list(dict.fromkeys(report.blockers))
    hard_fail = any(not c.passed for c in report.categories if c.name in ("Critical Files", "Module Imports", "Pipeline Dry-run"))

    if hard_fail or blockers:
        print(fail("NOT READY for full 37k image pipeline"))
        if blockers:
            print()
            print(_c("Blockers:", C.RED))
            for b in blockers:
                print(f"  - {b}")
        exit_code = 1
    else:
        if any(c.warnings for c in report.categories):
            print(warn("READY with warnings - address items above before unattended production run"))
            exit_code = 0
        else:
            print(ok("READY for full 37k image pipeline"))
            exit_code = 0

    print()
    passed_n = sum(1 for c in report.categories if c.passed)
    print(info(f" Categories: {passed_n}/{len(report.categories)} passed"))
    print(_c("=" * width, C.BOLD))
    print()
    return exit_code


def run_health_check(
    *,
    project_root: Path | None = None,
    skip_model_warm: bool = False,
    no_color: bool = False,
) -> int:
    _init_color(no_color)
    project_root = (project_root or ROOT).resolve()
    report = HealthReport(started_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    report.add(check_folders(project_root, {}))

    report.add(check_critical_files(project_root))

    cfg_cat, cfg = check_configuration(project_root)
    report.add(cfg_cat)
    if not cfg:
        report.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report.recommendations = ["Fix config.yaml before re-running health check."]
        return print_report(report, project_root)

    report.add(check_imports())
    report.add(check_gpu(cfg))
    if skip_model_warm:
        skip_cat = CategoryResult(name="Model Warm-up")
        skip_cat.messages.append(warn("Skipped (--skip-model-warm)"))
        report.add(skip_cat)
    else:
        report.add(check_model_warm(project_root, cfg))

    components = run_component_activation_check(cfg, project_root)
    comp_cat = CategoryResult(name="Component Activation")
    for name, info_dict in components.items():
        active = info_dict.get("active", True)
        note = info_dict.get("note", "")
        line = f"{name}: active={active}" + (f" ({note})" if note else "")
        if name == "deep_privacy2" and not active:
            comp_cat.warnings.append("DeepPrivacy2 inactive")
            comp_cat.messages.append(warn(line))
            comp_cat.blockers.append("DeepPrivacy2 not installed")
        elif name == "gpu_cuda" and not active:
            comp_cat.messages.append(warn(line))
        elif not active:
            comp_cat.messages.append(warn(line))
        else:
            comp_cat.messages.append(ok(line))
    report.add(comp_cat)

    report.add(check_rclone(cfg))
    report.add(check_input_queue(project_root, cfg))
    report.add(check_pipeline_dry_run(project_root))

    report.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report.recommendations = build_recommendations(report, project_root, cfg)
    return print_report(report, project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Production readiness health check for dataset_anonymizer.")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Project root directory")
    parser.add_argument("--skip-model-warm", action="store_true", help="Skip heavy model warm-up")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()
    raise SystemExit(
        run_health_check(
            project_root=args.project_root,
            skip_model_warm=bool(args.skip_model_warm),
            no_color=bool(args.no_color),
        )
    )


if __name__ == "__main__":
    main()
