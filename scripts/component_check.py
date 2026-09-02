"""Startup confirmation for all major pipeline components (logged, no secrets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .shared_models import resolve_dp2_repo_path
from .utils import get_logger

logger = get_logger(__name__)


def run_component_activation_check(cfg: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    """Return activation matrix and log one line per component."""
    root = Path(project_root)
    report: Dict[str, Any] = {}

    # GPU
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        report["gpu_cuda"] = {
            "active": cuda,
            "device": str(cfg.get("gpu", {}).get("device", cfg.get("device", "cuda"))),
            "note": "ok" if cuda else "cpu_fallback",
        }
    except ImportError:
        report["gpu_cuda"] = {"active": False, "note": "torch_not_installed"}

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR  # noqa: F401

        report["paddleocr"] = {"active": True, "shared_singleton": True}
    except ImportError:
        report["paddleocr"] = {"active": False, "note": "install paddleocr + paddlepaddle"}

    # DeepPrivacy2
    dp_path = resolve_dp2_repo_path(dict(cfg), root)
    dp_ok = dp_path.is_dir()
    report["deep_privacy2"] = {
        "active": dp_ok,
        "repo_root": str(dp_path) if dp_ok else str(dp_path),
        "note": "ok" if dp_ok else "clone vendor/deep_privacy2",
    }

    # LaMa / IOPaint
    lama = dict(cfg.get("lama") or {})
    backend = str(lama.get("backend", "lama_cleaner"))
    lama_active = backend != "none"
    report["lama"] = {"active": lama_active, "backend": backend}

    # InsightFace
    ic = dict(cfg.get("insightface") or {})
    report["insightface"] = {
        "active": bool(ic.get("enabled", False)),
        "model": ic.get("model_name", "buffalo_l"),
    }

    # QA
    qa = dict(cfg.get("qa") or {})
    report["qa_deterministic"] = {"active": True, "crewai_llm": bool(qa.get("use_crewai_llm", False))}

    # Security
    sec = dict(cfg.get("security") or {})
    report["security_hardening"] = {
        "active": True,
        "level": sec.get("level", "standard"),
        "crypt_enabled": bool(sec.get("crypt_enabled", False)),
        "verify_ingest_checksums": bool(sec.get("verify_ingest_checksums", True)),
    }

    # Rclone / B2
    try:
        from .rclone_integration import load_b2_config

        load_b2_config(
            yaml_cfg=cfg.get("backblaze") or {},
            security_cfg=sec,
        )
        report["rclone_b2"] = {"active": True, "note": "credentials_loaded"}
    except Exception as exc:  # noqa: BLE001
        report["rclone_b2"] = {"active": False, "note": str(exc)[:120]}

    # Audit / reporting
    report["audit_reporting"] = {
        "active": True,
        "master_csv": (cfg.get("reports") or {}).get("master_csv", "reports/master_summary.csv"),
    }

    for name, info in report.items():
        active = info.get("active", True)
        logger.info("component_activation", component=name, active=bool(active), **{k: v for k, v in info.items() if k != "active"})

    return report
