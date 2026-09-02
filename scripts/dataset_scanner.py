"""
Scan local folders or B2 remotes for image counts (UI dataset configuration).

Does not download B2 objects — uses ``rclone lsf`` with read-only remote only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .utils import discover_images, get_logger, load_config, resolve_pipeline_paths, resolve_project_root

logger = get_logger(__name__)

UI_CONFIG_REL = "reports/ui_dataset_config.json"
IMAGE_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _config_path(project_root: Path) -> Path:
    return project_root / UI_CONFIG_REL


def default_dataset_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(project_root or resolve_project_root()).resolve()
    yaml_cfg = load_config(root / "config.yaml") if (root / "config.yaml").is_file() else {}
    bz = yaml_cfg.get("backblaze") or {}
    return {
        "source_mode": "local",
        "local_path": "input_raw",
        "local_absolute": str((root / "input_raw").resolve()),
        "b2_remote_path": str(bz.get("ingest_remote_path") or "datasets/raw"),
        "b2_ingest_on_start": False,
        "b2_export_remote_path": str(bz.get("export_remote_path") or "datasets/anonymized"),
        "b2_export_on_complete": False,
        "b2_transfer_batch_size": int(bz.get("transfer_batch_size") or 32),
        "sync_to_input_raw": True,
        "last_scan": None,
    }


def load_dataset_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(project_root or resolve_project_root()).resolve()
    path = _config_path(root)
    base = default_dataset_config(root)
    if not path.is_file():
        return base
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            base.update(data)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ui_dataset_config_read_failed", error=str(exc))
    return base


def save_dataset_config(project_root: Path, config: Dict[str, Any]) -> Path:
    root = Path(project_root).resolve()
    path = _config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**default_dataset_config(root), **config}
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def scan_local_folder(
    folder: Path,
    *,
    image_extensions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    folder = Path(folder).resolve()
    exts = list(image_extensions or IMAGE_EXTS_DEFAULT)
    images = discover_images(folder, exts) if folder.is_dir() else []
    return {
        "ok": folder.is_dir(),
        "source": "local",
        "image_count": len(images),
        "path_display": str(folder.name) if len(str(folder)) > 80 else str(folder),
        "path_relative": _safe_relative_display(folder),
        "error": None if folder.is_dir() else "folder_not_found",
    }


def _safe_relative_display(path: Path) -> str:
    try:
        root = resolve_project_root()
        rel = path.resolve().relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        name = path.name
        return f"…/{name}" if name else "external_path"


def scan_b2_remote(
    remote_path: str,
    project_root: Optional[Path] = None,
    *,
    yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Count image-like files on read-only B2 remote via ``rclone lsf`` (no download).
    """
    from .rclone_integration import (
        REMOTE_READONLY,
        count_remote_image_files,
        load_b2_config,
        write_rclone_config,
    )

    root = Path(project_root or resolve_project_root()).resolve()
    cfg = yaml_cfg or load_config(root / "config.yaml")
    remote = (remote_path or "").strip()
    if not remote:
        b2 = cfg.get("backblaze") or {}
        remote = str(b2.get("ingest_remote_path") or "datasets/raw")

    try:
        b2_cfg = load_b2_config(
            yaml_cfg=cfg.get("backblaze") or {},
            security_cfg=cfg.get("security") or {},
        )
        write_rclone_config(b2_cfg)
        result = count_remote_image_files(b2_cfg, remote)
        result["source"] = "b2"
        result["remote_display"] = f"{REMOTE_READONLY}:{b2_cfg.readonly_bucket}/{remote.lstrip('/')}"
        result["read_only_reminder"] = (
            "Uses B2 read-only application key only — never the write key for ingest listing."
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("b2_scan_failed", error=str(exc))
        return {
            "ok": False,
            "source": "b2",
            "image_count": 0,
            "error": str(exc),
            "remote_display": remote,
            "read_only_reminder": "Configure B2_READONLY_KEY in .env before scanning.",
        }


def run_scan(
    dataset_cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
    *,
    yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(project_root or resolve_project_root()).resolve()
    cfg = yaml_cfg or load_config(root / "config.yaml")
    exts = cfg.get("image_extensions", IMAGE_EXTS_DEFAULT)
    mode = str(dataset_cfg.get("source_mode", "local")).lower()

    if mode == "b2":
        remote = str(dataset_cfg.get("b2_remote_path") or "").strip()
        scan = scan_b2_remote(remote, root, yaml_cfg=cfg)
    else:
        local = str(dataset_cfg.get("local_path") or "input_raw").strip()
        folder = Path(local)
        if not folder.is_absolute():
            folder = (root / folder).resolve()
        scan = scan_local_folder(folder, image_extensions=exts)

    scan["source_mode"] = mode
    dataset_cfg = {**dataset_cfg, "last_scan": scan}
    save_dataset_config(root, dataset_cfg)
    return scan


def apply_dataset_to_pipeline_overrides(
    dataset_cfg: Dict[str, Any],
    project_root: Path,
    *,
    yaml_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build ``config_overrides`` and pipeline kwargs fragments for ``run_pipeline``.
    """
    root = Path(project_root).resolve()
    cfg = yaml_cfg or load_config(root / "config.yaml")
    paths = resolve_pipeline_paths(root, cfg)
    overrides: Dict[str, Any] = {"paths": {}}
    pipeline_kw: Dict[str, Any] = {}

    mode = str(dataset_cfg.get("source_mode", "local")).lower()
    if mode == "b2" and bool(dataset_cfg.get("b2_ingest_on_start")):
        remote = str(dataset_cfg.get("b2_remote_path") or "").strip()
        if not remote:
            remote = str((cfg.get("backblaze") or {}).get("ingest_remote_path") or "")
        pipeline_kw["ingest_from_b2"] = remote or ""
        overrides["paths"]["input_raw"] = str(paths["input_raw"])
    if bool(dataset_cfg.get("b2_export_on_complete")):
        export_remote = str(dataset_cfg.get("b2_export_remote_path") or "").strip()
        if not export_remote:
            export_remote = str((cfg.get("backblaze") or {}).get("export_remote_path") or "")
        pipeline_kw["export_to_b2"] = export_remote or ""
    if mode != "b2" or not bool(dataset_cfg.get("b2_ingest_on_start")):
        local = str(dataset_cfg.get("local_path") or "input_raw").strip()
        folder = Path(local)
        if not folder.is_absolute():
            folder = (root / folder).resolve()
        if bool(dataset_cfg.get("sync_to_input_raw", True)):
            overrides["paths"]["input_raw"] = str(paths["input_raw"])
        else:
            overrides["paths"]["input_raw"] = str(folder)

    return {"overrides": overrides, "pipeline": pipeline_kw, "source_mode": mode}


def persist_ui_section_yaml(project_root: Path, dataset_cfg: Dict[str, Any]) -> None:
    """Merge UI dataset prefs into config.yaml under ``ui:`` (non-destructive deep merge)."""
    import yaml

    cfg_path = project_root / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    data.setdefault("ui", {})
    ui = data["ui"]
    ui["dataset"] = {
        "source_mode": dataset_cfg.get("source_mode", "local"),
        "local_path": dataset_cfg.get("local_path", "input_raw"),
        "b2_remote_path": dataset_cfg.get("b2_remote_path", ""),
        "b2_ingest_on_start": bool(dataset_cfg.get("b2_ingest_on_start")),
        "b2_export_remote_path": dataset_cfg.get("b2_export_remote_path", ""),
        "b2_export_on_complete": bool(dataset_cfg.get("b2_export_on_complete")),
        "sync_to_input_raw": bool(dataset_cfg.get("sync_to_input_raw", True)),
    }
    bz = data.setdefault("backblaze", {})
    if dataset_cfg.get("b2_remote_path"):
        bz["ingest_remote_path"] = dataset_cfg["b2_remote_path"]
    if dataset_cfg.get("b2_export_remote_path"):
        bz["export_remote_path"] = dataset_cfg["b2_export_remote_path"]
    cfg_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
