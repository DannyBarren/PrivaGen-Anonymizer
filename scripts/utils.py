"""
Shared helpers for the dataset_anonymizer pipeline.

Responsibilities:
- Filesystem scaffolding for the exact buyer-facing folder layout
- Robust metadata stripping (ExifTool primary, Pillow+piexif fallback)
- Image IO, JSON audit sidecars, structlog wiring, lightweight reporting helpers

Security note:
- Originals under input_raw/ should be treated as read-only. This module never
  deletes them; the orchestrator moves/copies only derived artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import structlog
from PIL import Image, ImageOps
import piexif

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

_CONFIGURED = False


def close_pipeline_logging() -> None:
    """Release log file handles (needed on Windows before temp dir cleanup)."""
    import logging

    global _CONFIGURED
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass
        root.removeHandler(handler)
    _CONFIGURED = False


def configure_structlog(log_level: str = "INFO", logfile: Optional[str] = None) -> None:
    """
    Configure structlog once per process.

    When logfile is provided, also attach a JSON-lines FileHandler to the stdlib root logger.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    import logging

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: List[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    def _redact_event(_logger: Any, _method: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        from .security import redact_secrets_obj

        return redact_secrets_obj(event_dict)

    log_level_u = (log_level or "INFO").upper()
    level = getattr(logging, log_level_u, logging.INFO) if hasattr(logging, log_level_u) else logging.INFO

    structlog.configure(
        processors=shared
        + [
            _redact_event,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(int(level)),
        cache_logger_on_first_use=True,
    )

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(int(level))
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(int(level))
        fh.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


logger = get_logger(__name__)

_INPUT_RAW_GUARD: Optional[Path] = None


def set_input_raw_guard(input_raw: Path) -> None:
    """Register read-only ``input_raw/`` root; all writes under it are rejected."""
    global _INPUT_RAW_GUARD
    _INPUT_RAW_GUARD = Path(input_raw).resolve()


def clear_input_raw_guard() -> None:
    global _INPUT_RAW_GUARD
    _INPUT_RAW_GUARD = None


def _assert_write_allowed(target: Path) -> None:
    if _INPUT_RAW_GUARD is not None:
        from .security import assert_not_input_raw

        assert_not_input_raw(target, _INPUT_RAW_GUARD)


# --------------------------------------------------------------------------------------
# Paths / layout
# --------------------------------------------------------------------------------------

PROJECT_ROOT_MARKER_FILES = ("config.yaml", "requirements.txt")


def resolve_project_root(start: Optional[Path] = None) -> Path:
    """
    Best-effort project root discovery (directory containing config.yaml).
    """
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "config.yaml").is_file():
            return candidate
    return p


def resolve_pipeline_paths(project_root: Path, cfg: Mapping[str, Any]) -> Dict[str, Path]:
    """
    Resolve configurable I/O directories (relative to project root or absolute).

    Keys: input_raw, final_clean, quarantine, manual_review, temp_processed, logs, reports.
    """
    root = Path(project_root).resolve()
    pc = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}

    def one(key: str, default_rel: str) -> Path:
        raw = pc.get(key)
        if raw is None or str(raw).strip() == "":
            p = root / default_rel
        else:
            p = Path(str(raw).strip())
            if not p.is_absolute():
                p = root / p
        return p.resolve()

    return {
        "input_raw": one("input_raw", "input_raw"),
        "final_clean": one("final_clean", "final_clean"),
        "quarantine": one("quarantine", "quarantine"),
        "manual_review": one("manual_review", "manual_review"),
        "temp_processed": one("temp_processed", "temp_processed"),
        "logs": one("logs", "logs"),
        "reports": one("reports", "reports"),
    }


def resolve_report_artifact(project_root: Path, reports_dir: Path, configured_rel: str, default_rel: str) -> Path:
    """Map config paths like ``reports/master_summary.csv`` onto ``reports_dir`` when layouts are overridden."""
    raw = (configured_rel or default_rel).strip()
    p = Path(raw)
    if p.is_absolute():
        return p
    root = Path(project_root).resolve()
    rdir = Path(reports_dir).resolve()
    parts = p.parts
    if parts and parts[0] == "reports":
        sub = Path(*parts[1:]) if len(parts) > 1 else Path(p.name)
        return (rdir / sub).resolve()
    return (root / p).resolve()


def resolve_log_artifacts(project_root: Path, logs_dir: Path, cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve ``logging.logfile`` and ``logging.batch_summary_dir`` against ``logs_dir``."""
    root = Path(project_root).resolve()
    ldir = Path(logs_dir).resolve()
    log_cfg = cfg.get("logging") if isinstance(cfg.get("logging"), dict) else {}

    def _under_logs(configured: str, default: str) -> Path:
        raw = (configured or default).strip()
        p = Path(raw)
        if p.is_absolute():
            return p.resolve()
        parts = p.parts
        if parts and parts[0] == "logs":
            sub = Path(*parts[1:]) if len(parts) > 1 else Path(p.name)
            return (ldir / sub).resolve()
        return (root / p).resolve()

    logfile = _under_logs(str(log_cfg.get("logfile", "logs/processing.log")), "logs/processing.log")
    batch_summary = _under_logs(str(log_cfg.get("batch_summary_dir", "logs/batch_summaries")), "logs/batch_summaries")
    return logfile, batch_summary


def setup_project_folders(project_root: Optional[Path] = None, cfg: Optional[Mapping[str, Any]] = None) -> Path:
    """
    Create the full directory tree required by the buyer-facing contract.

    When ``cfg`` is provided, folder locations follow ``resolve_pipeline_paths`` (including optional
    ``paths:`` overrides). When ``cfg`` is omitted, default relative layout under project root is used.

    Returns the resolved project root.
    """
    root = project_root or resolve_project_root()
    if cfg is not None:
        paths = resolve_pipeline_paths(root, dict(cfg))
        processing_log, batch_summary_dir = resolve_log_artifacts(root, paths["logs"], cfg)
        folders = [
            paths["input_raw"],
            paths["temp_processed"],
            paths["final_clean"],
            paths["quarantine"],
            paths["manual_review"],
            paths["logs"],
            batch_summary_dir,
            paths["reports"],
            root / "models",
            root / "scripts",
        ]
        reports_block = cfg.get("reports") if isinstance(cfg.get("reports"), dict) else {}
        audit_json = resolve_report_artifact(
            root,
            paths["reports"],
            str((reports_block or {}).get("audit_json", "reports/anonymization_audit.json")),
            "reports/anonymization_audit.json",
        )
        batch_summary_dir.mkdir(parents=True, exist_ok=True)
        inventories = (paths["input_raw"], root / "models", paths["quarantine"], paths["manual_review"])
    else:
        folders = [
            root / "input_raw",
            root / "temp_processed",
            root / "final_clean",
            root / "quarantine",
            root / "manual_review",
            root / "logs",
            root / "logs" / "batch_summaries",
            root / "reports",
            root / "models",
            root / "scripts",
        ]
        processing_log = root / "logs" / "processing.log"
        audit_json = root / "reports" / "anonymization_audit.json"
        inventories = (root / "input_raw", root / "models", root / "quarantine", root / "manual_review")

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    if not processing_log.exists():
        processing_log.parent.mkdir(parents=True, exist_ok=True)
        processing_log.write_text("", encoding="utf-8")

    if not audit_json.exists():
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text("[]", encoding="utf-8")

    for inventory in inventories:
        write_gitkeep(inventory)

    logger.info("project_folders_ready", root=str(root))
    return root


def write_gitkeep(path: Path) -> None:
    """Create a .gitkeep so empty inventory folders survive in git."""
    path.mkdir(parents=True, exist_ok=True)
    gk = path / ".gitkeep"
    if not gk.exists():
        gk.write_text("", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must deserialize to a mapping at the top level.")
    return cfg


# --------------------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------------------

def imread_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8 HxWx3 using Pillow (consistent EXIF orientation)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write RGB uint8 array using Pillow."""
    path = Path(path)
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=95)


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file on disk (for audit integrity)."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, ::-1].copy()


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1].copy()


# --------------------------------------------------------------------------------------
# Metadata stripping
# --------------------------------------------------------------------------------------

def strip_all_metadata(
    input_path: Path,
    output_path: Path,
    *,
    exiftool_binary: str = "exiftool",
    prefer_exiftool: bool = True,
) -> Dict[str, Any]:
    """
    Strip EXIF/IPTC/XMP as aggressively as practical.

    Strategy:
    1) If ExifTool is available and preferred, write a cleaned temp file and atomically replace.
    2) Else remove EXIF via Pillow+piexif while re-encoding the raster (metadata-free container).

    Returns a small dict describing which branch ran (for audit sidecars).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_exiftool:
        try:
            # ExifTool: strip tags and write a fresh output file (safer than editing in-place on Windows).
            in_place = input_path.resolve() == output_path.resolve()
            exif_out = output_path
            if in_place:
                exif_out = output_path.with_name(f"{output_path.stem}._exiftool_tmp{output_path.suffix}")
            elif output_path.exists():
                output_path.unlink()
            cmd = [
                exiftool_binary,
                "-all=",
                "-q",
                "-o",
                str(exif_out),
                str(input_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if in_place:
                exif_out.replace(output_path)
            return {"method": "exiftool", "binary": exiftool_binary}
        except Exception as exc:  # noqa: BLE001 - broad: best-effort fallback
            logger.warning("exiftool_failed_fallback_to_pillow", error=str(exc))
            if input_path.resolve() == output_path.resolve() and not input_path.is_file():
                raise FileNotFoundError(
                    f"Metadata strip failed and source raster was removed: {input_path}"
                ) from exc

    # Pillow + piexif fallback: save without EXIF payload.
    with Image.open(input_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        # Drop EXIF completely
        im.save(output_path, format="JPEG", quality=95, optimize=True)
    return {"method": "pillow_reencode", "note": "piexif_not_required_when_exif_omitted"}


# --------------------------------------------------------------------------------------
# Audit JSON helpers
# --------------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_update(base: MutableMapping[str, Any], extra: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v  # type: ignore[index]
    return base


def json_sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


def save_audit_json(image_path: Path, audit: MutableMapping[str, Any]) -> Path:
    """
    Persist/update per-image JSON sidecar next to the raster file.
    """
    image_path = Path(image_path)
    json_path = image_path.with_suffix(".json")
    _assert_write_allowed(json_path)
    audit.setdefault("schema", "dataset_anonymizer.audit.v1")
    audit.setdefault("updated_at", utc_now_iso())
    audit.setdefault("image_path", str(image_path.as_posix()))
    payload = json_sanitize(dict(audit))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("audit_json_written", json_path=str(json_path))
    return json_path


def load_audit_json(image_path: Path) -> Dict[str, Any]:
    json_path = Path(image_path).with_suffix(".json")
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def append_master_audit(project_root: Path, record: Mapping[str, Any], audit_path: Path) -> None:
    """
    Append one record to reports/anonymization_audit.json (JSON lines or JSON array).

    We store a JSON array for easy pandas ingestion while remaining human-readable.
    """
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Any] = []
    if audit_path.is_file():
        try:
            records = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except json.JSONDecodeError:
            records = []
    records.append(dict(record))
    audit_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def write_batch_summary_csv(batch_dir: Path, rows: Sequence[Mapping[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(out_csv, index=False)


def ensure_model_assets(project_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional one-time fetches for auxiliary weights (YOLOv8-face, etc.).

    IOPaint / PaddleOCR / InsightFace generally manage their own caches; this hook exists so you can
    pin corporate-mirror URLs inside ``config.yaml`` without scattering ``urllib`` calls across the codebase.
    """
    meta: Dict[str, Any] = {"status": "ok"}
    root = Path(project_root)
    models_dir = root / str(cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    md = cfg.get("model_downloads") or {}
    url = (md.get("yolov8_face_weights_url") or "").strip()
    if bool(md.get("auto_download_yolov8_face", False)) and url:
        dest = models_dir / "yolov8n-face.pt"
        if not dest.is_file():
            try:
                import urllib.request

                urllib.request.urlretrieve(url, str(dest))  # noqa: S310 — explicit opt-in from config
                meta["downloaded_yolov8_face"] = str(dest)
            except Exception as exc:  # noqa: BLE001
                meta["downloaded_yolov8_face"] = f"failed:{exc}"
        else:
            meta["downloaded_yolov8_face"] = "already_present"
    return meta


def compute_master_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_success_rate: float = 0.99,
) -> Dict[str, Any]:
    """
    Aggregate KPIs for buyer PDF/CSV headers (success rate, text regions, faces, QA targets).
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {
            "images_in_master": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "meets_target_success_rate": False,
            "target_success_rate": float(target_success_rate),
        }

    passed = int((df.get("final_decision", "") == "pass").sum()) if "final_decision" in df.columns else 0
    failed = int((df.get("final_decision", "") == "fail").sum()) if "final_decision" in df.columns else 0
    n = int(len(df))
    rate = float(passed / n) if n else 0.0

    def _sum_col(name: str) -> int:
        if name not in df.columns:
            return 0
        s = pd.to_numeric(df[name], errors="coerce").fillna(0)
        return int(s.sum())

    return {
        "images_in_master": n,
        "passed": passed,
        "failed": failed,
        "success_rate": round(rate, 6),
        "meets_target_success_rate": bool(rate + 1e-9 >= float(target_success_rate)),
        "target_success_rate": float(target_success_rate),
        "sum_text_regions_inpainted": _sum_col("text_regions_inpainted"),
        "sum_text_regions_detected": _sum_col("text_regions_detected"),
        "sum_original_face_count": _sum_col("original_face_count"),
        "sum_face_gan_applied_flags": int(df["face_gan_applied"].fillna(False).astype(bool).sum())
        if "face_gan_applied" in df.columns
        else 0,
    }


def _write_master_pdf_reportlab(pdf_path: Path, stats: Mapping[str, Any], df: pd.DataFrame) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    story.append(Paragraph("<b>dataset_anonymizer — master anonymization report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    kv = [[k, str(v)] for k, v in stats.items()]
    t = Table([["Metric", "Value"], *kv], colWidths=[3.2 * inch, 3.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Per-image evidence (detections, LaMa/DP2 stages, QA scores, retries) lives beside each raster "
            "as a <b>.json</b> sidecar and in <b>reports/anonymization_audit.json</b>.",
            styles["BodyText"],
        )
    )

    if not df.empty and len(df) <= 200:
        story.append(Spacer(1, 0.15 * inch))
        sub = df.head(200)
        data = [list(sub.columns)] + sub.astype(str).values.tolist()
        t2 = Table(data, repeatRows=1)
        t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.2, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6)]))
        story.append(t2)

    doc.build(story)


def _write_master_pdf_matplotlib(pdf_path: Path, stats: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("dataset_anonymizer — master summary", fontsize=14, pad=12)
        ax.text(0.05, 0.92, json.dumps(dict(stats), indent=2), fontsize=10, va="top", family="monospace", transform=ax.transAxes)
        ax.text(
            0.05,
            0.55,
            "Per-image details live in master_summary.csv and per-image JSON sidecars.",
            fontsize=10,
            transform=ax.transAxes,
        )
        pdf.savefig(fig)
        plt.close(fig)


def write_master_summary_csv_and_pdf(
    project_root: Path,
    rows: Sequence[Mapping[str, Any]],
    csv_path: Path,
    pdf_path: Path,
    *,
    pdf_engine: str = "reportlab",
    aggregate_stats: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Buyer-facing aggregate report: CSV always; PDF via **reportlab** (preferred) or matplotlib fallback.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(csv_path, index=False)

    stats: Dict[str, Any] = dict(aggregate_stats or {})
    if not stats:
        stats = compute_master_statistics(rows)

    eng = (pdf_engine or "reportlab").lower()
    if eng == "reportlab":
        try:
            _write_master_pdf_reportlab(pdf_path, stats, df)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("reportlab_pdf_failed_using_matplotlib", error=str(exc))

    _write_master_pdf_matplotlib(pdf_path, stats)


@dataclass
class PipelineCounters:
    """Lightweight counters for master reporting."""

    images_seen: int = 0
    images_passed: int = 0
    images_failed: int = 0
    faces_detected_input_total: int = 0
    text_boxes_inpaint_total: int = 0
    retries_total: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discover_images(folder: Path, extensions: Sequence[str]) -> List[Path]:
    exts = {e.lower() for e in extensions}
    out: List[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return out


def safe_move(src: Path, dst: Path) -> None:
    _assert_write_allowed(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    js = src.with_suffix(".json")
    if js.is_file():
        shutil.move(str(js), str(dst.with_suffix(".json")))
