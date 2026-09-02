"""
Secure Backblaze B2 integration via rclone.

Two remotes enforce key separation:
  - ``b2-readonly`` + read-only application key → ingest only
  - ``b2-write`` + write application key → export only

Secrets load from environment variables (never hardcoded). Bucket paths and
transfer policy merge from ``config.yaml`` → ``backblaze`` section when provided.

Required env vars:
    B2_READONLY_KEY, B2_WRITE_KEY
    B2_READONLY_KEY_ID + B2_WRITE_KEY_ID (least-privilege), or a shared B2_KEY_ID
    B2_READONLY_BUCKET (or set via config backblaze.source_bucket)
    B2_WRITE_BUCKET (or set via config backblaze.dest_bucket)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils import get_logger

logger = get_logger(__name__)

REMOTE_READONLY = "b2-readonly"
REMOTE_WRITE = "b2-write"
REMOTE_CRYPT = "b2-crypt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXPORT_EXTENSIONS = IMAGE_EXTENSIONS | {".json"}

# --------------------------------------------------------------------------------------
# Safe rclone defaults
# --------------------------------------------------------------------------------------
# These constants hard-set the safe defaults so the repo is preconfigured out of the box.
# Bucket/path values are placeholders — set your own via config.yaml or env. The ingest
# command is image-only and protective by default: it excludes originals (orig_*),
# thumbnails (thumb_*), and any stray video files, and runs with --dry-run first
# (remove --dry-run for the real copy).

#: rclone remote name used in the human-facing command strings.
DEFAULT_RCLONE_REMOTE = "backblaze"

#: Source (read-only) bucket + path holding the original uploads to anonymize.
DEFAULT_SOURCE_BUCKET = "your-source-bucket"
DEFAULT_INGEST_REMOTE_PATH = "datasets/raw"

#: Destination (read/write) bucket + path receiving the anonymized outputs.
DEFAULT_DEST_BUCKET = "your-dest-bucket"
DEFAULT_EXPORT_REMOTE_PATH = "datasets/anonymized"

#: Protective ingest filters — skip originals, thumbnails, and any video files so only
#: source images are ever listed/copied (image-only anonymization run).
DEFAULT_INGEST_EXCLUDES: Tuple[str, ...] = (
    "orig_*",
    "thumb_*",
    "*.mp4",
    "*.mov",
    "*.avi",
)

#: Default export parallelism.
DEFAULT_EXPORT_TRANSFERS = 16

#: Local directory whose anonymized outputs are uploaded on export.
DEFAULT_LOCAL_EXPORT_DIR = "local_clean"


def default_ingest_command(*, dry_run: bool = True) -> str:
    """
    Return the exact image-only ingest command string.

    ``dry_run=True`` (default) appends ``--dry-run`` so the first run only lists what
    would be copied. Remove it (``dry_run=False``) for the real ingest after testing.
    """
    parts = [
        "rclone",
        "ls",
        f"{DEFAULT_RCLONE_REMOTE}:{DEFAULT_SOURCE_BUCKET}/{DEFAULT_INGEST_REMOTE_PATH}",
        "--fast-list",
    ]
    for pattern in DEFAULT_INGEST_EXCLUDES:
        parts.append(f'--exclude "{pattern}"')
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def default_export_command() -> str:
    """Return the exact export command string."""
    return (
        f"rclone copy {DEFAULT_LOCAL_EXPORT_DIR}/ "
        f"{DEFAULT_RCLONE_REMOTE}:{DEFAULT_DEST_BUCKET}/{DEFAULT_EXPORT_REMOTE_PATH} "
        f"--checksum --fast-list --transfers {DEFAULT_EXPORT_TRANSFERS}"
    )


def default_rclone_commands() -> Dict[str, Any]:
    """Structured view of the locked-in rclone defaults (no secrets)."""
    return {
        "source_bucket": DEFAULT_SOURCE_BUCKET,
        "ingest_remote_path": DEFAULT_INGEST_REMOTE_PATH,
        "dest_bucket": DEFAULT_DEST_BUCKET,
        "export_remote_path": DEFAULT_EXPORT_REMOTE_PATH,
        "ingest_excludes": list(DEFAULT_INGEST_EXCLUDES),
        "export_transfers": DEFAULT_EXPORT_TRANSFERS,
        "ingest_command_dry_run": default_ingest_command(dry_run=True),
        "ingest_command_real": default_ingest_command(dry_run=False),
        "export_command": default_export_command(),
    }


def ingest_exclude_flags(cfg: "B2Config") -> List[str]:
    """rclone ``--exclude`` argv flags applied to read-only ingest listings/copies."""
    flags: List[str] = []
    for pattern in cfg.ingest_excludes:
        flags.extend(["--exclude", pattern])
    return flags

_TRANSIENT_PATTERNS = re.compile(
    r"timeout|timed out|connection reset|connection refused|connection lost|"
    r"temporary|too many requests|429|503|502|504|500 internal|rate limit|"
    r"i/o timeout|broken pipe|service unavailable|slow down|"
    r"exceeded maximum|upload failed|download failed|server closed",
    re.IGNORECASE,
)

_ERROR_CLASS_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("rate_limit", re.compile(r"429|too many requests|rate limit|slow down", re.I)),
    ("service_unavailable", re.compile(r"503|service unavailable", re.I)),
    ("gateway_error", re.compile(r"502|504|500 internal", re.I)),
    ("connection", re.compile(r"connection reset|connection refused|connection lost|broken pipe", re.I)),
    ("timeout", re.compile(r"timeout|timed out|i/o timeout", re.I)),
    ("b2_upload", re.compile(r"upload failed|b2.*error", re.I)),
]


@dataclass(frozen=True)
class B2Config:
    key_id: str
    readonly_key: str
    write_key: str
    readonly_bucket: str
    write_bucket: str
    rclone_config: Path
    ingest_remote_path: str
    export_remote_path: str
    # Least-privilege: each application key has its own key ID. Fall back to the
    # shared ``key_id`` when a per-remote ID is not supplied (backward compatible).
    readonly_key_id: str = ""
    write_key_id: str = ""
    transfer_batch_size: int = 32
    max_transfer_retries: int = 3
    require_confirm: bool = True
    quarantine_rel: str = "quarantine/b2_transfer_failures"
    crypt_enabled: bool = False
    crypt_password: str = ""
    crypt_password2: str = ""
    crypt_salt: str = ""
    transfers: int = DEFAULT_EXPORT_TRANSFERS
    checkers: int = 64
    upload_concurrency: int = 12
    chunk_size: str = "48M"
    bwlimit: str = "0"
    verify_after_export: bool = True
    #: Protective read-only ingest filters (image-only defaults).
    ingest_excludes: Tuple[str, ...] = DEFAULT_INGEST_EXCLUDES


class RcloneIntegrationError(RuntimeError):
    """Raised when rclone configuration or transfer fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_b2_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    yaml_cfg: Optional[Mapping[str, Any]] = None,
    security_cfg: Optional[Mapping[str, Any]] = None,
) -> B2Config:
    """Load B2 settings from env (secrets) and optional ``config.yaml`` ``backblaze`` block."""
    src = dict(env if env is not None else os.environ)
    yc = dict(yaml_cfg or {})
    sec = dict(security_cfg or {})

    def _require(name: str, yaml_key: Optional[str] = None) -> str:
        value = (src.get(name) or yc.get(yaml_key or "") or "").strip()
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise RcloneIntegrationError(
                f"Missing required setting: env {name}"
                + (f" or config backblaze.{yaml_key}" if yaml_key else "")
            )
        return str(value)

    readonly_bucket = (
        (src.get("B2_READONLY_BUCKET") or src.get("B2_SOURCE_BUCKET") or "")
        or str(yc.get("source_bucket") or yc.get("b2_source_bucket") or "")
        or DEFAULT_SOURCE_BUCKET
    ).strip()
    write_bucket = (
        (src.get("B2_WRITE_BUCKET") or src.get("B2_DEST_BUCKET") or "")
        or str(yc.get("dest_bucket") or yc.get("b2_dest_bucket") or "")
        or DEFAULT_DEST_BUCKET
    ).strip()
    if not readonly_bucket:
        raise RcloneIntegrationError("Missing source bucket (B2_READONLY_BUCKET / backblaze.source_bucket)")
    if not write_bucket:
        raise RcloneIntegrationError("Missing dest bucket (B2_WRITE_BUCKET / backblaze.dest_bucket)")

    default_config = Path.home() / ".config" / "rclone" / "rclone.conf"
    config_path = Path(
        (src.get("RCLONE_CONFIG") or str(yc.get("rclone_config") or default_config)).strip()
    ).expanduser()

    ingest_path = (
        src.get("B2_INGEST_REMOTE_PATH")
        or yc.get("ingest_remote_path")
        or DEFAULT_INGEST_REMOTE_PATH
    )
    export_path = (
        src.get("B2_EXPORT_REMOTE_PATH")
        or yc.get("export_remote_path")
        or DEFAULT_EXPORT_REMOTE_PATH
    )

    crypt_pw_env = str(sec.get("crypt_password_env", "RCLONE_CRYPT_PASSWORD"))
    crypt_pw2_env = str(sec.get("crypt_password2_env", "RCLONE_CRYPT_PASSWORD2"))
    crypt_salt_env = str(sec.get("crypt_salt_env", "RCLONE_CRYPT_SALT"))

    # Least-privilege key model: prefer separate application key IDs for the
    # read-only and write keys; fall back to a single shared B2_KEY_ID.
    readonly_key_id = (src.get("B2_READONLY_KEY_ID") or src.get("B2_KEY_ID") or "").strip()
    write_key_id = (src.get("B2_WRITE_KEY_ID") or src.get("B2_KEY_ID") or "").strip()
    shared_key_id = (src.get("B2_KEY_ID") or readonly_key_id or write_key_id or "").strip()
    if not readonly_key_id:
        raise RcloneIntegrationError(
            "Missing required setting: env B2_READONLY_KEY_ID (or shared B2_KEY_ID)"
        )
    if not write_key_id:
        raise RcloneIntegrationError(
            "Missing required setting: env B2_WRITE_KEY_ID (or shared B2_KEY_ID)"
        )

    return B2Config(
        key_id=shared_key_id,
        readonly_key_id=readonly_key_id,
        write_key_id=write_key_id,
        readonly_key=_require("B2_READONLY_KEY"),
        write_key=_require("B2_WRITE_KEY"),
        readonly_bucket=readonly_bucket,
        write_bucket=write_bucket,
        rclone_config=config_path,
        ingest_remote_path=str(ingest_path).strip().strip("/"),
        export_remote_path=str(export_path).strip().strip("/"),
        transfer_batch_size=int(yc.get("transfer_batch_size", src.get("B2_TRANSFER_BATCH_SIZE", 32)) or 32),
        max_transfer_retries=int(yc.get("max_transfer_retries", 3) or 3),
        require_confirm=bool(yc.get("require_confirm_real_transfer", True)),
        quarantine_rel=str(yc.get("quarantine_dir", "quarantine/b2_transfer_failures")),
        crypt_enabled=bool(sec.get("crypt_enabled", False)),
        crypt_password=(src.get(crypt_pw_env) or "").strip(),
        crypt_password2=(src.get(crypt_pw2_env) or "").strip(),
        crypt_salt=(src.get(crypt_salt_env) or "").strip(),
        transfers=int(yc.get("transfers", DEFAULT_EXPORT_TRANSFERS) or DEFAULT_EXPORT_TRANSFERS),
        checkers=int(yc.get("checkers", 64) or 64),
        upload_concurrency=int(yc.get("upload_concurrency", 12) or 12),
        chunk_size=str(yc.get("chunk_size", "48M") or "48M"),
        bwlimit=str(yc.get("bwlimit", "0") or "0"),
        verify_after_export=bool(yc.get("verify_after_export", True)),
        ingest_excludes=tuple(yc.get("ingest_excludes") or DEFAULT_INGEST_EXCLUDES),
    )


def render_rclone_config(cfg: B2Config) -> str:
    base = (
        f"[{REMOTE_READONLY}]\n"
        f"type = b2\n"
        f"account = {cfg.readonly_key_id or cfg.key_id}\n"
        f"key = {cfg.readonly_key}\n"
        f"hard_delete = false\n"
        f"\n"
        f"[{REMOTE_WRITE}]\n"
        f"type = b2\n"
        f"account = {cfg.write_key_id or cfg.key_id}\n"
        f"key = {cfg.write_key}\n"
        f"hard_delete = false\n"
    )
    if cfg.crypt_enabled:
        if not cfg.crypt_password:
            raise RcloneIntegrationError(
                "security.crypt_enabled requires RCLONE_CRYPT_PASSWORD (or crypt_password_env) in environment"
            )
        from .security_hardening import render_crypt_remote_section

        base += "\n" + render_crypt_remote_section(
            write_remote=REMOTE_WRITE,
            write_bucket=cfg.write_bucket,
            export_path="",
            password=cfg.crypt_password,
            password2=cfg.crypt_password2,
            salt=cfg.crypt_salt,
        )
    return base


def rclone_performance_flags(cfg: B2Config) -> List[str]:
    """High-volume transfer flags applied to all rclone copy/sync/check operations."""
    return [
        "--fast-list",
        "--checksum",
        "--skip-links",
        "--progress",
        f"--transfers={int(cfg.transfers)}",
        f"--checkers={int(cfg.checkers)}",
        f"--b2-upload-concurrency={int(cfg.upload_concurrency)}",
        f"--b2-chunk-size={cfg.chunk_size}",
        f"--bwlimit={cfg.bwlimit}",
    ]


def crypt_decryption_doc() -> str:
    """Buyer-facing note for decrypting crypt-wrapped B2 exports."""
    return (
        "Exports uploaded via the b2-crypt remote are filename-encrypted on B2. "
        "Decrypt with the same RCLONE_CRYPT_PASSWORD (and RCLONE_CRYPT_PASSWORD2 if set) "
        "using: rclone copy b2-crypt:bucket/path ./local_restore --config /path/to/rclone.conf"
    )


def write_rclone_config(cfg: B2Config, dest: Optional[Path] = None) -> Path:
    target = Path(dest or cfg.rclone_config).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_rclone_config(cfg), encoding="utf-8")
    remotes = [REMOTE_READONLY, REMOTE_WRITE]
    if cfg.crypt_enabled:
        remotes.append(REMOTE_CRYPT)
    logger.info("rclone_config_written", path=str(target), remotes=remotes)
    return target


def _resolve_rclone_binary() -> str:
    binary = (os.environ.get("RCLONE_BINARY") or "rclone").strip() or "rclone"
    if shutil.which(binary) is None:
        raise RcloneIntegrationError(
            f"rclone binary not found on PATH: {binary!r}. Install rclone or set RCLONE_BINARY."
        )
    return binary


def _remote_uri(remote: str, bucket: str, subpath: str) -> str:
    sub = (subpath or "").strip().strip("/")
    if sub:
        return f"{remote}:{bucket}/{sub}"
    return f"{remote}:{bucket}"


def _assert_ingest_remote(remote: str) -> None:
    if remote != REMOTE_READONLY:
        raise RcloneIntegrationError(
            f"Ingest must use read-only remote {REMOTE_READONLY!r}, got {remote!r}"
        )


def _assert_export_remote(remote: str) -> None:
    if remote != REMOTE_WRITE:
        raise RcloneIntegrationError(
            f"Export must use write remote {REMOTE_WRITE!r}, got {remote!r}"
        )


def _copy_common_flags(
    cfg: B2Config,
    *,
    progress: bool = True,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    flags = list(rclone_performance_flags(cfg))
    if not progress:
        flags = [f for f in flags if f != "--progress"]
    if extra:
        flags.extend(list(extra))
    return flags


def _classify_rclone_error(message: str) -> str:
    for label, pattern in _ERROR_CLASS_PATTERNS:
        if pattern.search(message or ""):
            return label
    return "unknown"


def _parse_check_mismatches(output: str) -> List[str]:
    """Extract relative file paths from ``rclone check`` output."""
    mismatched: List[str] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "differences found" in lower or "0 differences" in lower:
            continue
        if lower.startswith("failed to check") or "couldn't" in lower:
            continue
        if line.upper().startswith("ERROR"):
            payload = line.split(":", 1)[-1].strip() if ":" in line else line
            if payload.upper().startswith("ERROR"):
                payload = payload.split(":", 1)[-1].strip()
            rel = payload.split(":", 1)[0].strip()
            if rel and rel not in mismatched and not rel.isdigit():
                mismatched.append(rel)
        elif (" differ" in lower or "hash mismatch" in lower) and not lower.startswith("0 "):
            token = line.split()[0] if line.split() else line
            if token and token not in mismatched and not token.isdigit():
                mismatched.append(token)
    return mismatched


def _is_transient_error(message: str) -> bool:
    return bool(_TRANSIENT_PATTERNS.search(message or ""))


def _retry_delay_sec(attempt: int, *, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter for transient B2/rclone failures."""
    exp = min(base ** attempt, cap)
    return float(exp + random.uniform(0.0, min(1.0, exp * 0.25)))


def confirm_real_transfer(
    operation: str,
    *,
    remote_uri: str,
    file_count: int,
    total_bytes: int,
    dry_run: bool,
    require_confirm: bool = True,
) -> bool:
    """Prompt before non-dry-run transfers (skip when ``RCLONE_AUTO_CONFIRM=1`` or non-TTY CI)."""
    if dry_run:
        return True
    if os.environ.get("RCLONE_AUTO_CONFIRM", "").strip() in ("1", "true", "yes", "YES"):
        return True
    if not require_confirm:
        return True
    if not sys.stdin.isatty():
        raise RcloneIntegrationError(
            "Refusing unattended real B2 transfer. Set RCLONE_AUTO_CONFIRM=1 or use --ingest-dry-run / --export-dry-run."
        )

    mb = total_bytes / (1024 * 1024) if total_bytes else 0.0
    print(
        f"\n[rclone] REAL transfer: {operation}\n"
        f"  Remote: {remote_uri}\n"
        f"  Files:  {file_count}\n"
        f"  Size:   {mb:.2f} MiB (approx)\n"
    )
    answer = input("Type YES to proceed with real B2 transfer: ").strip()
    if answer != "YES":
        raise RcloneIntegrationError(f"Real transfer cancelled by user ({operation})")
    return True


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def compute_directory_hashes(
    root: Path,
    *,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    root = Path(root).resolve()
    if not root.is_dir():
        return {}
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or IMAGE_EXTENSIONS)}
    out: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = _sha256_file(path)
    return out


def build_transfer_manifest(
    operation: str,
    *,
    remote_uri: str,
    local_dir: Optional[Path],
    file_hashes: Mapping[str, str],
    dry_run: bool,
    batches: Sequence[Mapping[str, Any]],
    failed_files: Sequence[str],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    total_bytes = 0
    if local_dir is not None:
        root = Path(local_dir)
        for rel in file_hashes:
            p = root / rel
            if p.is_file():
                total_bytes += p.stat().st_size
    manifest = {
        "schema": "dataset_anonymizer.transfer_manifest.v1",
        "operation": operation,
        "remote_uri": remote_uri,
        "local_dir": str(local_dir) if local_dir else None,
        "dry_run": bool(dry_run),
        "timestamp": _utc_now(),
        "file_count": len(file_hashes),
        "total_bytes": int(total_bytes),
        "files": [
            {"path": rel, "sha256": h, "size_bytes": _file_size(local_dir, rel) if local_dir else None}
            for rel, h in sorted(file_hashes.items())
        ],
        "batches": list(batches),
        "failed_files": list(failed_files),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def _file_size(root: Optional[Path], rel: str) -> Optional[int]:
    if root is None:
        return None
    p = Path(root) / rel
    return int(p.stat().st_size) if p.is_file() else None


def write_transfer_manifest(project_root: Path, manifest: Dict[str, Any]) -> Path:
    reports = Path(project_root).resolve() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    op = str(manifest.get("operation", "transfer"))
    path = reports / f"transfer_manifest_{op}_{ts}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "transfer_manifest_written",
        path=str(path),
        file_count=manifest.get("file_count"),
        total_bytes=manifest.get("total_bytes"),
    )
    return path


def _append_transfer_log(project_root: Path, record: Dict[str, Any]) -> None:
    log_path = Path(project_root).resolve() / "logs" / "rclone_transfers.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _quarantine_failed(
    quarantine_dir: Path,
    operation: str,
    failed_files: Sequence[str],
    errors: Sequence[str],
) -> Path:
    quarantine_dir = Path(quarantine_dir).resolve()
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "operation": operation,
        "failed_files": list(failed_files),
        "errors": list(errors),
        "timestamp": _utc_now(),
    }
    path = quarantine_dir / f"failed_{operation}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.error("b2_transfer_failures_quarantined", path=str(path), n=len(failed_files))
    return path


def _run_rclone(
    args: List[str],
    *,
    cfg: B2Config,
    config_override: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    binary = _resolve_rclone_binary()
    config_path = config_override or cfg.rclone_config
    cmd = [binary, "--config", str(config_path), *args]
    logger.info("rclone_exec", command=cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise RcloneIntegrationError(f"Failed to execute rclone: {exc}") from exc

    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        err_class = _classify_rclone_error(msg)
        raise RcloneIntegrationError(f"rclone exited {proc.returncode} [{err_class}]: {msg}")
    return proc


def _run_rclone_with_retry(
    args: List[str],
    *,
    cfg: B2Config,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, int(cfg.max_transfer_retries) + 1):
        try:
            return _run_rclone(args, cfg=cfg, check=check)
        except RcloneIntegrationError as exc:
            last_exc = exc
            err_text = str(exc)
            if attempt >= cfg.max_transfer_retries or not _is_transient_error(err_text):
                raise
            delay = _retry_delay_sec(attempt)
            logger.warning(
                "rclone_transient_retry",
                attempt=attempt,
                delay_sec=delay,
                error_class=_classify_rclone_error(err_text),
                error=err_text[:500],
            )
            time.sleep(delay)
    raise RcloneIntegrationError(str(last_exc))


def _list_remote_files(cfg: B2Config, remote_uri: str, remote: str) -> List[str]:
    args = ["lsf", "--fast-list", remote_uri]
    if remote == REMOTE_READONLY:
        _assert_ingest_remote(remote)
        # Protective image-only listing: never surface originals,
        # thumbnails, or stray video files, so they can never be ingested/copied.
        args.extend(ingest_exclude_flags(cfg))
    else:
        _assert_export_remote(remote)
    proc = _run_rclone_with_retry(args, cfg=cfg)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def count_remote_image_files(cfg: B2Config, remote_path: str) -> Dict[str, Any]:
    """
    Count image-like paths on the read-only B2 remote (``rclone lsf`` only — no download).

    ``remote_path`` is the path segment under the read-only bucket (e.g. ``datasets/raw``).
    """
    remote = (remote_path or "").strip().strip("/")
    uri = _remote_uri(REMOTE_READONLY, cfg.readonly_bucket, remote)
    try:
        lines = _list_remote_files(cfg, uri, REMOTE_READONLY)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "image_count": 0,
            "error": str(exc),
        }
    image_count = 0
    for line in lines:
        low = line.lower()
        if any(low.endswith(ext) for ext in IMAGE_EXTENSIONS):
            image_count += 1
    return {
        "ok": True,
        "image_count": image_count,
        "error": None,
        "files_listed": len(lines),
    }


def verify_bucket_access(
    cfg: Optional[B2Config] = None,
    *,
    write_test: bool = True,
) -> Dict[str, Any]:
    """
    Confirm B2 connectivity for the two-key model before a large run.

    - **Read-only ingest:** lists the source bucket/path (``rclone lsf`` only —
      no download, and the read-only key cannot write, so the source is untouched).
    - **RW output:** uploads a tiny probe file to the destination bucket and then
      deletes it, proving the write key works. The probe never touches the source.

    Returns ``{"readonly_ok": bool, "write_ok": bool, "error": Optional[str]}``.
    Never raises — the caller decides how to surface a failure.
    """
    result: Dict[str, Any] = {"readonly_ok": False, "write_ok": False, "error": None}
    try:
        cfg = cfg or load_b2_config()
        write_rclone_config(cfg)
        _resolve_rclone_binary()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"config: {exc}"
        return result

    # 1) Read-only ingest bucket — list only (no download, no write).
    try:
        ro_uri = _remote_uri(REMOTE_READONLY, cfg.readonly_bucket, cfg.ingest_remote_path)
        _assert_ingest_remote(REMOTE_READONLY)
        _run_rclone(["lsf", "--max-depth", "1", ro_uri], cfg=cfg)
        result["readonly_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"readonly ingest bucket: {exc}"
        return result

    if not write_test:
        return result

    # 2) RW output bucket — upload a tiny probe then delete it (write confirmed).
    import tempfile

    probe_name = f".privagen_write_test_{int(time.time())}.txt"
    dest_uri = _remote_uri(REMOTE_WRITE, cfg.write_bucket, cfg.export_remote_path)
    probe_uri = f"{dest_uri}/{probe_name}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="privagen_wtest_"))
    tmp_file = tmp_dir / probe_name
    try:
        _assert_export_remote(REMOTE_WRITE)
        tmp_file.write_text("privagen write test\n", encoding="utf-8")
        _run_rclone(["copyto", str(tmp_file), probe_uri], cfg=cfg)
        # Remove the probe so the destination bucket stays pristine.
        _run_rclone(["deletefile", probe_uri], cfg=cfg)
        result["write_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"rw output bucket: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def _batched_copy(
    *,
    cfg: B2Config,
    src: str,
    dst: str,
    ingest: bool,
    file_names: Sequence[str],
    batch_size: int,
    dry_run: bool,
    operation: str,
    ingest_extra_flags: Optional[Sequence[str]] = None,
    export_via_crypt: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Run resumable batched ``rclone copy``; return batch records, transferred names, errors."""
    if ingest:
        _assert_ingest_remote(REMOTE_READONLY)
    elif not export_via_crypt:
        _assert_export_remote(REMOTE_WRITE)

    batch_records: List[Dict[str, Any]] = []
    transferred: List[str] = []
    errors: List[str] = []
    names = list(file_names)
    size = max(1, int(batch_size))

    perf = rclone_performance_flags(cfg)

    for offset in range(0, max(len(names), 1), size):
        batch = names[offset : offset + size]
        args = [
            "copy",
            src,
            dst,
            *_copy_common_flags(cfg, extra=ingest_extra_flags if ingest else None),
        ]
        if dry_run:
            args.append("--dry-run")
        if batch:
            for name in batch:
                args.extend(["--include", name])

        batch_t0 = time.monotonic()
        try:
            proc = _run_rclone_with_retry(args, cfg=cfg, check=not dry_run)
            transferred.extend(batch)
            batch_records.append(
                {
                    "offset": int(offset),
                    "count": len(batch),
                    "elapsed_sec": float(time.monotonic() - batch_t0),
                    "dry_run": bool(dry_run),
                    "stdout_tail": (proc.stdout or "")[-2000:],
                    "throughput_files_per_sec": float(len(batch) / max(1e-6, time.monotonic() - batch_t0)),
                    "performance_flags": perf,
                }
            )
            logger.info(
                "b2_batch_complete",
                operation=operation,
                offset=int(offset),
                n=len(batch),
                dry_run=bool(dry_run),
            )
        except RcloneIntegrationError as exc:
            err_msg = str(exc)
            errors.append(err_msg)
            batch_records.append(
                {
                    "offset": int(offset),
                    "count": len(batch),
                    "failed": True,
                    "error": err_msg,
                    "files": list(batch),
                }
            )
            logger.error("b2_batch_failed", operation=operation, offset=int(offset), error=err_msg)

    return batch_records, transferred, errors


def ingest_from_backblaze(
    remote_path: str,
    local_input_dir: Path,
    batch_size: Optional[int] = None,
    *,
    cfg: Optional[B2Config] = None,
    dry_run: bool = False,
    confirm: bool = True,
    project_root: Optional[Path] = None,
    quarantine_dir: Optional[Path] = None,
    enforce_readonly_ingest: bool = True,
    verify_checksums: bool = True,
) -> Dict[str, Any]:
    """
    Copy from read-only B2 remote into ``local_input_dir`` (resumable batches).
    """
    b2 = cfg or load_b2_config()
    _assert_ingest_remote(REMOTE_READONLY)
    local_input_dir = Path(local_input_dir).resolve()
    local_input_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(batch_size or b2.transfer_batch_size))

    remote_uri = _remote_uri(REMOTE_READONLY, b2.readonly_bucket, remote_path)
    pre_hashes = compute_directory_hashes(local_input_dir)
    remote_files = _list_remote_files(b2, remote_uri, REMOTE_READONLY)
    if not remote_files:
        logger.warning("b2_ingest_no_remote_files", remote=remote_uri)

    pre_manifest = build_transfer_manifest(
        "ingest_precheck",
        remote_uri=remote_uri,
        local_dir=local_input_dir,
        file_hashes={},
        dry_run=True,
        batches=[],
        failed_files=[],
        extra={"remote_file_count": len(remote_files)},
    )

    confirm_real_transfer(
        "ingest",
        remote_uri=remote_uri,
        file_count=len(remote_files),
        total_bytes=0,
        dry_run=dry_run,
        require_confirm=confirm and b2.require_confirm,
    )

    batch_records, transferred, errors = _batched_copy(
        cfg=b2,
        src=remote_uri,
        dst=str(local_input_dir),
        ingest=True,
        file_names=remote_files,
        batch_size=batch_size,
        dry_run=dry_run,
        operation="ingest",
        ingest_extra_flags=["--immutable", "--read-only"] if enforce_readonly_ingest else None,
    )

    post_hashes = compute_directory_hashes(local_input_dir)
    new_files = sorted(set(post_hashes) - set(pre_hashes))
    failed_files = sorted(set(remote_files) - set(transferred))

    if quarantine_dir is None and project_root is not None:
        quarantine_dir = Path(project_root) / b2.quarantine_rel
    if failed_files and quarantine_dir is not None and not dry_run:
        _quarantine_failed(Path(quarantine_dir), "ingest", failed_files, errors)

    manifest = build_transfer_manifest(
        "ingest",
        remote_uri=remote_uri,
        local_dir=local_input_dir,
        file_hashes={name: post_hashes[name] for name in new_files},
        dry_run=dry_run,
        batches=batch_records,
        failed_files=failed_files,
        extra={"remote_file_count": len(remote_files), "precheck": pre_manifest},
    )

    record = {
        "operation": "ingest",
        "remote": remote_uri,
        "local_dir": str(local_input_dir),
        "dry_run": bool(dry_run),
        "batch_size": int(batch_size),
        "manifest": manifest,
        "transfer_hashes": manifest["files"],
    }

    if project_root is not None:
        manifest_path = write_transfer_manifest(project_root, manifest)
        record["manifest_path"] = str(manifest_path)
        _append_transfer_log(project_root, record)

    if errors and not dry_run:
        raise RcloneIntegrationError(f"B2 ingest failed for {len(errors)} batch(es): {errors[0]}")

    if not dry_run and new_files and verify_checksums:
        from .security import verify_ingest_hashes

        verify_map = {f["path"]: f["sha256"] for f in manifest["files"]}
        verify_report = verify_ingest_hashes(local_input_dir, verify_map)
        record["hash_verification"] = verify_report
        if not verify_report.get("ok"):
            raise RcloneIntegrationError("B2 ingest hash verification failed")

    return record


def verify_export_with_rclone_check(
    local_final_dir: Path,
    remote_uri: str,
    *,
    cfg: B2Config,
    file_names: Optional[Sequence[str]] = None,
    quarantine_dir: Optional[Path] = None,
    retry_on_mismatch: bool = True,
) -> Dict[str, Any]:
    """
    Post-export ``rclone check --checksum --fast-list`` (local → remote).

    On mismatches: log, optional one retry copy for affected files, quarantine persistent failures.
    """
    local_final_dir = Path(local_final_dir).resolve()
    if quarantine_dir is None:
        quarantine_dir = Path(local_final_dir).parent / cfg.quarantine_rel

    check_args = [
        "check",
        str(local_final_dir),
        remote_uri,
        *_copy_common_flags(cfg, progress=False),
        "--one-way",
    ]
    t0 = time.monotonic()
    proc = _run_rclone(check_args, cfg=cfg, check=False)
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    mismatched = _parse_check_mismatches(combined)
    ok = proc.returncode == 0 and not mismatched

    report: Dict[str, Any] = {
        "ok": ok,
        "remote_uri": remote_uri,
        "returncode": int(proc.returncode),
        "mismatched_files": mismatched,
        "elapsed_sec": float(time.monotonic() - t0),
        "retried": False,
        "quarantined": [],
    }

    if ok:
        logger.info("b2_export_check_ok", remote=remote_uri, elapsed_sec=report["elapsed_sec"])
        return report

    logger.error(
        "b2_export_check_failed",
        remote=remote_uri,
        mismatched=mismatched[:20],
        error_class=_classify_rclone_error(combined),
    )

    retry_files = list(mismatched)
    if retry_on_mismatch and retry_files:
        report["retried"] = True
        _, retried, retry_errors = _batched_copy(
            cfg=cfg,
            src=str(local_final_dir),
            dst=remote_uri,
            ingest=False,
            file_names=retry_files,
            batch_size=max(1, min(len(retry_files), cfg.transfer_batch_size)),
            dry_run=False,
            operation="export_verify_retry",
            export_via_crypt=cfg.crypt_enabled,
        )
        report["retry_transferred"] = list(retried)
        if retry_errors:
            report["retry_errors"] = retry_errors

        proc2 = _run_rclone(check_args, cfg=cfg, check=False)
        combined2 = f"{proc2.stdout or ''}\n{proc2.stderr or ''}"
        mismatched = _parse_check_mismatches(combined2)
        ok = proc2.returncode == 0 and not mismatched
        report["ok"] = ok
        report["mismatched_files"] = mismatched
        report["post_retry_returncode"] = int(proc2.returncode)

    if not report["ok"] and mismatched:
        _quarantine_failed(Path(quarantine_dir), "export_check", mismatched, [combined[:4000]])
        report["quarantined"] = list(mismatched)
        logger.error("b2_export_check_quarantined", n=len(mismatched))

    return report


def export_to_backblaze(
    local_final_dir: Path,
    remote_dest: str,
    *,
    cfg: Optional[B2Config] = None,
    dry_run: bool = False,
    confirm: bool = True,
    project_root: Optional[Path] = None,
    quarantine_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    verify_after_export: Optional[bool] = None,
) -> Dict[str, Any]:
    """Upload ``local_final_dir`` (images + JSON sidecars) to write-only B2 remote."""
    b2 = cfg or load_b2_config()
    if not b2.crypt_enabled:
        _assert_export_remote(REMOTE_WRITE)
    local_final_dir = Path(local_final_dir).resolve()
    if not local_final_dir.is_dir():
        raise RcloneIntegrationError(f"Export source directory does not exist: {local_final_dir}")

    batch_size = max(1, int(batch_size or b2.transfer_batch_size))
    if b2.crypt_enabled:
        sub = (remote_dest or "").strip().strip("/")
        remote_uri = f"{REMOTE_CRYPT}:{sub}" if sub else REMOTE_CRYPT
    else:
        _assert_export_remote(REMOTE_WRITE)
        remote_uri = _remote_uri(REMOTE_WRITE, b2.write_bucket, remote_dest)
    pre_hashes = compute_directory_hashes(local_final_dir, extensions=EXPORT_EXTENSIONS)
    file_names = sorted(pre_hashes.keys())
    total_bytes = sum(
        (local_final_dir / rel).stat().st_size for rel in file_names if (local_final_dir / rel).is_file()
    )

    confirm_real_transfer(
        "export",
        remote_uri=remote_uri,
        file_count=len(file_names),
        total_bytes=total_bytes,
        dry_run=dry_run,
        require_confirm=confirm and b2.require_confirm,
    )

    record = _export_to_backblaze_core(
        local_final_dir,
        remote_uri,
        b2=b2,
        file_names=file_names,
        pre_hashes=pre_hashes,
        batch_size=batch_size,
        dry_run=dry_run,
        project_root=project_root,
        quarantine_dir=quarantine_dir,
    )

    do_verify = b2.verify_after_export if verify_after_export is None else bool(verify_after_export)
    if do_verify and not dry_run and file_names:
        check_report = verify_export_with_rclone_check(
            local_final_dir,
            remote_uri,
            cfg=b2,
            file_names=file_names,
            quarantine_dir=quarantine_dir,
            retry_on_mismatch=True,
        )
        record["post_export_check"] = check_report
        if not check_report.get("ok"):
            raise RcloneIntegrationError(
                f"B2 post-export checksum verification failed: {len(check_report.get('mismatched_files') or [])} mismatches"
            )

    return record


def _export_to_backblaze_core(
    local_final_dir: Path,
    remote_uri: str,
    *,
    b2: B2Config,
    file_names: Sequence[str],
    pre_hashes: Mapping[str, str],
    batch_size: int,
    dry_run: bool,
    project_root: Optional[Path],
    quarantine_dir: Optional[Path],
) -> Dict[str, Any]:
    batch_records, transferred, errors = _batched_copy(
        cfg=b2,
        src=str(local_final_dir),
        dst=remote_uri,
        ingest=False,
        file_names=list(file_names),
        batch_size=batch_size,
        dry_run=dry_run,
        operation="export",
        export_via_crypt=b2.crypt_enabled,
    )

    failed_files = sorted(set(file_names) - set(transferred))
    if quarantine_dir is None and project_root is not None:
        quarantine_dir = Path(project_root) / b2.quarantine_rel
    if failed_files and quarantine_dir is not None and not dry_run:
        _quarantine_failed(Path(quarantine_dir), "export", failed_files, errors)

    manifest = build_transfer_manifest(
        "export",
        remote_uri=remote_uri,
        local_dir=local_final_dir,
        file_hashes=dict(pre_hashes),
        dry_run=dry_run,
        batches=batch_records,
        failed_files=failed_files,
        extra={
            "crypt_enabled": b2.crypt_enabled,
            "performance": {
                "transfers": b2.transfers,
                "checkers": b2.checkers,
                "upload_concurrency": b2.upload_concurrency,
                "chunk_size": b2.chunk_size,
                "bwlimit": b2.bwlimit,
            },
        },
    )

    record: Dict[str, Any] = {
        "operation": "export",
        "local_dir": str(local_final_dir),
        "remote": remote_uri,
        "dry_run": bool(dry_run),
        "manifest": manifest,
        "transfer_hashes": dict(pre_hashes),
        "crypt_enabled": b2.crypt_enabled,
    }
    if b2.crypt_enabled:
        record["crypt_decryption_doc"] = crypt_decryption_doc()

    if project_root is not None:
        manifest_path = write_transfer_manifest(project_root, manifest)
        record["manifest_path"] = str(manifest_path)
        _append_transfer_log(project_root, record)

    if errors and not dry_run:
        raise RcloneIntegrationError(f"B2 export failed for {len(errors)} batch(es): {errors[0]}")

    return record


def verify_remote_against_manifest(
    remote_path: str,
    manifest: Mapping[str, Any],
    *,
    cfg: Optional[B2Config] = None,
    use_write_remote: bool = True,
) -> Dict[str, Any]:
    """
    Verify remote object names exist for every path in a transfer manifest.

    Uses ``rclone lsf --fast-list``; does not re-download for hash check (use
    ``verify_export_integrity`` for local↔remote hash after mock or sync).
    """
    b2 = cfg or load_b2_config()
    remote = REMOTE_WRITE if use_write_remote else REMOTE_READONLY
    bucket = b2.write_bucket if use_write_remote else b2.readonly_bucket
    _assert_export_remote(remote) if use_write_remote else _assert_ingest_remote(remote)

    remote_uri = _remote_uri(remote, bucket, remote_path)
    listed = set(_list_remote_files(b2, remote_uri, remote))
    expected = {str(f.get("path")) for f in (manifest.get("files") or [])}
    missing = sorted(expected - listed)
    extra = sorted(listed - expected)

    ok = not missing
    report = {
        "ok": ok,
        "remote_uri": remote_uri,
        "expected_count": len(expected),
        "listed_count": len(listed),
        "missing": missing,
        "extra_on_remote": extra[:50],
    }
    if not ok:
        logger.error("remote_manifest_verify_failed", missing=missing[:20])
    else:
        logger.info("remote_manifest_verify_ok", count=len(expected))
    return report


def verify_export_integrity(
    local_final_dir: Path,
    remote_staging_dir: Path,
    *,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare SHA256 of local export tree vs a local mirror of the remote (for tests or post-sync)."""
    local_hashes = compute_directory_hashes(local_final_dir, extensions=extensions or EXPORT_EXTENSIONS)
    remote_hashes = compute_directory_hashes(remote_staging_dir, extensions=extensions or EXPORT_EXTENSIONS)
    missing = sorted(set(local_hashes) - set(remote_hashes))
    mismatched = [
        k for k in local_hashes if k in remote_hashes and local_hashes[k] != remote_hashes[k]
    ]
    return {
        "ok": not missing and not mismatched,
        "local_count": len(local_hashes),
        "remote_count": len(remote_hashes),
        "missing_on_remote": missing,
        "hash_mismatched": mismatched,
    }


def _cmd_join(parts: Sequence[str]) -> str:
    return " ".join(str(p) for p in parts if p)


_PLACEHOLDER_SOURCE_BUCKET = "<source-bucket>"
_PLACEHOLDER_DEST_BUCKET = "<dest-bucket>"
_PLACEHOLDER_INGEST_PATH = "<ingest-path>"
_PLACEHOLDER_EXPORT_PATH = "<export-path>"


def _ui_bucket_or_placeholder(value: str, placeholder: str) -> str:
    v = (value or "").strip()
    return v if v else placeholder


def _ui_path_or_placeholder(value: str, placeholder: str) -> str:
    v = (value or "").strip().strip("/")
    return v if v else placeholder


def build_rclone_command_reference_for_ui(
    *,
    readonly_bucket: str = "",
    write_bucket: str = "",
    ingest_remote_path: str = "",
    export_remote_path: str = "",
    local_input_dir: str = "input_raw",
    local_final_dir: str = "final_clean",
    batch_size: int = 32,
    crypt_enabled: bool = False,
    rclone_config: Optional[str] = None,
    yaml_backblaze: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
    use_placeholders: bool = True,
) -> Dict[str, Any]:
    """
    Build rclone command preview for the Web UI.

    Works without valid B2 credentials — uses placeholders for unset buckets/paths
    so users can review read-only ingest vs write export before configuring .env.
    """
    yc = dict(yaml_backblaze or {})
    ro = _ui_bucket_or_placeholder(
        readonly_bucket,
        _PLACEHOLDER_SOURCE_BUCKET if use_placeholders else str(yc.get("source_bucket") or "source-bucket"),
    )
    wr = _ui_bucket_or_placeholder(
        write_bucket,
        _PLACEHOLDER_DEST_BUCKET if use_placeholders else str(yc.get("dest_bucket") or "dest-bucket"),
    )
    ingest_sub = _ui_path_or_placeholder(
        ingest_remote_path,
        _PLACEHOLDER_INGEST_PATH if use_placeholders else str(yc.get("ingest_remote_path") or "datasets/raw"),
    )
    export_sub = _ui_path_or_placeholder(
        export_remote_path,
        _PLACEHOLDER_EXPORT_PATH if use_placeholders else str(yc.get("export_remote_path") or "datasets/anonymized"),
    )
    default_config = Path.home() / ".config" / "rclone" / "rclone.conf"
    cfg_path = Path((rclone_config or yc.get("rclone_config") or default_config)).expanduser()

    preview_cfg = B2Config(
        key_id="<B2_KEY_ID>",
        readonly_key="<B2_READONLY_KEY>",
        write_key="<B2_WRITE_KEY>",
        readonly_bucket=ro,
        write_bucket=wr,
        rclone_config=cfg_path,
        ingest_remote_path=ingest_sub,
        export_remote_path=export_sub,
        transfer_batch_size=max(1, int(batch_size or yc.get("transfer_batch_size") or 32)),
        max_transfer_retries=int(yc.get("max_transfer_retries", 3) or 3),
        require_confirm=bool(yc.get("require_confirm_real_transfer", True)),
        crypt_enabled=bool(crypt_enabled),
        transfers=int(yc.get("transfers", 24) or 24),
        checkers=int(yc.get("checkers", 64) or 64),
        upload_concurrency=int(yc.get("upload_concurrency", 12) or 12),
        chunk_size=str(yc.get("chunk_size", "48M") or "48M"),
        bwlimit=str(yc.get("bwlimit", "0") or "0"),
        verify_after_export=bool(yc.get("verify_after_export", True)),
    )
    ref = build_rclone_command_reference(
        preview_cfg,
        ingest_remote_path=ingest_sub,
        export_remote_path=export_sub,
        local_input_dir=local_input_dir,
        local_final_dir=local_final_dir,
        batch_size=preview_cfg.transfer_batch_size,
        dry_run=dry_run,
    )
    ref["preview_mode"] = True
    ref["placeholders_active"] = bool(
        use_placeholders
        and (
            ro.startswith("<")
            or wr.startswith("<")
            or ingest_sub.startswith("<")
            or export_sub.startswith("<")
        )
    )
    ref["security_summary"] = {
        "ingest": (
            "Original data is read from B2 with the read-only key only. "
            "rclone copy pulls files into input_raw/ — never uploads or deletes on the source bucket."
        ),
        "export": (
            "Anonymized outputs upload with the write key to a separate bucket. "
            "The read-only key is never used for export."
        ),
    }
    return ref


def build_rclone_command_reference(
    cfg: B2Config,
    *,
    ingest_remote_path: Optional[str] = None,
    export_remote_path: Optional[str] = None,
    local_input_dir: str = "input_raw",
    local_final_dir: str = "final_clean",
    batch_size: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Human-readable rclone command templates for review (no secrets in output).

    Matches what ``ingest_from_backblaze`` / ``export_to_backblaze`` execute.
    """
    try:
        binary = _resolve_rclone_binary()
    except RcloneIntegrationError:
        binary = (os.environ.get("RCLONE_BINARY") or "rclone").strip() or "rclone"
    config = str(Path(cfg.rclone_config).expanduser())
    ingest_sub = (ingest_remote_path if ingest_remote_path is not None else cfg.ingest_remote_path).strip().strip("/")
    export_sub = (export_remote_path if export_remote_path is not None else cfg.export_remote_path).strip().strip("/")
    bs = max(1, int(batch_size or cfg.transfer_batch_size))

    ingest_uri = _remote_uri(REMOTE_READONLY, cfg.readonly_bucket, ingest_sub)
    export_uri_plain = _remote_uri(REMOTE_WRITE, cfg.write_bucket, export_sub)
    if cfg.crypt_enabled:
        export_sub_crypt = export_sub
        export_uri = f"{REMOTE_CRYPT}:{export_sub_crypt}" if export_sub_crypt else REMOTE_CRYPT
    else:
        export_uri = export_uri_plain

    perf = _copy_common_flags(cfg, progress=True)
    perf_no_progress = _copy_common_flags(cfg, progress=False)
    excludes = ingest_exclude_flags(cfg)
    ingest_flags = list(perf) + list(excludes) + ["--immutable", "--read-only"]
    dry = ["--dry-run"] if dry_run else []

    base = [binary, "--config", config]

    ingest_copy = base + ["copy", ingest_uri, f"{local_input_dir}/", *ingest_flags, *dry]
    export_copy = base + ["copy", f"{local_final_dir}/", export_uri, *perf, *dry]
    export_check = base + ["check", f"{local_final_dir}/", export_uri, *perf_no_progress, "--one-way"]

    return {
        "schema": "dataset_anonymizer.rclone_commands.v1",
        "rclone_binary": binary,
        "rclone_config": config,
        "dry_run": bool(dry_run),
        "batch_size": bs,
        "remotes": {
            "ingest": {
                "name": REMOTE_READONLY,
                "api_key": "B2_READONLY_KEY (read-only application key)",
                "bucket": cfg.readonly_bucket,
                "purpose": "List and copy source images into the pipeline only — no uploads or deletes",
            },
            "export": {
                "name": REMOTE_WRITE if not cfg.crypt_enabled else f"{REMOTE_WRITE} → {REMOTE_CRYPT}",
                "api_key": "B2_WRITE_KEY (read/write application key)",
                "bucket": cfg.write_bucket,
                "purpose": "Upload anonymized final_clean outputs to a separate bucket",
            },
        },
        "paths": {
            "ingest_remote_path": ingest_sub or "(bucket root)",
            "ingest_uri": ingest_uri,
            "export_remote_path": export_sub or "(bucket root)",
            "export_uri": export_uri,
            "export_uri_without_crypt": export_uri_plain,
            "local_input_dir": local_input_dir,
            "local_final_dir": local_final_dir,
        },
        "commands": {
            "write_config": (
                "# Generated by PrivaGen / dataset_anonymizer (never commit keys):\n"
                f"# [{REMOTE_READONLY}] account=<B2_KEY_ID> key=<B2_READONLY_KEY>\n"
                f"# [{REMOTE_WRITE}] account=<B2_KEY_ID> key=<B2_WRITE_KEY>"
                + (f"\n# [{REMOTE_CRYPT}] wraps {REMOTE_WRITE}:{cfg.write_bucket}" if cfg.crypt_enabled else "")
            ),
            "ingest_list": _cmd_join(base + ["lsf", "--fast-list", ingest_uri, *excludes]),
            "ingest_copy_per_batch": _cmd_join(ingest_copy + ["--include", "<filename>"]),
            "ingest_copy_all": _cmd_join(ingest_copy),
            "ingest_note": (
                f"Ingest runs in batches of {bs} files via repeated `rclone copy` with --include per file. "
                "Only b2-readonly remote; --immutable --read-only prevent destination→source writes."
            ),
            "export_copy_per_batch": _cmd_join(export_copy + ["--include", "<filename>"]),
            "export_copy_all": _cmd_join(export_copy),
            "export_note": (
                f"Export runs in batches of {bs} via `rclone copy` from final_clean to {export_uri}. "
                "Only b2-write (or b2-crypt) remote — read-only key is never used for export."
            ),
            "export_verify_check": _cmd_join(export_check),
            "export_verify_note": "Post-export integrity: one-way checksum check local → remote (optional).",
            "buyer_decrypt_crypt": crypt_decryption_doc() if cfg.crypt_enabled else None,
        },
        "performance_flags": perf,
        "ingest_excludes": list(cfg.ingest_excludes),
        "rclone_defaults": default_rclone_commands(),
        "security": {
            "ingest_remote_enforced": REMOTE_READONLY,
            "export_remote_enforced": REMOTE_WRITE,
            "separate_buckets": cfg.readonly_bucket != cfg.write_bucket,
            "readonly_bucket": cfg.readonly_bucket,
            "write_bucket": cfg.write_bucket,
        },
    }


def try_load_b2_config_for_ui(
    project_root: Path,
    *,
    yaml_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load B2 bucket/path overview for UI without raising if env is incomplete."""
    root = Path(project_root).resolve()
    yc = dict(yaml_cfg or {})
    if not yc:
        try:
            from .utils import load_config

            yc = load_config(root / "config.yaml").get("backblaze") or {}
        except Exception:  # noqa: BLE001
            yc = {}
    sec = {}
    try:
        from .utils import load_config

        sec = load_config(root / "config.yaml").get("security") or {}
    except Exception:  # noqa: BLE001
        pass
    try:
        cfg = load_b2_config(yaml_cfg=yc, security_cfg=sec)
        return {
            "configured": True,
            "readonly_bucket": cfg.readonly_bucket,
            "write_bucket": cfg.write_bucket,
            "ingest_remote_path": cfg.ingest_remote_path,
            "export_remote_path": cfg.export_remote_path,
            "transfer_batch_size": cfg.transfer_batch_size,
            "crypt_enabled": cfg.crypt_enabled,
            "rclone_config": str(cfg.rclone_config),
        }
    except RcloneIntegrationError as exc:
        return {
            "configured": False,
            "error": str(exc),
            "readonly_bucket": (os.environ.get("B2_READONLY_BUCKET") or yc.get("source_bucket") or "").strip(),
            "write_bucket": (os.environ.get("B2_WRITE_BUCKET") or yc.get("dest_bucket") or "").strip(),
            "ingest_remote_path": str(yc.get("ingest_remote_path") or "datasets/raw"),
            "export_remote_path": str(yc.get("export_remote_path") or "datasets/anonymized"),
        }


def secure_cleanup_temp(temp_processed_dir: Path, *, secure: bool = True) -> None:
    target = Path(temp_processed_dir).resolve()
    if not target.is_dir():
        logger.info("temp_cleanup_skipped", path=str(target), reason="not_a_directory")
        return
    if secure:
        from .security import secure_delete_tree

        secure_delete_tree(target)
    else:
        shutil.rmtree(target, ignore_errors=True)
    logger.info("temp_cleanup_complete", path=str(target), secure=bool(secure))
