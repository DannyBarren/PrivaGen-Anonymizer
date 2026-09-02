"""
Comprehensive security hardening: DLP, read-only ingest, at-rest protection, audit trail.

Integrates with ``rclone_integration`` and ``main_pipeline`` via ``SecurityContext`` hooks.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .security import redact_secrets_obj, verify_ingest_hashes
from .utils import get_logger, sha256_file, utc_now_iso

logger = get_logger(__name__)

PROCESSED_MANIFEST_SCHEMA = "dataset_anonymizer.processed_manifest.v1"
SECURITY_REPORT_NAME = "security_report.md"
REMOTE_CRYPT = "b2-crypt"

SECURITY_LEVELS = frozenset({"standard", "full"})


@dataclass
class SecurityHardeningConfig:
    """Resolved security policy from ``config.yaml`` + CLI ``--security-level``."""

    level: str = "standard"
    verify_ingest_checksums: bool = True
    enforce_readonly_ingest: bool = True
    crypt_enabled: bool = False
    crypt_password_env: str = "RCLONE_CRYPT_PASSWORD"
    crypt_salt_env: str = "RCLONE_CRYPT_SALT"
    umask: int = 0o077
    backup_before_cleanup: bool = True
    backups_rel: str = "backups"
    secure_wipe: bool = True
    secure_wipe_passes: int = 3
    shred_on_linux: bool = True
    processed_manifest_rel: str = "reports/processed_manifest.json"
    redact_logs: bool = True
    input_file_mode: int = 0o444
    dir_mode: int = 0o700

    @property
    def is_full(self) -> bool:
        return self.level == "full"


def load_security_hardening(
    cfg: Mapping[str, Any],
    *,
    security_level: Optional[str] = None,
) -> SecurityHardeningConfig:
    """Merge ``security:`` YAML block with optional CLI override."""
    sec = dict(cfg.get("security") or {})
    level = (security_level or sec.get("level") or "standard").strip().lower()
    if level not in SECURITY_LEVELS:
        level = "standard"

    full_defaults = level == "full"
    return SecurityHardeningConfig(
        level=level,
        verify_ingest_checksums=bool(sec.get("verify_ingest_checksums", True)),
        enforce_readonly_ingest=bool(sec.get("enforce_readonly_ingest", True)),
        crypt_enabled=bool(sec.get("crypt_enabled", False)),
        crypt_password_env=str(sec.get("crypt_password_env", "RCLONE_CRYPT_PASSWORD")),
        crypt_salt_env=str(sec.get("crypt_salt_env", "RCLONE_CRYPT_SALT")),
        umask=int(sec.get("umask", 0o077)),
        backup_before_cleanup=bool(sec.get("backup_before_cleanup", full_defaults or True)),
        backups_rel=str(sec.get("backups_dir", "backups")),
        secure_wipe=bool(sec.get("secure_wipe", full_defaults)),
        secure_wipe_passes=int(sec.get("secure_wipe_passes", 3)),
        shred_on_linux=bool(sec.get("shred_on_linux", True)),
        processed_manifest_rel=str(sec.get("processed_manifest", "reports/processed_manifest.json")),
        redact_logs=bool(sec.get("redact_logs", True)),
        input_file_mode=int(sec.get("input_file_mode", 0o444)),
        dir_mode=int(sec.get("dir_mode", 0o700)),
    )


@dataclass
class SecurityContext:
    """Stateful audit trail + hook implementations for a pipeline run."""

    project_root: Path
    config: SecurityHardeningConfig
    events: List[Dict[str, Any]] = field(default_factory=list)
    _umask_applied: bool = False
    last_backup_dir: Optional[Path] = None

    def log_event(self, name: str, **payload: Any) -> None:
        entry: Dict[str, Any] = {"event": name, "timestamp": utc_now_iso()}
        safe = redact_secrets_obj(payload) if self.config.redact_logs else payload
        entry.update(safe if isinstance(safe, dict) else payload)
        self.events.append(entry)
        logger.info(name, **(safe if isinstance(safe, dict) else payload))

    def processed_manifest_path(self) -> Path:
        p = Path(self.config.processed_manifest_rel)
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    def ingest_rclone_extra_flags(self) -> List[str]:
        if not self.config.enforce_readonly_ingest:
            return []
        return ["--immutable", "--read-only"]

    def apply_startup_hardening(self, paths: Mapping[str, Path]) -> None:
        if not self._umask_applied:
            os.umask(self.config.umask)
            self._umask_applied = True
            self.log_event("security_umask_applied", umask=oct(self.config.umask))

        for key in ("final_clean", "reports", "temp_processed", "logs"):
            p = paths.get(key)
            if p is not None:
                harden_directory_permissions(Path(p), mode=self.config.dir_mode)
        self.log_event("security_directory_permissions_applied", dirs=[k for k in paths if k in (
            "final_clean", "reports", "temp_processed", "logs",
        )])

    def pre_ingest(self, *, remote_path: str, local_input: Path, dry_run: bool) -> None:
        self.log_event(
            "security_pre_ingest",
            remote_path=remote_path,
            local_input=str(local_input),
            dry_run=bool(dry_run),
            verify_checksums=self.config.verify_ingest_checksums,
        )

    def post_ingest(
        self,
        *,
        local_input: Path,
        ingest_record: Mapping[str, Any],
        dry_run: bool,
    ) -> Dict[str, Any]:
        manifest = ingest_record.get("manifest") or {}
        files = manifest.get("files") or []
        verify_report: Dict[str, Any] = {"ok": True, "skipped": True}

        if not dry_run and self.config.verify_ingest_checksums and files:
            existing = ingest_record.get("hash_verification")
            if isinstance(existing, dict) and existing.get("ok"):
                verify_report = existing
            else:
                verify_map = {str(f["path"]): str(f["sha256"]) for f in files if f.get("path")}
                verify_report = verify_ingest_hashes(local_input, verify_map)
                if not verify_report.get("ok"):
                    self.log_event("security_ingest_checksum_failed", **verify_report)
                    raise RuntimeError("Ingest SHA256 verification failed (security.verify_ingest_checksums)")

        if not dry_run and self.config.enforce_readonly_ingest:
            n = enforce_input_raw_readonly(local_input, file_mode=self.config.input_file_mode)
            self.log_event("security_input_raw_readonly", files=n)

        ts = utc_now_iso()
        entries: List[Tuple[str, str]] = []
        for f in files:
            rel = str(f.get("path", ""))
            h = str(f.get("sha256", ""))
            if rel and h:
                entries.append((rel, h))
        updated = update_processed_manifest(self.processed_manifest_path(), entries, ingest_timestamp=ts)
        self.log_event("security_post_ingest", verified=verify_report.get("verified_count", 0), manifest_entries=updated)
        return verify_report

    def post_qa_batch(self, *, batch_dir: Path, routed_count: int) -> None:
        path = self.processed_manifest_path()
        data = _load_manifest_file(path)
        entries = data.setdefault("entries", {})
        for jpg in sorted(batch_dir.glob("*.jpg")):
            if not jpg.is_file():
                continue
            try:
                out_hash = sha256_file(jpg)
            except OSError:
                continue
            stem = jpg.stem
            existing = entries.get(stem) or entries.get(jpg.name) or {}
            existing["output_hash"] = out_hash
            existing["processed_at"] = utc_now_iso()
            entries[stem] = existing
        data["updated_at"] = utc_now_iso()
        _write_manifest_file(path, data)
        self.log_event("security_post_qa", batch=str(batch_dir), routed=int(routed_count))

    def pre_export(self, *, local_final: Path, remote_dest: str, dry_run: bool) -> None:
        if self.config.backup_before_cleanup and not dry_run:
            self.last_backup_dir = backup_critical_artifacts(
                self.project_root,
                backup_root=self.project_root / self.config.backups_rel,
            )
            self.log_event("security_pre_export_backup", backup_dir=str(self.last_backup_dir))
        self.log_event(
            "security_pre_export",
            local_final=str(local_final),
            remote_dest=remote_dest,
            dry_run=bool(dry_run),
            crypt_enabled=self.config.crypt_enabled,
        )

    def pre_cleanup(self, *, temp_dir: Path, reports_dir: Path) -> Optional[Path]:
        backup_dir: Optional[Path] = None
        if self.config.backup_before_cleanup:
            backup_dir = backup_critical_artifacts(
                self.project_root,
                backup_root=self.project_root / self.config.backups_rel,
            )
            self.last_backup_dir = backup_dir
            self.log_event("security_pre_cleanup_backup", backup_dir=str(backup_dir))
        return backup_dir

    def cleanup_temp(self, temp_dir: Path, *, pipeline_cfg: Optional[Mapping[str, Any]] = None) -> None:
        if self.config.secure_wipe:
            secure_wipe_dir(
                temp_dir,
                passes=self.config.secure_wipe_passes,
                shred_on_linux=self.config.shred_on_linux,
            )
            self.log_event("security_temp_secure_wiped", path=str(temp_dir))
            if pipeline_cfg is not None:
                try:
                    from .gpu_runtime import empty_cuda_cache_after_secure_wipe

                    empty_cuda_cache_after_secure_wipe(pipeline_cfg)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("cuda_cache_after_secure_wipe_failed", error=str(exc))
        elif temp_dir.is_dir():
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.log_event("security_temp_removed", path=str(temp_dir))

    def write_report(self, *, summary: Optional[Mapping[str, Any]] = None) -> Path:
        return write_security_report(
            self.project_root,
            self.config,
            self.events,
            summary=summary,
        )


def _load_manifest_file(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {
            "schema": PROCESSED_MANIFEST_SCHEMA,
            "created_at": utc_now_iso(),
            "entries": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"schema": PROCESSED_MANIFEST_SCHEMA, "created_at": utc_now_iso(), "entries": {}}


def _write_manifest_file(path: Path, data: MutableMapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_secrets_obj(dict(data))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def update_processed_manifest(
    manifest_path: Path,
    file_entries: Sequence[Tuple[str, str]],
    *,
    ingest_timestamp: str,
) -> int:
    """Merge ingest records (relative path, sha256) into ``processed_manifest.json``."""
    data = _load_manifest_file(manifest_path)
    entries = data.setdefault("entries", {})
    for rel, digest in file_entries:
        key = Path(rel).stem
        prev = entries.get(key) or {}
        prev["original_hash"] = digest
        prev["ingest_timestamp"] = ingest_timestamp
        prev["source_path"] = rel.replace("\\", "/")
        entries[key] = prev
    data["updated_at"] = utc_now_iso()
    _write_manifest_file(manifest_path, data)
    return len(file_entries)


def load_processed_manifest(manifest_path: Path) -> Dict[str, Any]:
    return _load_manifest_file(manifest_path)


def stems_fully_processed(manifest_path: Path, *, final_clean: Path) -> set[str]:
    """Stems with matching original_hash still present in final_clean (resume skip set)."""
    data = _load_manifest_file(manifest_path)
    entries = data.get("entries") or {}
    skip: set[str] = set()
    for stem, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        orig = entry.get("original_hash")
        if not orig:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = final_clean / f"{stem}{ext}"
            if candidate.is_file():
                skip.add(str(stem))
                break
            candidate = final_clean / stem
            if candidate.is_file() and sha256_file(candidate) == orig:
                skip.add(str(stem))
                break
    return skip


@contextmanager
def security_umask(cfg: SecurityHardeningConfig) -> Iterator[None]:
    prev = os.umask(cfg.umask)
    try:
        yield
    finally:
        os.umask(prev)


def harden_directory_permissions(directory: Path, *, mode: int = 0o700) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, mode)
    except OSError as exc:
        logger.warning("harden_dir_chmod_skipped", path=str(directory), error=str(exc))


def enforce_input_raw_readonly(input_raw: Path, *, file_mode: int = 0o444) -> int:
    """Set read-only permissions on all files under ``input_raw/``."""
    input_raw = Path(input_raw).resolve()
    count = 0
    if not input_raw.is_dir():
        return 0
    for path in input_raw.rglob("*"):
        if path.is_file():
            try:
                os.chmod(path, file_mode)
                count += 1
            except OSError as exc:
                logger.warning("input_raw_chmod_skipped", path=str(path), error=str(exc))
    try:
        os.chmod(input_raw, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    except OSError:
        pass
    return count


def backup_critical_artifacts(
    project_root: Path,
    *,
    backup_root: Path,
    timestamp: Optional[str] = None,
) -> Path:
    """Copy manifests and audit JSON into ``backups/{timestamp}/`` before cleanup."""
    project_root = Path(project_root).resolve()
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(backup_root) / ts
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    reports = project_root / "reports"
    if reports.is_dir():
        for src in sorted(reports.glob("*.json")):
            if src.name.startswith("transfer_manifest") or src.name in (
                "anonymization_audit.json",
                "processed_manifest.json",
            ):
                shutil.copy2(src, dest / src.name)
                copied += 1
        audit = reports / "anonymization_audit.json"
        if audit.is_file() and not (dest / audit.name).exists():
            shutil.copy2(audit, dest / audit.name)
            copied += 1
        pm = reports / "processed_manifest.json"
        if pm.is_file() and not (dest / pm.name).exists():
            shutil.copy2(pm, dest / pm.name)
            copied += 1

    log_jsonl = project_root / "logs" / "rclone_transfers.jsonl"
    if log_jsonl.is_file():
        shutil.copy2(log_jsonl, dest / log_jsonl.name)
        copied += 1

    meta = {
        "timestamp": ts,
        "files_copied": copied,
        "project_root": str(project_root),
    }
    (dest / "backup_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("security_backup_complete", dest=str(dest), files=copied)
    return dest


def _overwrite_file_windows(path: Path, *, passes: int) -> None:
    size = path.stat().st_size
    if size == 0:
        return
    with path.open("r+b") as fh:
        for _ in range(max(1, passes)):
            fh.seek(0)
            fh.write(os.urandom(size))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def _shred_file_linux(path: Path) -> bool:
    shred = shutil.which("shred")
    if not shred:
        return False
    try:
        subprocess.run([shred, "-u", "-z", "-n", "1", str(path)], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def secure_wipe_dir(
    root: Path,
    *,
    passes: int = 3,
    shred_on_linux: bool = True,
) -> None:
    """
    Secure-delete contents of ``root`` then remove the directory tree.

    Linux: optional ``shred`` when available; otherwise random overwrite.
    Windows: cipher-style random overwrite via ``os.urandom``.
    """
    root = Path(root).resolve()
    if not root.exists():
        return

    system = platform.system().lower()
    if root.is_file():
        _secure_wipe_file(root, passes=passes, shred_on_linux=shred_on_linux, system=system)
        return

    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            fp = Path(dirpath) / name
            _secure_wipe_file(fp, passes=passes, shred_on_linux=shred_on_linux, system=system)
        try:
            Path(dirpath).rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        shutil.rmtree(root, ignore_errors=True)


def _secure_wipe_file(path: Path, *, passes: int, shred_on_linux: bool, system: str) -> None:
    path = Path(path)
    if not path.is_file():
        return
    try:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
    except OSError:
        pass
    wiped = False
    if system == "linux" and shred_on_linux:
        wiped = _shred_file_linux(path)
    if not wiped:
        try:
            if system == "windows":
                _overwrite_file_windows(path, passes=passes)
            else:
                size = path.stat().st_size
                with path.open("r+b") as fh:
                    for _ in range(max(1, passes)):
                        fh.seek(0)
                        fh.write(os.urandom(size))
                        fh.flush()
        except OSError as exc:
            logger.warning("secure_wipe_overwrite_skipped", path=str(path), error=str(exc))
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("secure_wipe_unlink_failed", path=str(path), error=str(exc))


def render_crypt_remote_section(
    *,
    write_remote: str,
    write_bucket: str,
    export_path: str,
    password: str,
    password2: str = "",
    salt: str = "",
) -> str:
    """Append rclone crypt remote wrapping the write-only B2 remote."""
    sub = (export_path or "").strip().strip("/")
    remote_target = f"{write_remote}:{write_bucket}"
    if sub:
        remote_target = f"{remote_target}/{sub}"
    lines = [
        f"[{REMOTE_CRYPT}]",
        "type = crypt",
        f"remote = {remote_target}",
        f"password = {password}",
    ]
    if password2.strip():
        lines.append(f"password2 = {password2}")
    if salt.strip():
        lines.append(f"salt = {salt}")
    lines.append("")
    return "\n".join(lines)


def resolve_export_remote(cfg_crypt_enabled: bool) -> str:
    return REMOTE_CRYPT if cfg_crypt_enabled else "b2-write"


def write_security_report(
    project_root: Path,
    config: SecurityHardeningConfig,
    events: Sequence[Mapping[str, Any]],
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Emit ``reports/security_report.md`` after a pipeline run."""
    project_root = Path(project_root).resolve()
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / SECURITY_REPORT_NAME
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Security Report",
        "",
        f"Generated: {ts}",
        "",
        "## Configuration",
        "",
        f"- Security level: `{config.level}`",
        f"- Verify ingest checksums: `{config.verify_ingest_checksums}`",
        f"- Read-only ingest enforcement: `{config.enforce_readonly_ingest}`",
        f"- Crypt export enabled: `{config.crypt_enabled}`",
        f"- Backup before cleanup: `{config.backup_before_cleanup}`",
        f"- Secure wipe temp: `{config.secure_wipe}`",
        f"- Umask: `{oct(config.umask)}`",
        "",
        "## Event Summary",
        "",
    ]
    for ev in events:
        name = ev.get("event", "unknown")
        ets = ev.get("timestamp", "")
        detail = {k: v for k, v in ev.items() if k not in ("event", "timestamp")}
        lines.append(f"- **{name}** ({ets})")
        if detail:
            lines.append(f"  - `{json.dumps(redact_secrets_obj(detail), ensure_ascii=False)}`")
    lines.extend(["", "## Pipeline Summary", ""])
    if summary:
        for k, v in summary.items():
            lines.append(f"- **{k}**: `{redact_secrets_obj(v) if isinstance(v, (dict, list)) else v}`")
    else:
        lines.append("- _(no pipeline summary provided)_")

    rclone = (summary or {}).get("rclone") if isinstance(summary, dict) else None
    lines.extend(["", "## Rclone Integration", ""])
    if isinstance(rclone, dict):
        perf = rclone.get("performance") or {}
        lines.extend(
            [
                f"- Performance flags: transfers=`{perf.get('transfers')}` checkers=`{perf.get('checkers')}` "
                f"upload_concurrency=`{perf.get('upload_concurrency')}` chunk_size=`{perf.get('chunk_size')}` "
                f"bwlimit=`{perf.get('bwlimit')}`",
                f"- Post-export checksum verify: `{rclone.get('verify_after_export')}`",
                f"- Crypt export: `{rclone.get('crypt_enabled')}`",
                f"- Error handling: exponential backoff + jitter, classified retries "
                f"(`rate_limit`, `503`, `connection`, `timeout`)",
            ]
        )
        if rclone.get("crypt_decryption_doc"):
            lines.append(f"- Decryption: {rclone.get('crypt_decryption_doc')}")
        check = rclone.get("post_export_check")
        if isinstance(check, dict):
            lines.append(
                f"- Last export check: ok=`{check.get('ok')}` mismatches=`{len(check.get('mismatched_files') or [])}` "
                f"retried=`{check.get('retried')}`"
            )
    else:
        lines.append("- _(no rclone summary recorded)_")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    logger.info("security_report_written", path=str(path))
    return path
