"""
Security helpers: read-only originals, secret redaction, secure temp deletion.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from .utils import get_logger

logger = get_logger(__name__)

SECRET_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|token|b2[_-]?(?:readonly|write)?[_-]?key|authorization)\s*[:=]\s*\S+"
)
SECRET_ENV_NAMES = frozenset(
    {
        "B2_KEY_ID",
        "B2_READONLY_KEY",
        "B2_WRITE_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "RCLONE_CONFIG_PASS",
    }
)


class SecurityViolationError(PermissionError):
    """Raised when an operation would violate read-only or data-handling policy."""


def resolve_under(root: Path, path: Path) -> bool:
    """True if ``path`` resolves to a location under ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_not_input_raw(
    target: Path,
    input_raw: Path,
    *,
    operation: str = "write",
) -> None:
    """Refuse writes inside ``input_raw/`` (originals must stay read-only)."""
    if resolve_under(Path(input_raw), Path(target)):
        raise SecurityViolationError(
            f"Refusing {operation} inside read-only input_raw: {target}"
        )


def register_input_raw_guard(input_raw: Path) -> Path:
    """Return resolved ``input_raw`` path for downstream guards."""
    return Path(input_raw).resolve()


def redact_secrets_text(text: str) -> str:
    if not text:
        return text
    out = SECRET_KEY_PATTERN.sub(r"\1=***REDACTED***", text)
    for name in SECRET_ENV_NAMES:
        val = os.environ.get(name, "")
        if val and len(val) > 4:
            out = out.replace(val, "***REDACTED***")
    return out


def redact_secrets_obj(obj: Any) -> Any:
    """Recursively redact secret-like string values for logs and JSON exports."""
    if isinstance(obj, str):
        return redact_secrets_text(obj)
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if key.upper() in SECRET_ENV_NAMES or any(
                s in key.lower() for s in ("password", "secret", "api_key", "token", "private_key")
            ):
                out[key] = "***REDACTED***"
            else:
                out[key] = redact_secrets_obj(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_secrets_obj(v) for v in obj]
    return obj


def sanitize_audit_for_export(audit: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Strip/redact sensitive fields before persisting buyer-facing sidecars."""
    cleaned = redact_secrets_obj(dict(audit))
    if isinstance(cleaned, dict):
        cleaned.pop("decode_path", None)  # internal temp copy path — not needed in exports
    return cleaned  # type: ignore[return-value]


def secure_delete_file(path: Path, *, passes: int = 3) -> None:
    """
    Overwrite a file with random bytes then unlink.

    Falls back to unlink-only when overwrite is not permitted (e.g. some cloud FS).
    """
    path = Path(path)
    if not path.is_file():
        return
    try:
        size = path.stat().st_size
        with path.open("r+b") as fh:
            for _ in range(max(1, int(passes))):
                fh.seek(0)
                fh.write(os.urandom(size))
                fh.flush()
                os.fsync(fh.fileno())
    except OSError as exc:
        logger.warning("secure_delete_overwrite_skipped", path=str(path), error=str(exc))
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("secure_delete_unlink_failed", path=str(path), error=str(exc))


def secure_delete_tree(root: Path, *, passes: int = 3) -> None:
    """Secure-delete all files under ``root`` then remove directories."""
    root = Path(root)
    if not root.exists():
        return
    if root.is_file():
        secure_delete_file(root, passes=passes)
        return
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            secure_delete_file(Path(dirpath) / name, passes=passes)
        try:
            Path(dirpath).rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        shutil.rmtree(root, ignore_errors=True)


def verify_ingest_hashes(
    local_dir: Path,
    expected_hashes: Mapping[str, str],
    *,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Verify per-file SHA256 after ingest copy.

    ``expected_hashes`` maps relative posix paths to hex digests.
    """
    from .utils import sha256_file

    local_dir = Path(local_dir).resolve()
    allowed = extensions  # None = all files in expected_hashes keys

    missing: list[str] = []
    mismatched: list[str] = []
    verified: list[str] = []

    for rel, expected in expected_hashes.items():
        norm = rel.replace("\\", "/")
        path = local_dir / norm
        if not path.is_file():
            missing.append(norm)
            continue
        got = sha256_file(path)
        if got.lower() != str(expected).lower():
            mismatched.append(norm)
        else:
            verified.append(norm)

    ok = not missing and not mismatched
    report = {
        "ok": ok,
        "verified_count": len(verified),
        "missing": missing,
        "mismatched": mismatched,
    }
    if not ok:
        logger.error("ingest_hash_verification_failed", **report)
    else:
        logger.info("ingest_hash_verification_ok", verified_count=len(verified))
    return report
