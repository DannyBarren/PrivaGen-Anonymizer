"""
Startup secrets loader for the dataset_anonymizer pipeline.

Design goals (rented GPU VM deployment):
- Keep Backblaze B2 API keys OUT of the git repo and OUT of process logs.
- Support an ENCRYPTED secrets file (``.env.enc``, Fernet) as the preferred at-rest
  form, with a permission-locked plaintext ``.env`` / ``.env.local`` as fallback.
- Load secrets into ``os.environ`` at process startup, WITHOUT overwriting values
  already injected by the platform (platform-injected env always wins).
- Refuse to read a secrets file that is group/world readable (unless explicitly
  overridden), so a stray ``chmod`` cannot silently expose keys.
- Zero heavy dependencies at import time so this is safe to call during the
  minimal Flask UI boot. ``cryptography`` is imported lazily only when needed.

Typical operator workflow on the VM
-----------------------------------
    # 1) Generate a master key once (stored at ~/.config/dataset_anonymizer/secret.key, 0600)
    python -m scripts.secrets_manager gen-key

    # 2) Put real B2 keys in a local plaintext env file (0600), then encrypt it
    cp .env.example .env.local && chmod 600 .env.local && $EDITOR .env.local
    python -m scripts.secrets_manager encrypt --in .env.local --out .env.enc
    shred -u .env.local            # remove the plaintext once encrypted

    # 3) Verify the pipeline can load + see required B2 vars
    python -m scripts.secrets_manager check

At runtime ``load_secrets()`` (called automatically by app.py / main_pipeline.py)
decrypts ``.env.enc`` in memory and populates ``os.environ``.

The Fernet key itself may be provided via (in priority order):
  1. env ``DATASET_ANON_SECRET_KEY`` (base64 Fernet key) — e.g. platform secret
  2. key file ``DATASET_ANON_SECRET_KEY_FILE`` (0600)
  3. default key file ``~/.config/dataset_anonymizer/secret.key`` (0600)
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("dataset_anonymizer.secrets")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Env var that lets an operator bypass the file-permission hardening check.
ALLOW_INSECURE_ENV = "DATASET_ANON_ALLOW_INSECURE_ENV"

#: When truthy, refuse plaintext secret files — only the encrypted .env.enc is used.
USE_ENCRYPTED_ENV = "DATASET_ANON_USE_ENCRYPTED_ENV"

#: When truthy, run a B2 read-only + write bucket connectivity check on startup.
BUCKET_CONFIRMATION_ENV = "ENABLE_BUCKET_CONFIRMATION"

# --------------------------------------------------------------------------------------
# Run defaults (non-secret)
# --------------------------------------------------------------------------------------
# The two B2 application keys the loader expects. Their VALUES are secrets and load from
# ``.env.enc`` or injected env vars (never committed); only the NAMES are recorded here
# so the loader can report exactly which keys it expects:
#   - read-only key  → B2_READONLY_KEY (+ B2_READONLY_KEY_ID)  : source, read-only
#   - read/write key → B2_WRITE_KEY    (+ B2_WRITE_KEY_ID)     : output, read/write
READONLY_KEY_NAME = "b2-readonly"
WRITE_KEY_NAME = "b2-datasets-rw"

#: Non-secret bucket + path placeholders auto-applied at startup. Bucket names and
#: remote paths are NOT secrets. These are only applied when the operator/platform has
#: not already provided a value (so injected env or ``.env.enc`` always wins). The
#: bucket placeholders match ``_PLACEHOLDER_MARKERS``, so a real transfer is refused
#: until an operator supplies real bucket names.
RUN_DEFAULT_ENV: Dict[str, str] = {
    "B2_READONLY_BUCKET": "your-source-bucket",
    "B2_INGEST_REMOTE_PATH": "datasets/raw",
    "B2_WRITE_BUCKET": "your-dest-bucket",
    "B2_EXPORT_REMOTE_PATH": "datasets/anonymized",
}

#: First line of the startup banner — confirms the commands + paths are loaded.
READY_BANNER_HEADER = (
    "✅ rclone commands loaded | Source bucket protected (read-only) "
    "| Output bucket write verified"
)

#: Env var carrying a base64 Fernet key directly (highest priority).
SECRET_KEY_ENV = "DATASET_ANON_SECRET_KEY"

#: Env var pointing at a Fernet key file.
SECRET_KEY_FILE_ENV = "DATASET_ANON_SECRET_KEY_FILE"

#: Default location for the Fernet key file.
DEFAULT_KEY_FILE = Path.home() / ".config" / "dataset_anonymizer" / "secret.key"

#: Encrypted secrets filename (preferred at-rest form).
ENCRYPTED_ENV_NAME = ".env.enc"

#: Plaintext secret file candidates, in priority order (real values expected here).
#: NOTE: the committed ``.env`` is intentionally excluded — it is a mock/documentation
#: template only. Real secrets belong in these git-ignored files (or the encrypted
#: ``.env.enc``), so production keys never share a filename with a tracked template.
PLAINTEXT_CANDIDATES = (".env.local", ".env.production")

#: Secret env names whose VALUES must never be logged (redaction safety net).
SECRET_ENV_NAMES = frozenset(
    {
        "B2_KEY_ID",
        "B2_READONLY_KEY",
        "B2_WRITE_KEY",
        "RCLONE_CRYPT_PASSWORD",
        "RCLONE_CRYPT_PASSWORD2",
        "RCLONE_CRYPT_SALT",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        SECRET_KEY_ENV,
    }
)

#: Secret keys always required before a real B2 transfer can run.
REQUIRED_B2_VARS = (
    "B2_READONLY_KEY",
    "B2_WRITE_KEY",
)
#: One of each pair (either explicit or alias) must be present.
REQUIRED_B2_EITHER = (
    ("B2_READONLY_BUCKET", "B2_SOURCE_BUCKET"),
    ("B2_WRITE_BUCKET", "B2_DEST_BUCKET"),
)

#: Placeholder markers — values matching these are treated as "not real" and skipped
#: so a committed mock ``.env`` never masks a genuinely-missing production secret.
_PLACEHOLDER_MARKERS = (
    "mock",
    "replace_me",
    "replace-me",
    "changeme",
    "change-me",
    "your-",
    "your_",
    "<",
    "xxxx",
    "example",
    "placeholder",
)


class SecretsError(RuntimeError):
    """Raised for unrecoverable secrets configuration problems."""


def _is_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------

def _strip_inline_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines from dotenv-style text (comments/blank lines ignored)."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        out[key] = _strip_inline_quotes(value)
    return out


def is_placeholder(value: str) -> bool:
    """True if ``value`` looks like a mock/template placeholder rather than a real secret."""
    if value is None:
        return True
    v = value.strip()
    if not v:
        return True
    low = v.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


# --------------------------------------------------------------------------------------
# Permission hardening
# --------------------------------------------------------------------------------------

def _perms_ok(path: Path) -> bool:
    """POSIX: True unless the file is group/other readable or writable."""
    if os.name != "posix":
        return True  # Windows ACLs not enforced here
    try:
        mode = path.stat().st_mode
    except OSError:
        return True
    return not bool(mode & (stat.S_IRWXG | stat.S_IRWXO))


def _require_secure_perms(path: Path, *, kind: str) -> None:
    if _perms_ok(path):
        return
    if os.environ.get(ALLOW_INSECURE_ENV, "").strip() in ("1", "true", "yes"):
        logger.warning(
            "insecure_permissions_allowed kind=%s path=%s (override via %s)",
            kind,
            path,
            ALLOW_INSECURE_ENV,
        )
        return
    raise SecretsError(
        f"{kind} at {path} is group/world accessible. "
        f"Run: chmod 600 {path}  (or set {ALLOW_INSECURE_ENV}=1 to override)."
    )


# --------------------------------------------------------------------------------------
# Encryption (lazy cryptography import)
# --------------------------------------------------------------------------------------

def _fernet():
    try:
        from cryptography.fernet import Fernet  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise SecretsError(
            "The 'cryptography' package is required for encrypted secrets. "
            "Install it: pip install cryptography  (or use a plaintext .env.local instead)."
        ) from exc
    return Fernet


def generate_key() -> bytes:
    """Return a new base64-encoded Fernet key."""
    return _fernet().generate_key()


def _load_key() -> bytes:
    """Resolve the Fernet key from env or a 0600 key file."""
    env_key = os.environ.get(SECRET_KEY_ENV, "").strip()
    if env_key:
        return env_key.encode("utf-8")

    file_path = os.environ.get(SECRET_KEY_FILE_ENV, "").strip()
    key_file = Path(file_path).expanduser() if file_path else DEFAULT_KEY_FILE
    if not key_file.is_file():
        raise SecretsError(
            f"No decryption key. Set {SECRET_KEY_ENV}, or {SECRET_KEY_FILE_ENV}, "
            f"or create {DEFAULT_KEY_FILE} via: python -m scripts.secrets_manager gen-key"
        )
    _require_secure_perms(key_file, kind="secret key file")
    return key_file.read_bytes().strip()


def write_key_file(path: Path = DEFAULT_KEY_FILE, *, overwrite: bool = False) -> Path:
    """Generate and persist a Fernet key at ``path`` with 0600 permissions."""
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise SecretsError(f"Key file already exists: {path} (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_key()
    # Create with restrictive perms from the start (avoid a readable window).
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key + b"\n")
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


def encrypt_file(src: Path, dst: Path) -> Path:
    """Encrypt plaintext env file ``src`` → ``dst`` (.env.enc) using the resolved key."""
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    if not src.is_file():
        raise SecretsError(f"Plaintext env file not found: {src}")
    Fernet = _fernet()
    token = Fernet(_load_key()).encrypt(src.read_bytes())
    dst.write_bytes(token)
    return dst


def decrypt_text(path: Path) -> str:
    """Decrypt ``.env.enc`` at ``path`` and return the plaintext dotenv content."""
    Fernet = _fernet()
    from cryptography.fernet import InvalidToken  # type: ignore

    try:
        return Fernet(_load_key()).decrypt(Path(path).read_bytes()).decode("utf-8")
    except InvalidToken as exc:
        raise SecretsError(
            f"Failed to decrypt {path}: wrong key or corrupted file."
        ) from exc


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def _resolve_root(project_root: Optional[Path]) -> Path:
    if project_root is not None:
        return Path(project_root)
    # scripts/secrets_manager.py -> project root is parent of scripts/
    return Path(__file__).resolve().parent.parent


def _discover_source(root: Path) -> Tuple[Optional[Path], str]:
    """Return (path, kind) of the secrets source to use, preferring the encrypted file."""
    enc = root / ENCRYPTED_ENV_NAME
    if enc.is_file():
        return enc, "encrypted"
    for name in PLAINTEXT_CANDIDATES:
        candidate = root / name
        if candidate.is_file():
            return candidate, "plaintext"
    return None, "none"


def apply_run_defaults(*, override: bool = False) -> Dict[str, str]:
    """
    Apply the non-secret bucket/path defaults into ``os.environ``.

    Only fills values that are absent (or placeholder), so platform-injected env vars
    and real ``.env.enc`` values always take precedence. Returns ``{name: value}`` for
    the defaults that were actually applied. Bucket names and paths are not secrets.
    """
    applied: Dict[str, str] = {}
    for key, value in RUN_DEFAULT_ENV.items():
        current = (os.environ.get(key) or "").strip()
        if override or not current or is_placeholder(current):
            os.environ[key] = value
            applied[key] = value
    if applied:
        logger.info("run_defaults_applied keys=%s", sorted(applied.keys()))
    return applied


def ready_banner() -> str:
    """
    The exact startup banner shown once ingest + export bucket access is confirmed.

    Printed by :func:`confirm_bucket_access` after a successful two-key connectivity
    check, so operators get a single unambiguous "ready" signal before a real run.
    """
    return "\n".join(
        [
            READY_BANNER_HEADER,
            "✅ SUCCESS: Read-only ingest bucket ACCESSIBLE",
            "✅ SUCCESS: RW output bucket ACCESSIBLE + write confirmed",
            "🚀 PrivaGen ready for full image anonymization run",
        ]
    )


def load_secrets(
    project_root: Optional[Path] = None,
    *,
    override: bool = False,
) -> Dict[str, str]:
    """
    Load secrets into ``os.environ`` at startup.

    Returns a dict of ``{name: source}`` for the keys that were applied (values are
    never returned or logged). Placeholder values and vars already present in the
    environment are skipped unless ``override`` is True.

    This function never raises for a *missing* secrets file (a run may rely purely
    on platform-injected env vars); it only raises on an unreadable/insecure file
    or a decryption failure.
    """
    root = _resolve_root(project_root)
    source_path, kind = _discover_source(root)

    # Enforce encryption-at-rest: when required, only the encrypted .env.enc is
    # acceptable; a plaintext secrets file is ignored (never silently used).
    if _is_truthy(USE_ENCRYPTED_ENV) and kind == "plaintext":
        logger.warning(
            "encrypted_env_required path=%s (%s=1) — ignoring plaintext; provide .env.enc",
            getattr(source_path, "name", source_path),
            USE_ENCRYPTED_ENV,
        )
        return {name: "run_default" for name in apply_run_defaults(override=override)}

    if source_path is None:
        logger.info("secrets_no_file root=%s (relying on process environment)", root)
        return {name: "run_default" for name in apply_run_defaults(override=override)}

    if kind == "encrypted":
        # Encrypted bundles always hold real secrets — enforce strict perms.
        _require_secure_perms(source_path, kind="encrypted secrets file")
        try:
            text = decrypt_text(source_path)
        except SecretsError as exc:
            logger.error("secrets_decrypt_failed path=%s error=%s", source_path, exc)
            raise
        parsed = parse_env_text(text)
    else:
        # Plaintext file may be a committed mock template (safe to be world-readable)
        # or a real secrets file. Only enforce strict perms when it actually contains
        # real (non-placeholder) values, so a shipped mock .env never blocks startup.
        text = source_path.read_text(encoding="utf-8")
        parsed = parse_env_text(text)
        # Only enforce strict perms when a recognized SECRET var holds a real value —
        # non-sensitive config (paths, bucket names, rclone binary) never triggers it,
        # and a committed all-mock .env template stays usable at 0644.
        has_real_secret = any(
            key in SECRET_ENV_NAMES and not is_placeholder(value)
            for key, value in parsed.items()
        )
        if has_real_secret:
            _require_secure_perms(source_path, kind="secrets file")
    applied: Dict[str, str] = {}
    skipped_placeholder: List[str] = []
    skipped_present: List[str] = []

    for key, value in parsed.items():
        if is_placeholder(value):
            skipped_placeholder.append(key)
            continue
        if not override and os.environ.get(key):
            skipped_present.append(key)
            continue
        os.environ[key] = value
        applied[key] = str(source_path.name)

    # Apply the non-secret bucket/path defaults for any values the secrets file
    # did not provide (real file values above always win).
    for name, value in apply_run_defaults(override=override).items():
        applied.setdefault(name, "run_default")

    # Log NAMES only — never values.
    logger.info(
        "secrets_loaded source=%s kind=%s applied=%s skipped_placeholder=%s skipped_present=%s",
        source_path.name,
        kind,
        sorted(applied.keys()),
        sorted(skipped_placeholder),
        sorted(skipped_present),
    )
    return applied


def missing_required_b2_vars(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Return the list of required B2 settings that are absent/placeholder."""
    src = dict(env if env is not None else os.environ)
    missing: List[str] = []

    def _real(name: str) -> bool:
        val = (src.get(name) or "").strip()
        return bool(val) and not is_placeholder(val)

    for name in REQUIRED_B2_VARS:
        if not _real(name):
            missing.append(name)
    # Key ID: either a shared B2_KEY_ID, or BOTH per-remote least-privilege IDs.
    if not (_real("B2_KEY_ID") or (_real("B2_READONLY_KEY_ID") and _real("B2_WRITE_KEY_ID"))):
        missing.append("B2_KEY_ID (or B2_READONLY_KEY_ID + B2_WRITE_KEY_ID)")
    for pair in REQUIRED_B2_EITHER:
        if not any((src.get(n) or "").strip() and not is_placeholder(src.get(n, "")) for n in pair):
            missing.append(" or ".join(pair))
    return missing


def confirm_bucket_access(project_root: Optional[Path] = None) -> Optional[bool]:
    """
    On startup, verify B2 read-only ingest + RW export connectivity and print a
    clear confirmation for the operator (terminal / Jupyter UI).

    Gated by ``ENABLE_BUCKET_CONFIRMATION`` (truthy). Returns True on full success,
    False on a connectivity failure, and None when skipped. Never raises.
    """
    if not _is_truthy(BUCKET_CONFIRMATION_ENV):
        return None

    missing = missing_required_b2_vars()
    if missing:
        print("[bucket-check] skipped — missing B2 settings: " + ", ".join(missing))
        return None

    try:
        from .rclone_integration import verify_bucket_access
    except Exception:  # noqa: BLE001 - support non-package import contexts
        try:
            from rclone_integration import verify_bucket_access  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[bucket-check] unavailable: {exc}")
            return None

    try:
        result = verify_bucket_access()
    except Exception as exc:  # noqa: BLE001
        print(f"[bucket-check] error: {exc}")
        return False

    if not result.get("readonly_ok"):
        print(f"❌ FAILED: Read-only ingest bucket NOT accessible — {result.get('error')}")
        return False
    if not result.get("write_ok"):
        print("✅ SUCCESS: Read-only ingest bucket ACCESSIBLE")
        print(f"❌ FAILED: RW output bucket write test failed — {result.get('error')}")
        return False

    # Both keys verified — print the single "ready" banner.
    print(ready_banner())
    return True


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m scripts.secrets_manager",
        description="Manage encrypted/locked secrets for dataset_anonymizer.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-key", help="Generate a Fernet key file (0600)")
    p_gen.add_argument("--out", default=str(DEFAULT_KEY_FILE), help="Key file path")
    p_gen.add_argument("--force", action="store_true", help="Overwrite existing key file")

    p_enc = sub.add_parser("encrypt", help="Encrypt a plaintext env file to .env.enc")
    p_enc.add_argument("--in", dest="src", default=".env.local", help="Plaintext env file")
    p_enc.add_argument("--out", dest="dst", default=ENCRYPTED_ENV_NAME, help="Encrypted output")

    p_dec = sub.add_parser("decrypt", help="Decrypt .env.enc to stdout (verification)")
    p_dec.add_argument("--in", dest="src", default=ENCRYPTED_ENV_NAME, help="Encrypted env file")

    sub.add_parser("check", help="Load secrets and report required B2 vars status")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "gen-key":
            path = write_key_file(Path(args.out), overwrite=args.force)
            print(f"Wrote Fernet key (0600): {path}")
            print("Keep this file secret and backed up — it decrypts your .env.enc.")
            return 0

        if args.cmd == "encrypt":
            dst = encrypt_file(Path(args.src), Path(args.dst))
            print(f"Encrypted {args.src} -> {dst}")
            print("Now remove the plaintext, e.g.: shred -u " + args.src)
            return 0

        if args.cmd == "decrypt":
            sys.stdout.write(decrypt_text(Path(args.src)))
            return 0

        if args.cmd == "check":
            applied = load_secrets()
            missing = missing_required_b2_vars()
            print(f"Applied {len(applied)} secret(s)/default(s) into environment.")
            print(
                f"Resolved B2 routing:\n"
                f"  ingest = {os.environ.get('B2_READONLY_BUCKET', '')}/"
                f"{os.environ.get('B2_INGEST_REMOTE_PATH', '')}  "
                f"(key: {READONLY_KEY_NAME}, read-only)\n"
                f"  output = {os.environ.get('B2_WRITE_BUCKET', '')}/"
                f"{os.environ.get('B2_EXPORT_REMOTE_PATH', '')}  "
                f"(key: {WRITE_KEY_NAME}, read/write)"
            )
            if missing:
                print("MISSING required B2 settings: " + ", ".join(missing))
                return 1
            print("All required B2 settings are present.")
            return 0
    except SecretsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
