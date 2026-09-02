#!/usr/bin/env python3
"""
Safe, auditable cleanup of testing/debug artifacts in selected pipeline directories.

Targets ONLY these top-level folders (never walks outside them):
  - final_clean/
  - logs/batch_summaries/
  - temp_processed/

Default mode is dry-run. Use --execute to delete after explicit confirmation.
A timestamped deletion manifest (JSON + text) is written before any deletion.

Run from project root:
    python -m scripts.cleanup_testing_artifacts
    python -m scripts.cleanup_testing_artifacts --execute
    python -m scripts.cleanup_testing_artifacts --project-root /path/to/project
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Scope: only these roots (relative to project root)
# ---------------------------------------------------------------------------

TARGET_DIRS: Tuple[str, ...] = (
    "final_clean",
    "logs/batch_summaries",
    "temp_processed",
)

# ---------------------------------------------------------------------------
# Classification rules (conservative — when in doubt, preserve)
# ---------------------------------------------------------------------------

# Always preserve these basenames anywhere under target trees
PRESERVE_BASENAMES: frozenset[str] = frozenset(
    {
        ".gitkeep",
        ".gitignore",
    }
)

# Production manifest / metadata names (never delete by basename)
PRODUCTION_MANIFEST_BASENAMES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "manifest.csv",
        "master_manifest.json",
        "dataset_manifest.json",
        "inventory.json",
        "inventory.csv",
        "processed_manifest.json",
        "anonymization_audit.json",
        "master_summary.csv",
        "master_summary.pdf",
    }
)

# Substrings in basename (case-insensitive) that indicate testing/debug/temp
TEST_SUBSTRING_MARKERS: Tuple[str, ...] = (
    "test_",
    "_test",
    "_test_",
    "temp_",
    "_tmp",
    "_tmp_",
    "debug_",
    "_debug",
    "sample_",
    "_sample",
    "checkpoint",
    "_processed_tmp",
    "_chunk_",
    "_backup",
    "prod_test_",
    "prod_val_",
    "final_val_",
    "gpu_rt_",
    "perf_",
    "scale_",
    "roundtrip_",
)

# fnmatch patterns for known synthetic benchmark seeds (project test scripts)
TEST_GLOB_PATTERNS: Tuple[str, ...] = (
    "gpu_rt_*",
    "perf_*",
    "prod_test_*",
    "prod_val_*",
    "final_val_*",
    "scale_*",
    "roundtrip_*",
    "sample_*",
    "test_*",
    "*_test.*",
    "*_test_*",
    "temp_*",
    "*_tmp.*",
    "debug_*",
    "*_debug.*",
    "*checkpoint*",
    "*_processed_tmp*",
    "*_chunk_*",
    "*_backup*",
)

# Global junk extensions / dir names (safe to remove when under target trees)
JUNK_FILENAMES: frozenset[str] = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
JUNK_EXTENSIONS: frozenset[str] = frozenset({".pyc", ".pyo"})
JUNK_DIR_NAMES: frozenset[str] = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})

# temp_processed: operational batch layout (ephemeral — safe to remove between runs)
TEMP_BATCH_DIR_RE = re.compile(r"^batch_\d{5}$", re.IGNORECASE)
TEMP_EPHEMERAL_DIR_NAMES: frozenset[str] = frozenset(
    {
        "_source_copies",
        "_work",
        "_scratch",
        ".cache",
        "cache",
    }
)

# logs/batch_summaries: production batch CSV naming from main_pipeline
PRODUCTION_BATCH_SUMMARY_RE = re.compile(
    r"^batch_\d{5}(_resume)?\.csv$",
    re.IGNORECASE,
)

# Image extensions for production outputs (pipeline default sale format is .jpg)
PRODUCTION_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
)


@dataclass
class DeletionEntry:
    path: Path
    rel_path: str
    is_dir: bool
    size_bytes: int
    reason: str
    root_label: str


@dataclass
class DeletionPlan:
    entries: List[DeletionEntry] = field(default_factory=list)
    preserve_count: int = 0
    scan_errors: List[str] = field(default_factory=list)

    @property
    def files(self) -> List[DeletionEntry]:
        return [e for e in self.entries if not e.is_dir]

    @property
    def dirs(self) -> List[DeletionEntry]:
        return [e for e in self.entries if e.is_dir]

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_dirs(self) -> int:
        return len(self.dirs)

    @property
    def total_bytes(self) -> int:
        # Directory entries may carry aggregated subtree bytes (batch folders)
        return sum(e.size_bytes for e in self.entries)

    def add(self, entry: DeletionEntry) -> None:
        self.entries.append(entry)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f} MiB"
    return f"{num_bytes / 1024**3:.2f} GiB"


def _safe_stat(path: Path) -> Tuple[int, bool]:
    """Return (size_bytes, is_dir). size 0 for dirs or on error."""
    try:
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            return 0, True
        return int(st.st_size), False
    except OSError:
        return 0, path.is_dir()


def _basename_matches_test_markers(name: str) -> bool:
    lower = name.lower()
    stem = Path(name).stem.lower()
    for marker in TEST_SUBSTRING_MARKERS:
        if marker in lower or marker in stem:
            return True
    for pat in TEST_GLOB_PATTERNS:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(stem, pat):
            return True
    return False


def _is_junk_file(name: str) -> bool:
    lower = name.lower()
    if lower in JUNK_FILENAMES:
        return True
    return Path(name).suffix.lower() in JUNK_EXTENSIONS


def _is_production_manifest(name: str) -> bool:
    return name.lower() in {x.lower() for x in PRODUCTION_MANIFEST_BASENAMES}


def _is_production_media(path: Path) -> bool:
    """True if extension looks like a production image and name has no test markers."""
    ext = path.suffix.lower()
    if ext not in PRODUCTION_IMAGE_EXTENSIONS:
        return False
    return not _basename_matches_test_markers(path.name)


def _classify_final_clean(path: Path, *, is_dir: bool) -> Tuple[bool, str]:
    """
    final_clean: delete ONLY obvious test/debug artifacts.
    Preserve all production media, sidecars, and manifests unless explicitly test-marked.
    """
    name = path.name
    if name in PRESERVE_BASENAMES:
        return False, "preserve:gitkeep"
    if _is_production_manifest(name):
        return False, "preserve:production_manifest"

    if is_dir:
        if name.lower() in JUNK_DIR_NAMES:
            return True, "junk_dir"
        if _basename_matches_test_markers(name):
            return True, "test_dir_name"
        # Do not delete generic dirs (may contain production nested data)
        return False, "preserve:directory"

    if _is_junk_file(name):
        return True, "junk_file"
    if _basename_matches_test_markers(name):
        return True, "test_name_marker"
    if _is_production_media(path):
        return False, "preserve:production_media"
    # JSON sidecar next to production image — preserve unless test-marked
    if path.suffix.lower() == ".json" and not _basename_matches_test_markers(name):
        return False, "preserve:metadata_sidecar"
    # Unknown file in final_clean — preserve (conservative)
    return False, "preserve:unknown_conservative"


def _classify_batch_summaries(path: Path, *, is_dir: bool) -> Tuple[bool, str]:
    """
    logs/batch_summaries: keep production batch_*.csv; delete test/debug/junk only.
    """
    name = path.name
    if name in PRESERVE_BASENAMES:
        return False, "preserve:gitkeep"
    if is_dir:
        if name.lower() in JUNK_DIR_NAMES:
            return True, "junk_dir"
        if _basename_matches_test_markers(name):
            return True, "test_dir_name"
        return False, "preserve:directory"

    if _is_junk_file(name):
        return True, "junk_file"
    if PRODUCTION_BATCH_SUMMARY_RE.match(name):
        return False, "preserve:production_batch_summary"
    if _basename_matches_test_markers(name):
        return True, "test_name_marker"
    if name.endswith(".log") and ("test" in name.lower() or "debug" in name.lower()):
        return True, "test_log"
    # Unknown — preserve (may be operational)
    return False, "preserve:unknown_conservative"


def _classify_temp_processed(path: Path, *, is_dir: bool, rel_parts: Sequence[str]) -> Tuple[bool, str]:
    """
    temp_processed: batch workspaces and intermediates are ephemeral.
    Delete batch_* trees, _source_copies, test artifacts, junk.
    Preserve only .gitkeep and non-test lock files at root (optional conservatism).
    """
    name = path.name
    if name in PRESERVE_BASENAMES:
        return False, "preserve:gitkeep"

    if is_dir:
        if name.lower() in JUNK_DIR_NAMES:
            return True, "junk_dir"
        if TEMP_BATCH_DIR_RE.match(name):
            return True, "temp_batch_dir"
        if name in TEMP_EPHEMERAL_DIR_NAMES:
            return True, "ephemeral_dir"
        if _basename_matches_test_markers(name):
            return True, "test_dir_name"
        # Nested dir inside batch_* — handled when parent batch dir is deleted
        if len(rel_parts) >= 2 and TEMP_BATCH_DIR_RE.match(rel_parts[0]):
            return True, "inside_batch_dir"
        return False, "preserve:directory"

    if _is_junk_file(name):
        return True, "junk_file"
    if name.endswith(".processing.lock"):
        # Stale locks inside batch dirs go with batch; root-level locks preserved
        if rel_parts and TEMP_BATCH_DIR_RE.match(rel_parts[0]):
            return True, "batch_lock_file"
        return False, "preserve:root_lock"
    if _basename_matches_test_markers(name):
        return True, "test_name_marker"
    if len(rel_parts) >= 1 and TEMP_BATCH_DIR_RE.match(rel_parts[0]):
        return True, "inside_batch_dir"
    if len(rel_parts) >= 2 and rel_parts[0] in TEMP_EPHEMERAL_DIR_NAMES:
        return True, "ephemeral_content"
    # Loose files at temp_processed root (not gitkeep) — likely leftover test output
    if len(rel_parts) == 1:
        return True, "temp_root_loose_file"
    return False, "preserve:unknown_conservative"


CLASSIFIERS = {
    "final_clean": _classify_final_clean,
    "logs/batch_summaries": _classify_batch_summaries,
    "temp_processed": _classify_temp_processed,
}


def _iter_tree(root: Path) -> Iterator[Tuple[Path, Sequence[str]]]:
    """
    Efficient iterative walk using os.scandir (handles large trees better than rglob).
    Yields (absolute_path, relative_parts_from_root).
    """
    root = root.resolve()
    if not root.is_dir():
        return
    stack: List[Tuple[Path, Tuple[str, ...]]] = [(root, ())]
    while stack:
        current, parts = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError as exc:
            yield current, parts
            continue
        # Process files first, then push dirs (dirs deleted after contents in plan phase)
        subdirs: List[Tuple[Path, Tuple[str, ...]]] = []
        for entry in entries:
            name = entry.name
            child_parts = parts + (name,)
            child_path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                subdirs.append((child_path, child_parts))
            else:
                yield child_path, child_parts
        for child_path, child_parts in subdirs:
            yield child_path, child_parts
            stack.append((child_path, child_parts))


def _should_delete(
    root_label: str,
    path: Path,
    rel_parts: Sequence[str],
    *,
    is_dir: bool,
) -> Tuple[bool, str]:
    if root_label == "temp_processed":
        return _classify_temp_processed(path, is_dir=is_dir, rel_parts=rel_parts)
    classifier = CLASSIFIERS[root_label]
    return classifier(path, is_dir=is_dir)


def _subtree_file_bytes(root: Path) -> int:
    total = 0
    if not root.is_dir():
        return 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        try:
                            total += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            pass
        except OSError:
            pass
    return total


def build_deletion_plan(project_root: Path) -> DeletionPlan:
    project_root = project_root.resolve()
    plan = DeletionPlan()
    seen_delete: Set[str] = set()

    for rel_root in TARGET_DIRS:
        abs_root = project_root / rel_root
        root_label = rel_root.replace("\\", "/")
        if not abs_root.exists():
            plan.scan_errors.append(f"Target missing (skipped): {abs_root}")
            continue
        if not abs_root.is_dir():
            plan.scan_errors.append(f"Target is not a directory (skipped): {abs_root}")
            continue

        # batch_dirs_marked removed — using seen_delete only

        for path, rel_parts in _iter_tree(abs_root):
            try:
                rel_from_project = path.relative_to(project_root)
            except ValueError:
                continue
            rel_key = str(rel_from_project).replace("\\", "/")
            size, is_dir = _safe_stat(path)

            delete, reason = _should_delete(root_label, path, rel_parts, is_dir=is_dir)
            if not delete:
                plan.preserve_count += 1
                continue

            # Collapse temp_processed batch_* dirs: one entry with aggregated size
            if root_label == "temp_processed" and rel_parts and TEMP_BATCH_DIR_RE.match(rel_parts[0]):
                batch_parent = project_root / rel_root / rel_parts[0]
                try:
                    batch_rel = str(batch_parent.relative_to(project_root)).replace("\\", "/")
                except ValueError:
                    batch_rel = ""
                if batch_rel and batch_rel not in seen_delete:
                    if is_dir and path.name == rel_parts[0]:
                        seen_delete.add(batch_rel)
                        plan.add(
                            DeletionEntry(
                                path=batch_parent,
                                rel_path=batch_rel,
                                is_dir=True,
                                size_bytes=_subtree_file_bytes(batch_parent),
                                reason="temp_batch_dir",
                                root_label=root_label,
                            )
                        )
                continue

            if rel_key in seen_delete:
                continue
            seen_delete.add(rel_key)
            plan.add(
                DeletionEntry(
                    path=path,
                    rel_path=rel_key,
                    is_dir=is_dir,
                    size_bytes=size,
                    reason=reason,
                    root_label=root_label,
                )
            )

    # Sort: deepest paths first for execution (files before parents handled by sort key)
    plan.entries.sort(key=lambda e: (e.is_dir, -len(e.rel_path), e.rel_path))
    return plan


def _plan_to_manifest(plan: DeletionPlan, project_root: Path, *, execute: bool) -> Dict[str, object]:
    by_root: Dict[str, Dict[str, int]] = {}
    for e in plan.entries:
        bucket = by_root.setdefault(e.root_label, {"files": 0, "dirs": 0, "bytes": 0})
        if e.is_dir:
            bucket["dirs"] += 1
            bucket["bytes"] += e.size_bytes
        else:
            bucket["files"] += 1
            bucket["bytes"] += e.size_bytes

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root.resolve()),
        "mode": "execute" if execute else "dry_run",
        "summary": {
            "files": plan.total_files,
            "directories": plan.total_dirs,
            "total_bytes": plan.total_bytes,
            "total_human": _human_size(plan.total_bytes),
            "preserved_items_seen": plan.preserve_count,
        },
        "by_root": by_root,
        "scan_errors": plan.scan_errors,
        "entries": [
            {
                "path": e.rel_path,
                "is_dir": e.is_dir,
                "size_bytes": e.size_bytes,
                "reason": e.reason,
                "root": e.root_label,
            }
            for e in plan.entries
        ],
    }


def _write_manifest_files(
    plan: DeletionPlan,
    project_root: Path,
    stamp: str,
    *,
    execute: bool,
) -> Tuple[Path, Path]:
    out_dir = project_root / "reports" / "cleanup_plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cleanup_testing_artifacts_{stamp}.json"
    txt_path = out_dir / f"cleanup_testing_artifacts_{stamp}.txt"

    manifest = _plan_to_manifest(plan, project_root, execute=execute)
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "dataset_anonymizer — testing artifacts cleanup manifest",
        f"Generated: {manifest['generated_at']}",
        f"Project: {project_root}",
        f"Mode: {manifest['mode']}",
        "",
        "Summary",
        f"  Files to delete:      {plan.total_files}",
        f"  Directories to delete: {plan.total_dirs}",
        f"  Total size (files):   {_human_size(plan.total_bytes)} ({plan.total_bytes} bytes)",
        f"  Items preserved (scan): {plan.preserve_count}",
        "",
        "By root:",
    ]
    for root_label, counts in manifest.get("by_root", {}).items():
        lines.append(
            f"  {root_label}: {counts.get('files', 0)} files, "
            f"{counts.get('dirs', 0)} dirs, {_human_size(int(counts.get('bytes', 0)))}"
        )
    if plan.scan_errors:
        lines.extend(["", "Scan errors:"])
        lines.extend(f"  - {err}" for err in plan.scan_errors)
    lines.extend(["", "Entries:"])
    for e in plan.entries:
        kind = "DIR " if e.is_dir else "FILE"
        sz = _human_size(e.size_bytes) if not e.is_dir else "-"
        lines.append(f"  [{kind}] {e.rel_path}  ({e.reason})  {sz}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path


class CleanupLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._fp = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {message}"
        print(line)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def _chmod_writable(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _delete_path(path: Path, logger: CleanupLogger) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    _chmod_writable(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, onerror=_rmtree_onerror)
        else:
            path.unlink(missing_ok=True)
        logger.log(f"DELETED: {path}")
        return True
    except OSError as exc:
        logger.log(f"ERROR deleting {path}: {exc}")
        return False


def _rmtree_onerror(func, p, _exc_info) -> None:
    _chmod_writable(Path(p))
    try:
        func(p)
    except OSError:
        pass


def _confirm_execute(plan: DeletionPlan) -> bool:
    print()
    print("=" * 72)
    print("EXECUTE MODE - permanent deletion")
    print("=" * 72)
    print(
        f"  {plan.total_files} file(s) and {plan.total_dirs} folder(s) "
        f"({_human_size(plan.total_bytes)}) will be deleted."
    )
    print("  Scope: final_clean/, logs/batch_summaries/, temp_processed/ ONLY")
    print("  Production media without test markers is preserved.")
    print()
    print("Type exactly:  DELETE TESTING ARTIFACTS")
    try:
        typed = input("Confirmation: ").strip()
    except EOFError:
        print("Aborted (no input).")
        return False
    return typed == "DELETE TESTING ARTIFACTS"


def run_cleanup(
    *,
    project_root: Path,
    execute: bool = False,
) -> int:
    project_root = project_root.resolve()
    stamp = _utc_stamp()
    log_path = project_root / "logs" / f"cleanup_testing_artifacts_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CleanupLogger(log_path)

    logger.log(f"Starting cleanup_testing_artifacts (execute={execute})")
    logger.log(f"Project root: {project_root}")

    plan = build_deletion_plan(project_root)
    json_manifest, txt_manifest = _write_manifest_files(plan, project_root, stamp, execute=execute)
    logger.log(f"Manifest JSON: {json_manifest}")
    logger.log(f"Manifest text: {txt_manifest}")

    logger.log(
        f"Plan: {plan.total_files} files, {plan.total_dirs} dirs, "
        f"{_human_size(plan.total_bytes)} total; preserved scan count={plan.preserve_count}"
    )
    for err in plan.scan_errors:
        logger.log(f"SCAN ERROR: {err}")

    print()
    print("=" * 72)
    print("Testing artifacts cleanup - summary")
    print("=" * 72)
    print(f"  Project:     {project_root}")
    print(f"  Mode:        {'EXECUTE' if execute else 'DRY-RUN (no deletions)'}")
    print(f"  Files:       {plan.total_files}")
    print(f"  Directories: {plan.total_dirs}")
    print(f"  Size:        {_human_size(plan.total_bytes)} ({plan.total_bytes:,} bytes)")
    print(f"  Preserved:   {plan.preserve_count} items (classified keep)")
    print(f"  Manifest:    {txt_manifest}")
    print(f"  Log:         {log_path}")

    if not plan.entries:
        logger.log("Nothing to delete.")
        logger.close()
        print("\nNothing matched deletion rules.")
        return 0

    # Show sample paths
    print("\nSample paths (up to 25):")
    for e in plan.entries[:25]:
        kind = "DIR " if e.is_dir else "FILE"
        print(f"  [{kind}] {e.rel_path}  ({e.reason})")
    if len(plan.entries) > 25:
        print(f"  ... and {len(plan.entries) - 25} more (see manifest)")

    if not execute:
        logger.log("Dry-run complete; no files deleted.")
        logger.close()
        print("\nDry-run only. Re-run with --execute to delete after confirmation.")
        return 0

    if not _confirm_execute(plan):
        logger.log("Execute aborted by user.")
        logger.close()
        print("Aborted.")
        return 1

    logger.log("User confirmed; beginning deletion.")
    # Delete files first, then directories deepest-first
    files = [e for e in plan.entries if not e.is_dir]
    dirs = sorted([e for e in plan.entries if e.is_dir], key=lambda x: len(x.rel_path), reverse=True)

    ok_count = 0
    fail_count = 0
    for e in files:
        if _delete_path(project_root / e.rel_path, logger):
            ok_count += 1
        else:
            fail_count += 1

    for e in dirs:
        p = project_root / e.rel_path
        if not p.exists():
            logger.log(f"SKIP (already gone): {p}")
            ok_count += 1
            continue
        if _delete_path(p, logger):
            ok_count += 1
        else:
            fail_count += 1

    logger.log(f"Deletion finished: ok={ok_count} failed={fail_count}")
    logger.close()
    print(f"\nDeletion complete. ok={ok_count} failed={fail_count}")
    print(f"Log: {log_path}")
    return 0 if fail_count == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely remove testing/debug artifacts from final_clean/, "
            "logs/batch_summaries/, and temp_processed/ only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default is dry-run. Manifests are always written under reports/cleanup_plans/.\n"
            "Logs are written under logs/cleanup_testing_artifacts_YYYYMMDD_HHMM.log"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Project root (default: auto-detected)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion (requires typing DELETE TESTING ARTIFACTS)",
    )
    args = parser.parse_args()
    raise SystemExit(run_cleanup(project_root=args.project_root, execute=bool(args.execute)))


if __name__ == "__main__":
    main()
