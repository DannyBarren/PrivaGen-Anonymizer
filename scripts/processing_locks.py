"""
Batch processing locks for resume / idempotency across partial runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .utils import get_logger, utc_now_iso

logger = get_logger(__name__)

LOCK_FILENAME = ".processing.lock"
COMPLETE_FILENAME = ".processing.complete"
DEFAULT_STALE_SEC = 3600


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def write_processing_lock(
    batch_dir: Path,
    *,
    batch_index: int,
    stems: Iterable[str],
    source_paths: Optional[Iterable[str]] = None,
) -> Path:
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_index": int(batch_index),
        "stems": sorted({str(s) for s in stems}),
        "source_paths": list(source_paths or []),
        "started_at": utc_now_iso(),
        "pid": int(os.getpid()),
    }
    lock_path = batch_dir / LOCK_FILENAME
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("processing_lock_acquired", batch_dir=str(batch_dir), n=len(payload["stems"]))
    return lock_path


def mark_processing_complete(batch_dir: Path) -> None:
    batch_dir = Path(batch_dir)
    complete = batch_dir / COMPLETE_FILENAME
    complete.write_text(json.dumps({"completed_at": utc_now_iso()}), encoding="utf-8")
    lock = batch_dir / LOCK_FILENAME
    if lock.is_file():
        lock.unlink(missing_ok=True)
    logger.info("processing_lock_released", batch_dir=str(batch_dir))


def read_processing_lock(batch_dir: Path) -> Dict[str, Any]:
    lock = Path(batch_dir) / LOCK_FILENAME
    if not lock.is_file():
        return {}
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def is_lock_stale(lock_data: Dict[str, Any], *, stale_sec: int = DEFAULT_STALE_SEC) -> bool:
    if not lock_data:
        return True
    started = _parse_iso(str(lock_data.get("started_at", "")))
    if started is None:
        return True
    age = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
    return age > float(stale_sec)


def active_locked_stems(temp_processed: Path, *, stale_sec: int = DEFAULT_STALE_SEC) -> Set[str]:
    """Stems currently being processed (non-stale locks without .complete marker)."""
    temp_processed = Path(temp_processed)
    if not temp_processed.is_dir():
        return set()
    locked: Set[str] = set()
    for batch_dir in sorted(temp_processed.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        if (batch_dir / COMPLETE_FILENAME).is_file():
            continue
        lock_data = read_processing_lock(batch_dir)
        if not lock_data:
            continue
        if is_lock_stale(lock_data, stale_sec=stale_sec):
            logger.warning("processing_lock_stale", batch_dir=str(batch_dir))
            continue
        locked.update(str(s) for s in lock_data.get("stems") or [])
    return locked


def update_lock_completed_stems(batch_dir: Path, stems: Iterable[str]) -> None:
    """Append successfully processed stems to the batch lock for partial-batch resume."""
    batch_dir = Path(batch_dir)
    lock_path = batch_dir / LOCK_FILENAME
    data = read_processing_lock(batch_dir) if lock_path.is_file() else {}
    if not data:
        return
    done = set(str(s) for s in data.get("completed_stems") or [])
    done.update(str(s) for s in stems)
    data["completed_stems"] = sorted(done)
    data["updated_at"] = utc_now_iso()
    lock_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def filter_unfinished_paths(batch_dir: Path, paths: Sequence[Path]) -> List[Path]:
    """Skip stems that already have a sale output in this batch folder (partial resume)."""
    batch_dir = Path(batch_dir)
    lock_data = read_processing_lock(batch_dir)
    completed = set(str(s) for s in lock_data.get("completed_stems") or [])
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.stem in completed:
            continue
        if (batch_dir / f"{path.stem}.jpg").is_file():
            completed.add(path.stem)
            continue
        out.append(path)
    if completed and lock_data:
        update_lock_completed_stems(batch_dir, completed)
    return out


def discover_stale_partial_batches(temp_processed: Path, *, stale_sec: int = DEFAULT_STALE_SEC) -> List[Path]:
    """Stale locked batches with some outputs — candidates to re-process remaining stems."""
    temp_processed = Path(temp_processed)
    out: List[Path] = []
    if not temp_processed.is_dir():
        return out
    for batch_dir in sorted(temp_processed.glob("batch_*")):
        if not batch_dir.is_dir() or (batch_dir / COMPLETE_FILENAME).is_file():
            continue
        lock_data = read_processing_lock(batch_dir)
        if not lock_data or not is_lock_stale(lock_data, stale_sec=stale_sec):
            continue
        if any(batch_dir.glob("*.jpg")) or any(batch_dir.glob("*.error_audit.json")):
            out.append(batch_dir)
    return out


def discover_resume_batch_dirs(temp_processed: Path, *, stale_sec: int = DEFAULT_STALE_SEC) -> List[Path]:
    """
    Batch folders with outputs but no completion marker — candidates for QA-only resume.
    """
    temp_processed = Path(temp_processed)
    out: List[Path] = []
    if not temp_processed.is_dir():
        return out
    for batch_dir in sorted(temp_processed.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        if (batch_dir / COMPLETE_FILENAME).is_file():
            continue
        if not any(batch_dir.glob("*.jpg")):
            continue
        lock_data = read_processing_lock(batch_dir)
        if lock_data and not is_lock_stale(lock_data, stale_sec=stale_sec):
            out.append(batch_dir)
    return out
