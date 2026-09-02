"""Unit tests for security_hardening helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.security_hardening import (
    SecurityContext,
    backup_critical_artifacts,
    load_security_hardening,
    update_processed_manifest,
    write_security_report,
)


def test_processed_manifest_and_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = load_security_hardening({"security": {"level": "full"}}, security_level="full")
        ctx = SecurityContext(project_root=root, config=cfg)
        manifest_path = ctx.processed_manifest_path()
        update_processed_manifest(
            manifest_path,
            [("img001.png", "abc123"), ("img002.png", "def456")],
            ingest_timestamp="2026-01-01T00:00:00Z",
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["entries"]["img001"]["original_hash"] == "abc123"
        assert data["entries"]["img001"]["ingest_timestamp"] == "2026-01-01T00:00:00Z"

        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "processed_manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        backup = backup_critical_artifacts(root, backup_root=root / "backups")
        assert (backup / "processed_manifest.json").is_file()

        ctx.log_event("test_event", detail="ok")
        report = write_security_report(root, cfg, ctx.events, summary={"ok": True})
        assert report.is_file()
        assert "Security Report" in report.read_text(encoding="utf-8")
