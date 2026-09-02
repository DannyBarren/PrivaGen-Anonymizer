"""
Roundtrip test for Backblaze B2 rclone integration (10 files).

Uses mocked rclone when the binary is not installed; runs live transfers when
B2_* env vars and rclone are available.

Run from project root:
    python -m scripts.test_rclone_roundtrip
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from scripts.rclone_integration import (
    REMOTE_READONLY,
    REMOTE_WRITE,
    B2Config,
    compute_directory_hashes,
    export_to_backblaze,
    ingest_from_backblaze,
    load_b2_config,
    write_rclone_config,
)


def _seed_images(directory: Path, count: int = 10) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        path = directory / f"roundtrip_{i:03d}.png"
        im = Image.new("RGB", (64, 48), color=(i * 20 % 255, 40, 80))
        im.save(path)


def _rclone_available() -> bool:
    return shutil.which(os.environ.get("RCLONE_BINARY", "rclone") or "rclone") is not None


def _b2_env_present() -> bool:
    required = (
        "B2_KEY_ID",
        "B2_READONLY_KEY",
        "B2_WRITE_KEY",
        "B2_READONLY_BUCKET",
        "B2_WRITE_BUCKET",
    )
    return all((os.environ.get(k) or "").strip() for k in required)


def _parse_rclone_argv(argv: List[str]) -> List[str]:
    args = list(argv)
    if args and (args[0].endswith("rclone") or args[0] == "rclone"):
        args = args[1:]
    if len(args) >= 2 and args[0] == "--config":
        args = args[2:]
    return args


def _mock_rclone_side_effect(staging_remote: Path):
    """Simulate rclone copy/lsf by copying between local dirs mapped to remotes."""

    def _side_effect(cmd, *args, **kwargs):
        argv = list(cmd)
        sub = _parse_rclone_argv(argv)
        action = sub[0] if sub else ""

        if action == "lsf":
            remote_uri = sub[-1]
            bucket, _, prefix = remote_uri.split(":", 1)[1].partition("/")
            src = staging_remote / bucket / prefix
            lines = []
            if src.is_dir():
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        lines.append(p.relative_to(src).as_posix())
            return subprocess.CompletedProcess(argv, 0, stdout="\n".join(lines) + "\n", stderr="")

        if action == "copy":
            src_token, dst_token = sub[1], sub[2]
            if src_token.startswith(f"{REMOTE_READONLY}:"):
                bucket, _, prefix = src_token.split(":", 1)[1].partition("/")
                src = staging_remote / bucket / prefix
                dst = Path(dst_token)
                dst.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    for p in src.rglob("*"):
                        if p.is_file():
                            rel = p.relative_to(src)
                            target = dst / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(p, target)
            elif dst_token.startswith(f"{REMOTE_WRITE}:"):
                bucket, _, prefix = dst_token.split(":", 1)[1].partition("/")
                dst = staging_remote / bucket / prefix
                dst.mkdir(parents=True, exist_ok=True)
                src = Path(src_token)
                for p in src.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(src)
                        target = dst / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, target)
            else:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown copy direction")
            return subprocess.CompletedProcess(argv, 0, stdout="Transferred\n", stderr="")

        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=f"unsupported: {action}")

    return _side_effect


def run_mock_roundtrip(count: int = 10) -> dict:
    with tempfile.TemporaryDirectory(prefix="b2_mock_") as tmp:
        tmp_path = Path(tmp)
        staging = tmp_path / "remote"
        ingest_src = staging / "readonly-bucket" / "datasets" / "raw"
        _seed_images(ingest_src, count)

        local_input = tmp_path / "input_raw"
        local_final = tmp_path / "final_clean"
        _seed_images(local_final, 3)

        cfg = B2Config(
            key_id="test-key-id",
            readonly_key="readonly-secret",
            write_key="write-secret",
            readonly_bucket="readonly-bucket",
            write_bucket="write-bucket",
            rclone_config=tmp_path / "rclone.conf",
            ingest_remote_path="datasets/raw",
            export_remote_path="datasets/anonymized",
        )
        write_rclone_config(cfg)

        side_effect = _mock_rclone_side_effect(staging)
        with mock.patch(
            "scripts.rclone_integration._resolve_rclone_binary", return_value="rclone"
        ), mock.patch("scripts.rclone_integration.subprocess.run", side_effect=side_effect):
            ingest_record = ingest_from_backblaze(
                "datasets/raw",
                local_input,
                batch_size=4,
                cfg=cfg,
                dry_run=False,
                confirm=False,
                project_root=tmp_path,
            )
            export_record = export_to_backblaze(
                local_final,
                "datasets/anonymized",
                cfg=cfg,
                dry_run=False,
                confirm=False,
                project_root=tmp_path,
            )

        ingested = compute_directory_hashes(local_input)
        exported_remote = staging / "write-bucket" / "datasets" / "anonymized"
        remote_hashes = compute_directory_hashes(exported_remote)
        log_path = tmp_path / "logs" / "rclone_transfers.jsonl"

        return {
            "mode": "mock",
            "ingested_count": len(ingested),
            "exported_remote_count": len(remote_hashes),
            "ingest_new_files": len((ingest_record.get("manifest") or {}).get("files") or []),
            "ingest_hashes_logged": bool(ingest_record.get("manifest")),
            "export_hashes_logged": bool(export_record.get("manifest")),
            "transfer_log_exists": log_path.is_file(),
            "transfer_log_lines": len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.is_file() else 0,
            "passed": len(ingested) == count and len(remote_hashes) == 3,
        }


def run_live_roundtrip(count: int = 10) -> dict:
    cfg = load_b2_config()
    write_rclone_config(cfg)

    with tempfile.TemporaryDirectory(prefix="b2_live_") as tmp:
        tmp_path = Path(tmp)
        local_input = tmp_path / "input_raw"
        local_final = tmp_path / "final_clean"
        test_prefix = f"roundtrip-test/{os.getpid()}"

        with mock.patch.dict(os.environ, {"RCLONE_AUTO_CONFIRM": "1"}, clear=False):
            ingest_record = ingest_from_backblaze(
                cfg.ingest_remote_path,
                local_input,
                cfg=cfg,
                dry_run=True,
                confirm=False,
                project_root=ROOT,
            )
            ingest_record = ingest_from_backblaze(
                cfg.ingest_remote_path,
                local_input,
                batch_size=min(4, count),
                cfg=cfg,
                dry_run=False,
                confirm=False,
                project_root=ROOT,
            )
        _seed_images(local_final, min(3, count))

        export_dest = f"{cfg.export_remote_path}/{test_prefix}"
        export_record = export_to_backblaze(
            local_final,
            export_dest,
            cfg=cfg,
            dry_run=False,
            confirm=False,
            project_root=ROOT,
        )

        ingested = compute_directory_hashes(local_input)
        return {
            "mode": "live",
            "ingested_count": len(ingested),
            "ingest_hashes_logged": bool(ingest_record.get("transfer_hashes")),
            "export_hashes_logged": bool(export_record.get("transfer_hashes")),
            "export_remote": export_dest,
            "passed": bool(export_record.get("transfer_hashes")),
        }


def main() -> None:
    count = 10
    if _rclone_available() and _b2_env_present():
        summary = run_live_roundtrip(count)
    else:
        summary = run_mock_roundtrip(count)

    print(json.dumps(summary, indent=2))
    if not summary.get("passed"):
        print("FAILED roundtrip test", file=sys.stderr)
        sys.exit(1)
    print(f"PASSED ({summary['mode']} mode)")


if __name__ == "__main__":
    main()
