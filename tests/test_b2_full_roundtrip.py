"""
Full B2 round-trip: ingest 100 → process + QA → export → verify destination integrity.

Uses mocked rclone when unavailable; set B2_* env + RCLONE_AUTO_CONFIRM=1 for live runs.

Run:
    python -m tests.test_b2_full_roundtrip
    python -m tests.test_b2_full_roundtrip --count 100 --mock-only
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

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main_pipeline import run_pipeline
from scripts.rclone_integration import (
    REMOTE_CRYPT,
    REMOTE_READONLY,
    REMOTE_WRITE,
    B2Config,
    EXPORT_EXTENSIONS,
    compute_directory_hashes,
    verify_export_integrity,
    verify_remote_against_manifest,
    write_rclone_config,
)
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_audit_json, load_config, resolve_pipeline_paths, setup_project_folders, sha256_file


def _seed_source_bucket(staging: Path, count: int) -> Path:
    src = staging / "source-bucket" / "datasets" / "raw"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        path = src / f"b2_roundtrip_{i:04d}.png"
        if path.is_file():
            continue
        im = Image.new("RGB", (320, 240), color=(i * 3 % 255, 60, 90))
        if i % 5 == 0:
            draw = ImageDraw.Draw(im)
            draw.text((20, 100), f"RT{i:04d}", fill=(255, 255, 255))
        im.save(path)
    return src


def _parse_rclone_argv(argv: List[str]) -> List[str]:
    args = list(argv)
    if args and (args[0].endswith("rclone") or args[0] == "rclone"):
        args = args[1:]
    if len(args) >= 2 and args[0] == "--config":
        args = args[2:]
    return args


def _mock_rclone_side_effect(staging: Path):
    def _remote_mirror(remote_uri: str) -> Path:
        if remote_uri.startswith(f"{REMOTE_CRYPT}:"):
            sub = remote_uri.split(":", 1)[1].strip("/")
            return staging / "dest-bucket" / sub
        bucket, _, prefix = remote_uri.split(":", 1)[1].partition("/")
        return staging / bucket / prefix

    def _side_effect(cmd, *args, **kwargs):
        argv = list(cmd)
        sub = _parse_rclone_argv(argv)
        action = sub[0] if sub else ""

        if action == "lsf":
            remote_uri = sub[-1]
            src = _remote_mirror(remote_uri)
            lines = []
            if src.is_dir():
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        lines.append(p.relative_to(src).as_posix())
            return subprocess.CompletedProcess(argv, 0, stdout="\n".join(lines) + "\n", stderr="")

        if action == "check":
            local_path = Path(sub[1])
            remote_uri = sub[2]
            remote_root = _remote_mirror(remote_uri)
            local_hashes = compute_directory_hashes(local_path, extensions=EXPORT_EXTENSIONS)
            remote_hashes = compute_directory_hashes(remote_root, extensions=EXPORT_EXTENSIONS)
            mismatched = [
                k for k in local_hashes if k not in remote_hashes or local_hashes[k] != remote_hashes[k]
            ]
            if mismatched:
                err_lines = "\n".join(f"ERROR : {m}: hash mismatch" for m in mismatched[:20])
                return subprocess.CompletedProcess(argv, 1, stdout=err_lines + "\n", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout="0 differences found\n", stderr="")

        if action == "copy":
            src_token, dst_token = sub[1], sub[2]
            dry_run = "--dry-run" in sub
            if src_token.startswith(f"{REMOTE_READONLY}:"):
                if dry_run:
                    return subprocess.CompletedProcess(argv, 0, stdout="Dry run ingest\n", stderr="")
                src = _remote_mirror(src_token)
                dst = Path(dst_token)
                dst.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    for p in src.rglob("*"):
                        if p.is_file():
                            rel = p.relative_to(src)
                            target = dst / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(p, target)
            elif dst_token.startswith(f"{REMOTE_WRITE}:") or dst_token.startswith(f"{REMOTE_CRYPT}:"):
                if dry_run:
                    return subprocess.CompletedProcess(argv, 0, stdout="Dry run export\n", stderr="")
                dst = _remote_mirror(dst_token)
                dst.mkdir(parents=True, exist_ok=True)
                src = Path(src_token)
                for p in src.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(src)
                        target = dst / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, target)
            return subprocess.CompletedProcess(argv, 0, stdout="Transferred\n", stderr="")

        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=f"unsupported: {action}")

    return _side_effect


def _test_env(rclone_config: Path) -> dict:
    return {
        "B2_KEY_ID": "test-key-id",
        "B2_READONLY_KEY": "readonly-secret",
        "B2_WRITE_KEY": "write-secret",
        "B2_READONLY_BUCKET": "source-bucket",
        "B2_WRITE_BUCKET": "dest-bucket",
        "B2_INGEST_REMOTE_PATH": "datasets/raw",
        "B2_EXPORT_REMOTE_PATH": "datasets/anonymized",
        "RCLONE_CONFIG": str(rclone_config),
        "RCLONE_AUTO_CONFIRM": "1",
    }


def run_full_roundtrip(*, count: int = 200, security_level: str = "full", crypt_enabled: bool = False) -> dict:
    count = max(200, int(count))
    errors: list[str] = []
    ingested: list = []
    final_jpgs: list = []
    ingest_record: dict = {}
    export_record: dict = {}
    manifest: dict = {}
    result: dict = {}

    with tempfile.TemporaryDirectory(prefix="b2_full_rt_") as tmp:
        test_root = Path(tmp)
        staging = test_root / "remote"
        _seed_source_bucket(staging, count)

        b2_cfg = B2Config(
            key_id="test-key-id",
            readonly_key="readonly-secret",
            write_key="write-secret",
            readonly_bucket="source-bucket",
            write_bucket="dest-bucket",
            rclone_config=test_root / "rclone.conf",
            ingest_remote_path="datasets/raw",
            export_remote_path="datasets/anonymized",
            transfer_batch_size=16,
            max_transfer_retries=3,
            require_confirm=False,
            transfers=24,
            checkers=64,
            upload_concurrency=12,
            chunk_size="48M",
            bwlimit="0",
            verify_after_export=True,
            crypt_enabled=crypt_enabled,
            crypt_password="test-crypt-password" if crypt_enabled else "",
            crypt_password2="test-crypt-password2" if crypt_enabled else "",
        )
        write_rclone_config(b2_cfg)

        cfg_base = load_config(ROOT / "config.yaml")
        overrides = {
            "paths": {
                "input_raw": str(test_root / "input_raw"),
                "temp_processed": str(test_root / "temp_processed"),
                "final_clean": str(test_root / "final_clean"),
                "quarantine": str(test_root / "quarantine"),
                "manual_review": str(test_root / "manual_review"),
                "logs": str(test_root / "logs"),
                "reports": str(test_root / "reports"),
            },
            "batch_size": 8,
            "max_qa_waves": 12,
            "device": "cpu",
            "insightface": {"enabled": False},
            "lama": {"backend": "simple_lama", "device": "cpu"},
            "qa": {
                "reuse_processing_ocr": True,
                "text_det_score_fail": 0.95,
                "artifact_ssim_min": 0.55,
                "edge_artifact_ratio_max": 0.25,
            },
            "backblaze": {
                "source_bucket": "source-bucket",
                "dest_bucket": "dest-bucket",
                "ingest_remote_path": "datasets/raw",
                "export_remote_path": "datasets/anonymized",
                "require_confirm_real_transfer": False,
                "verify_after_export": True,
                "transfers": 24,
                "checkers": 64,
                "upload_concurrency": 12,
                "chunk_size": "48M",
                "bwlimit": "0",
            },
            "security": {
                "level": security_level,
                "verify_ingest_checksums": True,
                "enforce_readonly_ingest": True,
                "backup_before_cleanup": True,
                "secure_wipe": security_level == "full",
                "redact_logs": True,
                "crypt_enabled": crypt_enabled,
            },
        }
        cfg = dict(cfg_base)
        deep_update(cfg, overrides)
        setup_project_folders(test_root, cfg)
        paths = resolve_pipeline_paths(test_root, cfg)

        import yaml

        config_path = test_root / "config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

        side_effect = _mock_rclone_side_effect(staging)
        export_prefix = "roundtrip-export"
        ingested: list = []
        final_jpgs: list = []
        ingest_record: dict = {}
        export_record: dict = {}
        manifest: dict = {}
        result: dict = {}
        ingest_manifests: list = []
        export_manifests: list = []

        with mock.patch.dict(os.environ, _test_env(b2_cfg.rclone_config), clear=False):
            with mock.patch("scripts.rclone_integration._resolve_rclone_binary", return_value="rclone"), mock.patch(
                "scripts.rclone_integration.subprocess.run", side_effect=side_effect
            ):
                result = run_pipeline(
                    config_path=config_path,
                    project_root=test_root,
                    test_mode=True,
                    security_level=security_level,
                    ingest_from_b2=b2_cfg.ingest_remote_path,
                    export_to_b2=export_prefix,
                    verify_after_export=True,
                )
                close_pipeline_logging()

                ingested = discover_images(paths["input_raw"], cfg.get("image_extensions", [".png"]))
                if len(ingested) < count:
                    errors.append(f"ingest: expected >={count}, got {len(ingested)}")

                final_jpgs = sorted(paths["final_clean"].glob("*.jpg"))
                if len(final_jpgs) < count * 0.95:
                    errors.append(f"pipeline: final_clean {len(final_jpgs)}/{count} (<95%)")

                for jpg in final_jpgs[:10]:
                    js = jpg.with_suffix(".json")
                    if not js.is_file():
                        errors.append(f"missing sidecar: {jpg.name}")
                        continue
                    audit = load_audit_json(jpg)
                    ih = audit.get("integrity_hashes") or {}
                    if ih.get("output_sha256") != sha256_file(jpg):
                        errors.append(f"hash mismatch: {jpg.name}")
                    if not audit.get("qa", {}).get("final_decision"):
                        errors.append(f"no qa decision: {jpg.name}")

                ingest_manifests = sorted((test_root / "reports").glob("transfer_manifest_ingest_*.json"))
                export_manifests = sorted((test_root / "reports").glob("transfer_manifest_export_*.json"))
                if not ingest_manifests:
                    errors.append("missing ingest transfer manifest")
                if not export_manifests:
                    errors.append("missing export transfer manifest")

                if export_manifests:
                    manifest = json.loads(export_manifests[-1].read_text(encoding="utf-8"))
                if ingest_manifests:
                    ingest_record = {"manifest": json.loads(ingest_manifests[-1].read_text(encoding="utf-8"))}

                remote_mirror = staging / "dest-bucket" / export_prefix
                remote_verify = verify_remote_against_manifest(
                    export_prefix,
                    manifest,
                    cfg=b2_cfg,
                    use_write_remote=True,
                )
                if not remote_verify.get("ok"):
                    errors.append(f"remote verify: missing={remote_verify.get('missing')[:5]}")

                integrity = verify_export_integrity(
                    paths["final_clean"],
                    remote_mirror,
                    extensions=EXPORT_EXTENSIONS,
                )
                if not integrity.get("ok"):
                    errors.append(
                        f"integrity: missing={len(integrity.get('missing_on_remote') or [])} "
                        f"mismatch={len(integrity.get('hash_mismatched') or [])}"
                    )

                if not (test_root / "reports" / "security_report.md").is_file():
                    errors.append("missing security_report.md")
                else:
                    sec_text = (test_root / "reports" / "security_report.md").read_text(encoding="utf-8")
                    if "Rclone Integration" not in sec_text:
                        errors.append("security_report missing Rclone Integration section")

                export_payload = result.get("b2_export") or {}
                check = export_payload.get("post_export_check") or {}
                if not check.get("ok"):
                    errors.append(f"post_export_check failed: {check}")

                perf = (export_payload.get("manifest") or {}).get("performance") or {}
                if perf.get("transfers") != 24:
                    errors.append(f"unexpected rclone transfers flag: {perf}")
                if not (test_root / "reports" / "processed_manifest.json").is_file():
                    errors.append("missing processed_manifest.json")
                backups = test_root / "backups"
                if security_level == "full" and (not backups.is_dir() or not any(backups.iterdir())):
                    errors.append("missing pre-cleanup backup directory")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "count": count,
            "ingested": len(ingested),
            "final_clean": len(final_jpgs),
            "ingest_manifest_files": len((ingest_record.get("manifest") or {}).get("files") or []),
            "export_manifest_files": manifest.get("file_count"),
            "pipeline_pass_rate": (result.get("stats") or {}).get("success_rate"),
            "stopped_early": result.get("stopped_early"),
            "security_report": str(test_root / "reports" / "security_report.md"),
            "security_level": security_level,
            "crypt_enabled": crypt_enabled,
            "post_export_check_ok": (result.get("b2_export") or {}).get("post_export_check", {}).get("ok"),
            "export_performance": (result.get("b2_export") or {}).get("manifest", {}).get("performance"),
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--crypt", action="store_true", help="Enable rclone crypt export mock")
    args = parser.parse_args()
    os.environ.setdefault("RCLONE_AUTO_CONFIRM", "1")
    summary = run_full_roundtrip(count=args.count, crypt_enabled=bool(args.crypt))
    print(json.dumps(summary, indent=2))
    if not summary.get("passed"):
        for e in summary["errors"]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"PASSED B2 full roundtrip ({summary['count']} images)")


if __name__ == "__main__":
    main()
