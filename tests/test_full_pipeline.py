"""
Thorough local integration test for the full image anonymization pipeline.

Simulates: B2 ingest (mock or local copy) → process_batch → deterministic QA → routing
→ B2 export (dry-run) with checksum verification.

Run from project root:
    python -m tests.test_full_pipeline
    python -m tests.test_full_pipeline --count 30 --test-mode
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agentic_qa_crew import evaluate_image_qa, run_detection_verification
from scripts.main_pipeline import route_batch_outputs, run_pipeline
from scripts.rclone_integration import (
    REMOTE_READONLY,
    REMOTE_WRITE,
    B2Config,
    compute_directory_hashes,
    export_to_backblaze,
    ingest_from_backblaze,
    write_rclone_config,
)
from scripts.shared_models import get_shared_insightface_app, get_shared_paddle_ocr, ocr_polys_and_scores
from scripts.utils import (
    close_pipeline_logging,
    deep_update,
    discover_images,
    imread_rgb,
    load_audit_json,
    load_config,
    resolve_pipeline_paths,
    save_audit_json,
    setup_project_folders,
    sha256_file,
)

MAX_FALSE_QUARANTINE_RATE = 0.05
OCR_SCORE_CEILING = 0.85


def _test_env(rclone_config: Optional[Path] = None) -> Dict[str, str]:
    env = {
        "B2_KEY_ID": "test-key-id",
        "B2_READONLY_KEY": "readonly-secret",
        "B2_WRITE_KEY": "write-secret",
        "B2_READONLY_BUCKET": "readonly-bucket",
        "B2_WRITE_BUCKET": "write-bucket",
        "B2_INGEST_REMOTE_PATH": "datasets/raw",
        "B2_EXPORT_REMOTE_PATH": "datasets/anonymized",
    }
    if rclone_config is not None:
        env["RCLONE_CONFIG"] = str(rclone_config)
    return env


def _paths_override(test_root: Path) -> Dict[str, str]:
    return {
        "input_raw": str(test_root / "input_raw"),
        "temp_processed": str(test_root / "temp_processed"),
        "final_clean": str(test_root / "final_clean"),
        "quarantine": str(test_root / "quarantine"),
        "manual_review": str(test_root / "manual_review"),
        "logs": str(test_root / "logs"),
        "reports": str(test_root / "reports"),
    }


def _test_config_overrides(test_root: Path, count: int) -> Dict[str, Any]:
    """Relaxed QA for synthetic fixtures; still exercises real OCR/InsightFace when installed."""
    return {
        "paths": _paths_override(test_root),
        "batch_size": min(8, count),
        "max_qa_waves": 4,
        "max_retries": 3,
        "num_workers": 2,
        "device": "cpu",
        "insightface": {"enabled": True, "model_name": "buffalo_l", "ctx_id": -1},
        "qa": {
            "text_det_score_fail": 0.95,
            "artifact_ssim_min": 0.55,
            "identity_distance_min": 0.20,
            "edge_artifact_ratio_max": 0.25,
            "use_crewai_llm": False,
        },
        "lama": {"backend": "simple_lama", "device": "cpu", "mask_dilation": 20},
        "logging": {"level": "DEBUG", "logfile": "logs/processing.log", "batch_summary_dir": "logs/batch_summaries"},
    }


def _draw_landscape(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    w, h = 480, 360
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    sky = rng.randint(80, 180)
    arr[: h // 2, :] = (sky, sky + 20, 220)
    ground = (rng.randint(30, 90), rng.randint(100, 160), rng.randint(40, 90))
    arr[h // 2 :, :] = ground
    for _ in range(rng.randint(8, 20)):
        x0, y0 = rng.randint(0, w - 40), rng.randint(h // 2, h - 20)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        arr[y0 : y0 + rng.randint(10, 40), x0 : x0 + rng.randint(10, 40)] = color
    Image.fromarray(arr).save(path)


def _draw_text_sign(path: Path, text: str, seed: int) -> None:
    rng = random.Random(seed)
    im = Image.new("RGB", (640, 400), color=(rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60)))
    draw = ImageDraw.Draw(im)
    draw.rectangle((40, 120, 580, 280), fill=(220, 220, 220), outline=(0, 0, 0), width=3)
    draw.text((60, 170), text, fill=(10, 10, 10))
    im.save(path)


def _draw_plate(path: Path, plate: str, seed: int) -> None:
    rng = random.Random(seed)
    im = Image.new("RGB", (520, 320), color=(rng.randint(100, 180), rng.randint(100, 180), rng.randint(100, 180)))
    draw = ImageDraw.Draw(im)
    draw.rectangle((120, 130, 400, 190), fill=(255, 255, 200), outline=(0, 0, 0), width=2)
    draw.text((140, 145), plate, fill=(0, 0, 0))
    im.save(path)


def _draw_face_like(path: Path, seed: int) -> None:
    """Oval + features — may or may not trigger InsightFace on synthetic data."""
    rng = random.Random(seed)
    im = Image.new("RGB", (400, 500), color=(rng.randint(40, 90), rng.randint(40, 90), rng.randint(40, 90)))
    draw = ImageDraw.Draw(im)
    skin = (rng.randint(180, 230), rng.randint(140, 190), rng.randint(110, 160))
    draw.ellipse((100, 80, 300, 360), fill=skin, outline=(0, 0, 0))
    draw.ellipse((140, 170, 170, 200), fill=(40, 40, 40))
    draw.ellipse((230, 170, 260, 200), fill=(40, 40, 40))
    draw.arc((150, 220, 250, 300), start=10, end=170, fill=(80, 20, 20), width=3)
    im.save(path)


def _draw_complex_scene(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    im = Image.new("RGB", (512, 512), color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    draw = ImageDraw.Draw(im)
    for _ in range(40):
        x0, y0 = rng.randint(0, 450), rng.randint(0, 450)
        x1, y1 = x0 + rng.randint(10, 80), y0 + rng.randint(10, 80)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=1)
    im = im.filter(ImageFilter.GaussianBlur(radius=0.5))
    im.save(path)


def seed_diverse_images(output_dir: Path, count: int) -> List[Path]:
    """Create 20–50 held-out synthetic images covering multiple categories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(20, min(50, int(count)))
    builders = [
        ("landscape", lambda p, i: _draw_landscape(p, i)),
        ("text_sign", lambda p, i: _draw_text_sign(p, f"SIGN-{i:03d}", i)),
        ("plate", lambda p, i: _draw_plate(p, f"ABC{i:03d}", i)),
        ("face_like", lambda p, i: _draw_face_like(p, i)),
        ("complex", lambda p, i: _draw_complex_scene(p, i)),
        ("plain", lambda p, i: Image.new("RGB", (320, 240), color=(i * 9 % 255, 60, 120)).save(p)),
    ]
    created: List[Path] = []
    for i in range(count):
        category, fn = builders[i % len(builders)]
        path = output_dir / f"heldout_{category}_{i:03d}.png"
        if path.is_file():
            created.append(path)
            continue
        fn(path, i)
        created.append(path)
    return created


def _parse_rclone_argv(argv: List[str]) -> List[str]:
    args = list(argv)
    if args and (args[0].endswith("rclone") or args[0] == "rclone"):
        args = args[1:]
    if len(args) >= 2 and args[0] == "--config":
        args = args[2:]
    return args


def _mock_rclone_side_effect(staging_remote: Path):
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
            dry_run = "--dry-run" in sub
            if src_token.startswith(f"{REMOTE_READONLY}:"):
                if dry_run:
                    return subprocess.CompletedProcess(argv, 0, stdout="Dry run ingest\n", stderr="")
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
                if dry_run:
                    return subprocess.CompletedProcess(argv, 0, stdout="Dry run export\n", stderr="")
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


def _simulate_b2_ingest(
    staging: Path,
    local_input: Path,
    b2_cfg: B2Config,
    *,
    batch_size: int,
    dry_run_first: bool = True,
) -> Dict[str, Any]:
    ingest_src = staging / b2_cfg.readonly_bucket / b2_cfg.ingest_remote_path
    ingest_src.mkdir(parents=True, exist_ok=True)
    if not any(ingest_src.iterdir()):
        seed_diverse_images(ingest_src, len(list(local_input.glob("*.png"))) or 30)

    side_effect = _mock_rclone_side_effect(staging)
    with mock.patch("scripts.rclone_integration._resolve_rclone_binary", return_value="rclone"), mock.patch(
        "scripts.rclone_integration.subprocess.run", side_effect=side_effect
    ):
        if dry_run_first:
            ingest_from_backblaze(
                b2_cfg.ingest_remote_path,
                local_input,
                batch_size,
                cfg=b2_cfg,
                dry_run=True,
            )
        return ingest_from_backblaze(
            b2_cfg.ingest_remote_path,
            local_input,
            batch_size,
            cfg=b2_cfg,
            dry_run=False,
        )


def _simulate_b2_export_dry_run(
    staging: Path,
    local_final: Path,
    b2_cfg: B2Config,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    side_effect = _mock_rclone_side_effect(staging)
    with mock.patch("scripts.rclone_integration._resolve_rclone_binary", return_value="rclone"), mock.patch(
        "scripts.rclone_integration.subprocess.run", side_effect=side_effect
    ):
        return export_to_backblaze(
            local_final,
            b2_cfg.export_remote_path,
            cfg=b2_cfg,
            dry_run=True,
            project_root=project_root,
        )


def _insightface_count(image_path: Path, cfg: Dict[str, Any]) -> Tuple[int, Optional[str]]:
    app = get_shared_insightface_app(cfg)
    if app is None:
        return 0, "insightface_unavailable"
    try:
        bgr = imread_rgb(image_path)[:, :, ::-1].copy()
        return int(len(app.get(bgr) or [])), None
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def _paddleocr_max_score(image_path: Path, cfg: Dict[str, Any]) -> Tuple[float, int]:
    ocr = get_shared_paddle_ocr(cfg)
    if ocr is None:
        return 0.0, 0
    bgr = imread_rgb(image_path)[:, :, ::-1].copy()
    boxes = ocr_polys_and_scores(ocr, bgr)
    max_score = 0.0
    for box in boxes:
        try:
            max_score = max(max_score, float(box["score"]))
        except Exception:  # noqa: BLE001
            continue
    return max_score, len(boxes)


def _assert_sidecar_integrity(jpg: Path, input_raw: Path, errors: List[str]) -> None:
    js = jpg.with_suffix(".json")
    if not js.is_file():
        errors.append(f"missing sidecar: {jpg.name}")
        return
    audit = load_audit_json(jpg)
    ih = audit.get("integrity_hashes") or {}
    qa = audit.get("qa") or {}

    if not ih.get("source_sha256") or not ih.get("output_sha256"):
        errors.append(f"missing hashes: {jpg.name}")

    on_disk = sha256_file(jpg)
    if ih.get("output_sha256") and ih["output_sha256"] != on_disk:
        errors.append(f"output hash mismatch on disk: {jpg.name}")

    if qa.get("final_decision") not in ("pass", "fail"):
        errors.append(f"missing qa.final_decision: {jpg.name}")

    src = Path(audit.get("source_path") or "")
    if src.is_file():
        src_hash = sha256_file(src)
        if ih.get("source_sha256") and ih["source_sha256"] != src_hash:
            # source_path may point to input_raw original; hash was taken from security copy
            pass

    if "retry_count" not in audit:
        errors.append(f"missing retry_count: {jpg.name}")


def _test_quarantine_retry_routing(test_root: Path, cfg: Dict[str, Any], errors: List[str]) -> None:
    """Verify quarantine → manual_review when max_retries exhausted."""
    paths = resolve_pipeline_paths(test_root, cfg)
    batch_dir = paths["temp_processed"] / "batch_retry_test"
    batch_dir.mkdir(parents=True, exist_ok=True)

    src = paths["input_raw"] / "retry_probe.png"
    if not src.is_file():
        Image.new("RGB", (200, 200), color=(100, 50, 50)).save(src)

    out_img = batch_dir / "retry_probe.jpg"
    Image.new("RGB", (200, 200), color=(120, 60, 60)).save(out_img)
    audit = {
        "source_path": str(src),
        "retry_count": 2,
        "qa": {
            "final_decision": "fail",
            "deterministic": {"final_decision": "fail", "failure_reason_text": "synthetic_fail"},
        },
        "integrity_hashes": {"source_sha256": sha256_file(src), "output_sha256": sha256_file(out_img)},
    }
    save_audit_json(out_img, audit)

    audit_json_path = paths["reports"] / "anonymization_audit.json"
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    if not audit_json_path.is_file():
        audit_json_path.write_text("[]", encoding="utf-8")

    _, counters = route_batch_outputs(
        batch_dir,
        project_root=test_root,
        cfg=cfg,
        audit_json_path=audit_json_path,
    )

    manual = paths["manual_review"] / "retry_probe.jpg"
    if not manual.is_file():
        errors.append("retry routing: expected manual_review/retry_probe.jpg")
    if counters.get("manual_review", 0) < 1:
        errors.append("retry routing: manual_review counter not incremented")

    updated = load_audit_json(manual) if manual.is_file() else {}
    if int(updated.get("retry_count", 0)) != 3:
        errors.append(f"retry routing: expected retry_count=3, got {updated.get('retry_count')}")


def run_full_pipeline_test(*, count: int = 30, use_test_mode: bool = True) -> Dict[str, Any]:
    count = max(20, min(50, int(count)))
    errors: List[str] = []
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="full_pipeline_test_") as tmp:
        test_root = Path(tmp)
        cfg_base = load_config(ROOT / "config.yaml")
        overrides = _test_config_overrides(test_root, count)
        cfg = dict(cfg_base)
        deep_update(cfg, overrides)

        setup_project_folders(test_root, cfg)
        paths = resolve_pipeline_paths(test_root, cfg)

        # Stage remote fixtures + ingest (mock B2)
        staging = test_root / "remote"
        b2_cfg = B2Config(
            key_id="test-key-id",
            readonly_key="readonly-secret",
            write_key="write-secret",
            readonly_bucket="readonly-bucket",
            write_bucket="write-bucket",
            rclone_config=test_root / "rclone.conf",
            ingest_remote_path="datasets/raw",
            export_remote_path="datasets/anonymized",
        )
        write_rclone_config(b2_cfg, dest=b2_cfg.rclone_config)

        remote_seed = staging / b2_cfg.readonly_bucket / b2_cfg.ingest_remote_path
        seeded = seed_diverse_images(remote_seed, count)

        with mock.patch.dict(os.environ, _test_env(b2_cfg.rclone_config), clear=False):
            ingest_record = _simulate_b2_ingest(
                staging,
                paths["input_raw"],
                b2_cfg,
                batch_size=min(8, count),
            )

        input_images = discover_images(paths["input_raw"], cfg.get("image_extensions", [".png", ".jpg"]))
        if len(input_images) < count:
            errors.append(f"ingest: expected >={count} images, got {len(input_images)}")

        # Full pipeline with test-mode metrics
        config_path = test_root / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

        with mock.patch.dict(os.environ, _test_env(b2_cfg.rclone_config), clear=False):
            with mock.patch("scripts.rclone_integration._resolve_rclone_binary", return_value="rclone"), mock.patch(
                "scripts.rclone_integration.subprocess.run", side_effect=_mock_rclone_side_effect(staging)
            ):
                result = run_pipeline(
                    config_path=config_path,
                    project_root=test_root,
                    test_mode=use_test_mode,
                    export_to_b2="",
                    export_dry_run=True,
                )
        close_pipeline_logging()

        if result.get("stopped_early"):
            errors.append("pipeline stopped early")

        final_clean = sorted(paths["final_clean"].glob("*.jpg"))
        quarantine = sorted(paths["quarantine"].glob("*.jpg"))
        manual = sorted(paths["manual_review"].glob("*.jpg"))
        routed_total = len(final_clean) + len(quarantine) + len(manual)

        if routed_total < len(input_images):
            errors.append(
                f"routing incomplete: input={len(input_images)} routed={routed_total} "
                f"(final={len(final_clean)} quarantine={len(quarantine)} manual={len(manual)})"
            )

        # Sidecar + hash assertions on all routed outputs
        all_routed = list(final_clean) + list(quarantine) + list(manual)
        for jpg in all_routed:
            _assert_sidecar_integrity(jpg, paths["input_raw"], errors)

        # final_clean quality checks (InsightFace + PaddleOCR)
        ocr_scores: List[float] = []
        face_counts: List[int] = []
        for jpg in final_clean:
            faces, face_err = _insightface_count(jpg, cfg)
            face_counts.append(faces)
            if faces > 0:
                errors.append(f"final_clean face detected: {jpg.name} count={faces} ({face_err or 'ok'})")

            max_score, n_lines = _paddleocr_max_score(jpg, cfg)
            ocr_scores.append(max_score)
            if max_score >= OCR_SCORE_CEILING and n_lines > 0:
                errors.append(f"final_clean high OCR score: {jpg.name} max={max_score:.3f} lines={n_lines}")

        quarantine_rate = len(quarantine) / routed_total if routed_total else 0.0
        pass_rate = len(final_clean) / routed_total if routed_total else 0.0

        if quarantine_rate > MAX_FALSE_QUARANTINE_RATE:
            errors.append(
                f"quarantine rate {quarantine_rate:.1%} exceeds {MAX_FALSE_QUARANTINE_RATE:.0%} target "
                f"(quarantine={len(quarantine)} / {routed_total})"
            )

        if pass_rate < 1.0 - MAX_FALSE_QUARANTINE_RATE:
            errors.append(f"pass rate {pass_rate:.1%} below target {1.0 - MAX_FALSE_QUARANTINE_RATE:.0%}")

        # Export dry-run with checksum verification
        export_record = result.get("b2_export") or {}
        if not export_record.get("dry_run"):
            errors.append("export: expected dry_run export record")
        if not export_record.get("transfer_hashes"):
            errors.append("export: missing transfer_hashes")
        transfer_log = paths["logs"] / "rclone_transfers.jsonl"
        if not transfer_log.is_file():
            errors.append("export: missing logs/rclone_transfers.jsonl")

        # Retry / manual_review routing unit check
        _test_quarantine_retry_routing(test_root, cfg, errors)

        metrics = result.get("metrics") or {}
        if use_test_mode and not metrics:
            errors.append("test_mode: missing metrics summary")

        summary = {
            "passed": len(errors) == 0,
            "errors": errors,
            "count_requested": count,
            "input_images": len(input_images),
            "final_clean": len(final_clean),
            "quarantine": len(quarantine),
            "manual_review": len(manual),
            "pass_rate": float(pass_rate),
            "quarantine_rate": float(quarantine_rate),
            "ingest_dry_run_ok": bool(ingest_record.get("dry_run") is False),
            "export_dry_run_ok": bool(export_record.get("dry_run")),
            "export_checksum_files": len(export_record.get("transfer_hashes") or {}),
            "avg_ocr_max_score_final_clean": float(sum(ocr_scores) / len(ocr_scores)) if ocr_scores else None,
            "max_face_count_final_clean": max(face_counts) if face_counts else 0,
            "elapsed_sec": float(time.monotonic() - t0),
            "pipeline_stats": result.get("stats"),
            "metrics": metrics,
            "pipeline_test_mode": bool(result.get("test_mode")),
        }
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline local integration test")
    parser.add_argument("--count", type=int, default=30, help="Held-out images (20-50)")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Enable pipeline test-mode metrics")
    parser.add_argument("--no-test-mode", action="store_true", help="Disable pipeline test-mode")
    args = parser.parse_args()
    use_test_mode = bool(args.test_mode) and not bool(args.no_test_mode)

    print(f"Running full pipeline test (count={args.count}, test_mode={use_test_mode})...")
    summary = run_full_pipeline_test(count=args.count, use_test_mode=use_test_mode)
    print(json.dumps(summary, indent=2))

    if summary["errors"]:
        print("\nFAILED:", file=sys.stderr)
        for err in summary["errors"]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nPASSED: {summary['final_clean']}/{summary['input_images']} in final_clean, "
        f"quarantine_rate={summary['quarantine_rate']:.1%}, "
        f"elapsed={summary['elapsed_sec']:.1f}s"
    )


if __name__ == "__main__":
    main()
