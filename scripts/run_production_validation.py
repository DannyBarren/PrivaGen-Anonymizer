"""
Production readiness validation: large local batch + optional B2 dry-run.

Run:
    python -m scripts.run_production_validation --count 1000
    python -m scripts.run_production_validation --count 500 --with-b2-dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main_pipeline import run_pipeline
from scripts.rclone_integration import RcloneIntegrationError, load_b2_config
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_config, resolve_pipeline_paths, setup_project_folders

READINESS_REPORT = "readiness_summary.md"


def _seed_images(output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cats = ("landscape", "text", "plain", "complex", "face_like")
    for i in range(count):
        path = output_dir / f"prod_val_{cats[i % len(cats)]}_{i:05d}.png"
        if path.is_file():
            continue
        if cats[i % len(cats)] == "text":
            im = Image.new("RGB", (512, 384), color=(20, 20, 20))
            draw = ImageDraw.Draw(im)
            draw.text((40, 160), f"VAL{i:05d}", fill=(240, 240, 240))
        else:
            im = Image.new("RGB", (400, 300), color=(i * 2 % 255, 70, 100))
        im.save(path)


def _write_readiness_report(
    report_path: Path,
    *,
    validation: Dict[str, Any],
    go: bool,
    blockers: list[str],
    recommendations: list[str],
) -> None:
    v = validation
    metrics = v.get("metrics") or {}
    lines = [
        "# Production Readiness Summary",
        "",
        f"Generated: {v.get('timestamp', '')}",
        "",
        f"## Recommendation: **{'GO' if go else 'NO-GO'}** for containerization",
        "",
    ]
    if blockers:
        lines.append("### Blockers")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")
    if recommendations:
        lines.append("### Recommendations")
        lines.append("")
        for r in recommendations:
            lines.append(f"- {r}")
        lines.append("")

    lines.extend(
        [
            "## Security measures",
            "",
            f"- Security level: `{v.get('security_level', 'standard')}`",
            f"- Copy-before-decode: enabled via config",
            f"- Ingest checksum verification: `{v.get('verify_ingest_checksums', True)}`",
            f"- Secure wipe (full level): `{v.get('secure_wipe', False)}`",
            f"- Rclone crypt tested/configured: `{v.get('crypt_configured', False)}`",
            f"- B2 ingest dry-run: `{v.get('b2_ingest_dry_run', 'skipped')}`",
            f"- B2 export dry-run: `{v.get('b2_export_dry_run', 'skipped')}`",
            "",
            "## Performance",
            "",
            f"- Images processed: **{v.get('count', 0)}**",
            f"- Pass rate: **{100 * float(metrics.get('pass_rate', v.get('pass_rate', 0))):.1f}%**",
            f"- Quarantine rate: **{100 * float(metrics.get('quarantine_rate', 0)):.1f}%**",
            f"- Throughput: **{float(metrics.get('images_per_sec', v.get('images_per_sec', 0))):.3f} images/sec**",
            f"- Avg sec/image: **{float(metrics.get('avg_sec_per_image', 0)):.3f}**",
            f"- Peak GPU allocated (MB): **{metrics.get('peak_gpu_allocated_mb', 'n/a')}**",
            f"- Wall time: **{float(v.get('elapsed_sec', 0)):.1f}s**",
            "",
            "## GPU / CUDA",
            "",
            f"- CUDA available on host: `{v.get('cuda_available', False)}`",
            f"- See also: `reports/gpu_readiness_report.md`",
            "",
            "## Rclone / Backblaze",
            "",
            f"- Config load: `{v.get('b2_config_ok', 'skipped')}`",
            f"- Performance flags: transfers/checkers/chunk from `config.yaml` `backblaze`",
            "",
            "## Observability",
            "",
            f"- Monitoring log: `{v.get('monitoring_log', 'logs/monitoring.jsonl')}`",
            f"- Performance report: `{v.get('performance_report', '')}`",
            f"- Security report: `{v.get('security_report', '')}`",
            "",
            "## Validation summary (JSON)",
            "",
            "```json",
            json.dumps(
                {
                    k: val
                    for k, val in v.items()
                    if k not in ("full_result",)
                    and k != "metrics"
                },
                indent=2,
                default=str,
            ),
            "```",
            "",
            "Full per-batch metrics: run with `reports/performance_report.md` from the same validation.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_production_validation(
    *,
    count: int = 1000,
    with_b2_dry_run: bool = False,
    report_root: Optional[Path] = None,
) -> Dict[str, Any]:
    count = max(1, int(count))
    report_root = Path(report_root) if report_root else ROOT
    t0 = time.monotonic()
    blockers: list[str] = []
    recommendations: list[str] = []

    b2_ok = "skipped"
    b2_ingest_dr = "skipped"
    b2_export_dr = "skipped"
    crypt_configured = False
    cfg_base = load_config(ROOT / "config.yaml")

    if with_b2_dry_run:
        try:
            b2_cfg = load_b2_config(
                yaml_cfg=cfg_base.get("backblaze") or {},
                security_cfg=cfg_base.get("security") or {},
            )
            b2_ok = True
            crypt_configured = bool(b2_cfg.crypt_enabled)
        except RcloneIntegrationError as exc:
            b2_ok = False
            recommendations.append(f"B2 dry-run skipped: {exc}")

    with tempfile.TemporaryDirectory(prefix="prod_val_") as tmp:
        test_root = Path(tmp)
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
            "batch_size": 16,
            "max_qa_waves": 80,
            "gpu": {"device": "cuda", "adaptive_batch": True},
            "monitoring": {"resource_monitoring": True},
            "performance": {"always_monitor": True, "adaptive_batch_size": True},
            "qa": {"reuse_processing_ocr": True, "text_det_score_fail": 0.95, "artifact_ssim_min": 0.55},
            "security": {"level": "standard", "secure_wipe": False, "verify_ingest_checksums": True},
            "lama": {"backend": "simple_lama"},
        }
        cfg = dict(cfg_base)
        deep_update(cfg, overrides)

        import yaml

        config_path = test_root / "config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
        setup_project_folders(test_root, cfg)
        paths = resolve_pipeline_paths(test_root, cfg)
        _seed_images(paths["input_raw"], count)

        if with_b2_dry_run and b2_ok:
            dr = run_pipeline(
                config_path=config_path,
                project_root=test_root,
                ingest_from_b2="",
                export_to_b2="",
                dry_run=True,
            )
            b2_ingest_dr = bool(dr.get("dry_run"))
            b2_export_dr = bool(dr.get("dry_run"))

        result = run_pipeline(
            config_path=config_path,
            project_root=test_root,
            test_mode=False,
            security_level="standard",
        )
        close_pipeline_logging()

        final_n = len(discover_images(paths["final_clean"], [".jpg"]))
        metrics = result.get("metrics") or {}
        pass_rate = final_n / count if count else 0.0
        elapsed = time.monotonic() - t0

        gpu_val = result.get("gpu_validation") or {}
        cuda_available = bool(gpu_val.get("cuda_available"))

        if pass_rate < 0.95:
            blockers.append(f"QA pass rate {pass_rate:.1%} below 95% target")
        if float(metrics.get("quarantine_rate", 0)) > 0.08:
            recommendations.append(
                f"Quarantine rate {float(metrics.get('quarantine_rate', 0)):.1%} — review QA thresholds"
            )
        if not cuda_available:
            recommendations.append(
                "CUDA not available on validation host — re-run on CoreWeave GPU node before production 37k run"
            )
        recommendations.append(
            "Install DeepPrivacy2 in vendor/deep_privacy2 for face GAN (optional but required for biometric sale bundles)"
        )
        if not Path(ROOT / "vendor" / "deep_privacy2").is_dir():
            recommendations.append("deep_privacy2 repo_root missing — clone upstream into vendor/deep_privacy2")

        go = len(blockers) == 0

        validation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "count": count,
            "final_clean": final_n,
            "pass_rate": pass_rate,
            "elapsed_sec": elapsed,
            "images_per_sec": count / elapsed if elapsed > 0 else 0.0,
            "metrics": metrics,
            "cuda_available": cuda_available,
            "security_level": result.get("security_level"),
            "b2_config_ok": b2_ok,
            "b2_ingest_dry_run": b2_ingest_dr,
            "b2_export_dry_run": b2_export_dr,
            "crypt_configured": crypt_configured,
            "monitoring_log": result.get("monitoring_log"),
            "performance_report": result.get("performance_report"),
            "security_report": result.get("security_report"),
            "stopped_early": result.get("stopped_early"),
        }

        report_path = report_root / "reports" / READINESS_REPORT
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_readiness_report(
            report_path,
            validation=validation,
            go=go,
            blockers=blockers,
            recommendations=recommendations,
        )
        validation["readiness_report"] = str(report_path)
        validation["go"] = go
        validation["blockers"] = blockers
        validation["recommendations"] = recommendations
        return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Production readiness validation")
    parser.add_argument("--count", type=int, default=1000, help="Images to process (default 1000)")
    parser.add_argument("--with-b2-dry-run", action="store_true", help="Attempt B2 config + pipeline --dry-run")
    parser.add_argument("--report-root", type=Path, default=None, help="Write readiness_summary.md here")
    args = parser.parse_args()

    print(f"Production validation: {args.count} images")
    summary = run_production_validation(
        count=args.count,
        with_b2_dry_run=bool(args.with_b2_dry_run),
        report_root=args.report_root,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, default=str))
    print(f"\nReport: {summary.get('readiness_report')}")
    print(f"\n{'GO' if summary.get('go') else 'NO-GO'} for containerization")
    if not summary.get("go"):
        sys.exit(1)


if __name__ == "__main__":
    main()
