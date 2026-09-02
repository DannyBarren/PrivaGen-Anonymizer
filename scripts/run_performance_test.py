"""
Performance benchmark: 500–1000 synthetic images with monitoring + performance_report.md.

Run:
    python -m scripts.run_performance_test --count 500
    python -m scripts.run_performance_test --count 1000 --batch-size 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main_pipeline import run_pipeline
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_config, resolve_pipeline_paths, setup_project_folders


def _seed_images(output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = ("landscape", "text", "plain", "complex", "face_like")
    for i in range(count):
        path = output_dir / f"perf_{categories[i % len(categories)]}_{i:05d}.png"
        if path.is_file():
            continue
        if categories[i % len(categories)] == "text":
            im = Image.new("RGB", (480, 320), color=(25, 25, 25))
            draw = ImageDraw.Draw(im)
            draw.text((30, 130), f"PERF{i:05d}", fill=(240, 240, 240))
        else:
            im = Image.new("RGB", (384, 288), color=(i * 2 % 255, 55, 85))
        im.save(path)


def run_performance_test(*, count: int, batch_size: int, project_root: Path | None = None) -> dict:
    count = max(1, int(count))
    if project_root is None:
        count = max(500, min(1000, count))
    test_root = Path(project_root) if project_root else ROOT
    t0 = time.monotonic()

    cfg_base = load_config(ROOT / "config.yaml")
    paths_override = {
        "input_raw": str(test_root / "input_raw"),
        "temp_processed": str(test_root / "temp_processed"),
        "final_clean": str(test_root / "final_clean"),
        "quarantine": str(test_root / "quarantine"),
        "manual_review": str(test_root / "manual_review"),
        "logs": str(test_root / "logs"),
        "reports": str(test_root / "reports"),
    }
    overrides = {
        "paths": paths_override,
        "batch_size": batch_size,
        "max_qa_waves": 30,
        "device": "cpu",
        "insightface": {"enabled": True, "ctx_id": -1},
        "lama": {"backend": "simple_lama", "device": "cpu"},
        "monitoring": {"resource_monitoring": True},
        "performance": {
            "always_monitor": True,
            "adaptive_batch_size": True,
            "ingest_screening": True,
            "monitor_threshold": 100,
        },
        "qa": {
            "reuse_processing_ocr": True,
            "text_det_score_fail": 0.95,
            "artifact_ssim_min": 0.55,
        },
        "security": {"level": "standard", "secure_wipe": False},
    }
    cfg = dict(cfg_base)
    deep_update(cfg, overrides)

    import yaml

    config_path = test_root / "config.yaml"
    if project_root is None:
        config_path = test_root / "config.performance_run.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    setup_project_folders(test_root, cfg)
    paths = resolve_pipeline_paths(test_root, cfg)
    _seed_images(paths["input_raw"], count)

    result = run_pipeline(
        config_path=config_path,
        project_root=test_root,
        test_mode=True,
        security_level="standard",
    )
    close_pipeline_logging()

    elapsed = time.monotonic() - t0
    final_n = len(discover_images(paths["final_clean"], [".jpg"]))
    metrics = result.get("metrics") or {}
    perf_report = Path(result.get("performance_report") or paths["reports"] / "performance_report.md")

    summary = {
        "passed": final_n >= count * 0.95 and not result.get("stopped_early"),
        "count": count,
        "final_clean": final_n,
        "pass_rate": final_n / count if count else 0.0,
        "elapsed_sec": elapsed,
        "images_per_sec": count / elapsed if elapsed > 0 else 0.0,
        "batch_size": batch_size,
        "adaptive_batch_used": resolve_adaptive_batch_size(cfg, count),
        "ocr_cache_hit_rate": (result.get("model_usage") or {}).get("ocr_cache_hit_rate"),
        "quarantine_rate": metrics.get("quarantine_rate"),
        "avg_sec_per_image": metrics.get("avg_sec_per_image"),
        "performance_report": str(perf_report),
        "metrics": {k: v for k, v in metrics.items() if k not in ("batch_timings", "gpu_snapshots", "resource_snapshots")},
    }
    return summary


def resolve_adaptive_batch_size(cfg, pending_count):
    from scripts.performance import resolve_adaptive_batch_size as _r

    return _r(cfg, pending_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-volume performance benchmark")
    parser.add_argument("--count", type=int, default=500, help="500–1000 images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--project-root", type=Path, default=None, help="Use project dir (persists reports)")
    args = parser.parse_args()

    print(f"Performance test: {args.count} images, batch_size={args.batch_size}")
    summary = run_performance_test(
        count=args.count,
        batch_size=args.batch_size,
        project_root=args.project_root,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    print(f"Report: {summary.get('performance_report')}")
    if not summary.get("passed"):
        sys.exit(1)
    print(f"PASSED performance test ({summary['final_clean']}/{summary['count']})")


if __name__ == "__main__":
    main()
