"""
Scale test: 200–500 images with resource monitoring and batch-size tuning.

Run:
    python -m scripts.run_scale_test --count 300
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main_pipeline import run_pipeline
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_config, resolve_pipeline_paths, setup_project_folders


def _seed_scale_images(output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = ("landscape", "text", "plain", "complex", "face_like")
    for i in range(count):
        path = output_dir / f"scale_{categories[i % len(categories)]}_{i:04d}.png"
        if path.is_file():
            continue
        if categories[i % len(categories)] == "text":
            im = Image.new("RGB", (480, 320), color=(30, 30, 30))
            draw = ImageDraw.Draw(im)
            draw.text((40, 140), f"SCALE{i:04d}", fill=(240, 240, 240))
        elif categories[i % len(categories)] == "complex":
            im = Image.new("RGB", (512, 512), color=(i * 3 % 255, 50, 80))
        else:
            im = Image.new("RGB", (320, 240), color=(i * 5 % 255, 70, 90))
        im.save(path)


def run_scale_test(*, count: int, initial_batch_size: int = 16) -> dict:
    count = max(200, min(500, int(count)))
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="scale_test_") as tmp:
        test_root = Path(tmp)
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
            "batch_size": initial_batch_size,
            "max_qa_waves": 20,
            "device": "cpu",
            "insightface": {"enabled": True, "ctx_id": -1},
            "lama": {"backend": "simple_lama", "device": "cpu"},
            "monitoring": {"resource_monitoring": True},
            "qa": {
                "reuse_processing_ocr": True,
                "text_det_score_fail": 0.95,
                "artifact_ssim_min": 0.55,
            },
        }
        cfg = dict(cfg_base)
        deep_update(cfg, overrides)

        import yaml

        config_path = test_root / "config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
        setup_project_folders(test_root, cfg)
        paths = resolve_pipeline_paths(test_root, cfg)
        _seed_scale_images(paths["input_raw"], count)

        result = run_pipeline(
            config_path=config_path,
            project_root=test_root,
            test_mode=True,
        )
        close_pipeline_logging()

        final_n = len(discover_images(paths["final_clean"], [".jpg"]))
        quarantine_n = len(discover_images(paths["quarantine"], cfg.get("image_extensions", [".png", ".jpg"])))
        metrics = result.get("metrics") or {}
        batch_timings = metrics.get("batch_timings") or []

        # Auto-tune hint: if per-image time low and batch small, recommend larger batch
        avg_spi = float(metrics.get("avg_sec_per_image") or 0.0)
        recommended_batch = int(cfg.get("batch_size", initial_batch_size))
        if avg_spi < 1.0 and recommended_batch < 32:
            recommended_batch = min(32, recommended_batch * 2)

        elapsed = time.monotonic() - t0
        pass_rate = final_n / count if count else 0.0
        summary = {
            "passed": pass_rate >= 0.95 and not result.get("stopped_early"),
            "count": count,
            "final_clean": final_n,
            "quarantine": quarantine_n,
            "pass_rate": pass_rate,
            "elapsed_sec": elapsed,
            "images_per_sec": count / elapsed if elapsed > 0 else 0.0,
            "initial_batch_size": initial_batch_size,
            "recommended_batch_size": recommended_batch,
            "metrics": metrics,
            "pipeline_stats": result.get("stats"),
        }
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print(f"Scale test: {args.count} images, batch_size={args.batch_size}")
    summary = run_scale_test(count=args.count, initial_batch_size=args.batch_size)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    if summary.get("metrics"):
        print("avg_sec_per_image:", summary["metrics"].get("avg_sec_per_image"))
        print("quarantine_rate:", summary["metrics"].get("quarantine_rate"))

    if not summary.get("passed"):
        print("FAILED scale test", file=sys.stderr)
        sys.exit(1)
    print(f"PASSED scale test ({summary['final_clean']}/{summary['count']} final_clean)")


if __name__ == "__main__":
    main()
